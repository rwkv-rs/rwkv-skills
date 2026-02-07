from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Callable, Iterable, List, Union

import numpy as np
import tqdm

from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.results.schema import make_eval_payload
from .execution import check_correctness


def _parse_index(value: Any, name: str) -> int:
    """解析 sample_index/repeat_index，失败时抛出异常而非默认值。"""
    if value is None:
        raise ValueError(f"{name} is required but got None")
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}: {value}") from e


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_completions(source: Iterable[dict] | str | Path) -> Iterable[dict]:
    if isinstance(source, (str, Path)):
        yield from _iter_jsonl(source)
        return
    yield from source


def _max_stage_index(payload: dict) -> int:
    stage = 0
    for key in payload:
        if key.startswith("completion") and key.removeprefix("completion").isdigit():
            stage = max(stage, int(key.removeprefix("completion")))
    return stage


def _extract_code(text: str) -> str:
    if not text:
        return ""
    if "```" not in text:
        return text.strip()
    parts = text.split("```")
    if len(parts) < 3:
        return text.strip()
    block = parts[-2]
    lines = block.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py"}:
        block = "\n".join(lines[1:])
    return block.strip()


def _parse_json_maybe(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _decode_private_tests(raw: str) -> list[dict]:
    import base64
    import pickle
    import zlib

    try:
        payload = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
        return json.loads(payload)
    except Exception:
        return []


def _parse_tests(raw: object) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return _decode_private_tests(raw)
    return []


def _build_lcb_sample(record: CodeGenerationRecord) -> tuple[dict, str]:
    metadata_raw = record.metadata.get("metadata")
    metadata = _parse_json_maybe(metadata_raw)
    if not isinstance(metadata, dict):
        metadata = {}

    public_tests = _parse_tests(record.metadata.get("public_test_cases"))
    private_tests = _parse_tests(record.metadata.get("private_test_cases"))
    tests = public_tests + private_tests
    inputs = [str(item.get("input", "")) for item in tests]
    outputs = [str(item.get("output", "")) for item in tests]
    sample = {
        "input_output": json.dumps(
            {
                "inputs": inputs,
                "outputs": outputs,
                "fn_name": metadata.get("func_name"),
            }
        )
    }
    return sample, sample["input_output"]


def _normalize_difficulty(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"easy", "medium", "hard"}:
        return lowered
    return None


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    def estimator(n: int, c: int, val: int) -> float:
        if n - c < val:
            return 1.0
        return 1.0 - np.prod(1.0 - val / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_livecodebench_streaming(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    pass_k: Iterable[int] = (1, 5),
    n_workers: int = 4,
    timeout: float = 6.0,
    on_eval: Callable[[dict], None] | None = None,
    max_pending: int = 256,
) -> dict[str, float]:
    """流式评测 LiveCodeBench，边评测��通过 on_eval 回调写入结果。

    Args:
        completions: completion payloads 的迭代器或 JSONL 文件路径
        dataset_path: 数据集 JSONL 路径
        pass_k: 要计算的 pass@k 值
        n_workers: 并行执行测试的线程数
        timeout: 每个测试的超时时间
        on_eval: 每条 eval 结果的回调函数，用于流式写入数据库
        max_pending: 最大待处理任务数，控制内存使用

    Returns:
        pass@k 指标字典
    """
    pass_k_values = tuple(int(val) for val in pass_k)
    dataset_records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    sample_map: dict[int, dict] = {}
    ref_map: dict[int, str] = {}
    difficulty_map: dict[int, str] = {}
    for idx, record in enumerate(dataset_records):
        sample, ref_answer = _build_lcb_sample(record)
        sample_map[idx] = sample
        ref_map[idx] = ref_answer
        difficulty = _normalize_difficulty(record.metadata.get("difficulty"))
        if difficulty:
            difficulty_map[idx] = difficulty

    # 去重：记录已处理的 (sample_index, repeat_index)
    seen_keys: set[tuple[int, int]] = set()
    grouped: dict[int, list[bool]] = defaultdict(list)

    # 流式处理：使用滑动窗口控制并发
    pending: dict[int, tuple[int, int, str, str]] = {}  # future_id -> (sample_index, repeat_index, code, ref_answer)
    future_id = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures: dict[Any, int] = {}
        completion_iter = iter(_iter_completions(completions))
        exhausted = False

        with tqdm.tqdm(desc="Running test suites") as pbar:
            while True:
                # 填充任务队列直到达到 max_pending 或输入耗尽
                while len(futures) < max_pending and not exhausted:
                    try:
                        payload = next(completion_iter)
                    except StopIteration:
                        exhausted = True
                        break

                    sample_index = _parse_index(payload.get("sample_index"), "sample_index")
                    repeat_index = _parse_index(payload.get("repeat_index"), "repeat_index")

                    # 去重
                    key = (sample_index, repeat_index)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    last_stage = _max_stage_index(payload)
                    completion = str(payload.get(f"completion{last_stage}", "") or "")
                    code = _extract_code(completion)
                    sample = sample_map.get(
                        sample_index,
                        {"input_output": json.dumps({"inputs": [], "outputs": [], "fn_name": None})},
                    )

                    future = executor.submit(
                        check_correctness,
                        sample_index,
                        repeat_index,
                        sample,
                        code,
                        timeout,
                    )
                    futures[future] = future_id
                    pending[future_id] = (sample_index, repeat_index, code, ref_map.get(sample_index, ""))
                    future_id += 1

                # 如果没有待处理任务，退出
                if not futures:
                    break

                # 等待至少一个任务完成
                done_futures = []
                for future in as_completed(futures):
                    done_futures.append(future)
                    # 每完成一个就处理，保持流式
                    break

                for future in done_futures:
                    fid = futures.pop(future)
                    sample_index, repeat_index, code, ref_answer = pending.pop(fid)

                    result = future.result()
                    passed = bool(result.get("passed"))
                    grouped[sample_index].append(passed)

                    # 流式回调写入 eval
                    if on_eval is not None:
                        fail_reason = ""
                        if not passed:
                            fail_reason = str(result.get("result") or "")
                        eval_payload = make_eval_payload(
                            {"sample_index": sample_index, "repeat_index": repeat_index},
                            is_passed=passed,
                            fail_reason=fail_reason,
                            answer=code,
                            ref_answer=ref_answer,
                        )
                        on_eval(eval_payload)

                    processed += 1
                    pbar.update(1)

    # 计算 pass@k
    total, correct = [], []
    for flags in grouped.values():
        total.append(len(flags))
        correct.append(sum(1 for flag in flags if flag))
    total_arr = np.array(total)
    correct_arr = np.array(correct)

    pass_at_k: dict[str, float] = {}
    for val in pass_k_values:
        mask = total_arr >= val
        if not mask.any():
            continue
        pass_at_k[f"pass@{val}"] = estimate_pass_at_k(total_arr[mask], correct_arr[mask], int(val)).mean()

    # 按难度计算 pass@k
    difficulty_totals: dict[str, list[int]] = defaultdict(list)
    difficulty_corrects: dict[str, list[int]] = defaultdict(list)
    for sample_index, flags in grouped.items():
        difficulty = difficulty_map.get(sample_index)
        if not difficulty:
            continue
        difficulty_totals[difficulty].append(len(flags))
        difficulty_corrects[difficulty].append(sum(1 for flag in flags if flag))

    for difficulty in ("easy", "medium", "hard"):
        totals = np.array(difficulty_totals.get(difficulty, []))
        corrects = np.array(difficulty_corrects.get(difficulty, []))
        if totals.size == 0:
            continue
        for val in pass_k_values:
            mask = totals >= val
            if not mask.any():
                continue
            key = f"pass@{val}_{difficulty}"
            pass_at_k[key] = estimate_pass_at_k(totals[mask], corrects[mask], int(val)).mean()

    return pass_at_k


def evaluate_livecodebench_dataset(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    pass_k: Iterable[int] = (1, 5),
    n_workers: int = 4,
    timeout: float = 6.0,
) -> tuple[dict[str, float], list[dict]]:
    """评测 LiveCodeBench 数据集（兼容旧接口，返回 eval_payloads 列表）。

    注意：此函数会将所有 eval_payloads 收集到内存中，大数据集建议使用
    evaluate_livecodebench_streaming 配合流式写入。
    """
    eval_payloads: list[dict] = []

    def collect_eval(payload: dict) -> None:
        eval_payloads.append(payload)

    metrics = evaluate_livecodebench_streaming(
        completions,
        dataset_path=dataset_path,
        pass_k=pass_k,
        n_workers=n_workers,
        timeout=timeout,
        on_eval=collect_eval,
    )
    return metrics, eval_payloads


__all__ = ["evaluate_livecodebench_dataset", "evaluate_livecodebench_streaming", "estimate_pass_at_k"]
