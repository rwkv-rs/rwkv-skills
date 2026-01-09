from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import orjson
import tqdm

from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.results.schema import make_eval_payload
from .execution import check_correctness


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


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


def evaluate_livecodebench_dataset(
    completions_path: str | Path,
    *,
    dataset_path: str | Path,
    eval_output_path: str | Path,
    pass_k: Iterable[int] = (1, 5),
    n_workers: int = 4,
    timeout: float = 6.0,
) -> dict[str, float]:
    dataset_records = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    sample_map: dict[int, dict] = {}
    ref_map: dict[int, str] = {}
    for idx, record in enumerate(dataset_records):
        sample, ref_answer = _build_lcb_sample(record)
        sample_map[idx] = sample
        ref_map[idx] = ref_answer

    results_by_key: dict[tuple[int, int], dict] = {}
    grouped: dict[int, list[bool]] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for payload in tqdm.tqdm(_iter_jsonl(completions_path), desc="Reading completions"):
            sample_index = int(payload.get("sample_index", 0))
            repeat_index = int(payload.get("repeat_index", 0))
            last_stage = _max_stage_index(payload)
            completion = str(payload.get(f"completion{last_stage}", "") or "")
            code = _extract_code(completion)
            sample = sample_map.get(
                sample_index,
                {"input_output": json.dumps({"inputs": [], "outputs": [], "fn_name": None})},
            )
            futures.append(
                executor.submit(
                    check_correctness,
                    sample_index,
                    repeat_index,
                    sample,
                    code,
                    timeout,
                )
            )

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Running test suites"):
            result = future.result()
            key = (int(result["sample_index"]), int(result["repeat_index"]))
            results_by_key[key] = result
            grouped[int(result["sample_index"])].append(bool(result["passed"]))

    total, correct = [], []
    for flags in grouped.values():
        total.append(len(flags))
        correct.append(sum(1 for flag in flags if flag))
    total_arr = np.array(total)
    correct_arr = np.array(correct)

    pass_at_k: dict[str, float] = {}
    for val in pass_k:
        mask = total_arr >= val
        if not mask.any():
            continue
        pass_at_k[f"pass@{val}"] = estimate_pass_at_k(total_arr[mask], correct_arr[mask], int(val)).mean()

    eval_output_path = Path(eval_output_path)
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_output_path.open("wb") as out_f:
        for payload in _iter_jsonl(completions_path):
            sample_index = int(payload.get("sample_index", 0))
            repeat_index = int(payload.get("repeat_index", 0))
            last_stage = _max_stage_index(payload)
            completion = str(payload.get(f"completion{last_stage}", "") or "")
            code = _extract_code(completion)
            result = results_by_key.get((sample_index, repeat_index))
            passed = bool(result.get("passed")) if result else False
            fail_reason = ""
            if result and not passed:
                fail_reason = str(result.get("result") or "")
            eval_payload = make_eval_payload(
                payload,
                is_passed=passed,
                fail_reason=fail_reason,
                answer=code,
                ref_answer=ref_map.get(sample_index, ""),
            )
            out_f.write(orjson.dumps(eval_payload, option=orjson.OPT_APPEND_NEWLINE))

    return pass_at_k


__all__ = ["evaluate_livecodebench_dataset", "estimate_pass_at_k"]
