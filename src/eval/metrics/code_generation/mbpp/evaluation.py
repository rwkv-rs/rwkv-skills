from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Union
import itertools
import os
from pathlib import Path

import numpy as np
import tqdm

from src.eval.metrics.code_generation.human_eval.data import stream_jsonl, write_jsonl, read_problems
from .execution import check_correctness


_MP_ARENA_DIRS_SANITIZED = False


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """Estimate pass@k of each problem and return as array."""

    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_mbpp(
    sample_file: str,
    k: Iterable[int] = (1, 10, 100),
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str | None = None,
):
    """Evaluate MBPP completions.

    - 对原始 MBPP（含 assertion/test_list）的数据，用内嵌断言执行。
    - 若数据包含 EvalPlus 的 base_input/plus_input，则按 EvalPlus 方式分别跑 base+plus 测试。
    """
    if problem_file is None:
        raise ValueError("problem_file is required for MBPP evaluation")

    problems = {str(key): value for key, value in read_problems(problem_file).items()}
    for task_id, problem in problems.items():
        problem["task_id"] = str(task_id)
    has_plus_inputs = any("plus_input" in prob for prob in problems.values())
    if has_plus_inputs:
        # MBPP+ 的 base_input/plus_input 在 EvalPlus 原始数据中包含 tuple/set/complex 等非 JSON 类型。
        # 我们保存为 jsonl 后会丢失这些类型信息（tuple -> list 等），需要在评测前反序列化恢复。
        from evalplus.data.mbpp import mbpp_deserialize_inputs

        for task_id, problem in problems.items():
            if "base_input" in problem:
                problem["base_input"] = mbpp_deserialize_inputs(task_id, problem.get("base_input") or [])
            if "plus_input" in problem:
                problem["plus_input"] = mbpp_deserialize_inputs(task_id, problem.get("plus_input") or [])
        return _evaluate_mbpp_plus(sample_file, k=k, n_workers=n_workers, timeout=timeout, problems=problems, problem_file=problem_file)

    return _evaluate_mbpp_base(sample_file, k=k, n_workers=n_workers, timeout=timeout, problems=problems)


def _evaluate_mbpp_base(
    sample_file: str,
    *,
    k: Iterable[int],
    n_workers: int,
    timeout: float,
    problems: dict[str, dict],
):
    """Original MBPP evaluation using embedded assertions."""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results: dict[str, list[tuple[int, dict]]] = defaultdict(list)

        for sample in tqdm.tqdm(stream_jsonl(sample_file), desc="Reading samples"):
            task_id = str(sample["task_id"])
            completion = sample["completion"]
            problem = problems[task_id]
            args = (problem, completion, timeout, completion_id[task_id])
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        if len(completion_id) != len(problems):
            missing = set(problems) - set(completion_id)
            print(f"⚠️ MBPP 子集评测：未覆盖全部题目，缺失 {len(missing)} 个（示例 {sorted(missing)[:5]}）")

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Running test suites"):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total_arr = np.array(total)
    correct_arr = np.array(correct)

    pass_at_k: dict[str, float] = {}
    for val in k:
        mask = total_arr >= val
        if not mask.any():
            continue
        pass_at_k[f"pass@{val}"] = estimate_pass_at_k(total_arr[mask], correct_arr[mask], val).mean()

    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    write_jsonl(out_file, combine_results())

    return pass_at_k, out_file


def _prepare_evalplus_cache(problem_file: str) -> Path:
    """Ensure EvalPlus uses a writable cache (defaults to repo data/cache)."""

    cache_root = Path(problem_file).expanduser().resolve().parent.parent / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("EVALPLUS_CACHE", str(cache_root))
    import evalplus.data.utils as evalplus_data_utils

    evalplus_data_utils.CACHE_DIR = str(cache_root)
    # Ground-truth cache lives in evalplus.evaluate.CACHE_DIR (not evalplus.data.utils.CACHE_DIR).
    import evalplus.evaluate as evalplus_evaluate

    evalplus_evaluate.CACHE_DIR = str(cache_root)
    return cache_root


def _sanitize_mp_arena_dir_candidates() -> None:
    """Avoid using /dev/shm when it's blocked (e.g. sandboxed environments)."""

    global _MP_ARENA_DIRS_SANITIZED
    if _MP_ARENA_DIRS_SANITIZED:
        return

    import multiprocessing.heap as mp_heap

    if hasattr(mp_heap, "Arena") and hasattr(mp_heap.Arena, "_dir_candidates"):
        # 避免在受限环境写 /dev/shm；直接强制落到 util.get_temp_dir() (/tmp)。
        mp_heap.Arena._dir_candidates = []
    _MP_ARENA_DIRS_SANITIZED = True


def _untrusted_check_mbpp(
    solution: str,
    *,
    inputs: list,
    entry_point: str,
    expected: list,
    atol: float,
    ref_time: list[float],
    timeout: float,
) -> tuple[str, np.ndarray]:
    """EvalPlus `untrusted_check` without relying on buggy env parsing / semlocks."""

    import multiprocessing
    from multiprocessing.sharedctypes import RawArray, RawValue

    from evalplus.eval import FAIL, PASS, TIMEOUT, _UNKNOWN, _mapping, unsafe_execute

    _sanitize_mp_arena_dir_candidates()

    time_limits = [max(0.1, 2.0 * t) for t in ref_time]
    overall_timeout = min(float(timeout), sum(time_limits)) + 1

    progress = RawValue("i", 0)
    stat = RawValue("i", _UNKNOWN)
    details = RawArray("b", [0 for _ in range(len(inputs))])

    proc = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            "mbpp",
            entry_point,
            solution,
            inputs,
            expected,
            time_limits,
            atol,
            True,  # fast_check
            stat,
            details,
            progress,
        ),
    )
    proc.start()
    proc.join(timeout=overall_timeout + 1)
    if proc.is_alive():
        proc.terminate()
        import time as _time

        _time.sleep(0.1)
    if proc.is_alive():
        proc.kill()
        import time as _time

        _time.sleep(0.1)

    status = _mapping.get(stat.value)
    detail_slice = details[: progress.value]
    detail_array = np.asarray(detail_slice, dtype=bool)

    if not status:
        status = TIMEOUT
    if status == PASS:
        if len(detail_array) != len(inputs) or not bool(detail_array.all()):
            status = FAIL

    return status, detail_array


def _evaluate_mbpp_plus(
    sample_file: str,
    *,
    k: Iterable[int],
    n_workers: int,
    timeout: float,
    problems: dict[str, dict],
    problem_file: str,
):
    """EvalPlus MBPP+ evaluation requiring both base_input 与 plus_input 通过。"""

    _prepare_evalplus_cache(problem_file)

    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    import hashlib

    with Path(problem_file).expanduser().resolve().open("rb") as handle:
        dataset_hash = hashlib.md5(handle.read()).hexdigest()
    expected_output = get_groundtruth(problems, dataset_hash, MBPP_OUTPUT_NOT_NONE_TASKS)

    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")

    futures = []
    completion_id = Counter()
    results: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    payloads: list[tuple[str, int, dict, str, dict, float]] = []
    for sample in tqdm.tqdm(stream_jsonl(sample_file), desc="Reading samples"):
        task_id = sample["task_id"]
        completion = sample.get("completion") or sample.get("solution") or ""
        if task_id not in problems:
            continue
        problem = problems[task_id]
        solution = completion
        payloads.append(
            (
                task_id,
                completion_id[task_id],
                problem,
                solution,
                expected_output[task_id],
                timeout,
            )
        )
        completion_id[task_id] += 1

    if len(completion_id) != len(problems):
        missing = set(problems) - set(completion_id)
        print(f"⚠️ MBPP+ 子集评测：未覆盖全部题目，缺失 {len(missing)} 个（示例 {sorted(missing)[:5]}）")

    if n_workers == 1:
        for args in tqdm.tqdm(payloads, total=len(payloads), desc="Running MBPP+ tests"):
            result = _mbpp_plus_check(*args)
            results[result["task_id"]].append((result["completion_id"], result))
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for args in payloads:
                futures.append(executor.submit(_mbpp_plus_check, *args))
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Running MBPP+ tests"):
                result = future.result()
                results[result["task_id"]].append((result["completion_id"], result))

    total, base_correct, plus_correct = [], [], []
    for result in results.values():
        result.sort()
        base_passed = [r[1]["base_passed"] for r in result]
        plus_passed = [r[1]["plus_passed"] for r in result]
        total.append(len(base_passed))
        base_correct.append(sum(base_passed))
        plus_correct.append(sum(bp and pp for bp, pp in zip(base_passed, plus_passed)))

    total_arr = np.array(total)
    base_arr = np.array(base_correct)
    plus_arr = np.array(plus_correct)

    pass_at_k: dict[str, float] = {}
    for val in k:
        mask = total_arr >= val
        if not mask.any():
            continue
        # MBPP+ pass@k 需要 base+plus 均通过
        pass_at_k[f"pass@{val}"] = estimate_pass_at_k(total_arr[mask], plus_arr[mask], val).mean()
        pass_at_k[f"base_pass@{val}"] = estimate_pass_at_k(total_arr[mask], base_arr[mask], val).mean()

    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            items = results.get(task_id)
            if not items:
                yield sample
                continue
            result = items.pop(0)[1]
            sample["base_status"] = result["base_status"]
            sample["plus_status"] = result["plus_status"]
            sample["base_passed"] = result["base_passed"]
            sample["plus_passed"] = result["plus_passed"]
            sample["passed"] = result["base_passed"] and result["plus_passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    write_jsonl(out_file, combine_results())

    return pass_at_k, out_file


def _mbpp_plus_check(
    task_id: str,
    completion_id: int,
    problem: dict,
    solution: str,
    expected_output: dict,
    timeout: float,
):
    from evalplus.eval import PASS

    base_inputs = problem.get("base_input", []) or []
    plus_inputs = problem.get("plus_input", []) or []
    entry_point = problem.get("entry_point", "") or ""
    atol = float(problem.get("atol", 0.0) or 0.0)

    base_stat, base_details = _untrusted_check_mbpp(
        solution,
        inputs=base_inputs,
        entry_point=entry_point,
        expected=expected_output["base"],
        atol=atol,
        ref_time=expected_output["base_time"],
        timeout=timeout,
    )
    plus_stat, plus_details = _untrusted_check_mbpp(
        solution,
        inputs=plus_inputs,
        entry_point=entry_point,
        expected=expected_output["plus"],
        atol=atol,
        ref_time=expected_output["plus_time"],
        timeout=timeout,
    )

    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "base_status": base_stat,
        "plus_status": plus_stat,
        "base_passed": base_stat == PASS,
        "plus_passed": plus_stat == PASS,
        "base_details": base_details,
        "plus_details": plus_details,
    }
