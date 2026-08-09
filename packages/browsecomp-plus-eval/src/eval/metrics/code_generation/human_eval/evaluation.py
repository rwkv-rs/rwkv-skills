from __future__ import annotations

from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List, Union
import itertools
import sys

import numpy as np
import tqdm

from .data import read_problems, stream_jsonl, write_jsonl
from .execution import check_correctness


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


def evaluate_functional_correctness(
    sample_file: str,
    k: Iterable[int] = (1, 10, 100),
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str | None = None,
):
    """Evaluate functional correctness of generated samples."""
    if problem_file is None:
        raise ValueError("problem_file is required for HumanEval evaluation")
    # Some HumanEval+ plus_input cases include very large integers; relax Python's
    # string-to-int guard so eval does not raise ValueError on those literals.
    if hasattr(sys, "set_int_max_str_digits"):
        sys.set_int_max_str_digits(0)  # 0 -> disable limit

    problems = read_problems(problem_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results: dict[str, list[tuple[int, dict]]] = defaultdict(list)

        for sample in tqdm.tqdm(stream_jsonl(sample_file), desc="Reading samples"):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            futures.append(executor.submit(check_correctness, *args))
            completion_id[task_id] += 1
            n_samples += 1

        if len(completion_id) != len(problems):
            missing = set(problems) - set(completion_id)
            print(f"⚠️ HumanEval 子集评测：未覆盖全部题目，缺失 {len(missing)} 个（示例 {sorted(missing)[:5]}）")

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
