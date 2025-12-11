from __future__ import annotations

import unittest

from src.eval.metrics.free_response import (
    FreeResponseSample,
    FreeResponseSampleResult,
    compute_avg_at_k,
    compute_pass_at_k,
)


def _make_sample(
    problem_idx: int,
    correct: bool,
    *,
    judge: bool | None = None,
    sample_id: int = 0,
) -> FreeResponseSampleResult:
    sample = FreeResponseSample(
        sample_index=problem_idx,
        dataset="demo",
        question="",
        answer="",
        prediction="",
        subject=None,
        cot=None,
        problem_index=problem_idx,
        sample_id=sample_id,
    )
    result = FreeResponseSampleResult(sample=sample, correct_exact=correct)
    if judge is not None:
        result.judge_correct = judge
    return result


class TestFreeResponsePassK(unittest.TestCase):
    def test_compute_pass_at_k_exact(self) -> None:
        samples: list[FreeResponseSampleResult] = []
        samples.append(_make_sample(0, True))
        samples.append(_make_sample(0, True))
        samples.append(_make_sample(1, False))
        samples.append(_make_sample(1, False))
        metrics = compute_pass_at_k(samples, (1, 2))
        self.assertEqual(metrics, {"pass@1": 0.5, "pass@2": 0.5})

    def test_compute_pass_at_k_with_judge_overrides(self) -> None:
        samples = [_make_sample(0, False, judge=True) for _ in range(2)]
        metrics = compute_pass_at_k(samples, (1,), use_judge=True)
        self.assertEqual(metrics["pass@1"], 1.0)

    def test_compute_pass_at_k_skips_without_problem_index(self) -> None:
        sample = _make_sample(0, True)
        sample.sample.problem_index = None
        metrics = compute_pass_at_k([sample], (1, 2))
        self.assertEqual(metrics, {"pass@1": 1.0})


class TestFreeResponseAvgK(unittest.TestCase):
    def test_compute_avg_at_k_exact(self) -> None:
        samples = [
            _make_sample(0, True, sample_id=0),
            _make_sample(0, False, sample_id=1),
            _make_sample(1, False, sample_id=0),
            _make_sample(1, False, sample_id=1),
        ]
        metrics = compute_avg_at_k(samples, (1, 2))
        self.assertEqual(metrics, {"avg@1": 0.5, "avg@2": 0.25})

    def test_compute_avg_at_k_uses_judge(self) -> None:
        samples = [
            _make_sample(0, False, judge=True, sample_id=0),
            _make_sample(0, False, judge=False, sample_id=1),
        ]
        metrics = compute_avg_at_k(samples, (2,), use_judge=True)
        self.assertEqual(metrics["avg@2"], 0.5)
