from __future__ import annotations

import unittest

from src.eval.execution_plan import (
    avg_k_metric_key,
    build_auto_avg_k_execution_plan,
    build_avg_k_execution_plan,
)
from src.eval.metrics.at_k import compute_avg_at_k


class ExecutionPlanTests(unittest.TestCase):
    def test_auto_plan_repeats_small_dataset_to_reach_target(self) -> None:
        plan = build_auto_avg_k_execution_plan("mmlu_test", 448)
        self.assertEqual(plan.avg_k, 12.0)
        self.assertEqual(plan.repeat_count, 12)
        self.assertEqual(plan.sample_size, 448)
        self.assertEqual(plan.effective_sample_count, 5376)
        self.assertEqual(plan.sample_indices[0], 0)
        self.assertEqual(plan.sample_indices[-1], 447)

    def test_auto_plan_subsamples_large_dataset_deterministically(self) -> None:
        first = build_auto_avg_k_execution_plan("mmlu_test", 6000)
        second = build_auto_avg_k_execution_plan("mmlu_test", 6000)
        self.assertAlmostEqual(first.avg_k, 5000 / 6000)
        self.assertEqual(first.repeat_count, 1)
        self.assertEqual(first.sample_size, 5000)
        self.assertEqual(first.sample_indices, second.sample_indices)
        self.assertEqual(first.sample_indices, tuple(sorted(first.sample_indices)))
        self.assertGreaterEqual(first.sample_indices[0], 0)
        self.assertLess(first.sample_indices[-1], 6000)

    def test_manual_ratio_plan_uses_float_metric_key(self) -> None:
        plan = build_avg_k_execution_plan("gpqa_test", 6000, avg_k=5000 / 6000)
        rows = [
            (plan.sample_indices[0], 0, True),
            (plan.sample_indices[1], 0, False),
        ]
        metrics = compute_avg_at_k(rows, (plan.avg_k,))
        self.assertEqual(avg_k_metric_key(plan.avg_k), "avg@0.833333")
        self.assertEqual(metrics, {"avg@0.833333": 0.5})


if __name__ == "__main__":
    unittest.main()
