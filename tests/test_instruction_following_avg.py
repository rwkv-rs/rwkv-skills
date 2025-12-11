from __future__ import annotations

import unittest

from src.eval.metrics.instruction_following.metrics import (
    InstructionFollowingSample,
    InstructionFollowingSampleResult,
    compute_avg_at_k,
)


def _make_sample(key: int, follow_all: bool, sample_id: int) -> InstructionFollowingSampleResult:
    sample = InstructionFollowingSample(
        key=key,
        prompt="",
        response="",
        instruction_ids=[],
        kwargs_list=[],
        sample_id=sample_id,
    )
    return InstructionFollowingSampleResult(sample=sample, follow_instruction_list=[follow_all])


class TestInstructionFollowingAvg(unittest.TestCase):
    def test_compute_avg_at_k(self) -> None:
        samples = [
            _make_sample(0, True, 0),
            _make_sample(0, False, 1),
            _make_sample(1, False, 0),
            _make_sample(1, False, 1),
        ]
        metrics = compute_avg_at_k(samples, (1, 2))
        self.assertEqual(metrics, {"avg@1": 0.5, "avg@2": 0.25})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
