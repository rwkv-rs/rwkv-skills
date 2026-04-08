from __future__ import annotations

import pytest

from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k


def test_compute_pass_at_k_deduplicates_resume_rows_by_sample_and_repeat() -> None:
    rows = [
        (0, 0, True),
        (0, 0, False),
        (0, 1, False),
        (1, 0, False),
        (1, 1, False),
    ]

    metrics = compute_pass_at_k(rows, (1, 2))

    assert metrics["pass@1"] == pytest.approx(0.25)
    assert metrics["pass@2"] == pytest.approx(0.5)


def test_compute_avg_at_k_deduplicates_resume_rows_by_sample_and_repeat() -> None:
    rows = [
        (0, 0, True),
        (0, 0, False),
        (0, 1, False),
        (1, 0, False),
        (1, 1, False),
    ]

    metrics = compute_avg_at_k(rows, (0.5, 1))

    assert metrics["avg@0.5"] == pytest.approx(0.25)
    assert metrics["avg@1"] == pytest.approx(0.5)
