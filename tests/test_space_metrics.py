from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.space.data import ScoreEntry
from src.space.metrics import _entry_method_tag, _parse_k_metric


def _score_entry(*, task: str | None, extra: dict | None = None) -> ScoreEntry:
    return ScoreEntry(
        task_id=1,
        dataset="mmlu_test",
        model="rwkv7-g1a-2.9b",
        metrics={},
        samples=5000,
        problems=5000,
        created_at=datetime(2026, 3, 30, 0, 0, 0),
        log_path="",
        cot=True,
        task=task,
        task_details=None,
        path=Path("<db>"),
        relative_path=Path("<db>"),
        domain="multi-choice系列",
        extra=extra or {},
        arch_version="rwkv7",
        data_version="g1a",
        num_params="2_9b",
    )


def test_parse_k_metric_accepts_fractional_avg_k() -> None:
    assert _parse_k_metric("avg@0.405186") == ("avg", 0.405186)
    assert _parse_k_metric("pass@8") == ("pass", 8.0)


def test_entry_method_tag_prefers_fake_cot_mode() -> None:
    entry = _score_entry(
        task="multi_choice_fake_cot",
        extra={"sampling_config": {"cot_mode": "fake_cot"}},
    )

    assert _entry_method_tag(entry) == "fake_cot"
