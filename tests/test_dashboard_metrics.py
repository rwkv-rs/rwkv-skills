from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.dashboard.core.data import ScoreEntry
from src.dashboard.core.domains import DOMAIN_CODING, DOMAIN_MMLU
from src.dashboard.core.metrics import (
    _detail_rows_for_entry,
    _display_metric_from_context,
    _primary_metric,
)


def _entry(
    *,
    dataset: str,
    task: str,
    metrics: dict[str, object],
    sampling_config: dict[str, object],
    domain: str,
    cot: bool = False,
) -> ScoreEntry:
    return ScoreEntry(
        task_id=1,
        dataset=dataset,
        model="rwkv7-g1x-1.5b-test",
        metrics=metrics,
        samples=1,
        problems=1,
        created_at=datetime(2026, 1, 1),
        log_path="",
        cot=cot,
        task=task,
        task_details=None,
        path=Path("<test>"),
        relative_path=Path("<test>"),
        domain=domain,
        extra={"sampling_config": sampling_config, "cot_mode": "cot" if cot else "no_cot"},
        arch_version="RWKV7",
        data_version="G1X",
        num_params="1_5b",
    )


def test_display_metric_uses_db_sampling_avg_k() -> None:
    key, value = _display_metric_from_context(
        {"accuracy": 0.5106, "avg@0.356075": 0.5106},
        sampling_config={"avg_k": 0.35607463324312777},
    )

    assert (key, value) == ("avg@0.356075", 0.5106)


def test_detail_row_uses_avg_metric_from_db_sampling() -> None:
    entry = _entry(
        dataset="human_eval_test",
        task="code_human_eval_naive",
        metrics={"avg@32": 0.5682164634146342},
        sampling_config={"avg_k": 32.0},
        domain=DOMAIN_CODING,
    )

    assert _detail_rows_for_entry(entry) == [
        ("human_eval_nocot", "exact_match", "avg@32", 0.5682164634146342)
    ]


def test_count_only_metrics_are_not_display_scores() -> None:
    metrics = {"swebench_harness_ran": 0.0, "swebench_predictions": 2294.0}

    assert _display_metric_from_context(metrics, sampling_config={"avg_k": 1.0}) == (None, None)
    assert _primary_metric(metrics) is None


def test_avg_metric_wins_over_accuracy_for_multi_choice() -> None:
    entry = _entry(
        dataset="mmlu_test",
        task="multi_choice_plain",
        metrics={"accuracy": 0.5106, "avg@0.356075": 0.5106},
        sampling_config={"avg_k": 0.35607463324312777},
        domain=DOMAIN_MMLU,
    )

    assert _detail_rows_for_entry(entry) == [
        ("mmlu_nocot", "logits", "avg@0.356075", 0.5106)
    ]
