from __future__ import annotations

from datetime import datetime, timedelta

from ops.g1i_strict46.recompute_math_from_completions import (
    _classify_existing_replays,
    _evaluation_preflight,
    _metric_ks,
    _replayed_metrics,
    _source_completion_preflight,
)
from src.eval.metrics.free_response import (
    STRATEGY_A,
    STRATEGY_B,
    STRATEGY_C,
    FreeResponseEvaluation,
)


def test_metric_ks_falls_back_to_failed_task_sampling_protocol() -> None:
    pass_ks, avg_ks = _metric_ks(
        {},
        {
            "avg_k": 8.0,
            "pass_ks": [1, 4],
        },
    )

    assert pass_ks == (1, 4)
    assert avg_ks == (8,)


def test_metric_ks_prefers_persisted_score_metric_families() -> None:
    pass_ks, avg_ks = _metric_ks(
        {"avg@16": 0.5, "pass@2": 0.75, "exact_accuracy": 0.5},
        {"avg_k": 8.0, "pass_ks": [1]},
    )

    assert pass_ks == (2,)
    assert avg_ks == (16,)


def test_replayed_metrics_restore_avg_at_k_without_source_score() -> None:
    rows = [
        (0, 0, True),
        (0, 1, False),
        (1, 0, True),
        (1, 1, True),
    ]

    metrics = _replayed_metrics(
        rows,
        {},
        exact_accuracy=0.75,
        sampling_config={"avg_k": 2.0, "pass_ks": []},
    )

    assert metrics == {"exact_accuracy": 0.75, "avg@2": 0.75}


def test_metric_ks_ignores_invalid_task_metadata() -> None:
    pass_ks, avg_ks = _metric_ks(
        {},
        {
            "avg_k": "not-a-number",
            "pass_ks": [0, -1, None, "bad"],
        },
    )

    assert pass_ks == ()
    assert avg_ks == ()


def _evaluation(*, unresolved: int = 0) -> FreeResponseEvaluation:
    groups = (STRATEGY_A, STRATEGY_B, STRATEGY_C)
    rows = [(0, 0, True), (1, 0, False)]
    payloads = [
        {"sample_index": 0, "repeat_index": 0, "answer": "7"},
        {"sample_index": 1, "repeat_index": 0, "answer": "8"},
    ]
    return FreeResponseEvaluation(
        metrics_by_group={group: {"exact_accuracy": 0.5} for group in groups},
        rows_by_group={group: list(rows) for group in groups},
        samples=2,
        payloads=list(payloads),
        payloads_by_group={group: list(payloads) for group in groups},
        math_verify_retry_stats_by_group={
            group: {
                "attempted_count": unresolved,
                "resolved_count": 0,
                "unresolved_count": unresolved,
                "rows": (
                    [
                        {
                            "sample_index": 1,
                            "repeat_index": 0,
                            "first_fail_reason": "math_verify_timeout",
                            "retry_fail_reason": "math_verify_timeout",
                            "resolved": False,
                        }
                    ]
                    if unresolved
                    else []
                ),
            }
            for group in groups
        },
        primary_group=STRATEGY_C,
    )


def test_evaluation_preflight_accepts_complete_retry_accounting() -> None:
    preflight = _evaluation_preflight(_evaluation(), expected_rows=2)

    assert preflight["passed"] is True
    assert preflight["blockers"] == []


def test_evaluation_preflight_blocks_any_unresolved_timeout() -> None:
    preflight = _evaluation_preflight(_evaluation(unresolved=1), expected_rows=2)

    assert preflight["passed"] is False
    assert preflight["blockers"] == [
        "strategy_a.unresolved_math_verify_timeouts:1",
        "strategy_b.unresolved_math_verify_timeouts:1",
        "strategy_c.unresolved_math_verify_timeouts:1",
    ]


def test_evaluation_preflight_blocks_duplicate_coordinates() -> None:
    evaluation = _evaluation()
    evaluation.rows_by_group[STRATEGY_C][1] = (0, 0, False)

    preflight = _evaluation_preflight(evaluation, expected_rows=2)

    assert preflight["passed"] is False
    assert "strategy_c.duplicate_coordinates:1" in preflight["blockers"]
    assert "strategy_c.coordinate_set_mismatch" in preflight["blockers"]


def test_source_preflight_requires_original_complete_coordinate_grid() -> None:
    payloads = [
        {
            "sample_index": sample,
            "repeat_index": repeat,
            "pass_index": 0,
            "prompt": f"question-{sample}",
            "completion": str(sample + repeat),
            "ref_answer": str(sample),
            "context": {},
        }
        for sample in range(2)
        for repeat in range(2)
    ]

    preflight = _source_completion_preflight(
        task={
            "status": "Completed",
            "is_tmp": False,
            "is_param_search": False,
            "desc": "original=true",
            "git_hash": "a" * 40,
        },
        benchmark={"num_samples": 2},
        sampling_config={"effective_sample_count": 4, "avg_k": 2},
        payloads=payloads,
    )

    assert preflight["passed"] is True
    assert preflight["blockers"] == []
    assert len(str(preflight["ordered_payload_sha256"])) == 64


def test_source_preflight_rejects_replay_chain_without_writing() -> None:
    preflight = _source_completion_preflight(
        task={
            "status": "Completed",
            "is_tmp": False,
            "is_param_search": False,
            "desc": "replay_source_task_id=100",
            "git_hash": "a" * 40,
        },
        benchmark={"num_samples": 1},
        sampling_config={"effective_sample_count": 1, "avg_k": 1},
        payloads=[
            {
                "sample_index": 0,
                "repeat_index": 0,
                "pass_index": 0,
                "context": {},
            }
        ],
    )

    assert preflight["passed"] is False
    assert "source_task_is_replay_chain" in preflight["blockers"]


def test_existing_replay_classifier_reuses_valid_and_retries_stale_attempts() -> None:
    now = datetime(2026, 8, 8, 12, 0, 0)
    valid = {
        "task_id": 300,
        "status": "Completed",
        "task_created_at": now - timedelta(hours=1),
        "task_git_hash": "a" * 40,
        "score_id": 1,
        "completion_count": 16,
        "eval_count": 16,
    }
    state, selected = _classify_existing_replays(
        [valid],
        expected_rows=16,
        expected_git_hash="a" * 40,
        now=now,
    )
    assert state == "valid"
    assert selected == valid

    pending = {
        **valid,
        "task_id": 301,
        "status": "Running",
        "task_created_at": now - timedelta(minutes=2),
        "score_id": None,
        "completion_count": 0,
        "eval_count": 0,
    }
    state, selected = _classify_existing_replays(
        [pending],
        expected_rows=16,
        expected_git_hash="a" * 40,
        now=now,
    )
    assert state == "pending"
    assert selected == pending

    pending["task_created_at"] = now - timedelta(hours=1)
    state, selected = _classify_existing_replays(
        [pending],
        expected_rows=16,
        expected_git_hash="a" * 40,
        now=now,
    )
    assert state == "retry"
    assert selected is None
