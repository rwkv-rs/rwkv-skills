from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import runpy


MONITOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "ops"
    / "g1i_strict46"
    / "monitor_new_scores.py"
)
MONITOR = runpy.run_path(str(MONITOR_PATH), run_name="g1i_score_monitor_test")


def test_score_cursor_round_trip_uses_monotonic_score_id(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    row = {
        "score_id": 812,
        "score_created_at": datetime(2026, 8, 6, 20, 0, 1),
        "task_id": 28566,
    }

    MONITOR["_save_cursor"](state, row)

    assert MONITOR["_load_cursor"](state) == 812
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload == {
        "score_id": 812,
        "score_created_at": "2026-08-06T20:00:01",
        "task_id": 28566,
    }


def test_legacy_timestamp_cursor_is_reinitialized(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "score_created_at": "2026-08-06T20:00:01",
                "task_id": 28566,
            }
        ),
        encoding="utf-8",
    )

    assert MONITOR["_load_cursor"](state) is None


def test_new_score_query_cannot_skip_same_timestamp_rows() -> None:
    query = MONITOR["NEW_QUERY"]

    assert "s.score_id > %s" in query
    assert "ORDER BY s.score_id" in query
    assert "s.created_at >" not in query


def test_event_log_is_append_only_jsonl(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"

    MONITOR["_append_event"](
        events,
        {"event": "test", "observed_at": datetime(2026, 8, 6, 20, 0, 1)},
    )
    MONITOR["_append_event"](events, {"event": "test-2"})

    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["test", "test-2"]
    assert rows[0]["observed_at"] == "2026-08-06T20:00:01"


def test_per_score_audit_captures_protocol_and_comparison_signals(
    tmp_path: Path,
) -> None:
    aggregate_path = tmp_path / "aggregate.json"
    output_dir = tmp_path / "scores"
    aggregate_path.write_text(
        json.dumps(
            {
                "database": "audit_db",
                "valid_task_rows": [
                    {
                        "task_id": 42,
                        "domain": "coding",
                        "blank_primary_generation_count": 0,
                        "leading_orphan_close_count": 0,
                        "missing_prediction_count": 0,
                        "overall_truncation_count": 1,
                    }
                ],
                "invalid_scored_tasks": [],
                "curve_comparisons": [
                    {
                        "smaller_task_id": 41,
                        "larger_task_id": 42,
                        "investigate": True,
                    }
                ],
                "reference_comparisons": [
                    {"g1i_task_id": 42, "investigate": True}
                ],
                "choice_bias_signals": [],
                "truncation_examples_by_task": {
                    "42": [
                        {
                            "sample_index": 3,
                            "completion_tail": "repeated output",
                            "stage1_stop_reason": "max_length",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    score = {
        "score_id": 7,
        "score_created_at": datetime(2026, 8, 7, 6, 0, 0),
        "task_id": 42,
    }

    paths = MONITOR["_write_per_score_audits"](
        aggregate_path, output_dir, [score]
    )

    assert paths == [output_dir / "task_42_strict_audit.json"]
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["strict_protocol_status"] == "accepted"
    assert payload["accepted_by_strict_audit"] is True
    assert payload["investigation_signals"] == [
        "overall_truncation_count:1",
        "parameter_curve_investigation",
        "architecture_reference_investigation",
    ]
    assert payload["truncation_policy"] == {
        "scope": "evaluator_facing_final_output",
        "field": "overall_truncation_count",
        "count": 1,
    }
    assert payload["truncation_examples"] == [
        {
            "sample_index": 3,
            "completion_tail": "repeated output",
            "stage1_stop_reason": "max_length",
        }
    ]


def test_per_score_audit_handles_missing_or_malformed_truncation_examples() -> None:
    base = {
        "database": "audit_db",
        "valid_task_rows": [{"task_id": 5, "domain": "knowledge"}],
        "invalid_scored_tasks": [],
        "curve_comparisons": [],
        "reference_comparisons": [],
        "choice_bias_signals": [],
    }

    missing = MONITOR["_build_per_score_audit"](
        base,
        {"score_id": 1, "task_id": 5},
    )
    malformed = MONITOR["_build_per_score_audit"](
        {**base, "truncation_examples_by_task": {"5": "not-a-list"}},
        {"score_id": 2, "task_id": 5},
    )

    assert missing["truncation_examples"] == []
    assert malformed["truncation_examples"] == []


def test_per_score_audit_records_global_invalid_reasons() -> None:
    payload = MONITOR["_build_per_score_audit"](
        {
            "database": "audit_db",
            "valid_task_rows": [],
            "invalid_scored_tasks": [
                {
                    "task_id": 99,
                    "invalid_reasons": ["knowledge_generation_protocol"],
                }
            ],
            "curve_comparisons": [],
            "reference_comparisons": [],
            "choice_bias_signals": [],
        },
        {"score_id": 9, "task_id": 99},
    )

    assert payload["strict_protocol_status"] == "invalid"
    assert payload["accepted_by_strict_audit"] is False
    assert payload["investigation_signals"] == [
        "invalid:knowledge_generation_protocol"
    ]


def test_math_per_score_audit_only_flags_final_stage_truncation() -> None:
    payload = MONITOR["_build_per_score_audit"](
        {
            "database": "audit_db",
            "valid_task_rows": [
                {
                    "task_id": 123,
                    "domain": "math",
                    "overall_truncation_count": 161,
                    "initial_generation_truncation_count": 2017,
                    "final_stage_truncation_count": 0,
                }
            ],
            "invalid_scored_tasks": [],
            "curve_comparisons": [],
            "reference_comparisons": [],
            "choice_bias_signals": [],
        },
        {"score_id": 10, "task_id": 123},
    )

    assert payload["investigation_signals"] == []
    assert payload["truncation_policy"] == {
        "scope": "math_final_stage_only",
        "field": "final_stage_truncation_count",
        "count": 0,
    }
