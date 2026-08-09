from datetime import datetime, timedelta, timezone
import os

from ops.g1i_strict46.monitor_core20_campaign import (
    _attach_runtime_activity,
    _new_score_rows,
    _task_issues,
    _transition_events,
)


NOW = datetime(2026, 8, 9, 0, 0, tzinfo=timezone.utc)


def _row(**overrides):
    row = {
        "task_id": 30000,
        "status": "Running",
        "evaluator": "multi_choice_cot_naive",
        "task_created_at": NOW - timedelta(hours=2),
        "model_name": "rwkv7-g1i-1.5b-20260805-ctx16384",
        "benchmark_name": "mmlu",
        "benchmark_split": "test",
        "expected": 100,
        "completion_count": 50,
        "completed_completion_count": 50,
        "eval_count": 0,
        "latest_completion_at": NOW - timedelta(minutes=5),
        "latest_eval_at": None,
        "score_id": None,
        "blank_primary_count": 0,
        "missing_prediction_count": 0,
        "noncompleted_completion_count": 0,
    }
    row.update(overrides)
    return row


def _issues(*rows):
    return _task_issues(
        list(rows),
        now=NOW,
        generation_stall=timedelta(minutes=90),
        evaluation_stall=timedelta(minutes=90),
        score_stall=timedelta(minutes=30),
    )


def test_active_generation_with_recent_completion_is_healthy():
    assert _issues(_row()) == {}


def test_naive_database_timestamp_uses_server_local_timezone():
    row = _row(
        latest_completion_at=None,
        task_created_at=datetime(2026, 8, 8, 22, 0),
    )
    issues = _issues(row)
    assert issues["generation_stalled:30000"]["kind"] == "generation_stalled"


def test_runtime_log_activity_prevents_false_stall_before_db_commit(tmp_path):
    row = _row(
        completion_count=0,
        completed_completion_count=0,
        latest_completion_at=None,
        task_created_at=NOW - timedelta(hours=4),
        sampling_config={"cot_mode": "CoT"},
        benchmark_name="mmlu_pro",
        benchmark_split="test",
    )
    log = (
        tmp_path
        / "campaign"
        / "rwkv7_g1i_1_5b_20260805_ctx16384"
        / "mmlu_pro_test__cot.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text("Generating CoT", encoding="utf-8")
    recent = (NOW - timedelta(minutes=5)).timestamp()
    os.utime(log, (recent, recent))

    _attach_runtime_activity([row], run_log_root=tmp_path)

    assert _issues(row) == {}


def test_stale_runtime_log_still_reports_generation_stall(tmp_path):
    row = _row(
        completion_count=0,
        completed_completion_count=0,
        latest_completion_at=None,
        task_created_at=NOW - timedelta(hours=4),
        sampling_config={"cot_mode": "CoT"},
        benchmark_name="ceval",
        benchmark_split="test",
    )
    log = (
        tmp_path
        / "campaign"
        / "rwkv7_g1i_1_5b_20260805_ctx16384"
        / "ceval_test__cot.log"
    )
    log.parent.mkdir(parents=True)
    log.write_text("Generating CoT", encoding="utf-8")
    stale = (NOW - timedelta(minutes=91)).timestamp()
    os.utime(log, (stale, stale))

    _attach_runtime_activity([row], run_log_root=tmp_path)

    issues = _issues(row)
    assert issues["generation_stalled:30000"]["kind"] == "generation_stalled"


def test_generation_stall_uses_latest_committed_completion():
    issues = _issues(
        _row(latest_completion_at=NOW - timedelta(minutes=91))
    )
    assert issues["generation_stalled:30000"]["kind"] == "generation_stalled"


def test_evaluation_and_score_stalls_are_distinct():
    eval_issue = _issues(
        _row(
            completion_count=100,
            completed_completion_count=100,
            latest_completion_at=NOW - timedelta(minutes=91),
        )
    )
    score_issue = _issues(
        _row(
            completion_count=100,
            completed_completion_count=100,
            eval_count=100,
            latest_eval_at=NOW - timedelta(minutes=31),
        )
    )
    assert set(eval_issue) == {"evaluation_stalled:30000"}
    assert set(score_issue) == {"score_stalled:30000"}


def test_scored_task_requires_complete_nonblank_one_to_one_evidence():
    issues = _issues(
        _row(
            status="Completed",
            completion_count=100,
            completed_completion_count=99,
            eval_count=99,
            score_id=12000,
            blank_primary_count=1,
            missing_prediction_count=2,
            noncompleted_completion_count=1,
        )
    )
    reasons = issues["invalid_score_evidence:30000"]["reasons"]
    assert "completed_completions:99!=100" in reasons
    assert "evals:99!=completions:100" in reasons
    assert "blank_primary:1" in reasons
    assert "missing_prediction:2" in reasons
    assert "noncompleted_completions:1" in reasons


def test_natural_truncation_is_not_a_monitor_error():
    row = _row(
        status="Completed",
        completion_count=100,
        completed_completion_count=100,
        eval_count=100,
        score_id=12001,
        overall_truncation_count=50,
        final_stage_truncation_count=0,
    )
    assert _issues(row) == {}


def test_non_choice_missing_answer_is_valid_wrong_model_output():
    row = _row(
        status="Completed",
        evaluator="free_response_naive",
        completion_count=100,
        completed_completion_count=100,
        eval_count=100,
        score_id=12002,
        missing_prediction_count=3,
        blank_eval_answer_count=3,
    )
    assert _issues(row) == {}


def test_failed_and_completed_without_score_are_reported():
    issues = _issues(
        _row(status="Failed"),
        _row(task_id=30001, status="Completed", completion_count=100),
    )
    assert set(issues) == {"failed_task:30000", "completed_without_score:30001"}


def test_transitions_and_new_scores_are_monotonic():
    previous = {"failed_task:1": {"kind": "failed_task", "task_id": 1}}
    current = {"failed_task:2": {"kind": "failed_task", "task_id": 2}}
    events = _transition_events(previous, current)
    assert [event["event"] for event in events] == [
        "core20_issue_started",
        "core20_issue_resolved",
    ]
    assert _new_score_rows(
        [{"score_id": 7}, {"score_id": 8}, {"score_id": None}], {7}
    ) == [{"score_id": 8}]
