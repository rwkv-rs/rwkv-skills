from datetime import datetime, timedelta, timezone

from ops.g1i_strict46.monitor_runtime_health import (
    _health_issues,
    _scheduler_process_issues,
    _transition_events,
    _without_baseline_failures,
)


NOW = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)


def _row(**overrides):
    row = {
        "task_id": 10,
        "model_name": "rwkv7-g1i-2.9b-20260805-ctx16384",
        "benchmark_name": "aime25",
        "benchmark_split": "test",
        "status": "Running",
        "task_created_at": NOW - timedelta(hours=2),
        "latest_completion_at": NOW - timedelta(minutes=5),
        "expected": 100,
        "completions": 50,
    }
    row.update(overrides)
    return row


def test_healthy_running_task_has_no_issue():
    issues = _health_issues(
        [_row()],
        {"rwkv7-g1i-2.9b-20260805-ctx16384": {"ok": True}},
        now=NOW,
        stall_after=timedelta(minutes=75),
    )
    assert issues == {}


def test_stall_and_endpoint_mismatch_are_reported_independently():
    issues = _health_issues(
        [_row(latest_completion_at=NOW - timedelta(minutes=90))],
        {
            "rwkv7-g1i-2.9b-20260805-ctx16384": {
                "ok": False,
                "error": "requested model is not served",
                "served_models": ["some-other-model"],
            }
        },
        now=NOW,
        stall_after=timedelta(minutes=75),
    )
    assert set(issues) == {
        "stalled_task:10",
        "endpoint:rwkv7-g1i-2.9b-20260805-ctx16384",
    }


def test_failed_task_is_reported_without_an_endpoint_probe_requirement():
    issues = _health_issues(
        [_row(status="Failed")],
        {},
        now=NOW,
        stall_after=timedelta(minutes=75),
    )
    assert set(issues) == {"failed_task:10"}


def test_issue_transitions_emit_only_edges():
    old = {"stalled_task:1": {"kind": "stalled_task", "task_id": 1}}
    new = {"endpoint:model": {"kind": "endpoint", "model_name": "model"}}
    events = _transition_events(old, new)

    assert [event["event"] for event in events] == [
        "runtime_issue_started",
        "runtime_issue_resolved",
    ]
    assert {event["issue_key"] for event in events} == {
        "stalled_task:1",
        "endpoint:model",
    }


def test_baseline_failures_do_not_hide_new_failures_or_other_issues():
    issues = {
        "failed_task:1": {"kind": "failed_task", "task_id": 1},
        "failed_task:2": {"kind": "failed_task", "task_id": 2},
        "stalled_task:3": {"kind": "stalled_task", "task_id": 3},
    }

    filtered = _without_baseline_failures(issues, {"failed_task:1"})

    assert set(filtered) == {"failed_task:2", "stalled_task:3"}


def test_stopped_scheduler_is_reported_but_sleeping_scheduler_is_not():
    issues = _scheduler_process_issues(
        [
            {"pid": 101, "state": "T", "command": "scheduler one"},
            {"pid": 102, "state": "t", "command": "scheduler two"},
            {"pid": 103, "state": "S", "command": "scheduler healthy"},
        ]
    )

    assert set(issues) == {"stopped_scheduler:101", "stopped_scheduler:102"}
    assert issues["stopped_scheduler:101"]["process_state"] == "T"
