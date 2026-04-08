from __future__ import annotations

from pathlib import Path

from src.eval.scheduler import actions
from src.eval.scheduler.actions import DispatchOptions, QueueOptions
from src.eval.scheduler.admin import SchedulerStartRequest
from src.eval.scheduler.cli import build_parser


class _FakeLeaseManager:
    def __init__(self, active_sequences: list[set[str]] | None = None) -> None:
        self._active_sequences = list(active_sequences or [])
        self.claim_calls: list[str] = []
        self.release_calls: list[tuple[str, ...]] = []
        self.renew_calls: list[tuple[str, ...]] = []

    def active_foreign_job_ids(self) -> set[str]:
        if self._active_sequences:
            return set(self._active_sequences.pop(0))
        return set()

    def claim(self, job_id: str, *, lease_meta=None) -> bool:
        self.claim_calls.append(job_id)
        return True

    def renew(self, job_ids) -> set[str]:
        normalized = tuple(sorted(str(job_id) for job_id in job_ids))
        self.renew_calls.append(normalized)
        return set(normalized)

    def release(self, job_ids) -> int:
        normalized = tuple(sorted(str(job_id) for job_id in job_ids))
        self.release_calls.append(normalized)
        return len(normalized)


def test_action_queue_filters_foreign_cluster_claims(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    lease_manager = _FakeLeaseManager(active_sequences=[{"free_response__claimed"}])

    monkeypatch.setattr(actions, "scan_completed_jobs", lambda: (set(), {}))
    monkeypatch.setattr(actions, "load_running", lambda _pid_dir: {})
    monkeypatch.setattr(actions, "derive_question_counts", lambda _records: {})
    monkeypatch.setattr(actions, "sort_queue_items", lambda items, **_kwargs: items)
    monkeypatch.setattr(actions, "_print_queue_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(actions, "_build_lease_manager", lambda _opts: lease_manager)

    def _fake_build_queue(**kwargs):
        captured["running"] = set(kwargs["running"])
        return []

    monkeypatch.setattr(actions, "build_queue", _fake_build_queue)

    actions.action_queue(
        QueueOptions(
            log_dir=tmp_path,
            pid_dir=tmp_path,
            job_order=("free_response",),
            distributed_claims=True,
        )
    )

    assert captured["running"] == {"free_response__claimed"}


def test_action_dispatch_waits_for_foreign_cluster_claims(monkeypatch, tmp_path: Path) -> None:
    lease_manager = _FakeLeaseManager(active_sequences=[{"job-foreign"}, set()])
    events: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(actions, "ensure_dirs", lambda *_args: None)
    monkeypatch.setattr(actions, "scan_completed_jobs", lambda: (set(), {}))
    monkeypatch.setattr(actions, "load_running", lambda _pid_dir: {})
    monkeypatch.setattr(actions, "derive_question_counts", lambda _records: {})
    monkeypatch.setattr(actions, "_build_lease_manager", lambda _opts: lease_manager)
    monkeypatch.setattr(actions, "time", type("_T", (), {"time": staticmethod(lambda: 0.0), "sleep": staticmethod(lambda _s: None)}))
    monkeypatch.setattr(actions.FAILURE_MONITOR, "wait_failure", lambda timeout=0: None)
    monkeypatch.setattr(actions.FAILURE_MONITOR, "reset", lambda: None)
    monkeypatch.setattr(actions, "log_job_event", lambda event, job_id, **payload: events.append((event, job_id, payload)))

    def _fake_build_queue(**kwargs):
        running = set(kwargs["running"])
        return [] if "job-foreign" in running else []

    monkeypatch.setattr(actions, "build_queue", _fake_build_queue)
    monkeypatch.setattr(actions, "sort_queue_items", lambda items, **_kwargs: items)

    opts = DispatchOptions(
        log_dir=tmp_path,
        pid_dir=tmp_path,
        run_log_dir=tmp_path,
        job_order=("free_response",),
        distributed_claims=True,
        dispatch_poll_seconds=1,
    )

    actions.action_dispatch(opts)

    wait_events = [payload for event, _job_id, payload in events if event == "dispatcher_wait"]
    assert any(payload.get("reason") == "cluster_running" for payload in wait_events)
    assert any(event == "dispatcher_done" for event, _job_id, _payload in events)


def test_scheduler_cli_accepts_distributed_claim_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "dispatch",
            "--distributed-claims",
            "--scheduler-node-id",
            "node-a",
            "--lease-duration-s",
            "321",
        ]
    )

    assert args.distributed_claims is True
    assert args.scheduler_node_id == "node-a"
    assert args.lease_duration_s == 321


def test_scheduler_start_request_builds_distributed_claim_options() -> None:
    request = SchedulerStartRequest(
        only_jobs=["free_response"],
        distributed_claims=True,
        scheduler_node_id="node-a",
        lease_duration_s=321,
    )

    opts = request.to_dispatch_options()

    assert opts.distributed_claims is True
    assert opts.scheduler_node_id == "node-a"
    assert opts.lease_duration_s == 321
