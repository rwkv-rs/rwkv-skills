from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Callable

from src.eval.scheduler.actions import DispatchOptions
from src.eval.scheduler.admin import (
    SchedulerAdminController,
    SchedulerStartRequest,
    _render_admin_shell,
    build_status_response,
)
from src.eval.scheduler.control import (
    DesiredState,
    ObservedStatus,
    SchedulerProgressSnapshot,
    SchedulerRuntimeControl,
)


def test_scheduler_runtime_control_roundtrip(tmp_path: Path) -> None:
    control = SchedulerRuntimeControl.from_dir(tmp_path)
    _ = control.write_desired_state(DesiredState.PAUSED)
    runtime = control.write_status(
        ObservedStatus.RUNNING,
        progress=SchedulerProgressSnapshot(
            pending_jobs=5,
            running_jobs=2,
            completed_jobs=3,
            queue_head=("job-a", "job-b"),
            active_jobs=("job-x",),
            available_gpus=("0",),
        ),
    )

    desired_state, loaded = control.snapshot()

    assert desired_state is DesiredState.PAUSED
    assert loaded is not None
    assert loaded.observed_status == ObservedStatus.RUNNING.value
    assert loaded.pending_jobs == 5
    assert loaded.queue_head == ["job-a", "job-b"]
    assert loaded.started_at_unix_ms == runtime.started_at_unix_ms


class _BlockingDispatchRunner:
    started: threading.Event
    allow_exit: threading.Event

    def __init__(self) -> None:
        self.started = threading.Event()
        self.allow_exit = threading.Event()

    def __call__(self, _opts: DispatchOptions, *, runtime_control: SchedulerRuntimeControl) -> None:
        running_progress = SchedulerProgressSnapshot(
            pending_jobs=3,
            running_jobs=1,
            completed_jobs=0,
            queue_head=("job-1", "job-2"),
            active_jobs=("job-0",),
            available_gpus=("0",),
        )
        paused_progress = SchedulerProgressSnapshot(
            pending_jobs=3,
            running_jobs=0,
            completed_jobs=0,
            queue_head=("job-1", "job-2"),
            active_jobs=(),
            available_gpus=("0",),
        )
        completed_progress = SchedulerProgressSnapshot(completed_jobs=3)
        _ = runtime_control.write_status(ObservedStatus.RUNNING, progress=running_progress)
        self.started.set()

        while not self.allow_exit.is_set():
            desired_state = runtime_control.desired_state()
            if desired_state is DesiredState.PAUSED:
                _ = runtime_control.write_status(ObservedStatus.PAUSED, progress=paused_progress)
                while runtime_control.desired_state() is DesiredState.PAUSED and not self.allow_exit.is_set():
                    time.sleep(0.01)
                if self.allow_exit.is_set():
                    break
                _ = runtime_control.write_status(ObservedStatus.RUNNING, progress=running_progress)
                continue
            if desired_state is DesiredState.CANCELLED:
                _ = runtime_control.write_status(ObservedStatus.CANCELLED, progress=paused_progress)
                return
            time.sleep(0.01)

        _ = runtime_control.write_status(ObservedStatus.COMPLETED, progress=completed_progress)


def test_scheduler_admin_controller_transitions(tmp_path: Path) -> None:
    runner = _BlockingDispatchRunner()
    controller = SchedulerAdminController(state_dir=tmp_path, dispatch_runner=runner)

    _ = controller.start(SchedulerStartRequest(only_jobs=["free_response"]))
    assert runner.started.wait(timeout=1.0)

    paused = controller.pause()
    assert paused.desired_state is DesiredState.PAUSED
    _wait_for(lambda: _snapshot_status(controller) == "paused")

    resumed = controller.resume()
    assert resumed.desired_state is DesiredState.RUNNING
    _wait_for(lambda: _snapshot_status(controller) == "running")

    cancelled = controller.cancel()
    assert cancelled.desired_state is DesiredState.CANCELLED
    _wait_for(lambda: _snapshot_status(controller) == "cancelled")


def test_scheduler_admin_idle_status_and_shell_render() -> None:
    payload = build_status_response(None)
    shell = _render_admin_shell(SchedulerStartRequest().to_dict())

    assert payload["status"] == "idle"
    assert payload["run_id"] is None
    assert "RWKV Scheduler Admin" in shell
    assert "/api/v1/admin/eval/start" in shell


def _snapshot_status(controller: SchedulerAdminController) -> str | None:
    snapshot = controller.snapshot()
    if snapshot is None:
        return None
    return snapshot.runtime.observed_status


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met before timeout")
