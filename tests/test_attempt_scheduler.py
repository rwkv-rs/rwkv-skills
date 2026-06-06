from __future__ import annotations

import threading
import time

from src.eval.attempt_scheduler import (
    AttemptKey,
    AttemptResult,
    AttemptStatus,
    AttemptWorkItem,
    TaskRunState,
    run_attempt_scheduler,
)


def test_attempt_scheduler_respects_model_concurrency_and_finalize_gate() -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()
    attempts = [
        AttemptKey(task_run_id="task-a", sample_index=index, avg_repeat_index=0, pass_index=0, seed=index)
        for index in range(4)
    ]
    task = TaskRunState.from_attempts(
        task_run_id="task-a",
        model_key="rwkv-test",
        attempts=attempts,
        max_attempts_per_model=2,
    )

    def runner(work: AttemptWorkItem) -> AttemptResult:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return AttemptResult(key=work.key, payload={"sample_index": work.key.sample_index}, passed=True)

    completed: list[AttemptKey] = []
    run_attempt_scheduler([task], runner=runner, on_result=lambda _task, result: completed.append(result.key))

    assert max_active == 2
    assert task.ready_to_finalize is True
    assert task.missing_result_keys(task.expected_keys) == set()
    assert set(completed) == set(attempts)


def test_attempt_scheduler_retries_failures_then_records_final_status() -> None:
    key = AttemptKey(task_run_id="task-b", sample_index=7, avg_repeat_index=1, pass_index=0, seed=123)
    task = TaskRunState.from_attempts(
        task_run_id="task-b",
        model_key="rwkv-test",
        attempts=[key],
        max_attempts_per_model=1,
        max_retries=1,
    )
    calls = 0

    def runner(work: AttemptWorkItem) -> AttemptResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return AttemptResult(
                key=work.key,
                status=AttemptStatus.FAILED,
                error_type="TransientError",
                error_message="temporary",
            )
        return AttemptResult(key=work.key, payload={"ok": True}, passed=True)

    run_attempt_scheduler([task], runner=runner)

    assert calls == 2
    assert task.ready_to_finalize is True
    assert task.results[key].successful is True
    assert task.results[key].retry_count == 1


def test_attempt_scheduler_keeps_final_failed_attempt_after_retry_limit() -> None:
    key = AttemptKey(task_run_id="task-c", sample_index=3, avg_repeat_index=0, pass_index=0, seed=55)
    task = TaskRunState.from_attempts(
        task_run_id="task-c",
        model_key="rwkv-test",
        attempts=[key],
        max_attempts_per_model=1,
        max_retries=1,
    )

    def runner(work: AttemptWorkItem) -> AttemptResult:
        return AttemptResult(
            key=work.key,
            status=AttemptStatus.JUDGE_FAILED,
            error_type="JudgeError",
            error_message="judge unavailable",
        )

    run_attempt_scheduler([task], runner=runner)

    assert task.ready_to_finalize is True
    assert task.results[key].status == AttemptStatus.JUDGE_FAILED
    assert task.results[key].retry_count == 1
