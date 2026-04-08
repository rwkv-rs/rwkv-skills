from __future__ import annotations

import pytest

from src.eval.evaluating.runtime import TaskRunController, TaskRunState
from src.eval.evaluating.task_persistence import RunMode


class _FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], str]] = []

    def record_score_payload(self, *, payload: dict[str, object], task_id: str) -> None:
        self.calls.append((payload, task_id))


def _make_state() -> TaskRunState:
    return TaskRunState(
        task_id="42",
        run_mode=RunMode.NEW,
        expected_attempt_count=1,
        completed_attempts=set(),
        pending_attempts=[],
    )


def test_record_score_accepts_mapping_and_marks_state_completed() -> None:
    service = _FakeService()
    state = _make_state()
    runtime = TaskRunController(service=service, state=state)

    runtime.record_score({"dataset": "mmlu_test", "metrics": {"accuracy": 1.0}})

    assert service.calls == [({"dataset": "mmlu_test", "metrics": {"accuracy": 1.0}}, "42")]
    assert state.completed is True
    assert state.failed_error is None


def test_record_score_rejects_non_mapping_payload_with_clear_error() -> None:
    runtime = TaskRunController(service=_FakeService(), state=_make_state())

    with pytest.raises(TypeError, match="score payload must be a mapping"):
        runtime.record_score(({"dataset": "mmlu_test"},))  # type: ignore[arg-type]
