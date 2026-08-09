from __future__ import annotations

"""Shared task runtime state machine aligned with rwkv-rs scheduler semantics."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import signal
from typing import Any, Callable, Protocol, runtime_checkable

from src.db.async_writer import CompletionWriteWorker
from src.db.eval_db_service import EvalDbService

from .checker import collect_pending_checker_inputs, run_checker_rows
from .task_persistence import RunMode, TaskExecutionState

AttemptTuple = tuple[int, int, int]


@runtime_checkable
class SupportsAttemptTuple(Protocol):
    def as_tuple(self) -> tuple[int, int, int]:
        ...


def _coerce_attempt_index(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        return int(stripped) if stripped else 0
    raise TypeError(f"unsupported attempt index: {value!r}")


def attempt_tuple(value: object) -> AttemptTuple:
    if isinstance(value, SupportsAttemptTuple):
        raw = value.as_tuple()
        return (
            _coerce_attempt_index(raw[0]),
            _coerce_attempt_index(raw[1]),
            _coerce_attempt_index(raw[2]),
        )
    if isinstance(value, Mapping):
        return (
            _coerce_attempt_index(value.get("sample_index", 0)),
            _coerce_attempt_index(value.get("repeat_index", 0)),
            _coerce_attempt_index(value.get("pass_index", 0)),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 3:
        return (
            _coerce_attempt_index(value[0]),
            _coerce_attempt_index(value[1]),
            _coerce_attempt_index(value[2]),
        )
    raise TypeError(f"unsupported attempt identity: {value!r}")


def _mapping_payload(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


@dataclass(slots=True)
class TaskRunState:
    task_id: str
    run_mode: RunMode
    expected_attempt_count: int
    completed_attempts: set[AttemptTuple]
    pending_attempts: list[AttemptTuple]
    task_results: dict[AttemptTuple, bool] = field(default_factory=dict)
    pending_checks: list[AttemptTuple] = field(default_factory=list)
    checker_running: bool = False
    completed: bool = False
    failed_error: str | None = None
    completion_count: int = 0
    eval_count: int = 0
    checker_count: int = 0

    @classmethod
    def from_task_execution(
        cls,
        *,
        execution_state: TaskExecutionState,
        attempt_keys: Sequence[Any],
        expected_attempt_count: int,
    ) -> "TaskRunState":
        completed = {attempt_tuple(item) for item in execution_state.skip_keys}
        pending: list[AttemptTuple] = []
        for item in attempt_keys:
            key = attempt_tuple(item)
            if key in completed:
                continue
            pending.append(key)
        return cls(
            task_id=execution_state.task_id,
            run_mode=execution_state.run_mode,
            expected_attempt_count=int(expected_attempt_count),
            completed_attempts=completed,
            pending_attempts=pending,
            completion_count=len(completed),
        )

    def remaining_attempts(self) -> int:
        return len(self.pending_attempts)

    def is_terminal(self) -> bool:
        return self.completed or self.failed_error is not None


class TaskRunController:
    def __init__(self, *, service: EvalDbService, state: TaskRunState) -> None:
        self._service = service
        self.state = state

    def create_writer(
        self,
        *,
        max_queue: int = 4096,
        drain_every: int = 0,
    ) -> CompletionWriteWorker:
        return CompletionWriteWorker(
            service=self._service,
            task_id=self.state.task_id,
            max_queue=max_queue,
            drain_every=drain_every,
        )

    def complete_attempt_stage(
        self,
        writer: CompletionWriteWorker,
        *,
        timeout_s: float | None = None,
        on_after_close: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        self._close_writer(writer, timeout_s=timeout_s)
        if on_after_close is not None:
            on_after_close()
        return self.sync_completion_state()

    def handle_attempt_stage_failure(
        self,
        writer: CompletionWriteWorker,
        *,
        timeout_s: float | None = None,
        interrupted: bool = False,
        error: str | None = None,
        on_after_close: Callable[[], None] | None = None,
    ) -> None:
        close_error = self._try_close_writer(writer, timeout_s=timeout_s)
        after_error = self._run_after_close(on_after_close)
        self.sync_completion_state()
        status = "failed" if interrupted else (
            "completed" if self.state.completion_count == self.state.expected_attempt_count else "failed"
        )
        self._service.update_task_status(task_id=self.state.task_id, status=status)
        self.state.completed = status == "completed"
        if status != "completed":
            self.state.failed_error = self._merge_errors(error, close_error, after_error) or "attempt_stage_failed"

    def sync_completion_state(self) -> list[dict[str, Any]]:
        payloads = self._service.list_completion_payloads(task_id=self.state.task_id, status="Completed")
        completed = {attempt_tuple(payload) for payload in payloads}
        self.state.completed_attempts = completed
        self.state.pending_attempts = [key for key in self.state.pending_attempts if key not in completed]
        self.state.completion_count = len(payloads)
        return payloads

    def ingest_eval_payloads(self, payloads: Sequence[Mapping[str, Any]]) -> int:
        rows = [_mapping_payload(item, name="eval payload") for item in payloads]
        inserted = self._service.ingest_eval_payloads(payloads=rows, task_id=self.state.task_id)
        self.state.task_results = {
            attempt_tuple(payload): bool(payload.get("is_passed", False))
            for payload in rows
        }
        self.state.eval_count = len(self.state.task_results)
        return inserted

    def run_checker(self, *, model_name: str) -> int:
        checker_inputs = collect_pending_checker_inputs(
            service=self._service,
            task_id=self.state.task_id,
            model_name=model_name,
        )
        self.state.pending_checks = [attempt_tuple(payload) for payload in checker_inputs]
        if not checker_inputs:
            self.state.pending_checks = []
            self.state.checker_count = 0
            return 0

        self.state.checker_running = True
        try:
            checker_rows = run_checker_rows(checker_inputs)
            if not checker_rows:
                self.state.checker_count = 0
                return 0
            inserted = self._service.ingest_checker_payloads(
                payloads=checker_rows,
                task_id=self.state.task_id,
            )
            self.state.checker_count = inserted
            return inserted
        finally:
            self.state.checker_running = False
            self.state.pending_checks = []

    def record_score(self, payload: Mapping[str, Any]) -> None:
        self._service.record_score_payload(
            payload=_mapping_payload(payload, name="score payload"),
            task_id=self.state.task_id,
        )
        self.state.completed = True
        self.state.failed_error = None

    def fail_task(
        self,
        *,
        error: str | None = None,
        writer: CompletionWriteWorker | None = None,
        timeout_s: float | None = None,
        on_after_close: Callable[[], None] | None = None,
    ) -> None:
        close_error = None
        if writer is not None:
            close_error = self._try_close_writer(writer, timeout_s=timeout_s)
        after_error = self._run_after_close(on_after_close)
        self._service.update_task_status(task_id=self.state.task_id, status="failed")
        self.state.completed = False
        self.state.failed_error = self._merge_errors(error, close_error, after_error) or "task_failed"

    @staticmethod
    def _run_after_close(callback: Callable[[], None] | None) -> str | None:
        if callback is None:
            return None
        try:
            callback()
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return None

    @staticmethod
    def _merge_errors(*values: str | None) -> str | None:
        parts = [value.strip() for value in values if isinstance(value, str) and value.strip()]
        if not parts:
            return None
        return " | ".join(parts)

    @staticmethod
    def _close_writer(writer: CompletionWriteWorker, *, timeout_s: float | None) -> None:
        if timeout_s is None:
            writer.close()
            return
        writer.close(timeout_s=timeout_s)

    def _try_close_writer(self, writer: CompletionWriteWorker, *, timeout_s: float | None) -> str | None:
        try:
            self._close_writer(writer, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return None


class TaskRunSignalGuard:
    def __init__(
        self,
        *,
        controller: TaskRunController,
        writer: CompletionWriteWorker,
        close_timeout_s: float | None = None,
        signals_to_handle: Sequence[signal.Signals] = (signal.SIGINT, signal.SIGTERM),
        on_interrupt: Callable[[str], None] | None = None,
    ) -> None:
        self._controller = controller
        self._writer = writer
        self._close_timeout_s = close_timeout_s
        self._signals = tuple(signals_to_handle)
        self._on_interrupt = on_interrupt
        self._original_handlers: dict[signal.Signals, Any] = {}
        self.interrupted = False

    def __enter__(self) -> "TaskRunSignalGuard":
        for sig in self._signals:
            self._original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)
        self._original_handlers.clear()

    def _handle_signal(self, signum: int, _frame: object) -> None:
        if self.interrupted:
            raise SystemExit(128 + signum)
        self.interrupted = True
        signame = signal.Signals(signum).name
        callback = self._on_interrupt
        self._controller.fail_task(
            error=f"received {signame}",
            writer=self._writer,
            timeout_s=self._close_timeout_s,
            on_after_close=_signal_callback(callback, signame) if callback is not None else None,
        )
        raise SystemExit(128 + signum)


def _signal_callback(callback: Callable[[str], None], signame: str) -> Callable[[], None]:
    def _run() -> None:
        callback(signame)

    return _run


__all__ = [
    "AttemptTuple",
    "TaskRunController",
    "TaskRunSignalGuard",
    "TaskRunState",
    "attempt_tuple",
]
