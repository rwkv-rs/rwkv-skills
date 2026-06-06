from __future__ import annotations

"""Shared attempt-level scheduling primitives for benchmark evaluators.

The scheduler operates on single-sample attempts. It is intentionally separate
from model request batching: a benchmark may still batch prompts internally, but
the unit of retry, resume, persistence, and score finalization is an attempt.
"""

from collections import deque
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Any


@dataclass(frozen=True, order=True, slots=True)
class AttemptKey:
    task_run_id: str
    sample_index: int
    avg_repeat_index: int = 0
    pass_index: int = 0
    seed: int | None = None

    @property
    def repeat_index(self) -> int:
        """Compatibility key for existing completion/eval tables."""

        return int(self.avg_repeat_index)

    def completion_key(self) -> tuple[int, int]:
        return (int(self.sample_index), int(self.repeat_index))


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    JUDGE_FAILED = "judge_failed"


@dataclass(slots=True)
class AttemptResult:
    key: AttemptKey
    status: AttemptStatus = AttemptStatus.SUCCESS
    payload: Any | None = None
    passed: bool | None = None
    error_type: str = ""
    error_message: str = ""
    retry_count: int = 0

    @property
    def successful(self) -> bool:
        return self.status == AttemptStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class AttemptWorkItem:
    task_run_id: str
    model_key: str
    key: AttemptKey
    retry_count: int = 0


@dataclass(slots=True)
class TaskRunState:
    task_run_id: str
    model_key: str
    pending_attempts: deque[AttemptKey]
    expected_attempts: set[AttemptKey] = field(default_factory=set)
    max_attempts_per_model: int = 1
    max_retries: int = 0
    inflight: int = 0
    results: dict[AttemptKey, AttemptResult] = field(default_factory=dict)
    retry_counts: dict[AttemptKey, int] = field(default_factory=dict)

    @classmethod
    def from_attempts(
        cls,
        *,
        task_run_id: str,
        model_key: str,
        attempts: Iterable[AttemptKey],
        completed_keys: set[AttemptKey] | None = None,
        max_attempts_per_model: int = 1,
        max_retries: int = 0,
    ) -> TaskRunState:
        all_attempts = tuple(attempts)
        completed = completed_keys or set()
        pending = deque(key for key in all_attempts if key not in completed)
        return cls(
            task_run_id=task_run_id,
            model_key=model_key,
            pending_attempts=pending,
            expected_attempts=set(all_attempts),
            max_attempts_per_model=max(1, int(max_attempts_per_model)),
            max_retries=max(0, int(max_retries)),
        )

    @property
    def expected_keys(self) -> set[AttemptKey]:
        if self.expected_attempts:
            return set(self.expected_attempts)
        return set(self.pending_attempts) | set(self.results)

    @property
    def ready_to_finalize(self) -> bool:
        return (
            not self.pending_attempts
            and self.inflight == 0
            and self.expected_keys.issubset(set(self.results))
        )

    def missing_result_keys(self, expected_keys: Iterable[AttemptKey]) -> set[AttemptKey]:
        return set(expected_keys) - set(self.results)


AttemptRunner = Callable[[AttemptWorkItem], AttemptResult]
AttemptResultCallback = Callable[[TaskRunState, AttemptResult], None]


class ModelRuntimeLimiter:
    """Named semaphore registry for model/checker stage limits."""

    def __init__(self, limits: Mapping[str, int] | None = None) -> None:
        self._limits = {
            str(key): threading.BoundedSemaphore(max(1, int(value)))
            for key, value in (limits or {}).items()
        }
        self._lock = threading.Lock()

    def semaphore(self, key: str, default: int = 1) -> threading.BoundedSemaphore:
        normalized = str(key)
        with self._lock:
            sem = self._limits.get(normalized)
            if sem is None:
                sem = threading.BoundedSemaphore(max(1, int(default)))
                self._limits[normalized] = sem
            return sem


def run_attempt_scheduler(
    task_runs: Iterable[TaskRunState],
    *,
    runner: AttemptRunner,
    on_result: AttemptResultCallback | None = None,
    max_workers: int | None = None,
) -> list[TaskRunState]:
    """Run attempts until every task has no pending or inflight work.

    A model's concurrency limit is taken from the largest
    `max_attempts_per_model` requested by task runs for that model. Failed
    attempts are requeued until the owning task's `max_retries` is exhausted.
    """

    tasks = list(task_runs)
    if not tasks:
        return []
    model_limits: dict[str, int] = {}
    for task in tasks:
        model_limits[task.model_key] = max(
            model_limits.get(task.model_key, 1),
            int(task.max_attempts_per_model),
        )
    semaphores = {
        model_key: threading.BoundedSemaphore(max(1, int(limit)))
        for model_key, limit in model_limits.items()
    }

    total_workers = max_workers or sum(max(1, int(task.max_attempts_per_model)) for task in tasks)
    total_workers = max(1, int(total_workers))
    futures: dict[Future[AttemptResult], tuple[TaskRunState, AttemptKey, threading.BoundedSemaphore, int]] = {}

    def submit_available(executor: ThreadPoolExecutor) -> None:
        for task in tasks:
            sem = semaphores[task.model_key]
            while task.pending_attempts and sem.acquire(blocking=False):
                key = task.pending_attempts.popleft()
                retry_count = int(task.retry_counts.get(key, 0))
                task.inflight += 1
                work = AttemptWorkItem(
                    task_run_id=task.task_run_id,
                    model_key=task.model_key,
                    key=key,
                    retry_count=retry_count,
                )
                future = executor.submit(_run_with_result_wrapping, runner, work)
                futures[future] = (task, key, sem, retry_count)

    with ThreadPoolExecutor(max_workers=total_workers, thread_name_prefix="benchmark-attempt") as executor:
        submit_available(executor)
        while futures:
            done, _pending = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                task, key, sem, retry_count = futures.pop(future)
                try:
                    result = future.result()
                finally:
                    sem.release()
                    task.inflight -= 1
                result.retry_count = retry_count
                if not result.successful and retry_count < task.max_retries:
                    task.results.pop(key, None)
                    task.retry_counts[key] = retry_count + 1
                    task.pending_attempts.append(key)
                else:
                    task.results[key] = result
                    if on_result is not None:
                        on_result(task, result)
            submit_available(executor)
    return tasks


def _run_with_result_wrapping(runner: AttemptRunner, work: AttemptWorkItem) -> AttemptResult:
    try:
        result = runner(work)
    except TimeoutError as exc:
        return AttemptResult(
            key=work.key,
            status=AttemptStatus.TIMEOUT,
            error_type=type(exc).__name__,
            error_message=str(exc),
            retry_count=work.retry_count,
        )
    except Exception as exc:
        return AttemptResult(
            key=work.key,
            status=AttemptStatus.FAILED,
            error_type=type(exc).__name__,
            error_message=str(exc),
            retry_count=work.retry_count,
        )
    if result.key != work.key:
        raise ValueError(f"attempt runner returned mismatched key: {result.key!r} != {work.key!r}")
    return result


__all__ = [
    "AttemptKey",
    "AttemptResult",
    "AttemptStatus",
    "AttemptWorkItem",
    "ModelRuntimeLimiter",
    "TaskRunState",
    "run_attempt_scheduler",
]
