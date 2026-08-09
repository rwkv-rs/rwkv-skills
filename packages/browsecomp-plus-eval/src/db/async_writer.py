from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .eval_db_service import EvalDbService


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.001, value)


_DEFAULT_FLUSH_ROWS = _positive_int_env("RWKV_COMPLETION_WRITE_FLUSH_ROWS", 64)
_DEFAULT_FLUSH_INTERVAL_S = _positive_float_env("RWKV_COMPLETION_WRITE_FLUSH_INTERVAL_S", 0.25)


class CompletionWriteWorker:
    def __init__(
        self,
        *,
        service: EvalDbService,
        task_id: str,
        max_queue: int = 4096,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        drain_every: int = 0,
        flush_rows: int | None = None,
        flush_interval_s: float | None = None,
    ) -> None:
        self._service = service
        self._task_id = task_id
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self._exc: BaseException | None = None
        self._enqueued_count = 0
        self._flushed_count = 0
        self._closed = False
        self._close_lock = threading.Lock()
        self._cv = threading.Condition()
        self._thread = threading.Thread(
            target=self._run,
            name=f"CompletionWriteWorker[{task_id}]",
            daemon=True,
        )
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_s = float(retry_backoff_s)
        self._drain_every = max(0, int(drain_every))
        self._flush_rows = max(1, int(flush_rows or _DEFAULT_FLUSH_ROWS))
        self._flush_interval_s = max(
            0.001,
            float(_DEFAULT_FLUSH_INTERVAL_S if flush_interval_s is None else flush_interval_s),
        )
        self._thread.start()

    def enqueue(self, payload: dict[str, Any]) -> None:
        should_drain = False
        with self._close_lock:
            self._raise_if_failed()
            with self._cv:
                if self._closed:
                    raise RuntimeError("DB writer is closed")
            self._queue.put(payload)
            with self._cv:
                self._enqueued_count += 1
                if self._drain_every > 0 and self._enqueued_count % self._drain_every == 0:
                    should_drain = True
        if should_drain:
            drained = self.drain()
            if not drained:
                raise TimeoutError("DB writer drain timed out")

    def drain(self, timeout_s: float | None = None) -> bool:
        self._raise_if_failed()
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        with self._cv:
            while self._flushed_count < self._enqueued_count:
                self._raise_if_failed()
                if deadline is None:
                    self._cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
        self._raise_if_failed()
        return True

    def close(self, timeout_s: float | None = None) -> None:
        with self._close_lock:
            if self._closed:
                self._raise_if_failed()
                return
            self._closed = True
            drained = self.drain(timeout_s=timeout_s)
            if not drained:
                raise TimeoutError("DB writer drain timed out before close")
            self._queue.put(self._stop)
            join_timeout = None if timeout_s is None else max(0.0, float(timeout_s))
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                raise TimeoutError("DB writer thread did not stop in time")
        self._raise_if_failed()

    def _run(self) -> None:
        pending: list[dict[str, Any]] = []
        try:
            while True:
                try:
                    item = self._queue.get(timeout=self._flush_interval_s if pending else None)
                except queue.Empty:
                    if pending:
                        self._flush_batch(pending)
                        pending = []
                    continue
                try:
                    if item is self._stop:
                        if pending:
                            self._flush_batch(pending)
                            pending = []
                        break
                    if isinstance(item, dict):
                        pending.append(item)
                        if len(pending) >= self._flush_rows:
                            self._flush_batch(pending)
                            pending = []
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            with self._cv:
                self._exc = exc
                self._cv.notify_all()

    def _flush(self, payload: dict[str, Any]) -> None:
        self._flush_batch([payload])

    def _flush_batch(self, payloads: list[dict[str, Any]]) -> None:
        if not payloads:
            return
        attempts = 0
        while True:
            try:
                self._service.insert_completion_payloads_batch(payloads=payloads, task_id=self._task_id)
                with self._cv:
                    self._flushed_count += len(payloads)
                    self._cv.notify_all()
                return
            except Exception as exc:
                attempts += 1
                if attempts >= self._max_retries:
                    raise exc
                time.sleep(self._retry_backoff_s * (2 ** (attempts - 1)))

    def _raise_if_failed(self) -> None:
        if self._exc:
            raise RuntimeError("DB writer thread failed") from self._exc


__all__ = ["CompletionWriteWorker"]
