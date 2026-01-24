from __future__ import annotations

import queue
import threading
import time
from typing import Any, Iterable

from .eval_db_service import EvalDbService


class CompletionWriteWorker:
    def __init__(
        self,
        *,
        service: EvalDbService,
        version_id: str,
        is_param_search: bool,
        batch_size: int = 128,
        max_queue: int = 4096,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
    ) -> None:
        self._service = service
        self._version_id = version_id
        self._is_param_search = is_param_search
        self._batch_size = max(1, int(batch_size))
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"CompletionWriteWorker[{version_id}]",
            daemon=True,
        )
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_s = float(retry_backoff_s)
        self._thread.start()

    def enqueue(self, payload: dict[str, Any]) -> None:
        if self._exc:
            raise RuntimeError("DB writer thread failed") from self._exc
        self._queue.put(payload)

    def close(self) -> None:
        self._queue.put(self._stop)
        self._thread.join()
        if self._exc:
            raise RuntimeError("DB writer thread failed") from self._exc

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._stop:
                        if batch:
                            self._flush(batch)
                            batch = []
                        break
                    if isinstance(item, dict):
                        batch.append(item)
                        if len(batch) >= self._batch_size:
                            self._flush(batch)
                            batch = []
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._exc = exc

    def _flush(self, batch: Iterable[dict[str, Any]]) -> None:
        attempts = 0
        while True:
            try:
                self._service.ingest_completion_payloads(
                    payloads=batch,
                    version_id=self._version_id,
                    is_param_search=self._is_param_search,
                )
                return
            except Exception as exc:
                attempts += 1
                if attempts >= self._max_retries:
                    raise exc
                time.sleep(self._retry_backoff_s * (2 ** (attempts - 1)))


__all__ = ["CompletionWriteWorker"]
