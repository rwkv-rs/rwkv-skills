from __future__ import annotations

import queue
import threading
import time
from typing import Any

from .eval_db_service import EvalDbService


class CompletionWriteWorker:
    def __init__(
        self,
        *,
        service: EvalDbService,
        task_id: str,
        max_queue: int = 4096,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        put_timeout: float = 1.0,
        close_timeout: float = 30.0,
    ) -> None:
        self._service = service
        self._task_id = task_id
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self._exc: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"CompletionWriteWorker[{task_id}]",
            daemon=True,
        )
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_s = float(retry_backoff_s)
        self._put_timeout = float(put_timeout)
        self._close_timeout = float(close_timeout)
        self._thread.start()

    def enqueue(self, payload: dict[str, Any]) -> None:
        if self._exc:
            raise RuntimeError("DB writer thread failed") from self._exc
        # 使用带超时的 put，定期检查线程是否存活
        while True:
            if self._exc:
                raise RuntimeError("DB writer thread failed") from self._exc
            if not self._thread.is_alive():
                raise RuntimeError("DB writer thread died unexpectedly") from self._exc
            try:
                self._queue.put(payload, timeout=self._put_timeout)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        deadline = time.monotonic() + self._close_timeout

        while self._thread.is_alive():
            if self._exc:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._queue.put(self._stop, timeout=min(self._put_timeout, remaining))
                break
            except queue.Full:
                continue

        remaining = max(0.0, deadline - time.monotonic())
        self._thread.join(timeout=remaining)

        if self._thread.is_alive():
            raise RuntimeError(
                "DB writer thread did not finish within timeout; data may not be fully persisted"
            )
        if self._exc:
            raise RuntimeError("DB writer thread failed") from self._exc

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._stop:
                        break
                    if isinstance(item, dict):
                        self._flush(item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._exc = exc

    def _flush(self, payload: dict[str, Any]) -> None:
        attempts = 0
        while True:
            try:
                self._service.insert_completion_payload(payload=payload, task_id=self._task_id)
                return
            except Exception as exc:
                attempts += 1
                if attempts >= self._max_retries:
                    raise exc
                time.sleep(self._retry_backoff_s * (2 ** (attempts - 1)))


class EvalWriteWorker:
    """异步逐条写入 Eval 记录，每条立即入库。"""

    def __init__(
        self,
        *,
        service: EvalDbService,
        task_id: str,
        max_queue: int = 8192,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        put_timeout: float = 1.0,
        close_timeout: float = 30.0,
    ) -> None:
        self._service = service
        self._task_id = task_id
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self._exc: BaseException | None = None
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_s = float(retry_backoff_s)
        self._put_timeout = float(put_timeout)
        self._close_timeout = float(close_timeout)
        self._inserted = 0
        self._mapping: dict[tuple[int, int], int] | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"EvalWriteWorker[{task_id}]",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, payload: dict[str, Any]) -> None:
        if self._exc:
            raise RuntimeError("Eval writer thread failed") from self._exc
        # 使用带超时的 put，定期检查线程是否存活
        while True:
            if self._exc:
                raise RuntimeError("Eval writer thread failed") from self._exc
            if not self._thread.is_alive():
                raise RuntimeError("Eval writer thread died unexpectedly") from self._exc
            try:
                self._queue.put(payload, timeout=self._put_timeout)
                return
            except queue.Full:
                continue

    def close(self) -> int:
        """关闭写入器，返回成功写入的记录数。"""
        deadline = time.monotonic() + self._close_timeout

        while self._thread.is_alive():
            if self._exc:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._queue.put(self._stop, timeout=min(self._put_timeout, remaining))
                break
            except queue.Full:
                continue

        remaining = max(0.0, deadline - time.monotonic())
        self._thread.join(timeout=remaining)

        if self._thread.is_alive():
            raise RuntimeError(
                "Eval writer thread did not finish within timeout; data may not be fully persisted"
            )
        if self._exc:
            raise RuntimeError("Eval writer thread failed") from self._exc
        return self._inserted

    @property
    def inserted(self) -> int:
        return self._inserted

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._stop:
                        break
                    if isinstance(item, dict):
                        self._flush(item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._exc = exc

    def _flush(self, payload: dict[str, Any]) -> None:
        if self._mapping is None:
            self._mapping = self._service.fetch_completion_id_map(task_id=self._task_id)

        sample_index = payload.get("sample_index")
        repeat_index = payload.get("repeat_index")
        if sample_index is None or repeat_index is None:
            return

        completions_id = self._mapping.get((int(sample_index), int(repeat_index)))
        if completions_id is None:
            return

        attempts = 0
        while True:
            try:
                self._service.insert_eval_payload(
                    payload=payload,
                    completions_id=completions_id,
                )
                self._inserted += 1
                return
            except Exception as exc:
                attempts += 1
                if attempts >= self._max_retries:
                    raise exc
                time.sleep(self._retry_backoff_s * (2 ** (attempts - 1)))


class CheckerWriteWorker:
    """异步批量更新 eval.fail_reason（checker 结果）。

    与 EvalWriteWorker 类似，但执行的是 update 而非 insert。
    支持批量提交以提高效率。
    """

    def __init__(
        self,
        *,
        service: EvalDbService,
        checker_type: str = "llm_checker",
        batch_size: int = 50,
        max_queue: int = 4096,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
        put_timeout: float = 1.0,
        close_timeout: float = 30.0,
    ) -> None:
        self._service = service
        self._checker_type = checker_type
        self._batch_size = max(1, int(batch_size))
        self._queue: queue.Queue[tuple[int, dict[str, Any]] | object] = queue.Queue(maxsize=max_queue)
        self._stop = object()
        self._exc: BaseException | None = None
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_s = float(retry_backoff_s)
        self._put_timeout = float(put_timeout)
        self._close_timeout = float(close_timeout)
        self._updated = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"CheckerWriteWorker[{checker_type}]",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, eval_id: int, checker_result: dict[str, Any]) -> None:
        """将 checker 结果加入队列。"""
        if self._exc:
            raise RuntimeError("Checker writer thread failed") from self._exc
        while True:
            if self._exc:
                raise RuntimeError("Checker writer thread failed") from self._exc
            if not self._thread.is_alive():
                raise RuntimeError("Checker writer thread died unexpectedly") from self._exc
            try:
                self._queue.put((eval_id, checker_result), timeout=self._put_timeout)
                return
            except queue.Full:
                continue

    def close(self) -> int:
        """关闭写入器，返回成功更新的记录数。"""
        deadline = time.monotonic() + self._close_timeout

        while self._thread.is_alive():
            if self._exc:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                self._queue.put(self._stop, timeout=min(self._put_timeout, remaining))
                break
            except queue.Full:
                continue

        remaining = max(0.0, deadline - time.monotonic())
        self._thread.join(timeout=remaining)

        if self._thread.is_alive():
            raise RuntimeError(
                "Checker writer thread did not finish within timeout; data may not be fully persisted"
            )
        if self._exc:
            raise RuntimeError("Checker writer thread failed") from self._exc
        return self._updated

    @property
    def updated(self) -> int:
        return self._updated

    def _run(self) -> None:
        batch: list[tuple[int, str, dict[str, Any]]] = []
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is self._stop:
                        # 处理剩余的 batch
                        if batch:
                            self._flush_batch(batch)
                        break
                    if isinstance(item, tuple) and len(item) == 2:
                        eval_id, checker_result = item
                        batch.append((eval_id, self._checker_type, checker_result))
                        if len(batch) >= self._batch_size:
                            self._flush_batch(batch)
                            batch = []
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._exc = exc

    def _flush_batch(self, batch: list[tuple[int, str, dict[str, Any]]]) -> None:
        attempts = 0
        while True:
            try:
                count = self._service.bulk_update_eval_fail_reason(updates=batch)
                self._updated += count
                return
            except Exception as exc:
                attempts += 1
                if attempts >= self._max_retries:
                    raise exc
                time.sleep(self._retry_backoff_s * (2 ** (attempts - 1)))


__all__ = ["CompletionWriteWorker", "EvalWriteWorker", "CheckerWriteWorker"]
