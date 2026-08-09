"""Small fail-fast helpers for runner-side episode concurrency."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

_LOG = logging.getLogger(__name__)


def run_episodes(
    items: Iterable[T],
    worker: Callable[[T], R],
    *,
    max_workers: int = 1,
    on_result: Callable[[R], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    label: str = "episode",
    collect_results: bool = True,
) -> list[R]:
    """Run independent episode workers and re-raise infrastructure failures.

    Worker exceptions are never converted into sample-level failures here. A
    runner that wants a scoreable failed sample should return that payload from
    its worker instead of raising. In threaded mode, a failure cancels work that
    has not started and raises without joining in-flight workers. Python cannot
    interrupt threads already blocked inside backend/generate HTTP calls, so
    those calls may continue until their backend/HTTP timeout and may still
    delay process shutdown.
    """

    rows = list(items)
    if not rows:
        return []
    workers = max(1, int(max_workers))
    if workers == 1:
        return _run_serial(
            rows,
            worker,
            on_result=on_result,
            on_progress=on_progress,
            label=label,
            collect_results=collect_results,
        )
    return _run_threaded(
        rows,
        worker,
        max_workers=workers,
        on_result=on_result,
        on_progress=on_progress,
        label=label,
        collect_results=collect_results,
    )


def _run_serial(
    rows: list[T],
    worker: Callable[[T], R],
    *,
    on_result: Callable[[R], None] | None,
    on_progress: Callable[[int, int], None] | None,
    label: str,
    collect_results: bool,
) -> list[R]:
    results: list[R] = []
    total = len(rows)
    for index, item in enumerate(rows):
        try:
            result = worker(item)
        except Exception:
            _LOG.exception("%s %s failed", label, index)
            raise
        if collect_results:
            results.append(result)
        if on_result is not None:
            on_result(result)
        if on_progress is not None:
            on_progress(index + 1, total)
    return results


def _run_threaded(
    rows: list[T],
    worker: Callable[[T], R],
    *,
    max_workers: int,
    on_result: Callable[[R], None] | None,
    on_progress: Callable[[int, int], None] | None,
    label: str,
    collect_results: bool,
) -> list[R]:
    results_by_index: dict[int, R] = {}
    total = len(rows)
    executor = ThreadPoolExecutor(max_workers=min(max_workers, len(rows)))
    futures: dict[Future[R], int] = {
        executor.submit(worker, item): index
        for index, item in enumerate(rows)
    }
    failed = False
    done = 0
    try:
        for future in as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except Exception:
                failed = True
                _LOG.exception(
                    "%s %s failed; cancelling pending episodes. In-flight workers already inside "
                    "backend/generate calls cannot be interrupted and may run until the backend/HTTP timeout.",
                    label,
                    index,
                )
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                raise
            try:
                if collect_results:
                    results_by_index[index] = result
                if on_result is not None:
                    on_result(result)
                done += 1
                if on_progress is not None:
                    on_progress(done, total)
            except Exception:
                failed = True
                _LOG.exception(
                    "%s %s result handler failed; cancelling pending episodes. In-flight workers already "
                    "inside backend/generate calls cannot be interrupted and may run until the backend/HTTP timeout.",
                    label,
                    index,
                )
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
                raise
    finally:
        executor.shutdown(wait=not failed, cancel_futures=True)
    if not collect_results:
        return []
    return [results_by_index[index] for index in range(len(rows)) if index in results_by_index]


__all__ = ["run_episodes"]
