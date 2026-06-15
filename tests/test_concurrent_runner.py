from __future__ import annotations

import logging
import threading
import time

import pytest

from src.eval.concurrent_runner import run_episodes


def test_run_episodes_defaults_to_serial_order() -> None:
    seen: list[int] = []
    emitted: list[int] = []

    result = run_episodes(
        [1, 2, 3],
        lambda value: seen.append(value) or value * 10,
        on_result=emitted.append,
    )

    assert seen == [1, 2, 3]
    assert emitted == [10, 20, 30]
    assert result == [10, 20, 30]


def test_run_episodes_threads_results_without_reordering_return_value() -> None:
    emitted: list[int] = []

    def worker(value: int) -> int:
        time.sleep(0.01 * (3 - value))
        return value * 10

    result = run_episodes([1, 2, 3], worker, max_workers=3, on_result=emitted.append)

    assert sorted(emitted) == [10, 20, 30]
    assert result == [10, 20, 30]


def test_run_episodes_can_emit_without_collecting_results() -> None:
    emitted: list[int | None] = []

    result = run_episodes(
        [1, 2],
        lambda value: None if value == 1 else value,
        max_workers=2,
        on_result=emitted.append,
        collect_results=False,
    )

    assert set(emitted) == {None, 2}
    assert result == []


def test_run_episodes_logs_traceback_and_reraises(caplog: pytest.LogCaptureFixture) -> None:
    def worker(value: int) -> int:
        if value == 2:
            raise RuntimeError("boom")
        return value

    caplog.set_level(logging.ERROR, logger="src.eval.concurrent_runner")

    with pytest.raises(RuntimeError, match="boom"):
        run_episodes([1, 2, 3], worker, max_workers=2, label="bfcl_v3 episode")

    assert "bfcl_v3 episode" in caplog.text
    assert "HTTP timeout" in caplog.text
    assert "Traceback" in caplog.text
    assert "RuntimeError: boom" in caplog.text


def test_run_episodes_reraises_before_slow_inflight_worker_finishes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    slow_started = threading.Event()
    slow_release = threading.Event()
    slow_finished = threading.Event()

    def worker(value: str) -> str:
        if value == "slow":
            slow_started.set()
            try:
                slow_release.wait(timeout=2)
            finally:
                slow_finished.set()
            return value
        assert slow_started.wait(timeout=1)
        raise RuntimeError("boom")

    caplog.set_level(logging.ERROR, logger="src.eval.concurrent_runner")

    started_at = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            run_episodes(["slow", "fail"], worker, max_workers=2, label="bfcl_v3 episode")
        elapsed = time.monotonic() - started_at
    finally:
        slow_release.set()
        slow_finished.wait(timeout=1)

    assert elapsed < 1
    assert "In-flight workers" in caplog.text
    assert "HTTP timeout" in caplog.text
