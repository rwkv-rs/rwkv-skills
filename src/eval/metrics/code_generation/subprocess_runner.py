from __future__ import annotations

import multiprocessing
from multiprocessing.connection import Connection
from typing import Any, Callable, Sequence


class _PipeResultSink:
    """Minimal ``list.append`` compatible sink backed by a one-way pipe."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._sent = False

    def append(self, value: Any) -> None:
        if self._sent:
            return
        self._connection.send(value)
        self._sent = True


def run_isolated(
    target: Callable[..., None],
    args: Sequence[Any],
    *,
    timeout: float,
    fallback: str = "timed out",
) -> Any:
    """Run one code-evaluation target and always reap its child process.

    A one-way pipe is enough for the single result value.  This avoids creating
    one ``multiprocessing.Manager`` server per completion, which can accumulate
    thousands of lock-waiting processes under a threaded evaluator.
    """

    ctx = multiprocessing.get_context()
    receive_conn, send_conn = ctx.Pipe(duplex=False)
    sink = _PipeResultSink(send_conn)
    process = ctx.Process(target=target, args=(*args, sink))
    result: Any = fallback
    received = False

    try:
        process.start()
        send_conn.close()
        if receive_conn.poll(max(0.0, float(timeout)) + 1.0):
            try:
                result = receive_conn.recv()
                received = True
            except EOFError:
                result = fallback

        # The result is emitted just before final sandbox cleanup.  Give that
        # cleanup a short grace period and then guarantee process reclamation.
        process.join(timeout=0.25 if received else 0.0)
        if process.is_alive():
            kill = getattr(process, "kill", process.terminate)
            kill()
            process.join(timeout=1.0)
    finally:
        receive_conn.close()
        send_conn.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        try:
            process.close()
        except (AttributeError, ValueError):
            pass

    return result


__all__ = ["run_isolated"]
