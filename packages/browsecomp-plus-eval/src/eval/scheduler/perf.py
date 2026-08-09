"""Local perf logging shim so scheduler can run without the server package."""

from __future__ import annotations


class _NoopPerfLogger:
    enabled = False

    def log(self, *_args, **_kwargs) -> None:  # pragma: no cover - noop shim
        return None


perf_logger = _NoopPerfLogger()


__all__ = ["perf_logger"]
