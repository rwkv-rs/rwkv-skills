from __future__ import annotations

from dataclasses import dataclass
import threading
import time

try:  # pragma: no cover - environment dependent
    import pynvml
except Exception:  # pragma: no cover - environment dependent
    pynvml = None


@dataclass(slots=True)
class GpuMemorySample:
    baseline_used_gb: float | None
    peak_used_gb: float | None
    peak_delta_gb: float | None


class GpuMemoryMonitor:
    def __init__(self, *, gpu_index: int, sample_interval_s: float = 0.01) -> None:
        self.gpu_index = int(gpu_index)
        self.sample_interval_s = max(float(sample_interval_s), 0.001)
        self._handle = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_bytes: int | None = None
        self._peak_bytes: int | None = None

    @property
    def available(self) -> bool:
        return pynvml is not None

    def start(self) -> None:
        if not self.available:
            return
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._baseline_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        self._peak_bytes = self._baseline_bytes
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> GpuMemorySample:
        if not self.available or self._handle is None:
            return GpuMemorySample(None, None, None)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            current = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
            if self._peak_bytes is None or current > self._peak_bytes:
                self._peak_bytes = current
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

        baseline = self._baseline_bytes
        peak = self._peak_bytes
        if baseline is None or peak is None:
            return GpuMemorySample(None, None, None)
        return GpuMemorySample(
            baseline_used_gb=baseline / (1024**3),
            peak_used_gb=peak / (1024**3),
            peak_delta_gb=max(0, peak - baseline) / (1024**3),
        )

    def _run(self) -> None:
        assert self._handle is not None
        while not self._stop.is_set():
            try:
                used = int(pynvml.nvmlDeviceGetMemoryInfo(self._handle).used)
                if self._peak_bytes is None or used > self._peak_bytes:
                    self._peak_bytes = used
            except Exception:
                pass
            time.sleep(self.sample_interval_s)


__all__ = ["GpuMemoryMonitor", "GpuMemorySample"]
