from __future__ import annotations

"""Remote inference concurrency probes for scheduler sizing."""

from dataclasses import asdict, dataclass
import json
import threading
import time
from pathlib import Path
from typing import Sequence

from src.infer.backend import RemoteInferenceBackend, RemoteInferenceConfig, RemoteInferenceProtocol
from src.infer.sampling import SamplingConfig

try:  # pragma: no cover - environment dependent
    import pynvml
except Exception:  # pragma: no cover - environment dependent
    pynvml = None


DEFAULT_REMOTE_CONCURRENCY_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_REMOTE_PROBE_PROMPT = "User: Reply with exactly one short sentence.\n\nAssistant:"


@dataclass(slots=True, frozen=True)
class RemoteProbePoint:
    concurrency: int
    status: str
    elapsed_s: float
    request_count: int
    output_chars: int
    successful_request_count: int | None = None
    rps: float | None = None
    output_chars_per_s: float | None = None
    avg_latency_s: float | None = None
    p50_latency_s: float | None = None
    p95_latency_s: float | None = None
    max_latency_s: float | None = None
    avg_gpu_utilization: float | None = None
    peak_gpu_utilization: float | None = None
    peak_memory_used_mb: float | None = None
    peak_memory_fraction: float | None = None
    error: str | None = None


@dataclass(slots=True, frozen=True)
class RemoteProbeResult:
    base_url: str
    model: str
    protocol: str
    selected_concurrency: int | None
    throughput_best_concurrency: int | None
    gpu_full_concurrency: int | None
    largest_successful_concurrency: int | None
    suggested_infer_max_workers: int | None
    suggested_remote_batch_size: int | None
    target_gpu_utilization: float | None
    saturating_concurrency: int | None
    points: tuple[RemoteProbePoint, ...]
    cold_first_request_latency_s: float | None = None
    cold_first_request_error: str | None = None
    warmup_request_count: int = 0
    warmup_elapsed_s: float | None = None
    warmup_output_chars: int = 0
    warmup_error: str | None = None
    max_p95_latency_s: float | None = None
    min_throughput_gain: float = 0.03

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["points"] = [asdict(point) for point in self.points]
        return payload


def probe_remote_inference(
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    timeout_s: float = 600.0,
    protocol: RemoteInferenceProtocol = "openai",
    candidates: Sequence[int] = DEFAULT_REMOTE_CONCURRENCY_CANDIDATES,
    prompt: str = DEFAULT_REMOTE_PROBE_PROMPT,
    max_tokens: int = 16,
    temperature: float = 1e-5,
    top_p: float = 0.8,
    top_k: int = 50,
    stop_suffix: str | None = None,
    gpu_index: int | None = None,
    target_gpu_utilization: float | None = 90.0,
    gpu_sample_interval_s: float = 0.05,
    warmup_requests: int = 1,
    max_p95_latency_s: float | None = None,
    min_throughput_gain: float = 0.03,
) -> RemoteProbeResult:
    normalized_candidates = tuple(sorted({int(value) for value in candidates if int(value) > 0}))
    if not normalized_candidates:
        raise ValueError("at least one positive concurrency candidate is required")
    latency_lock = threading.Lock()
    latency_sink: list[float] | None = None

    def _record_request_latency(_url: str, elapsed_s: float, ok: bool) -> None:
        if not ok:
            return
        with latency_lock:
            if latency_sink is not None:
                latency_sink.append(float(elapsed_s))

    def _begin_latency_capture() -> None:
        nonlocal latency_sink
        with latency_lock:
            latency_sink = []

    def _end_latency_capture() -> tuple[float, ...]:
        nonlocal latency_sink
        with latency_lock:
            values = tuple(latency_sink or ())
            latency_sink = None
        return values

    backend = RemoteInferenceBackend(
        RemoteInferenceConfig(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
            max_workers=max(normalized_candidates),
            protocol=protocol,
            request_latency_callback=_record_request_latency,
        )
    )
    sampling = SamplingConfig(
        max_generate_tokens=max(1, int(max_tokens)),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=max(1, int(top_k)),
    )
    cold_first_request_latency_s: float | None = None
    cold_first_request_error: str | None = None
    _begin_latency_capture()
    cold_started = time.perf_counter()
    try:
        cold_outputs = backend.generate(
            [prompt],
            sampling=sampling,
            batch_size=1,
            prompt_stop_suffixes=[(stop_suffix,)] if stop_suffix else None,
            show_progress=False,
        )
        cold_elapsed_s = max(0.0, time.perf_counter() - cold_started)
        cold_latencies = _end_latency_capture()
        cold_first_request_latency_s = _first_latency_or_elapsed(cold_latencies, cold_elapsed_s)
        if len(cold_outputs) != 1:
            cold_first_request_error = f"expected 1 cold output, got {len(cold_outputs)}"
    except BaseException as exc:
        cold_elapsed_s = max(0.0, time.perf_counter() - cold_started)
        cold_latencies = _end_latency_capture()
        cold_first_request_latency_s = _first_latency_or_elapsed(cold_latencies, cold_elapsed_s)
        cold_first_request_error = str(exc)

    normalized_warmup_requests = max(0, int(warmup_requests))
    warmup_elapsed_s: float | None = None
    warmup_output_chars = 0
    warmup_error: str | None = None
    if normalized_warmup_requests:
        warmup_prompts = [prompt] * normalized_warmup_requests
        warmup_stop_suffixes = [(stop_suffix,)] * normalized_warmup_requests if stop_suffix else None
        _begin_latency_capture()
        warmup_started = time.perf_counter()
        try:
            warmup_outputs = backend.generate(
                warmup_prompts,
                sampling=sampling,
                batch_size=min(normalized_warmup_requests, max(normalized_candidates)),
                prompt_stop_suffixes=warmup_stop_suffixes,
                show_progress=False,
            )
            warmup_elapsed_s = max(0.0, time.perf_counter() - warmup_started)
            warmup_output_chars = sum(len(output.text) for output in warmup_outputs)
            if len(warmup_outputs) != normalized_warmup_requests:
                warmup_error = f"expected {normalized_warmup_requests} warmup outputs, got {len(warmup_outputs)}"
        except BaseException as exc:
            warmup_elapsed_s = max(0.0, time.perf_counter() - warmup_started)
            warmup_error = str(exc)
        finally:
            _end_latency_capture()

    points: list[RemoteProbePoint] = []
    largest_successful: int | None = None
    saturating: int | None = None
    for concurrency in normalized_candidates:
        prompts = [prompt] * concurrency
        stop_suffixes = None
        if stop_suffix:
            stop_suffixes = [(stop_suffix,)] * concurrency
        monitor = (
            _GpuUtilizationMonitor(gpu_index=gpu_index, sample_interval_s=gpu_sample_interval_s)
            if gpu_index is not None
            else None
        )
        if monitor is not None:
            monitor.start()
        _begin_latency_capture()
        started = time.perf_counter()
        try:
            outputs = backend.generate(
                prompts,
                sampling=sampling,
                batch_size=concurrency,
                prompt_stop_suffixes=stop_suffixes,
                show_progress=False,
            )
            elapsed_s = max(0.0, time.perf_counter() - started)
            request_latencies = _end_latency_capture()
            latency_stats = _latency_stats(request_latencies)
            gpu_sample = monitor.stop() if monitor is not None else _GpuProbeSample()
            output_chars = sum(len(output.text) for output in outputs)
            if len(outputs) != concurrency:
                raise RuntimeError(f"expected {concurrency} outputs, got {len(outputs)}")
            rps = concurrency / elapsed_s if elapsed_s > 0 else None
            chars_per_s = output_chars / elapsed_s if elapsed_s > 0 else None
            if (
                saturating is None
                and target_gpu_utilization is not None
                and gpu_sample.peak_gpu_utilization is not None
                and gpu_sample.peak_gpu_utilization >= float(target_gpu_utilization)
            ):
                saturating = concurrency
            points.append(
                RemoteProbePoint(
                    concurrency=concurrency,
                    status="ok",
                    elapsed_s=elapsed_s,
                    request_count=concurrency,
                    successful_request_count=len(outputs),
                    output_chars=output_chars,
                    rps=rps,
                    output_chars_per_s=chars_per_s,
                    avg_latency_s=latency_stats["avg"],
                    p50_latency_s=latency_stats["p50"],
                    p95_latency_s=latency_stats["p95"],
                    max_latency_s=latency_stats["max"],
                    avg_gpu_utilization=gpu_sample.avg_gpu_utilization,
                    peak_gpu_utilization=gpu_sample.peak_gpu_utilization,
                    peak_memory_used_mb=gpu_sample.peak_memory_used_mb,
                    peak_memory_fraction=gpu_sample.peak_memory_fraction,
                )
            )
            largest_successful = concurrency
        except BaseException as exc:
            elapsed_s = max(0.0, time.perf_counter() - started)
            request_latencies = _end_latency_capture()
            latency_stats = _latency_stats(request_latencies)
            gpu_sample = monitor.stop() if monitor is not None else _GpuProbeSample()
            points.append(
                RemoteProbePoint(
                    concurrency=concurrency,
                    status="failed",
                    elapsed_s=elapsed_s,
                    request_count=concurrency,
                    successful_request_count=0,
                    output_chars=0,
                    avg_latency_s=latency_stats["avg"],
                    p50_latency_s=latency_stats["p50"],
                    p95_latency_s=latency_stats["p95"],
                    max_latency_s=latency_stats["max"],
                    avg_gpu_utilization=gpu_sample.avg_gpu_utilization,
                    peak_gpu_utilization=gpu_sample.peak_gpu_utilization,
                    peak_memory_used_mb=gpu_sample.peak_memory_used_mb,
                    peak_memory_fraction=gpu_sample.peak_memory_fraction,
                    error=str(exc),
                )
            )
            break

    throughput_best = _select_throughput_best_concurrency(points)
    gpu_full = _select_gpu_full_concurrency(points, target_gpu_utilization=target_gpu_utilization)
    selected = _select_probe_concurrency(
        points,
        saturating_concurrency=saturating,
        max_p95_latency_s=max_p95_latency_s,
        min_throughput_gain=min_throughput_gain,
    ) or gpu_full or throughput_best or largest_successful
    return RemoteProbeResult(
        base_url=base_url,
        model=model,
        protocol=protocol,
        selected_concurrency=selected,
        throughput_best_concurrency=throughput_best,
        gpu_full_concurrency=gpu_full,
        largest_successful_concurrency=largest_successful,
        suggested_infer_max_workers=selected,
        suggested_remote_batch_size=selected,
        target_gpu_utilization=target_gpu_utilization,
        saturating_concurrency=saturating,
        cold_first_request_latency_s=cold_first_request_latency_s,
        cold_first_request_error=cold_first_request_error,
        warmup_request_count=normalized_warmup_requests,
        warmup_elapsed_s=warmup_elapsed_s,
        warmup_output_chars=warmup_output_chars,
        warmup_error=warmup_error,
        max_p95_latency_s=max_p95_latency_s,
        min_throughput_gain=float(min_throughput_gain),
        points=tuple(points),
    )


def _select_probe_concurrency(
    points: Sequence[RemoteProbePoint],
    *,
    saturating_concurrency: int | None,
    max_p95_latency_s: float | None = None,
    min_throughput_gain: float = 0.03,
) -> int | None:
    ok_points = _latency_eligible_points(points, max_p95_latency_s=max_p95_latency_s)
    if not ok_points:
        return None
    if saturating_concurrency is not None:
        for point in ok_points:
            if int(point.concurrency) == int(saturating_concurrency):
                return int(point.concurrency)
    best = max(ok_points, key=_throughput_value)
    best_throughput = _throughput_value(best)
    if best_throughput <= 0.0:
        return int(best.concurrency)
    threshold = best_throughput * max(0.0, 1.0 - float(min_throughput_gain))
    plateau = [point for point in ok_points if _throughput_value(point) >= threshold]
    if plateau:
        return min(int(point.concurrency) for point in plateau)
    return int(best.concurrency)


def _select_throughput_best_concurrency(points: Sequence[RemoteProbePoint]) -> int | None:
    ok_points = [point for point in points if point.status == "ok"]
    if not ok_points:
        return None
    best = max(ok_points, key=lambda point: (_throughput_value(point), int(point.concurrency)))
    return int(best.concurrency)


def _latency_eligible_points(
    points: Sequence[RemoteProbePoint],
    *,
    max_p95_latency_s: float | None,
) -> list[RemoteProbePoint]:
    ok_points = [point for point in points if point.status == "ok"]
    if max_p95_latency_s is None:
        return ok_points
    limit = float(max_p95_latency_s)
    return [
        point
        for point in ok_points
        if point.p95_latency_s is None or float(point.p95_latency_s) <= limit
    ]


def _throughput_value(point: RemoteProbePoint) -> float:
    if point.output_chars_per_s is not None:
        return float(point.output_chars_per_s)
    if point.rps is not None:
        return float(point.rps)
    return 0.0


def _select_gpu_full_concurrency(
    points: Sequence[RemoteProbePoint],
    *,
    target_gpu_utilization: float | None,
) -> int | None:
    if target_gpu_utilization is None:
        return None
    target = float(target_gpu_utilization)
    ok_points = [point for point in points if point.status == "ok"]
    avg_full = [
        point
        for point in ok_points
        if point.avg_gpu_utilization is not None and float(point.avg_gpu_utilization) >= target
    ]
    if avg_full:
        return max(int(point.concurrency) for point in avg_full)
    peak_full = [
        point
        for point in ok_points
        if point.peak_gpu_utilization is not None and float(point.peak_gpu_utilization) >= target
    ]
    if peak_full:
        return max(int(point.concurrency) for point in peak_full)
    return None


def _latency_stats(latencies: Sequence[float]) -> dict[str, float | None]:
    values = sorted(float(value) for value in latencies if float(value) >= 0.0)
    if not values:
        return {"avg": None, "p50": None, "p95": None, "max": None}
    return {
        "avg": sum(values) / len(values),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
        "max": max(values),
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * min(max(float(percentile), 0.0), 100.0) / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def _first_latency_or_elapsed(latencies: Sequence[float], elapsed_s: float) -> float:
    if latencies:
        return float(latencies[0])
    return float(elapsed_s)


def write_remote_probe_result(path: Path, result: RemoteProbeResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@dataclass(slots=True, frozen=True)
class _GpuProbeSample:
    avg_gpu_utilization: float | None = None
    peak_gpu_utilization: float | None = None
    peak_memory_used_mb: float | None = None
    peak_memory_fraction: float | None = None


class _GpuUtilizationMonitor:
    def __init__(self, *, gpu_index: int, sample_interval_s: float = 0.05) -> None:
        self.gpu_index = int(gpu_index)
        self.sample_interval_s = max(float(sample_interval_s), 0.01)
        self._handle = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._util_samples: list[float] = []
        self._memory_samples: list[tuple[int, int]] = []

    def start(self) -> None:
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._sample_once()
        except Exception:
            self._handle = None
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="remote-infer-gpu-probe", daemon=True)
        self._thread.start()

    def stop(self) -> _GpuProbeSample:
        if self._handle is None:
            return _GpuProbeSample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample_once()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        avg_util = sum(self._util_samples) / len(self._util_samples) if self._util_samples else None
        peak_util = max(self._util_samples) if self._util_samples else None
        peak_memory = max((used for used, _total in self._memory_samples), default=None)
        total_memory = max((total for _used, total in self._memory_samples), default=None)
        peak_memory_mb = None if peak_memory is None else peak_memory / (1024**2)
        peak_memory_fraction = None
        if peak_memory is not None and total_memory:
            peak_memory_fraction = peak_memory / total_memory
        return _GpuProbeSample(
            avg_gpu_utilization=avg_util,
            peak_gpu_utilization=peak_util,
            peak_memory_used_mb=peak_memory_mb,
            peak_memory_fraction=peak_memory_fraction,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            time.sleep(self.sample_interval_s)

    def _sample_once(self) -> None:
        if self._handle is None:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        except Exception:
            return
        self._util_samples.append(float(getattr(util, "gpu", 0.0)))
        self._memory_samples.append((int(mem.used), int(mem.total)))


__all__ = [
    "DEFAULT_REMOTE_CONCURRENCY_CANDIDATES",
    "DEFAULT_REMOTE_PROBE_PROMPT",
    "RemoteProbePoint",
    "RemoteProbeResult",
    "probe_remote_inference",
    "write_remote_probe_result",
]
