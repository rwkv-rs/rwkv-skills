from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

from src.eval.performance.gpu_monitor import GpuMemoryMonitor
from src.eval.performance.schema import (
    PerfBenchmarkResult,
    PerfPointResult,
    PerfRunRecord,
    ServiceRequestRecord,
    summarize_values,
)
from src.eval.performance.service_client import OpenAIChatServiceClient
from src.eval.performance.tokenizers import BenchmarkTokenizer
from src.eval.performance.workload import build_prompt_for_target_tokens, repeat_prompts


@dataclass(slots=True)
class ServiceBenchmarkConfig:
    base_url: str
    model: str
    api_key: str
    protocol: str
    stack_name: str
    engine_name: str
    precision: str
    tokenizer_label: str
    tokenizer: BenchmarkTokenizer
    ctx_lens: tuple[int, ...]
    concurrency_levels: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    output_tokens: int
    warmup_runs: int
    measure_runs: int
    temperature: float
    top_p: float
    timeout_s: float
    base_seed: int | None
    gpu_index: int | None
    service_metadata: dict[str, Any]
    hardware_metadata: dict[str, Any]


def _build_client(config: ServiceBenchmarkConfig) -> OpenAIChatServiceClient:
    if config.protocol != "openai-chat":
        raise ValueError(f"暂不支持的协议: {config.protocol}")
    return OpenAIChatServiceClient(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout_s=config.timeout_s,
    )


def _summarize_point(point: PerfPointResult) -> dict[str, Any]:
    measured_runs = [run for run in point.runs if run.phase == "measure"]
    ttft_values = [record.ttft_s for run in measured_runs for record in run.request_records if record.status == "ok"]
    e2el_values = [record.e2el_s for run in measured_runs for record in run.request_records if record.status == "ok"]
    output_token_values = [
        float(record.output_tokens)
        for run in measured_runs
        for record in run.request_records
        if record.status == "ok"
    ]
    input_tps_values = [float(run.input_tps) for run in measured_runs if run.input_tps is not None]
    output_tps_values = [float(run.output_tps) for run in measured_runs if run.output_tps is not None]
    rps_values = [float(run.rps) for run in measured_runs if run.rps is not None]
    peak_vram_values = [float(run.peak_vram_gb) for run in measured_runs if run.peak_vram_gb is not None]
    peak_delta_values = [
        float(run.peak_vram_delta_gb) for run in measured_runs if run.peak_vram_delta_gb is not None
    ]
    total_requests = sum(run.request_count for run in measured_runs)
    failed_requests = sum(
        1
        for run in measured_runs
        for record in run.request_records
        if record.status != "ok"
    )
    return {
        "ttft_s": summarize_values(ttft_values),
        "e2el_s": summarize_values(e2el_values),
        "output_tokens": summarize_values(output_token_values),
        "input_tps": summarize_values(input_tps_values),
        "output_tps": summarize_values(output_tps_values),
        "rps": summarize_values(rps_values),
        "peak_vram_gb": summarize_values(peak_vram_values),
        "peak_vram_delta_gb": summarize_values(peak_delta_values),
        "measured_runs": len(measured_runs),
        "request_count": total_requests,
        "failed_requests": failed_requests,
        "failure_rate": (failed_requests / total_requests) if total_requests > 0 else None,
    }


def _measure_run(
    *,
    client: OpenAIChatServiceClient,
    config: ServiceBenchmarkConfig,
    prompts: list[str],
    prompt_token_count: int,
    parallel_requests: int,
    run_index: int,
    phase: str,
) -> PerfRunRecord:
    monitor = GpuMemoryMonitor(gpu_index=config.gpu_index) if config.gpu_index is not None else None
    if monitor is not None:
        monitor.start()
    wall_started = time.perf_counter()
    results = client.benchmark_many(
        prompts=prompts,
        max_tokens=config.output_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        base_seed=config.base_seed,
        max_workers=parallel_requests,
    )
    wall_finished = time.perf_counter()
    memory_sample = monitor.stop() if monitor is not None else None

    request_records: list[ServiceRequestRecord] = []
    ok_results = [result for result in results if not result.error]
    total_output_tokens = 0
    ttft_values = []
    e2el_values = []
    for result in results:
        if result.error:
            request_records.append(
                ServiceRequestRecord(
                    request_index=result.request_index,
                    prompt_tokens=prompt_token_count,
                    output_tokens=0,
                    ttft_s=0.0,
                    e2el_s=0.0,
                    status="error",
                    finish_reason=result.finish_reason,
                    error=result.error,
                )
            )
            continue
        output_tokens = len(config.tokenizer.encode(result.text))
        total_output_tokens += output_tokens
        ttft_values.append(result.ttft_s)
        e2el_values.append(result.e2el_s)
        request_records.append(
            ServiceRequestRecord(
                request_index=result.request_index,
                prompt_tokens=prompt_token_count,
                output_tokens=output_tokens,
                ttft_s=result.ttft_s,
                e2el_s=result.e2el_s,
                status="ok",
                finish_reason=result.finish_reason,
            )
        )

    prefill_window_s = max(ttft_values) if ttft_values else None
    decode_windows = [
        max(0.0, result.e2el_s - result.ttft_s)
        for result in ok_results
    ]
    decode_window_s = max(decode_windows) if decode_windows else None
    input_tps = None
    if prefill_window_s and prefill_window_s > 0:
        input_tps = (prompt_token_count * len(ok_results)) / prefill_window_s
    output_tps = None
    if decode_window_s and decode_window_s > 0:
        output_tps = total_output_tokens / decode_window_s
    rps = None
    total_wall_s = max(0.0, wall_finished - wall_started)
    if total_wall_s > 0:
        rps = len(ok_results) / total_wall_s

    return PerfRunRecord(
        run_index=run_index,
        phase=phase,
        request_count=len(results),
        prompt_tokens_per_request=prompt_token_count,
        output_tokens_target_per_request=config.output_tokens,
        request_records=request_records,
        input_tps=input_tps,
        output_tps=output_tps,
        rps=rps,
        baseline_vram_gb=None if memory_sample is None else memory_sample.baseline_used_gb,
        peak_vram_gb=None if memory_sample is None else memory_sample.peak_used_gb,
        peak_vram_delta_gb=None if memory_sample is None else memory_sample.peak_delta_gb,
    )


def _run_point(
    *,
    client: OpenAIChatServiceClient,
    config: ServiceBenchmarkConfig,
    point_kind: str,
    ctx_len: int,
    prompt_text: str,
    prompt_token_count: int,
    load_value: int,
) -> PerfPointResult:
    point = PerfPointResult(
        point_kind=point_kind,
        ctx_len=ctx_len,
        output_tokens=config.output_tokens,
        status="ok",
        load_value=load_value,
        concurrency=load_value if point_kind == "concurrency" else None,
        batch_size=load_value if point_kind == "batch_size" else None,
    )
    prompts = repeat_prompts(prompt_text, load_value)
    try:
        for run_index in range(config.warmup_runs):
            point.runs.append(
                _measure_run(
                    client=client,
                    config=config,
                    prompts=prompts,
                    prompt_token_count=prompt_token_count,
                    parallel_requests=load_value,
                    run_index=run_index,
                    phase="warmup",
                )
            )
        for run_index in range(config.measure_runs):
            point.runs.append(
                _measure_run(
                    client=client,
                    config=config,
                    prompts=prompts,
                    prompt_token_count=prompt_token_count,
                    parallel_requests=load_value,
                    run_index=run_index,
                    phase="measure",
                )
            )
    except BaseException as exc:
        point.status = "failed"
        point.failure_reason = str(exc)
    point.summary = _summarize_point(point)
    return point


def run_service_benchmark(config: ServiceBenchmarkConfig) -> PerfBenchmarkResult:
    client = _build_client(config)
    prompt_cache: dict[int, tuple[str, int]] = {}
    points: list[PerfPointResult] = []
    matrices: list[tuple[str, tuple[int, ...]]] = []

    if config.concurrency_levels:
        matrices.append(
            (
                "concurrency",
                tuple(sorted(set(int(value) for value in config.concurrency_levels))),
            )
        )
    if config.batch_sizes:
        matrices.append(
            (
                "batch_size",
                tuple(sorted(set(int(value) for value in config.batch_sizes))),
            )
        )

    for ctx_len in sorted(set(int(value) for value in config.ctx_lens)):
        prompt_text, prompt_token_count = prompt_cache.get(ctx_len, ("", 0))
        if not prompt_text:
            prompt_text, prompt_token_count = build_prompt_for_target_tokens(config.tokenizer, ctx_len)
            prompt_cache[ctx_len] = (prompt_text, prompt_token_count)

        for point_kind, levels in matrices:
            for load_value in levels:
                points.append(
                    _run_point(
                        client=client,
                        config=config,
                        point_kind=point_kind,
                        ctx_len=ctx_len,
                        prompt_text=prompt_text,
                        prompt_token_count=prompt_token_count,
                        load_value=load_value,
                    )
                )

    workload = {
        "ctx_lens": list(sorted(set(int(value) for value in config.ctx_lens))),
        "concurrency_levels": list(sorted(set(int(value) for value in config.concurrency_levels))),
        "batch_sizes": list(sorted(set(int(value) for value in config.batch_sizes))),
        "point_kinds": [point_kind for point_kind, _ in matrices],
        "output_tokens": int(config.output_tokens),
        "warmup_runs": int(config.warmup_runs),
        "measure_runs": int(config.measure_runs),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "seed": config.base_seed,
    }

    return PerfBenchmarkResult(
        schema_version=2,
        protocol=config.protocol,
        stack_name=config.stack_name,
        engine_name=config.engine_name,
        model_name=config.model,
        precision=config.precision,
        service=config.service_metadata,
        hardware=config.hardware_metadata,
        tokenizer={
            "label": config.tokenizer_label,
        },
        workload=workload,
        points=points,
    )


def write_benchmark_result(path: Path, result: PerfBenchmarkResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


__all__ = [
    "ServiceBenchmarkConfig",
    "run_service_benchmark",
    "write_benchmark_result",
]
