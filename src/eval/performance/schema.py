from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any, Sequence
import math


def _quantile(values: Sequence[float], q: float) -> float | None:
    cleaned = sorted(float(v) for v in values)
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    q = min(1.0, max(0.0, float(q)))
    index = max(0, math.ceil(q * len(cleaned)) - 1)
    return cleaned[index]


def summarize_values(values: Sequence[float]) -> dict[str, float | int | None]:
    cleaned = [float(v) for v in values]
    if not cleaned:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    return {
        "count": len(cleaned),
        "mean": fmean(cleaned),
        "min": min(cleaned),
        "max": max(cleaned),
        "p50": _quantile(cleaned, 0.50),
        "p90": _quantile(cleaned, 0.90),
        "p99": _quantile(cleaned, 0.99),
    }


@dataclass(slots=True)
class ServiceRequestRecord:
    request_index: int
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    e2el_s: float
    status: str
    finish_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class PerfRunRecord:
    run_index: int
    phase: str
    request_count: int
    prompt_tokens_per_request: int
    output_tokens_target_per_request: int
    request_records: list[ServiceRequestRecord] = field(default_factory=list)
    input_tps: float | None = None
    output_tps: float | None = None
    rps: float | None = None
    baseline_vram_gb: float | None = None
    peak_vram_gb: float | None = None
    peak_vram_delta_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        ttft_values = [record.ttft_s for record in self.request_records if record.status == "ok"]
        e2el_values = [record.e2el_s for record in self.request_records if record.status == "ok"]
        output_token_values = [float(record.output_tokens) for record in self.request_records if record.status == "ok"]
        payload["ttft_summary"] = summarize_values(ttft_values)
        payload["e2el_summary"] = summarize_values(e2el_values)
        payload["output_tokens_summary"] = summarize_values(output_token_values)
        return payload


@dataclass(slots=True)
class PerfPointResult:
    point_kind: str
    ctx_len: int
    output_tokens: int
    status: str
    load_value: int
    concurrency: int | None = None
    batch_size: int | None = None
    runs: list[PerfRunRecord] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_kind": self.point_kind,
            "ctx_len": self.ctx_len,
            "load_value": self.load_value,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "output_tokens": self.output_tokens,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "summary": self.summary,
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(slots=True)
class PerfBenchmarkResult:
    schema_version: int
    protocol: str
    stack_name: str
    engine_name: str
    model_name: str
    precision: str
    service: dict[str, Any]
    hardware: dict[str, Any]
    tokenizer: dict[str, Any]
    workload: dict[str, Any]
    points: list[PerfPointResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol,
            "stack_name": self.stack_name,
            "engine_name": self.engine_name,
            "model_name": self.model_name,
            "precision": self.precision,
            "service": self.service,
            "hardware": self.hardware,
            "tokenizer": self.tokenizer,
            "workload": self.workload,
            "points": [point.to_dict() for point in self.points],
        }


__all__ = [
    "PerfBenchmarkResult",
    "PerfPointResult",
    "PerfRunRecord",
    "ServiceRequestRecord",
    "summarize_values",
]
