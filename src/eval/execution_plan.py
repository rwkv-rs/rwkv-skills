from __future__ import annotations

"""avg@k execution planning for knowledge-style evaluators."""

from dataclasses import dataclass
import math
import random

from src.eval.scheduler.dataset_utils import canonical_slug


TARGET_EVAL_ATTEMPTS = 5000
_AVG_K_SAMPLE_BASE_SEED = 0xA11CE5EED5EED123


@dataclass(frozen=True, slots=True)
class AvgKExecutionPlan:
    avg_k: float
    repeat_count: int
    sample_indices: tuple[int, ...]

    @property
    def sample_size(self) -> int:
        return len(self.sample_indices)

    @property
    def effective_sample_count(self) -> int:
        return self.repeat_count * self.sample_size


def build_auto_avg_k_execution_plan(
    dataset_slug: str,
    dataset_len: int,
    *,
    target_attempts: int = TARGET_EVAL_ATTEMPTS,
) -> AvgKExecutionPlan:
    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0")
    if target_attempts <= 0:
        raise ValueError("target_attempts must be > 0")

    if dataset_len > target_attempts:
        avg_k = target_attempts / dataset_len
        return build_avg_k_execution_plan(dataset_slug, dataset_len, avg_k=avg_k)

    repeat_count = max(1, math.ceil(target_attempts / dataset_len))
    return AvgKExecutionPlan(
        avg_k=float(repeat_count),
        repeat_count=repeat_count,
        sample_indices=tuple(range(dataset_len)),
    )


def build_avg_k_execution_plan(
    dataset_slug: str,
    dataset_len: int,
    *,
    avg_k: float,
) -> AvgKExecutionPlan:
    if not math.isfinite(avg_k) or avg_k <= 0.0:
        raise ValueError(f"invalid avg_k={avg_k!r}; avg_k must be finite and > 0")
    if dataset_len <= 0:
        raise ValueError("dataset_len must be > 0")

    if avg_k < 1.0:
        sample_size = compute_ratio_sample_size(dataset_len, avg_k)
        ratio_key = int(round(avg_k * 1_000_000_000))
        seed = _AVG_K_SAMPLE_BASE_SEED ^ fnv1a_hash64(canonical_slug(dataset_slug).encode("utf-8")) ^ ratio_key
        sample_indices = deterministic_sample_indices(dataset_len, sample_size, seed)
        return AvgKExecutionPlan(
            avg_k=float(avg_k),
            repeat_count=1,
            sample_indices=tuple(sample_indices),
        )

    rounded = round(avg_k)
    if abs(avg_k - rounded) > 1e-9:
        raise ValueError(f"invalid avg_k={avg_k!r}; avg_k >= 1 must be an integer repeat count")
    repeat_count = max(1, int(rounded))
    return AvgKExecutionPlan(
        avg_k=float(repeat_count),
        repeat_count=repeat_count,
        sample_indices=tuple(range(dataset_len)),
    )


def compute_ratio_sample_size(total_len: int, ratio: float) -> int:
    return max(1, min(total_len, int(round(total_len * ratio))))


def deterministic_sample_indices(total_len: int, sample_size: int, seed: int) -> list[int]:
    if sample_size <= 0 or sample_size > total_len:
        raise ValueError(f"invalid sample_size={sample_size} for total_len={total_len}")
    rng = random.Random(seed)
    sampled = rng.sample(range(total_len), sample_size)
    sampled.sort()
    return sampled


def fnv1a_hash64(raw: bytes) -> int:
    value = 0xCBF29CE484222325
    for byte in raw:
        value ^= int(byte)
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def format_avg_k(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


def avg_k_metric_key(value: float) -> str:
    return f"avg@{format_avg_k(value)}"


__all__ = [
    "AvgKExecutionPlan",
    "TARGET_EVAL_ATTEMPTS",
    "avg_k_metric_key",
    "build_auto_avg_k_execution_plan",
    "build_avg_k_execution_plan",
    "compute_ratio_sample_size",
    "deterministic_sample_indices",
    "format_avg_k",
]
