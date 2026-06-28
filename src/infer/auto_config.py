from __future__ import annotations

"""Hardware- and model-aware defaults for RWKV vLLM inference services."""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AutoConfigMode = Literal["off", "balanced", "throughput"]


@dataclass(frozen=True, slots=True)
class GpuProfile:
    name: str
    total_memory_mb: int


@dataclass(frozen=True, slots=True)
class InferAutoConfig:
    max_batch_size: int
    batch_collect_ms: int
    max_num_seqs: int
    max_num_batched_tokens: int
    rwkv_prefill_token_budget: int
    rwkv_prefill_max_batch_size: int
    max_state_slots: int
    gpu_memory_utilization: float
    model_params_b: float | None
    gpu_total_memory_mb: int | None
    mode: AutoConfigMode
    reason: str


def detect_visible_gpu_profile() -> GpuProfile | None:
    """Return the profile for CUDA device 0 after CUDA_VISIBLE_DEVICES remapping."""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return None
    total_mb = int(math.floor(float(getattr(props, "total_memory", 0)) / (1024 * 1024)))
    name = str(getattr(props, "name", "") or "cuda:0")
    if total_mb <= 0:
        return None
    return GpuProfile(name=name, total_memory_mb=total_mb)


def infer_model_params_b(*, model_path: str | Path, model_name: str | None = None) -> float | None:
    text = " ".join(part for part in (str(model_name or ""), Path(str(model_path)).stem) if part)
    match = re.search(r"(?P<params>\d+(?:[._]\d+)?)\s*b\b", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(?P<params>\d+(?:[._]\d+)?)b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group("params").replace("_", "."))
    except ValueError:
        return None


def choose_infer_auto_config(
    *,
    model_path: str | Path,
    model_name: str | None = None,
    gpu_profile: GpuProfile | None = None,
    mode: AutoConfigMode = "balanced",
) -> InferAutoConfig:
    if mode == "off":
        raise ValueError("auto config mode 'off' does not select a profile")

    params_b = infer_model_params_b(model_path=model_path, model_name=model_name)
    memory_mb = gpu_profile.total_memory_mb if gpu_profile is not None else None
    memory_gb = float(memory_mb or 0) / 1024.0

    tier = _memory_tier(memory_gb)
    tier = _adjust_tier_for_model_size(tier, params_b=params_b)
    if mode == "balanced":
        tier = max(0, tier - 1)

    spec = _tier_spec(tier)
    reason = (
        f"mode={mode}, gpu_memory_mb={memory_mb if memory_mb is not None else 'unknown'}, "
        f"model_params_b={params_b if params_b is not None else 'unknown'}, tier={tier}"
    )
    return InferAutoConfig(
        max_batch_size=spec["max_batch_size"],
        batch_collect_ms=spec["batch_collect_ms"],
        max_num_seqs=spec["max_num_seqs"],
        max_num_batched_tokens=spec["max_num_batched_tokens"],
        rwkv_prefill_token_budget=spec["rwkv_prefill_token_budget"],
        rwkv_prefill_max_batch_size=spec["rwkv_prefill_max_batch_size"],
        max_state_slots=spec["max_state_slots"],
        gpu_memory_utilization=spec["gpu_memory_utilization"],
        model_params_b=params_b,
        gpu_total_memory_mb=memory_mb,
        mode=mode,
        reason=reason,
    )


def _memory_tier(memory_gb: float) -> int:
    if memory_gb >= 80.0:
        return 4
    if memory_gb >= 48.0:
        return 3
    if memory_gb >= 32.0:
        return 2
    if memory_gb >= 20.0:
        return 1
    return 0


def _adjust_tier_for_model_size(tier: int, *, params_b: float | None) -> int:
    if params_b is None:
        return tier
    if params_b >= 14.0:
        return max(0, tier - 2)
    if params_b >= 7.0 and tier < 4:
        return max(0, tier - 1)
    return tier


def _tier_spec(tier: int) -> dict[str, int | float]:
    specs: tuple[dict[str, int | float], ...] = (
        {
            "max_batch_size": 128,
            "batch_collect_ms": 25,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 8192,
            "rwkv_prefill_token_budget": 2048,
            "rwkv_prefill_max_batch_size": 128,
            "max_state_slots": 2048,
            "gpu_memory_utilization": 0.90,
        },
        {
            "max_batch_size": 256,
            "batch_collect_ms": 35,
            "max_num_seqs": 512,
            "max_num_batched_tokens": 16384,
            "rwkv_prefill_token_budget": 4096,
            "rwkv_prefill_max_batch_size": 256,
            "max_state_slots": 4096,
            "gpu_memory_utilization": 0.94,
        },
        {
            "max_batch_size": 512,
            "batch_collect_ms": 50,
            "max_num_seqs": 1024,
            "max_num_batched_tokens": 24576,
            "rwkv_prefill_token_budget": 4096,
            "rwkv_prefill_max_batch_size": 512,
            "max_state_slots": 6144,
            "gpu_memory_utilization": 0.96,
        },
        {
            "max_batch_size": 768,
            "batch_collect_ms": 50,
            "max_num_seqs": -1,
            "max_num_batched_tokens": 32768,
            "rwkv_prefill_token_budget": 8192,
            "rwkv_prefill_max_batch_size": 768,
            "max_state_slots": -1,
            "gpu_memory_utilization": 0.97,
        },
        {
            "max_batch_size": 1024,
            "batch_collect_ms": 50,
            "max_num_seqs": -1,
            "max_num_batched_tokens": 32768,
            "rwkv_prefill_token_budget": 8192,
            "rwkv_prefill_max_batch_size": 1024,
            "max_state_slots": -1,
            "gpu_memory_utilization": 0.98,
        },
    )
    return specs[max(0, min(int(tier), len(specs) - 1))]
