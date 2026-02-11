from __future__ import annotations

"""Loader for the rapid-sampling CUDA/ROCm extension."""

from functools import lru_cache
import os
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.cpp_extension import load


class RapidSamplingModule(Protocol):
    def setup_rand(self, seed: int, batch_size: int) -> torch.Tensor:  # pragma: no cover - extension protocol
        ...

    def batch_sampling_temperature_topk_topp(
        self,
        logits: torch.Tensor,
        states: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:  # pragma: no cover - extension protocol
        ...

    def batch_sampling_repetition_temperature_topk_topp(
        self,
        logits: torch.Tensor,
        penalties: torch.Tensor,
        states: torch.Tensor,
        presence_penalty: float,
        repetition_penalty: float,
        penalty_decay: float,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:  # pragma: no cover - extension protocol
        ...


@lru_cache(maxsize=1)
def get_rapid_sampling_module() -> RapidSamplingModule:
    """JIT-compile and load the rapid-sampling extension on first use."""

    if not torch.cuda.is_available():
        raise RuntimeError("rapid-sampling requires a CUDA/ROCm-enabled torch runtime")

    base_dir = Path(__file__).resolve().parent / "rapid_sampling"
    rocm = torch.version.hip is not None

    if rocm:
        name = "rwkv_rapid_sampling_rocm"
        sources = [base_dir / "hip" / "sampling_op.hip", base_dir / "hip" / "sampling.hip"]
        extra_cuda_cflags = ["-fopenmp", "-ffast-math", "-O3", "-munsafe-fp-atomics"]
    else:
        name = "rwkv_rapid_sampling_cuda"
        sources = [base_dir / "sampling.cpp", base_dir / "sampling.cu"]
        extra_cuda_cflags = ["-O3", "-res-usage", "--extra-device-vectorization"]
        if os.name != "nt":
            extra_cuda_cflags.append("-Xptxas -O3")

    module = load(
        name=name,
        sources=[str(path) for path in sources],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
    return module  # type: ignore[return-value]


__all__ = ["get_rapid_sampling_module"]
