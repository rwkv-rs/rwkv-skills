from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Iterable, Mapping, Sequence

from src.infer.sampling import SamplingConfig


COT_GRID: dict[str, tuple[float, ...]] = {
    "temperature": (0.3, 0.4, 0.6, 0.8),
    "top_p": (0.3, 0.4, 0.5, 0.6),
    "alpha_presence": (0.5, 1.0, 1.5, 2.0),
    "alpha_frequency": (0.1, 0.3, 0.5),
    "alpha_decay": (0.99,),
}


def _normalize_grid(grid: Mapping[str, Sequence[float]] | None) -> dict[str, tuple[float, ...]]:
    source = COT_GRID if grid is None else grid
    required = ("temperature", "top_p", "alpha_presence", "alpha_frequency", "alpha_decay")
    normalized: dict[str, tuple[float, ...]] = {}
    for key in required:
        values = source.get(key)
        if not values:
            raise ValueError(f"参数网格缺少字段: {key}")
        normalized[key] = tuple(float(v) for v in values)
    return normalized


def grid_size(grid: Mapping[str, Sequence[float]] | None = None) -> int:
    normalized = _normalize_grid(grid)
    return (
        len(normalized["temperature"])
        * len(normalized["top_p"])
        * len(normalized["alpha_presence"])
        * len(normalized["alpha_frequency"])
        * len(normalized["alpha_decay"])
    )


def iter_cot_sampling_grid(
    base: SamplingConfig,
    grid: Mapping[str, Sequence[float]] | None = None,
) -> Iterable[tuple[int, SamplingConfig, dict[str, object]]]:
    normalized = _normalize_grid(grid)
    trial_idx = 0
    search_space = product(
        normalized["temperature"],
        normalized["top_p"],
        normalized["alpha_presence"],
        normalized["alpha_frequency"],
        normalized["alpha_decay"],
    )
    for temperature, top_p, alpha_presence, alpha_frequency, alpha_decay in search_space:
        cfg = replace(
            base,
            temperature=float(temperature),
            top_p=float(top_p),
            alpha_presence=float(alpha_presence),
            alpha_frequency=float(alpha_frequency),
            alpha_decay=float(alpha_decay),
        )
        params = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "alpha_presence": float(alpha_presence),
            "alpha_frequency": float(alpha_frequency),
            "alpha_decay": float(alpha_decay),
        }
        yield trial_idx, cfg, params
        trial_idx += 1


__all__ = [
    "COT_GRID",
    "grid_size",
    "iter_cot_sampling_grid",
]
