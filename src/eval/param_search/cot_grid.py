from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Iterable

from src.infer.sampling import SamplingConfig


NORMAL_COT_GRID: dict[str, tuple[float, ...]] = {
    "temperature": (0.3, 0.4, 0.6, 0.8),
    "top_p": (0.3, 0.4, 0.5, 0.6),
    "alpha_presence": (0.5, 1.0, 1.5, 2.0),
    "alpha_frequency": (0.1, 0.3, 0.5),
    "alpha_decay": (0.99,),
}

SIMPLE_COT_GRID: dict[str, tuple[float, ...]] = {
    "temperature": (0.8, 0.6, 0.4, 0.3),
    "noise": (1.0, 2.0, 3.0),
}


def grid_size_by_mode() -> dict[str, int]:
    normal = (
        len(NORMAL_COT_GRID["temperature"])
        * len(NORMAL_COT_GRID["top_p"])
        * len(NORMAL_COT_GRID["alpha_presence"])
        * len(NORMAL_COT_GRID["alpha_frequency"])
        * len(NORMAL_COT_GRID["alpha_decay"])
    )
    simple = len(SIMPLE_COT_GRID["temperature"]) * len(SIMPLE_COT_GRID["noise"])
    return {"normal": normal, "simple": simple}


def total_grid_size() -> int:
    sizes = grid_size_by_mode()
    return int(sizes["normal"]) + int(sizes["simple"])


def grid_size(scan_mode: str = "both") -> int:
    mode = (scan_mode or "both").strip().lower()
    sizes = grid_size_by_mode()
    if mode == "both":
        return int(sizes["normal"]) + int(sizes["simple"])
    if mode in sizes:
        return int(sizes[mode])
    raise ValueError(f"未知的 scan_mode: {scan_mode!r} (expected: both/normal/simple)")


def iter_cot_sampling_grid(
    base: SamplingConfig,
    *,
    scan_mode: str = "both",
) -> Iterable[tuple[int, SamplingConfig, dict[str, object]]]:
    """Yield (trial_index, cot_sampling_cfg, params_dict) in a stable order.

    Order: all normal grid points first, then all simple grid points.
    """

    mode = (scan_mode or "both").strip().lower()
    if mode not in {"both", "normal", "simple"}:
        raise ValueError(f"未知的 scan_mode: {scan_mode!r} (expected: both/normal/simple)")
    include_normal = mode in {"both", "normal"}
    include_simple = mode in {"both", "simple"}

    trial_idx = 0
    normal_grid = product(
        NORMAL_COT_GRID["temperature"],
        NORMAL_COT_GRID["top_p"],
        NORMAL_COT_GRID["alpha_presence"],
        NORMAL_COT_GRID["alpha_frequency"],
        NORMAL_COT_GRID["alpha_decay"],
    )
    for temperature, top_p, alpha_presence, alpha_frequency, alpha_decay in normal_grid:
        cfg = replace(
            base,
            sample_mode="normal",
            noise=0.0,
            temperature=float(temperature),
            top_p=float(top_p),
            alpha_presence=float(alpha_presence),
            alpha_frequency=float(alpha_frequency),
            alpha_decay=float(alpha_decay),
        )
        params = {
            "sample_mode": "normal",
            "temperature": float(temperature),
            "top_p": float(top_p),
            "alpha_presence": float(alpha_presence),
            "alpha_frequency": float(alpha_frequency),
            "alpha_decay": float(alpha_decay),
        }
        if include_normal:
            yield trial_idx, cfg, params
        trial_idx += 1

    simple_grid = product(
        SIMPLE_COT_GRID["temperature"],
        SIMPLE_COT_GRID["noise"],
    )
    for temperature, noise in simple_grid:
        cfg = replace(
            base,
            sample_mode="simple",
            temperature=float(temperature),
            noise=float(noise),
        )
        params = {
            "sample_mode": "simple",
            "temperature": float(temperature),
            "noise": float(noise),
        }
        if include_simple:
            yield trial_idx, cfg, params
        trial_idx += 1


__all__ = [
    "NORMAL_COT_GRID",
    "SIMPLE_COT_GRID",
    "grid_size_by_mode",
    "grid_size",
    "total_grid_size",
    "iter_cot_sampling_grid",
]
