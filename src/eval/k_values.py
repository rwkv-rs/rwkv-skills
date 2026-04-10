from __future__ import annotations

"""Shared helpers for pass@k / avg@k configuration and metric filtering."""

from typing import TypeAlias

NumericK: TypeAlias = int | float


def format_k_value(value: NumericK) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.12g}"


def parse_metric_suffix(key: str, prefix: str) -> float | None:
    if not key.startswith(prefix):
        return None
    suffix_text = key[len(prefix) :].strip()
    if not suffix_text:
        return None
    try:
        return float(suffix_text)
    except ValueError:
        return None


def filter_metrics_by_k(
    metric_map: dict[str, float] | None,
    ks: tuple[NumericK, ...],
    prefix: str,
) -> dict[str, float]:
    if not metric_map or not ks:
        return {}
    allowed = {
        format_k_value(k)
        for k in ks
        if isinstance(k, (int, float)) and not isinstance(k, bool) and float(k) > 0
    }
    filtered: dict[str, float] = {}
    for key, value in metric_map.items():
        suffix = parse_metric_suffix(key, prefix)
        if suffix is None:
            continue
        if format_k_value(suffix) in allowed:
            filtered[key] = value
    return filtered


def max_generation_k(values: tuple[NumericK, ...] | None) -> int:
    if not values:
        return 0
    best = 0
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if number < 1 or not number.is_integer():
            continue
        best = max(best, int(number))
    return best


__all__ = [
    "NumericK",
    "filter_metrics_by_k",
    "format_k_value",
    "max_generation_k",
    "parse_metric_suffix",
]
