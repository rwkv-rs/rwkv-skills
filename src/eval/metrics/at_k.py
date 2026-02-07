from __future__ import annotations

"""Shared helpers for pass@k / avg@k style aggregation.

These utilities operate over canonical evaluator outputs keyed by:
- sample_index (problem index)
- repeat_index (repeat/sample id)
"""

from collections import defaultdict
from typing import Iterable, Sequence


def estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    if total - correct < k:
        return 1.0
    product = 1.0
    for n in range(total - correct + 1, total + 1):
        product *= 1.0 - k / n
    return 1.0 - product


def compute_pass_at_k(
    rows: Iterable[tuple[int, int, bool]],
    ks: Sequence[int],
) -> dict[str, float]:
    # 去重：对于相同的 (sample_index, repeat_index)，只保留第一个
    seen: set[tuple[int, int]] = set()
    grouped: dict[int, list[bool]] = defaultdict(list)
    for sample_index, repeat_index, passed in rows:
        key = (int(sample_index), int(repeat_index))
        if key in seen:
            continue
        seen.add(key)
        grouped[int(sample_index)].append(bool(passed))

    totals = [len(flags) for flags in grouped.values()]
    corrects = [sum(1 for flag in flags if flag) for flags in grouped.values()]

    metrics: dict[str, float] = {}
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        values: list[float] = []
        for total, correct in zip(totals, corrects):
            if total < k:
                continue
            values.append(estimate_pass_at_k(total, correct, k))
        if values:
            metrics[f"pass@{k}"] = sum(values) / len(values)
    return metrics


def compute_avg_at_k(
    rows: Iterable[tuple[int, int, bool]],
    ks: Sequence[int],
) -> dict[str, float]:
    grouped: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for sample_index, repeat_index, passed in rows:
        grouped[int(sample_index)].append((int(repeat_index), bool(passed)))

    metrics: dict[str, float] = {}
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        correct = 0
        total = 0
        for entries in grouped.values():
            ordered = sorted(entries, key=lambda pair: pair[0])
            if len(ordered) < k:
                continue
            selected = ordered[:k]
            correct += sum(1 for _, flag in selected if flag)
            total += k
        if total > 0:
            metrics[f"avg@{k}"] = correct / total
    return metrics


__all__ = ["estimate_pass_at_k", "compute_avg_at_k", "compute_pass_at_k"]

