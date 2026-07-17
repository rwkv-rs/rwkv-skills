from __future__ import annotations

"""Tooltip formatting and cell-id helpers shared with the web serializer."""

import hashlib
from typing import Any

from .constants import SUBDOMAIN_ORDER
from .data import ScoreEntry
from .domains import is_instruction_following_domain, is_multi_choice_domain
from .metrics import (
    _dataset_base,
    _format_metric_value,
    _map_subject_to_subdomain,
    _numeric_value,
)


def _sorted_numeric_items(raw_map: dict[str, Any], *, limit: int = 8) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for key, value in raw_map.items():
        num = _numeric_value(value)
        if num is None:
            continue
        items.append((str(key), num))
    items.sort(key=lambda item: (-item[1], item[0]))
    return items[: max(1, limit)]


def _format_tooltip_lines(title: str, items: list[tuple[str, float]]) -> str | None:
    if not items:
        return None
    lines = [title]
    for key, value in items:
        lines.append(f"{key}: {_format_metric_value(value)}")
    if len(lines) <= 1:
        return None
    return "\n".join(lines)


def _mmlu_tooltip(entry: ScoreEntry) -> str | None:
    details = entry.task_details or {}
    accuracy_by_subject = details.get("accuracy_by_subject")
    if not isinstance(accuracy_by_subject, dict):
        return None

    grouped: dict[str, list[float]] = {}
    for raw_subject, raw_value in accuracy_by_subject.items():
        num = _numeric_value(raw_value)
        if num is None:
            continue
        subdomain = _map_subject_to_subdomain(str(raw_subject))
        grouped.setdefault(subdomain, []).append(num)

    items: list[tuple[str, float]] = []
    for subdomain in SUBDOMAIN_ORDER:
        values = grouped.get(subdomain)
        if not values:
            continue
        items.append((subdomain.replace("_", " "), sum(values) / len(values)))

    return _format_tooltip_lines("MMLU 子领域", items[:8])


def _ifeval_tooltip(entry: ScoreEntry) -> str | None:
    details = entry.task_details or {}
    tier0 = details.get("tier0_accuracy")
    if not isinstance(tier0, dict):
        return None
    items = _sorted_numeric_items(tier0, limit=10)
    pretty = [(name.replace("_", " "), value) for name, value in items]
    return _format_tooltip_lines("IFEval 子领域", pretty)


def _metric_fallback_tooltip(entry: ScoreEntry) -> str | None:
    items = _sorted_numeric_items(entry.metrics, limit=8)
    return _format_tooltip_lines("指标明细", items)


def _tooltip_for_entry(entry: ScoreEntry) -> str | None:
    dataset = _dataset_base(entry.dataset).lower()
    if dataset.startswith("mmlu") or is_multi_choice_domain(entry.domain):
        tooltip = _mmlu_tooltip(entry)
        if tooltip:
            return tooltip
    if dataset.startswith("ifeval") or is_instruction_following_domain(entry.domain):
        tooltip = _ifeval_tooltip(entry)
        if tooltip:
            return tooltip
    return _metric_fallback_tooltip(entry)


def _make_cell_id(*parts: str) -> str:
    token = "|".join(parts)
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:16]
    return f"cell-{digest}"
