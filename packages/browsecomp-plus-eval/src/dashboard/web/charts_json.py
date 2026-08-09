"""Chart series as JSON for the React frontend (replaces Plotly figures).

Each builder reuses the same data-extraction logic the old Plotly charts used,
but returns plain dictionaries that a lightweight charting library (Recharts /
ECharts) can render directly.
"""

from __future__ import annotations

from typing import Any

from ..core.constants import AIME_BASES, INSTRUCTION_DOMAIN_ORDER, SelectionState
from ..core.domains import (
    is_coding_domain,
    is_instruction_following_domain,
    is_knowledge_group_domain,
)
from ..core.metrics import (
    _best_numeric_metric,
    _dataset_base,
    _extract_pass_curve,
    _normalize_subject_label,
    _numeric_value,
)
from ..core.selection import _model_display_name, _series_sort_key


def _ordered_models(models: set[str], label_to_model: dict[str, str]) -> list[str]:
    return sorted(models, key=lambda label: _series_sort_key(label_to_model.get(label, label)))


def _knowledge_bar(selection: SelectionState) -> dict[str, Any] | None:
    subject_scores: dict[str, dict[str, float]] = {}
    for entry in selection.entries:
        if not is_knowledge_group_domain(entry.domain):
            continue
        acc_map = (entry.task_details or {}).get("accuracy_by_subject")
        if not isinstance(acc_map, dict):
            continue
        model = _model_display_name(entry.model)
        for raw_name, raw_val in acc_map.items():
            canonical = _normalize_subject_label(str(raw_name))
            score = _numeric_value(raw_val)
            if canonical and score is not None:
                bucket = subject_scores.setdefault(model, {})
                if canonical not in bucket or score > bucket[canonical]:
                    bucket[canonical] = score

    data: list[dict[str, Any]] = []
    subject_sums: dict[str, list[float]] = {}
    label_to_model = {_model_display_name(e.model): e.model for e in selection.entries}
    for model, scores in subject_scores.items():
        for subject, score in scores.items():
            label = subject.replace("_", " ").title()
            data.append({"model": model, "subject": label, "score": score})
            subject_sums.setdefault(label, []).append(score)
    if not data:
        return None

    subject_order = sorted(subject_sums, key=lambda s: sum(subject_sums[s]) / len(subject_sums[s]), reverse=True)
    models = _ordered_models({d["model"] for d in data}, label_to_model)
    return {"type": "knowledge_bar", "subjects": subject_order, "models": models, "data": data}


def _aime_curves(selection: SelectionState) -> dict[str, Any] | None:
    series: dict[str, list[dict[str, float]]] = {}
    order: dict[str, tuple[Any, ...]] = {}
    ks: set[int] = set()
    for entry in selection.entries:
        base = _dataset_base(entry.dataset).lower()
        if base not in AIME_BASES:
            continue
        curve = _extract_pass_curve(entry)
        if not curve:
            continue
        name = f"{base.upper()} · {_model_display_name(entry.model)}"
        order.setdefault(name, (base.upper(), *_series_sort_key(entry.model)))
        points = series.setdefault(name, [])
        for k, acc in curve.items():
            points.append({"k": int(k), "acc": float(acc)})
            ks.add(int(k))
    if not series:
        return None
    ordered = sorted(series, key=lambda name: order[name])
    return {
        "type": "aime_line",
        "ks": sorted(ks),
        "series": [{"name": name, "points": sorted(series[name], key=lambda p: p["k"])} for name in ordered],
    }


def _instruction_bar(selection: SelectionState) -> dict[str, Any] | None:
    entries = [e for e in selection.entries if is_instruction_following_domain(e.domain)]
    if not entries:
        return None
    label_to_model = {_model_display_name(e.model): e.model for e in entries}
    data: list[dict[str, Any]] = []
    seen_domains: set[str] = set()
    for entry in entries:
        details = entry.task_details or {}
        buckets: dict[str, list[float]] = {}
        for tier_key in ("tier0_accuracy", "tier1_accuracy"):
            acc_map = details.get(tier_key)
            if not isinstance(acc_map, dict):
                continue
            for name, value in acc_map.items():
                num = _numeric_value(value)
                if num is None:
                    continue
                prefix = str(name).split(":", 1)[0]
                buckets.setdefault(prefix, []).append(num)
        for domain, scores in buckets.items():
            label = domain.replace("_", " ")
            seen_domains.add(label)
            data.append({"domain": label, "model": _model_display_name(entry.model), "score": sum(scores) / len(scores)})
    if not data:
        return None
    domain_order = [d.replace("_", " ") for d in INSTRUCTION_DOMAIN_ORDER if d.replace("_", " ") in seen_domains]
    for d in sorted(seen_domains):
        if d not in domain_order:
            domain_order.append(d)
    models = _ordered_models({d["model"] for d in data}, label_to_model)
    return {"type": "instruction_bar", "domains": domain_order, "models": models, "data": data}


def _coding_bar(selection: SelectionState) -> dict[str, Any] | None:
    entries = [e for e in selection.entries if is_coding_domain(e.domain)]
    if not entries:
        return None
    label_to_model = {_model_display_name(e.model): e.model for e in entries}
    data: list[dict[str, Any]] = []
    datasets: set[str] = set()
    for entry in entries:
        base = _dataset_base(entry.dataset)
        metric_key, metric_value = _best_numeric_metric(entry, dataset_base=base)
        if metric_value is None:
            continue
        label = base.upper()
        datasets.add(label)
        data.append(
            {
                "dataset": label,
                "model": _model_display_name(entry.model),
                "score": metric_value,
                "metric": metric_key or "score",
            }
        )
    if not data:
        return None
    models = _ordered_models({d["model"] for d in data}, label_to_model)
    return {"type": "coding_bar", "datasets": sorted(datasets), "models": models, "data": data}


def serialize_charts(selection: SelectionState) -> dict[str, Any]:
    """Return chart series keyed by domain group for the dashboard."""

    return {
        "knowledge": _knowledge_bar(selection),
        "math": _aime_curves(selection),
        "instruction_following": _instruction_bar(selection),
        "coding": _coding_bar(selection),
    }


__all__ = ["serialize_charts"]
