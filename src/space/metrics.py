"""Metric extraction, score formatting, cell styling, and subdomain mapping.

This module is a pure-logic layer with no I/O or Gradio dependency.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from .data import ScoreEntry
from .constants import (
    AIME_BASES,
    IFEVAL_BASES,
    MATH500_BASES,
    PRIMARY_KEYS,
    SUBDOMAIN_KEYWORDS,
    SUBDOMAIN_ORDER,
    DetailPoint,
)


# ---------------------------------------------------------------------------
# Naming / normalisation helpers
# ---------------------------------------------------------------------------

def _dataset_base(name: str) -> str:
    for suffix in ("_test", "_eval", "_val"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _normalize_token(text: str) -> str:
    """Normalise a subject/label string for keyword matching.

    NOTE: This variant *preserves* underscores so that compound keywords like
    ``computer_science`` survive intact.  ``data.py`` has its own
    ``_normalize_token`` that strips all non-alnum characters — the two serve
    different purposes and both are intentionally kept.
    """
    return "".join(ch.lower() if ch.isalnum() or ch == "_" else "_" for ch in text)


def _normalize_subject_label(text: str) -> str:
    return text.replace("_", " ").strip().lower()


def _method_tag(is_cot: bool) -> str:
    return "cot" if is_cot else "nocot"


def _benchmark_name(entry: ScoreEntry) -> str:
    return f"{_dataset_base(entry.dataset)}_{_method_tag(entry.cot)}"


def _format_param(token: str | None) -> str:
    if not token:
        return "?"
    return token.replace("_", ".")


# ---------------------------------------------------------------------------
# Numeric extraction
# ---------------------------------------------------------------------------

def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    if -1.0 <= value <= 1.0:
        return value * 100.0
    return value


def _max_percent(values: Iterable[float | None]) -> float | None:
    best: float | None = None
    for value in values:
        percent = _score_to_percent(value)
        if percent is None:
            continue
        if best is None or percent > best:
            best = percent
    return best


def _parse_pass_suffix(key: str) -> int | None:
    token = str(key).strip().lower()
    if not token.startswith("pass"):
        return None
    token = token[len("pass"):]
    if token.startswith("@"):
        token = token[1:]
    if token.startswith("at"):
        token = token[2:]
    try:
        return int(token)
    except ValueError:
        return None


def _parse_k_metric(key: str) -> tuple[str, float] | None:
    token = str(key).strip().lower()
    if token.startswith("pass@"):
        try:
            return "pass", float(token.split("@", 1)[1])
        except ValueError:
            return None
    if token.startswith("avg@"):
        try:
            return "avg", float(token.split("@", 1)[1])
        except ValueError:
            return None
    return None


def _preferred_k_metric(metrics: dict[str, Any]) -> str:
    avg_candidates: list[tuple[float, str]] = []
    pass_candidates: list[tuple[float, str]] = []
    for key, value in metrics.items():
        parsed = _parse_k_metric(str(key))
        if parsed is None or _numeric_value(value) is None:
            continue
        kind, k = parsed
        if kind == "avg":
            avg_candidates.append((k, str(key)))
        elif kind == "pass":
            pass_candidates.append((k, str(key)))
    if avg_candidates:
        return max(avg_candidates, key=lambda item: item[0])[1]
    if pass_candidates:
        return max(pass_candidates, key=lambda item: item[0])[1]
    return "pass@1"


def _preferred_numeric(metrics: dict[str, Any], keys: Sequence[str]) -> tuple[str | None, float | None]:
    for key in keys:
        value = metrics.get(key)
        number = _numeric_value(value)
        if number is not None:
            return key, number
    return None, None


def _primary_numeric_metric(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    key, value = _preferred_numeric(metrics, PRIMARY_KEYS)
    if key:
        return key, value
    for key, value in metrics.items():
        num = _numeric_value(value)
        if num is not None:
            return key, num
    for key, value in metrics.items():
        if value is not None:
            num = _numeric_value(value)
            if num is not None:
                return key, num
    return None, None


def _best_numeric_metric(entry: ScoreEntry, *, dataset_base: str | None = None) -> tuple[str | None, float | None]:
    base = (dataset_base or _dataset_base(entry.dataset)).lower()
    metrics = entry.metrics

    if base.startswith("aime"):
        key, value = _preferred_numeric(metrics, ("pass@8", "avg@16"))
        if key:
            return key, value

    if base in MATH500_BASES:
        key, value = _preferred_numeric(metrics, ("avg@4",))
        if key:
            return key, value

    if base.startswith("ifeval"):
        key, value = _preferred_numeric(metrics, ("avg@4", "instruction_accuracy", "prompt_accuracy"))
        if key:
            return key, value

    if entry.domain == "coding系列":
        for k in (1, 2, 4, 8, 16):
            key, value = _preferred_numeric(metrics, (f"pass@{k}",))
            if key:
                return key, value
        best_key: str | None = None
        best_val: float | None = None
        for key, value in metrics.items():
            num = _numeric_value(value)
            if num is None:
                continue
            if best_val is None or num > best_val:
                best_key, best_val = key, num
        if best_key:
            return best_key, best_val

    return _primary_numeric_metric(metrics)

# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "✓" if value else "✕"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and 0 <= value <= 1:
            return f"{value * 100:.1f}%"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if isinstance(value, int):
            return f"{value:d}"
        return f"{value:.3f}"
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, str] | None:
    for key in PRIMARY_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, _format_metric_value(value)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return key, _format_metric_value(value)
    for key in PRIMARY_KEYS:
        value = metrics.get(key)
        if value is not None:
            return key, _format_metric_value(value)
    for key, value in metrics.items():
        if value is not None:
            return key, _format_metric_value(value)
    return None


# ---------------------------------------------------------------------------
# Score formatting (1 decimal place)
# ---------------------------------------------------------------------------

def _format_score_1dp(value: float | None) -> str:
    normalized = _score_to_percent(value)
    if normalized is None:
        return "—"
    return f"{normalized:.1f}"


def _score_display_percent_1dp(value: float | None) -> float | None:
    normalized = _score_to_percent(value)
    if normalized is None:
        return None
    return float(f"{normalized:.1f}")


def _delta_from_display_scores(latest: float | None, previous: float | None) -> float | None:
    latest_n = _score_display_percent_1dp(latest)
    prev_n = _score_display_percent_1dp(previous)
    if latest_n is None or prev_n is None:
        return None
    return latest_n - prev_n


def _format_delta_1dp(latest: float | None, previous: float | None) -> str:
    delta = _delta_from_display_scores(latest, previous)
    if delta is None:
        return "—"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"


def _format_delta_value(value: float | None) -> str:
    """Format a delta value with proper sign prefix.

    Uses Unicode minus sign (U+2212) instead of hyphen for negative values.
    """
    if value is None:
        return "—"
    if abs(value) < 1e-6:
        return "0.0"
    sign = "+" if value > 0 else "\u2212"
    return f"{sign}{abs(value):.1f}"


def _parse_display_number(value: Any) -> float | None:
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text == "—":
        return None
    text = text.replace("%", "").replace(",", "")
    if text.startswith("+"):
        text = text[1:]
    # Handle Unicode minus sign
    if text.startswith("\u2212"):
        text = "-" + text[1:]
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Cell styling — simplified (CSS-class based, no box-shadow)
# ---------------------------------------------------------------------------

def _styled_score_cell(value: float | None) -> str:
    """Return formatted score text. No inline style — row cycling handles bg."""
    return _format_score_1dp(value)


def _styled_score_cell_norm(value: float | None, min_v: float | None, max_v: float | None) -> str:
    """Return formatted score text. No inline style — row cycling handles bg."""
    return _format_score_1dp(value)


def _styled_delta_cell(delta_value: float | None) -> tuple[str, str]:
    """Return ``(formatted_text, css_class)`` for a delta cell.

    css_class is one of ``"cell-delta-pos"``, ``"cell-delta-neg"``,
    ``"cell-delta-zero"``, or ``"cell-na"``.
    """
    text = _format_delta_value(delta_value)
    if delta_value is None:
        return text, "cell-na"
    if delta_value > 1e-6:
        return text, "cell-delta-pos"
    if delta_value < -1e-6:
        return text, "cell-delta-neg"
    return text, "cell-delta-zero"


# ---------------------------------------------------------------------------
# Cell metric value (for pivot table)
# ---------------------------------------------------------------------------

def _cell_metric_value(entry: ScoreEntry | None, *, dataset_base: str) -> str:
    if entry is None:
        return "—"

    base = dataset_base.lower()
    metrics = entry.metrics

    def _format_specific(key: str) -> str | None:
        value = metrics.get(key)
        if isinstance(value, (int, float, bool)):
            return _format_metric_value(value)
        return None

    if _is_multi_choice_entry(entry):
        preferred_k = _preferred_k_metric(metrics)
        formatted = _format_specific(preferred_k)
        if formatted is not None:
            return f"{preferred_k} {formatted}"

    if base.startswith("aime"):
        parts: list[str] = []
        for key in ("avg@16", "pass@8"):
            formatted = _format_specific(key)
            if formatted is not None:
                parts.append(f"{key} {formatted}")
        if parts:
            return " / ".join(parts)

    if base in MATH500_BASES:
        formatted = _format_specific("avg@4")
        if formatted is not None:
            return f"avg@4 {formatted}"

    if base.startswith("ifeval"):
        formatted = _format_specific("avg@4")
        if formatted is not None:
            return f"avg@4 {formatted}"

    primary = _primary_metric(entry.metrics)
    if not primary:
        return "—"
    return primary[1]


def _cell_numeric_value(entry: ScoreEntry | None, *, dataset_base: str) -> float | None:
    if entry is None:
        return None

    base = dataset_base.lower()
    metrics = entry.metrics

    def _numeric_specific(key: str) -> float | None:
        return _numeric_value(metrics.get(key))

    if _is_multi_choice_entry(entry):
        preferred_k = _preferred_k_metric(metrics)
        value = _numeric_specific(preferred_k)
        if value is not None:
            return value

    if base.startswith("aime"):
        candidates: list[float] = []
        for key in ("avg@16", "pass@8"):
            value = _numeric_specific(key)
            if value is not None:
                candidates.append(value)
        if candidates:
            return max(candidates)

    if base in MATH500_BASES:
        value = _numeric_specific("avg@4")
        if value is not None:
            return value

    if base.startswith("ifeval"):
        value = _numeric_specific("avg@4")
        if value is not None:
            return value

    _, value = _primary_numeric_metric(metrics)
    return value


# ---------------------------------------------------------------------------
# Metric score — promoted to module level (was nested inside loop)
# ---------------------------------------------------------------------------

def _metric_score(item: ScoreEntry | None) -> float | None:
    """Return the first numeric metric value from an entry, or None."""
    if item is None:
        return None
    for val in item.metrics.values():
        if isinstance(val, (int, float)):
            return float(val)
    return None


# ---------------------------------------------------------------------------
# Multi-choice detection & eval-method scoring
# ---------------------------------------------------------------------------

def _is_multi_choice_entry(entry: ScoreEntry) -> bool:
    task = (entry.task or "").lower()
    if "multi" in task and "choice" in task:
        return True
    if task in {"multi_choice_plain", "multi_choice_cot"}:
        return True
    job_hint = entry.domain in {"mmlu系列", "multi-choice系列"}
    return job_hint and _numeric_value(entry.metrics.get("accuracy")) is not None


def _prefer_llm_judge(entry: ScoreEntry) -> bool:
    task = (entry.task or "").lower()
    if "judge" in task:
        return True
    return _numeric_value(entry.metrics.get("judge_accuracy")) is not None


def _score_for_eval_method(entry: ScoreEntry, method: str, k_metric: str) -> float | None:
    metrics = entry.metrics
    if method == "logits":
        k_value = _numeric_value(metrics.get(k_metric))
        if k_value is not None:
            return k_value
        acc = _numeric_value(metrics.get("accuracy"))
        if acc is not None:
            return acc
        _, fallback = _best_numeric_metric(entry, dataset_base=_dataset_base(entry.dataset))
        return fallback
    if method == "llm_judge":
        judge_score = _numeric_value(metrics.get("judge_accuracy"))
        if judge_score is not None:
            return judge_score
        k_value = _numeric_value(metrics.get(k_metric))
        if k_value is not None:
            return k_value
        for key in ("exact_accuracy", "instruction_accuracy", "prompt_accuracy", "accuracy"):
            value = _numeric_value(metrics.get(key))
            if value is not None:
                return value
        _, fallback = _best_numeric_metric(entry, dataset_base=_dataset_base(entry.dataset))
        return fallback

    # exact_match
    for key in ("exact_accuracy", "instruction_accuracy", "prompt_accuracy"):
        number = _numeric_value(metrics.get(key))
        if number is not None:
            return number
    if not _is_multi_choice_entry(entry):
        acc = _numeric_value(metrics.get("accuracy"))
        if acc is not None:
            return acc
    k_value = _numeric_value(metrics.get(k_metric))
    if k_value is not None:
        return k_value
    _, fallback = _best_numeric_metric(entry, dataset_base=_dataset_base(entry.dataset))
    return fallback


def _detail_rows_for_entry(entry: ScoreEntry) -> list[tuple[str, str, str, float | None]]:
    benchmark = _benchmark_name(entry)
    k_metric = _preferred_k_metric(entry.metrics)
    methods: list[str] = []
    if _is_multi_choice_entry(entry):
        methods.append("logits")
    else:
        if _prefer_llm_judge(entry):
            methods.append("llm_judge")
        else:
            methods.append("exact_match")

    rows: list[tuple[str, str, str, float | None]] = []
    for method in methods:
        score = _score_for_eval_method(entry, method, k_metric)
        if score is None:
            continue
        rows.append((benchmark, method, k_metric, score))
    return rows


def _field_primary_score(entry: ScoreEntry) -> float | None:
    if _is_multi_choice_entry(entry):
        return _score_for_eval_method(entry, "logits", _preferred_k_metric(entry.metrics))
    if _prefer_llm_judge(entry):
        return _score_for_eval_method(entry, "llm_judge", _preferred_k_metric(entry.metrics))
    return _score_for_eval_method(entry, "exact_match", _preferred_k_metric(entry.metrics))


def _detail_sort_key(row_key: tuple[str, str, str]) -> tuple[Any, ...]:
    benchmark, method, k_metric = row_key
    method_rank = {"llm_judge": 0, "exact_match": 1, "logits": 2}.get(method, 9)
    parsed = _parse_k_metric(k_metric)
    if parsed is None:
        k_rank = (9, 0)
    else:
        kind, k = parsed
        kind_rank = 0 if kind == "avg" else 1
        k_rank = (kind_rank, k)
    return benchmark, method_rank, k_rank, k_metric


def _field_average_score(entries: Iterable[ScoreEntry]) -> float | None:
    values: list[float] = []
    for entry in entries:
        score = _field_primary_score(entry)
        if score is not None:
            values.append(score)
    if not values:
        return None
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Subdomain mapping
# ---------------------------------------------------------------------------

def _map_subject_to_subdomain(subject: str) -> str:
    token = _normalize_token(subject)
    for domain, keywords in SUBDOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in token:
                return domain
    parts = token.split("_")
    if parts and parts[0]:
        head = parts[0]
        for domain, keywords in SUBDOMAIN_KEYWORDS.items():
            if head in keywords:
                return domain
    return "other"


def _collect_subject_metrics(entry: ScoreEntry) -> dict[str, float]:
    details = entry.task_details or {}
    accuracy_by_subject = details.get("accuracy_by_subject")
    if isinstance(accuracy_by_subject, dict):
        results: dict[str, float] = {}
        for subject, value in accuracy_by_subject.items():
            num = _numeric_value(value)
            if num is not None:
                results[str(subject)] = num
        return results
    return {}


def _extract_pass_curve(entry: ScoreEntry) -> dict[int, float]:
    curve: dict[int, float] = {}

    def _ingest(source: Any) -> None:
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            suffix = _parse_pass_suffix(key)
            score = _numeric_value(value)
            if suffix is not None and score is not None:
                curve.setdefault(suffix, score)

    if entry.task_details:
        _ingest(entry.task_details.get("pass_curve"))
    _ingest(entry.metrics)
    return dict(sorted(curve.items()))
