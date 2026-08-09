"""Metric extraction, score formatting, cell styling, and subdomain mapping.

This module is a pure-logic layer with no I/O or Gradio dependency.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from src.eval.execution_plan import avg_k_metric_key

from .data import ScoreEntry
from .constants import (
    SUBDOMAIN_KEYWORDS,
    SUBDOMAIN_ORDER,
)
from .domains import is_multi_choice_domain


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


def _method_tag(is_cot: bool, *, cot_mode: str | None = None) -> str:
    normalized = str(cot_mode or "").strip().lower()
    if normalized == "fake_cot":
        return "fake_cot"
    if normalized in {"no_cot", "nocot"}:
        return "nocot"
    if normalized == "cot":
        return "cot"
    return "cot" if is_cot else "nocot"


def _entry_cot_mode(entry: ScoreEntry) -> str | None:
    if entry.task_details:
        mode = entry.task_details.get("cot_mode")
        if isinstance(mode, str) and mode.strip():
            return mode
    mode = entry.extra.get("cot_mode")
    if isinstance(mode, str) and mode.strip():
        return mode
    sampling_config = entry.extra.get("sampling_config")
    if isinstance(sampling_config, dict):
        nested = sampling_config.get("cot_mode")
        if isinstance(nested, str) and nested.strip():
            return nested
    task = (entry.task or "").strip().lower()
    if "fake_cot" in task:
        return "fake_cot"
    if task.endswith("_plain"):
        return "no_cot"
    return "cot" if entry.cot else "no_cot"


def _entry_method_tag(entry: ScoreEntry) -> str:
    return _method_tag(entry.cot, cot_mode=_entry_cot_mode(entry))


def _benchmark_name(entry: ScoreEntry) -> str:
    return f"{_dataset_base(entry.dataset)}_{_entry_method_tag(entry)}"


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
            return "pass", float(int(token.split("@", 1)[1]))
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


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _avg_metric_key_from_value(value: Any) -> str | None:
    number = _numeric_value(value)
    if number is None or number <= 0:
        return None
    return avg_k_metric_key(number)


def _iter_sequence(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if value is None:
        return ()
    return (value,)


def _configured_metric_keys(*contexts: Any) -> list[str]:
    keys: list[str] = []

    def add(key: str | None) -> None:
        if key and key not in keys:
            keys.append(key)

    for raw_context in contexts:
        context = _as_mapping(raw_context)
        if not context:
            continue
        for field in ("display_metric_key", "score_metric_key", "primary_metric_key", "metric_key"):
            value = context.get(field)
            if isinstance(value, str):
                add(value)
        for field in ("report_avg_k", "avg_k"):
            for item in _iter_sequence(context.get(field)):
                add(_avg_metric_key_from_value(item))
        for field in ("report_pass_k", "pass_ks", "pass_k"):
            for item in _iter_sequence(context.get(field)):
                number = _numeric_value(item)
                if number is not None and number > 0 and float(number).is_integer():
                    add(f"pass@{int(number)}")
    return keys


def _is_avg_metric_key(key: str) -> bool:
    return str(key).strip().lower().startswith("avg@")


def _is_pass_metric_key(key: str) -> bool:
    return str(key).strip().lower().startswith("pass@")


def _scoreish_fallback_rank(key: str) -> tuple[int, str] | None:
    token = str(key).strip().lower()
    if not token:
        return None
    # Diagnostics can be numeric rates, but they are not leaderboard scores.
    non_score_fragments = (
        "count",
        "compaction",
        "diagnostic",
        "error",
        "fail",
        "harness",
        "latency",
        "predictions",
        "prompt",
        "token",
        "tool_route",
        "turn",
    )
    if any(fragment in token for fragment in non_score_fragments):
        return None
    if token == "official_score" or token.endswith("_official_score"):
        return (0, token)
    if token == "success_rate" or token.endswith("_success_rate"):
        return (1, token)
    if token == "resolution_rate" or token.endswith("_resolution_rate"):
        return (2, token)
    if token == "f1" or token.endswith("_f1"):
        return (3, token)
    if token == "accuracy" or token.endswith("_accuracy"):
        return (4, token)
    if token == "score" or token.endswith("_score"):
        return (5, token)
    if token == "rate" or token.endswith("_rate"):
        return (6, token)
    return None


def _k_metric_sort_value(key: str) -> tuple[float, str]:
    parsed = _parse_k_metric(key)
    if parsed is None:
        return (0.0, str(key))
    _, k = parsed
    return (k, str(key))


def _display_metric_from_context(
    metrics: dict[str, Any],
    *,
    sampling_config: Any = None,
    task_details: Any = None,
    extra: Any = None,
) -> tuple[str | None, float | None]:
    """Select the score-bearing metric for display.

    The DB row owns the metrics and the task sampling config owns the avg/pass
    plan. UI code must not invent benchmark-specific score keys here.
    """
    if not metrics:
        return None, None

    for key in _configured_metric_keys(extra, task_details, sampling_config):
        value = _numeric_value(metrics.get(key))
        if value is not None:
            return key, value

    avg_candidates = [
        (str(key), _numeric_value(value))
        for key, value in metrics.items()
        if _is_avg_metric_key(str(key)) and _numeric_value(value) is not None
    ]
    if avg_candidates:
        key, value = max(avg_candidates, key=lambda item: _k_metric_sort_value(item[0]))
        return key, value

    pass_candidates = [
        (str(key), _numeric_value(value))
        for key, value in metrics.items()
        if _is_pass_metric_key(str(key)) and _numeric_value(value) is not None
    ]
    if pass_candidates:
        key, value = max(pass_candidates, key=lambda item: _k_metric_sort_value(item[0]))
        return key, value

    scoreish: list[tuple[tuple[int, str], str, float]] = []
    for key, value in metrics.items():
        rank = _scoreish_fallback_rank(str(key))
        number = _numeric_value(value)
        if rank is not None and number is not None:
            scoreish.append((rank, str(key), number))
    if scoreish:
        _, key, value = min(scoreish, key=lambda item: item[0])
        return key, value

    return None, None


def _display_metric_for_entry(entry: ScoreEntry | None) -> tuple[str | None, float | None]:
    if entry is None:
        return None, None
    return _display_metric_from_context(
        entry.metrics,
        sampling_config=(entry.extra or {}).get("sampling_config"),
        task_details=entry.task_details,
        extra=entry.extra,
    )


def _primary_numeric_metric(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    return _display_metric_from_context(metrics)


def _best_numeric_metric(entry: ScoreEntry, *, dataset_base: str | None = None) -> tuple[str | None, float | None]:
    del dataset_base
    return _display_metric_for_entry(entry)


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
    key, value = _display_metric_from_context(metrics)
    if key is not None and value is not None:
        return key, _format_metric_value(value)
    return None


# ---------------------------------------------------------------------------
# Multi-choice detection & eval-method scoring
# ---------------------------------------------------------------------------

def _is_multi_choice_entry(entry: ScoreEntry) -> bool:
    task = (entry.task or "").lower()
    if "multi" in task and "choice" in task:
        return True
    job_hint = is_multi_choice_domain(entry.domain)
    return job_hint and _numeric_value(entry.metrics.get("accuracy")) is not None


def _prefer_llm_judge(entry: ScoreEntry) -> bool:
    task = (entry.task or "").lower()
    if "judge" in task:
        return True
    return _numeric_value(entry.metrics.get("judge_accuracy")) is not None


def _score_for_eval_method(entry: ScoreEntry, method: str, k_metric: str) -> float | None:
    del method
    value = _numeric_value(entry.metrics.get(k_metric))
    if value is not None:
        return value
    _, fallback = _display_metric_for_entry(entry)
    return fallback


def _detail_rows_for_entry(entry: ScoreEntry) -> list[tuple[str, str, str, float]]:
    benchmark = _benchmark_name(entry)
    k_metric, metric_value = _display_metric_for_entry(entry)
    if k_metric is None or metric_value is None:
        return []
    methods: list[str] = []
    if _is_multi_choice_entry(entry):
        methods.append("logits")
    else:
        if _prefer_llm_judge(entry):
            methods.append("llm_judge")
        else:
            methods.append("exact_match")

    rows: list[tuple[str, str, str, float]] = []
    for method in methods:
        score = _score_for_eval_method(entry, method, k_metric)
        if score is None:
            continue
        rows.append((benchmark, method, k_metric, score))
    return rows


def _field_primary_score(entry: ScoreEntry) -> float | None:
    _, value = _display_metric_for_entry(entry)
    return value


def _detail_sort_key(row_key: tuple[str, str, str]) -> tuple[Any, ...]:
    benchmark, method, k_metric = row_key
    method_rank = {"llm_judge": 0, "exact_match": 1, "logits": 2}.get(method, 9)
    parsed = _parse_k_metric(k_metric)
    if parsed is None:
        k_rank = (9, 0.0)
    else:
        kind, k = parsed
        kind_rank = 0 if kind == "avg" else 1
        k_rank = (kind_rank, k)
    return benchmark, method_rank, k_rank, k_metric


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
