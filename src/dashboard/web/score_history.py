"""Score-history page service (DB-backed; does NOT use score_index.jsonl).

For a chosen model + benchmark, returns score points split into at most two
charts by ``cot_mode`` (NoCoT / CoT). The default response is compact: repeated
runs for the same cot/evaluator/board/metric group collapse to the latest score.
Clients can request ``compact=false`` for the full official score history. Each
point carries its score_id/task_id so the frontend can open per-task detail.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ..core.boards import BOARD_NAIVE, BOARD_NORMAL, is_naive_meta
from ..core.metrics import _display_metric_from_context, _score_to_percent
from ..core.vocab import token_id_to_display
from .context_display import clean_context_for_display
from .eval_service import _extract_context_object
from .store import DashboardStore


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _metric_percent(metrics: Any, *, sampling_config: Any = None) -> tuple[str | None, float | None]:
    md = _as_dict(metrics)
    name, value = _display_metric_from_context(md, sampling_config=sampling_config)
    if name is None or value is None:
        return None, None
    return name, _score_to_percent(value)


def _sampling_summary(sampling_config: Any) -> str:
    nested = _as_dict(_as_dict(sampling_config).get("sampling_config"))
    stage = next((v for v in nested.values() if isinstance(v, dict)), {})
    parts: list[str] = []
    if stage.get("temperature") is not None:
        parts.append(f"T={stage['temperature']}")
    if stage.get("top_p") is not None:
        parts.append(f"top_p={stage['top_p']}")
    if stage.get("top_k") is not None:
        parts.append(f"top_k={stage['top_k']}")
    return " ".join(parts)


def _store_or_default(store: DashboardStore | None = None) -> DashboardStore:
    return store or DashboardStore()


def score_history_options(*, store: DashboardStore | None = None) -> dict[str, Any]:
    pairs = _store_or_default(store).list_score_history_pairs()
    models = sorted({str(p["model"]) for p in pairs})
    benchmarks = sorted({str(p["dataset"]) for p in pairs})
    return {"models": models, "benchmarks": benchmarks, "pairs": pairs}


def _compact_latest_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for point in points:
        key = (
            point.get("cot_mode"),
            point.get("evaluator"),
            point.get("board"),
            point.get("metric"),
        )
        previous = latest_by_key.get(key)
        current_sort = (point.get("created_at") or "", point.get("score_id") or 0)
        previous_sort = (
            (previous.get("created_at") or "") if previous else "",
            (previous.get("score_id") or 0) if previous else 0,
        )
        if previous is None or current_sort > previous_sort:
            latest_by_key[key] = point
    return list(latest_by_key.values())


def score_history(
    *,
    model: str,
    benchmark: str,
    compact: bool = True,
    store: DashboardStore | None = None,
) -> dict[str, Any]:
    rows = _store_or_default(store).list_score_history(model=model, dataset=benchmark)
    points: list[dict[str, Any]] = []
    for row in rows:
        metric_name, percent = _metric_percent(row.get("metrics"), sampling_config=row.get("sampling_config"))
        if metric_name is None or percent is None:
            continue
        evaluator = row.get("evaluator")
        naive = is_naive_meta(evaluator, row.get("sampling_config"))
        points.append(
            {
                "score_id": row.get("score_id"),
                "task_id": row.get("task_id"),
                "cot_mode": row.get("cot_mode"),
                "evaluator": evaluator,
                "board": BOARD_NAIVE if naive else BOARD_NORMAL,
                "percent": percent,
                "metric": metric_name,
                "created_at": _iso(row.get("created_at")),
                "sampling_summary": _sampling_summary(row.get("sampling_config")),
                "model": row.get("model"),
                "benchmark": row.get("dataset"),
            }
        )
    raw_total = len(points)
    if compact:
        points = _compact_latest_points(points)

    # Split into at most two charts by cot_mode (NoCoT / CoT). Within each chart,
    # naive points first, then created_at ascending.
    buckets: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        key = "NoCoT" if point["cot_mode"] == "NoCoT" else "CoT"
        buckets.setdefault(key, []).append(point)

    groups: list[dict[str, Any]] = []
    for key in ("NoCoT", "CoT"):
        bucket = buckets.get(key)
        if not bucket:
            continue
        bucket.sort(key=lambda p: (0 if p["board"] == BOARD_NAIVE else 1, p["created_at"] or ""))
        groups.append({"cot_mode": key, "points": bucket})

    return {
        "model": model,
        "benchmark": benchmark,
        "total": len(points),
        "raw_total": raw_total,
        "compact": compact,
        "groups": groups,
    }


def _stop_token_rows(stop_tokens: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(stop_tokens, list):
        for token_id in stop_tokens:
            try:
                tid = int(token_id)
            except (TypeError, ValueError):
                continue
            rows.append({"id": tid, "token": token_id_to_display(tid)})
    return rows


def score_history_detail(*, task_id: int, store: DashboardStore | None = None) -> dict[str, Any]:
    detail = _store_or_default(store).get_score_history_detail(task_id=str(task_id))
    if detail is None:
        return {"found": False, "task_id": task_id}

    score = detail.get("score") or {}
    task = detail.get("task") or {}
    metric_name, percent = _metric_percent(score.get("metrics"), sampling_config=task.get("sampling_config"))
    outer = _as_dict(task.get("sampling_config"))
    nested = _as_dict(outer.get("sampling_config"))
    evaluator = task.get("evaluator")
    naive = is_naive_meta(evaluator, outer)

    # Per-stage generation params.
    stages_sampling: dict[str, Any] = {}
    for stage_name, stage_cfg in nested.items():
        cfg = stage_cfg if isinstance(stage_cfg, dict) else {}
        stages_sampling[str(stage_name)] = {
            "temperature": cfg.get("temperature"),
            "top_k": cfg.get("top_k"),
            "top_p": cfg.get("top_p"),
            "max_tokens": cfg.get("max_new_tokens"),
            "stop_tokens": _stop_token_rows(cfg.get("stop_tokens")),
            "penalties": {
                "presence_penalty": cfg.get("presence_penalty"),
                "repetition_penalty": cfg.get("repetition_penalty"),
                "penalty_decay": cfg.get("penalty_decay"),
            },
        }

    sampling = {
        "stages": stages_sampling,
        "effective_sample_count": outer.get("effective_sample_count"),
        "avg_k": outer.get("avg_k"),
        "pass_ks": outer.get("pass_ks"),
        "n_shot": outer.get("n_shot"),
        "sample_limit": outer.get("sample_limit"),
        "prompt_profile": outer.get("prompt_profile"),
    }

    # Representative prompt stages from one completion's context.
    prompt_stages: list[dict[str, Any]] = []
    context = _extract_context_object(detail.get("context"))
    if isinstance(context, dict) and isinstance(context.get("stages"), list):
        context = clean_context_for_display(context)
        for stage in context["stages"]:
            if isinstance(stage, dict):
                prompt_stages.append(
                    {
                        "prompt": str(stage.get("prompt") or ""),
                        "completion": str(stage.get("completion") or ""),
                        "stop_reason": stage.get("stop_reason"),
                    }
                )

    return {
        "found": True,
        "task_id": int(task_id),
        "model": score.get("model"),
        "benchmark": score.get("dataset"),
        "cot_mode": score.get("cot_mode"),
        "evaluator": evaluator,
        "board": BOARD_NAIVE if naive else BOARD_NORMAL,
        "metric": metric_name,
        "percent": percent,
        "metrics": _as_dict(score.get("metrics")),
        "sampling": sampling,
        "stages": prompt_stages,
    }


__all__ = ["score_history", "score_history_detail", "score_history_options"]
