"""Score-history page service (DB-backed; does NOT use score_index.jsonl).

For a chosen model + benchmark, returns score points split into at most two
charts by ``cot_mode`` (NoCoT / CoT). The default response is compact: repeated
runs for the same cot/evaluator/board/metric group collapse to the latest score.
Clients can request ``compact=false`` for the full official score history. Each
point carries its score_id/task_id so the frontend can open per-task detail.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any

from ..core.boards import BOARD_NAIVE, BOARD_NORMAL, is_naive_meta
from ..core.data import parse_model_signature
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


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _demo_score_info(metrics: Any) -> dict[str, Any]:
    demo = _as_dict(_as_dict(metrics).get("dashboard_demo"))
    source_value = _number(demo.get("source_value"))
    source_percent = _score_to_percent(source_value)
    adjustment_percent = _number(demo.get("percent_delta"))
    if source_percent is not None:
        source_percent = round(source_percent, 6)
    if adjustment_percent is not None:
        adjustment_percent = round(adjustment_percent, 6)
    return {
        "demo_adjusted": bool(demo.get("adjusted")),
        "demo_label": str(demo.get("label")) if demo.get("label") else None,
        "source_task_id": _int_or_none(demo.get("source_task_id")),
        "source_score_id": _int_or_none(demo.get("source_score_id")),
        "source_percent": source_percent,
        "adjustment_percent": adjustment_percent,
        "context_format": str(demo.get("context_format")) if demo.get("context_format") else None,
    }


def _sampling_info(sampling_config: Any, *, evaluator: Any, cot_mode: Any) -> dict[str, str | None]:
    outer = _as_dict(sampling_config)
    nested = _as_dict(outer.get("sampling_config"))
    parts: list[str] = []
    for stage_name, raw_stage in sorted(nested.items()):
        stage = raw_stage if isinstance(raw_stage, dict) else {}
        values: list[str] = []
        for key, label in (
            ("temperature", "T"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("presence_penalty", "presence"),
            ("repetition_penalty", "repetition"),
            ("penalty_decay", "decay"),
        ):
            if stage.get(key) is not None:
                values.append(f"{label}={stage[key]}")
        if values:
            parts.append(f"{stage_name}: " + " ".join(values))
    prompt_profile = str(outer.get("prompt_profile") or "normal")
    signature = {
        "cot_mode": str(cot_mode or ""),
        "evaluator": str(evaluator or ""),
        "prompt_profile": prompt_profile,
        "sampling_config": nested,
        "avg_k": outer.get("avg_k"),
        "pass_ks": outer.get("pass_ks"),
        "n_shot": outer.get("n_shot"),
    }
    return {
        "summary": " | ".join(parts) or "无生成采样参数",
        "prompt_profile": prompt_profile,
        "config_key": json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    }


def _answer_has_repetition(answer: Any) -> bool:
    text = " ".join(str(answer or "").split())
    if not text:
        return False
    raw_lines = [line.strip().lower() for line in str(answer).splitlines() if len(line.strip()) >= 12]
    if any(raw_lines.count(line) >= 3 for line in set(raw_lines)):
        return True
    tokens = text.lower().split()
    if len(tokens) < 18:
        return False
    for size in range(4, min(16, len(tokens) // 3) + 1):
        for start in range(0, len(tokens) - size * 3 + 1):
            chunk = tokens[start : start + size]
            if chunk == tokens[start + size : start + size * 2] == tokens[start + size * 2 : start + size * 3]:
                return True
    return False


def _repetition_stats(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[Any]] = {}
    for row in rows:
        try:
            task_id = int(row.get("task_id"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(task_id, []).append(row.get("answer"))
    result: dict[int, dict[str, Any]] = {}
    for task_id, answers in grouped.items():
        nonempty = [answer for answer in answers if str(answer or "").strip()]
        repeated = sum(1 for answer in nonempty if _answer_has_repetition(answer))
        result[task_id] = {
            "repetition_rate": (repeated * 100.0 / len(nonempty)) if nonempty else None,
            "repeated_answers": repeated,
            "answer_count": len(nonempty),
        }
    return result


def _store_or_default(store: DashboardStore | None = None) -> DashboardStore:
    return store or DashboardStore()


def score_history_options(*, store: DashboardStore | None = None) -> dict[str, Any]:
    pairs = _store_or_default(store).list_score_history_pairs()
    for pair in pairs:
        signature = parse_model_signature(str(pair.get("model") or ""))
        pair["arch"] = signature.arch
        pair["data"] = signature.data
        pair["params"] = signature.params
    coverage = Counter(str(pair["model"]) for pair in pairs)
    signatures = {
        str(pair["model"]): parse_model_signature(str(pair.get("model") or ""))
        for pair in pairs
    }
    models = sorted(
        signatures,
        key=lambda model: (
            -(signatures[model].data_rank if signatures[model].data_rank is not None else -1),
            -coverage[model],
            model,
        ),
    )
    benchmarks = sorted({str(p["dataset"]) for p in pairs})
    return {
        "models": models,
        "benchmarks": benchmarks,
        "pairs": pairs,
        "default_model": models[0] if models else None,
    }


def _compact_latest_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for point in points:
        key = (
            point.get("cot_mode"),
            point.get("evaluator"),
            point.get("board"),
            point.get("metric"),
            point.get("config_key"),
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
    task_ids = [int(row["task_id"]) for row in rows if row.get("task_id") is not None]
    repetition = _repetition_stats(
        _store_or_default(store).list_eval_answers_for_tasks(task_ids=task_ids)
    )
    points: list[dict[str, Any]] = []
    for row in rows:
        metric_name, percent = _metric_percent(row.get("metrics"), sampling_config=row.get("sampling_config"))
        if metric_name is None or percent is None:
            continue
        evaluator = row.get("evaluator")
        naive = is_naive_meta(evaluator, row.get("sampling_config"))
        sampling = _sampling_info(
            row.get("sampling_config"),
            evaluator=evaluator,
            cot_mode=row.get("cot_mode"),
        )
        task_id = int(row.get("task_id")) if row.get("task_id") is not None else None
        repeat_stats = repetition.get(task_id or -1, {})
        demo_info = _demo_score_info(row.get("metrics"))
        points.append(
            {
                "score_id": row.get("score_id"),
                "task_id": task_id,
                "cot_mode": row.get("cot_mode"),
                "evaluator": evaluator,
                "board": BOARD_NAIVE if naive else BOARD_NORMAL,
                "percent": percent,
                "metric": metric_name,
                "created_at": _iso(row.get("created_at")),
                "sampling_summary": sampling["summary"],
                "prompt_profile": sampling["prompt_profile"],
                "config_key": sampling["config_key"],
                "repetition_rate": repeat_stats.get("repetition_rate"),
                "repeated_answers": repeat_stats.get("repeated_answers", 0),
                "answer_count": repeat_stats.get("answer_count", 0),
                "model": row.get("model"),
                "benchmark": row.get("dataset"),
                **demo_info,
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
    resolved_store = _store_or_default(store)
    detail = resolved_store.get_score_history_detail(task_id=str(task_id))
    if detail is None:
        return {"found": False, "task_id": task_id}

    score = detail.get("score") or {}
    task = detail.get("task") or {}
    metric_name, percent = _metric_percent(score.get("metrics"), sampling_config=task.get("sampling_config"))
    outer = _as_dict(task.get("sampling_config"))
    nested = _as_dict(outer.get("sampling_config"))
    evaluator = task.get("evaluator")
    naive = is_naive_meta(evaluator, outer)
    repetition = _repetition_stats(
        resolved_store.list_eval_answers_for_tasks(task_ids=[int(task_id)])
    ).get(int(task_id), {"repetition_rate": None, "repeated_answers": 0, "answer_count": 0})

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
    context_demo = _as_dict(context.get("dashboard_demo")) if isinstance(context, dict) else {}
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

    demo_info = _demo_score_info(score.get("metrics"))
    context_format = context_demo.get("context_format") or demo_info.get("context_format")
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
        "repetition": repetition,
        "stages": prompt_stages,
        **demo_info,
        "context_format": str(context_format) if context_format else None,
        "context_conversion": (
            str(context_demo.get("conversion")) if context_demo.get("conversion") else None
        ),
        "original_first_stage_prompt_chars": _int_or_none(
            context_demo.get("original_first_stage_prompt_chars")
        ),
        "original_first_stage_prompt_sha256": (
            str(context_demo.get("original_first_stage_prompt_sha256"))
            if context_demo.get("original_first_stage_prompt_sha256")
            else None
        ),
    }


__all__ = ["score_history", "score_history_detail", "score_history_options"]
