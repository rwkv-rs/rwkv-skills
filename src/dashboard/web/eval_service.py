"""Eval-record and context access for the FastAPI API.

Thin wrapper over :class:`src.db.eval_db_service.EvalDbService` that reuses the
project's existing Python data layer (no Rust backend, no raw SQL here). Returns
plain JSON-serialisable structures for the React frontend.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.constants import EVAL_PAGE_SIZE
from ..core.vocab import token_id_to_display
from .context_display import clean_context_for_display, clean_legacy_role_message_labels
from .store import DashboardStore


def _store_or_default(store: DashboardStore | None = None) -> DashboardStore:
    return store or DashboardStore()


def rebuild_score_index_from_db(
    *,
    include_param_search: bool = False,
    store: DashboardStore | None = None,
) -> int:
    """Rebuild ``score_index.jsonl`` from the latest DB scores.

    The leaderboard reader (``load_scores``) consumes the jsonl unchanged; this
    only refreshes the *source* file so newly-completed evals show up. Rows from
    ``list_latest_scores_for_space`` already carry exactly the keys the reader
    parses (dataset/model/cot/task/task_id/metrics/samples/problems/created_at/
    log_path/task_details/is_param_search). Returns the number of entries written.
    """
    return _store_or_default(store).rebuild_score_index_from_db(
        include_param_search=include_param_search,
    )


def _extract_context_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _build_stop_tokens_mapping(sampling_config: Any) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(sampling_config, dict):
        return mapping
    for stage_name, stage_cfg in sampling_config.items():
        if not isinstance(stage_name, str) or not isinstance(stage_cfg, dict):
            continue
        stop_tokens = stage_cfg.get("stop_tokens")
        if not isinstance(stop_tokens, list):
            continue
        rows: list[dict[str, Any]] = []
        for token_id in stop_tokens:
            try:
                tid = int(token_id)
            except (TypeError, ValueError):
                continue
            rows.append({"id": tid, "token": token_id_to_display(tid)})
        mapping[stage_name] = rows
    return mapping


def load_eval_records(
    *,
    task_id: int,
    only_wrong: bool = False,
    limit: int = EVAL_PAGE_SIZE,
    offset: int = 0,
    store: DashboardStore | None = None,
) -> dict[str, Any]:
    """Return one page of eval records for a task (no full context payload)."""

    records = _store_or_default(store).list_eval_records_for_space(
        task_id=str(task_id),
        only_wrong=only_wrong,
        limit=limit + 1,  # fetch one extra to detect has_more
        offset=offset,
        include_context=False,
        include_preview=True,
    )
    has_more = len(records) > limit
    page = records[:limit]
    for record in page:
        preview = record.get("context_preview")
        if isinstance(preview, str):
            record["context_preview"] = clean_legacy_role_message_labels(preview)
    return {
        "task_id": task_id,
        "records": page,
        "offset": offset,
        "limit": limit,
        "next_offset": offset + len(page),
        "has_more": has_more,
    }


def load_eval_context(
    *,
    task_id: int,
    sample_index: int,
    repeat_index: int,
    pass_index: int = 0,
    store: DashboardStore | None = None,
) -> dict[str, Any]:
    """Return the full structured context for a single attempt (modal payload)."""

    event: dict[str, Any] = {
        "view": "text",
        "raw_text": "",
        "context": None,
        "stop_tokens": {},
        "errors": [],
    }
    try:
        resolved_store = _store_or_default(store)
        context_value = resolved_store.get_eval_context_for_space(
            task_id=str(task_id),
            sample_index=sample_index,
            repeat_index=repeat_index,
            pass_index=pass_index,
        )
    except Exception as exc:  # noqa: BLE001
        event["raw_text"] = f"读取 context 失败：{exc}"
        event["errors"].append("read_context_failed")
        return event

    if context_value is None:
        event["raw_text"] = "当前样本没有 context 内容。"
        return event

    if isinstance(context_value, str):
        event["raw_text"] = context_value
    else:
        try:
            event["raw_text"] = json.dumps(context_value, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            event["raw_text"] = str(context_value)

    context_obj = _extract_context_object(context_value)
    if context_obj is None:
        return event
    if isinstance(context_value, str):
        try:
            event["raw_text"] = json.dumps(context_obj, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass
    context_obj = clean_context_for_display(context_obj)

    sampling_config = context_obj.get("sampling_config")
    if not isinstance(sampling_config, dict) or not sampling_config:
        try:
            task_bundle = resolved_store.get_task_bundle(task_id=str(task_id))
        except Exception as exc:  # noqa: BLE001
            task_bundle = None
            event["errors"].append(f"read_task_sampling_failed:{exc}")
        task = task_bundle.get("task") if isinstance(task_bundle, dict) else None
        task_sampling = task.get("sampling_config") if isinstance(task, dict) else None
        if isinstance(task_sampling, dict) and task_sampling:
            context_obj["sampling_config"] = task_sampling

    stages = context_obj.get("stages")
    sampling_config = context_obj.get("sampling_config")
    if isinstance(stages, list) or isinstance(sampling_config, dict):
        event["view"] = "structured"
        event["context"] = context_obj
        event["stop_tokens"] = _build_stop_tokens_mapping(sampling_config)
        try:
            event["raw_text"] = json.dumps(context_obj, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass
    return event


__all__ = [
    "load_eval_records",
    "load_eval_context",
    "rebuild_score_index_from_db",
]
