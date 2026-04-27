"""Eval record loading, rendering, and context modal payload helpers."""

from __future__ import annotations

import html as html_module
import json
from datetime import datetime
from typing import Any

from src.db.eval_db_service import EvalDbService

from .constants import (
    EVAL_CONTEXT_PREVIEW_LIMIT,
    EVAL_FETCH_ROWS,
    EVAL_OVERSCAN_ROWS,
    EVAL_PAGE_SIZE,
    EVAL_PRELOAD_ROWS,
)
from .vocab import token_id_to_display


def _html(text: Any) -> str:
    return html_module.escape(str(text), quote=True)


def _parse_click_payload(payload: str) -> str | None:
    if not payload:
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    cell_id = parsed.get("cell_id")
    if isinstance(cell_id, str) and cell_id.strip():
        return cell_id.strip()
    return None


def _parse_context_payload(payload: str) -> tuple[str | None, int | None, int | None]:
    if not payload:
        return None, None, None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None, None, None
    if not isinstance(parsed, dict):
        return None, None, None

    cell_id = parsed.get("cell_id")
    sample_index = parsed.get("sample_index")
    repeat_index = parsed.get("repeat_index")
    if not isinstance(cell_id, str) or not cell_id.strip():
        return None, None, None
    try:
        sample = int(sample_index)
        repeat = int(repeat_index)
    except (TypeError, ValueError):
        return None, None, None
    if sample < 0 or repeat < 0:
        return None, None, None
    return cell_id.strip(), sample, repeat


def _context_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(value)


def _truncate_preview(text: str, limit: int = EVAL_CONTEXT_PREVIEW_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _empty_eval_loader_state() -> dict[str, Any]:
    return {
        "cell_id": None,
        "task_id": None,
        "only_wrong": False,
        "next_offset": 0,
        "page_size": EVAL_FETCH_ROWS,
        "has_more": False,
        "meta": None,
    }


def _render_eval_records_html(
    *,
    meta: dict[str, Any] | None,
    records: list[dict[str, Any]],
    only_wrong: bool,
    is_loading: bool = False,
) -> str:
    if meta is None:
        return """
<div class="space-section-card space-eval-panel">
  <div class="space-section-header">
    <h3 class="space-section-title">Eval 记录</h3>
    <div class="subtitle">点击上方分数后，这里会展示对应样本记录。</div>
  </div>
</div>
""".strip()

    benchmark = _html(meta.get("benchmark_name") or "N/A")
    method = _html(meta.get("eval_method") or "N/A")
    model = _html(meta.get("model") or "N/A")
    filter_tag = "仅错题" if only_wrong else "全部样本"

    if not records:
        loading_text = " · 还有更多，请点击\"下一页\"" if is_loading else ""
        return f"""
<div class="space-section-card space-eval-panel">
  <div class="space-section-header">
    <h3 class="space-section-title">Eval 记录</h3>
    <div class="subtitle">{benchmark} · {method} · {model} · {filter_tag}{loading_text}</div>
  </div>
  <div class="space-table-empty">当前筛选下没有样本记录。</div>
</div>
""".strip()

    header_cells = "".join(
        f"<th>{name}</th>"
        for name in ("#", "model_output", "ref_answer", "is_passed", "fail_reason", "context")
    )
    body_rows: list[str] = []
    sorted_rows = sorted(
        records,
        key=lambda item: (
            int(item.get("sample_index", 0)),
            int(item.get("repeat_index", 0)),
        ),
    )
    meta_cell_id = meta.get("cell_id") if isinstance(meta.get("cell_id"), str) else ""
    for row in sorted_rows:
        try:
            sample_index = int(row.get("sample_index", 0))
        except (TypeError, ValueError):
            sample_index = 0
        try:
            repeat_index = int(row.get("repeat_index", 0))
        except (TypeError, ValueError):
            repeat_index = 0

        preview_source = str(row.get("context_preview") or "")
        if not preview_source and "context" in row:
            preview_source = _context_text(row.get("context"))
        preview = _truncate_preview(preview_source, limit=EVAL_CONTEXT_PREVIEW_LIMIT)

        if meta_cell_id:
            context_button = (
                f'<button type="button" class="space-context-open" '
                f'data-context-cell-id="{_html(meta_cell_id)}" '
                f'data-sample-index="{sample_index}" '
                f'data-repeat-index="{repeat_index}">'
                f'{_html(preview)}</button>'
            )
        else:
            context_button = _html(preview)

        is_passed = bool(row.get("is_passed"))
        pass_text = "PASS" if is_passed else "FAIL"
        pass_class = "pass-yes" if is_passed else "pass-no"

        cells = "".join(
            [
                f"<td>{_html(sample_index)}/{_html(repeat_index)}</td>",
                f"<td>{_html(row.get('answer') or '')}</td>",
                f"<td>{_html(row.get('ref_answer') or '')}</td>",
                f'<td class="{pass_class}">{pass_text}</td>',
                f"<td>{_html(row.get('fail_reason') or '')}</td>",
                f"<td>{context_button}</td>",
            ]
        )
        body_rows.append(f"<tr>{cells}</tr>")

    rows_html = "".join(body_rows)
    load_status = (
        f"已加载 {len(sorted_rows)} 条（每页 {EVAL_PAGE_SIZE} 条，点击\"下一页\"继续加载）"
        if is_loading
        else f"已加载 {len(sorted_rows)} 条（每页 {EVAL_PAGE_SIZE} 条，已全部加载）"
    )
    return f"""
<div class="space-section-card space-eval-panel">
  <div class="space-section-header">
    <h3 class="space-section-title">Eval 记录</h3>
    <div class="subtitle">{benchmark} · {method} · {model} · {filter_tag}</div>
    <div class="subtitle">{load_status}</div>
  </div>
  <div class="space-table-wrapper">
    <table class="eval-table">
      <thead><tr>{header_cells}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
""".strip()


def _start_eval_records_load(
    *,
    cell_id: str | None,
    only_wrong: bool,
    interaction_meta: dict[str, dict[str, Any]],
) -> tuple[str, str | None, list[dict[str, Any]], dict[str, Any], bool]:
    if not cell_id:
        return _render_eval_records_html(meta=None, records=[], only_wrong=only_wrong), None, [], _empty_eval_loader_state(), False

    meta = interaction_meta.get(cell_id)
    if not meta:
        return _render_eval_records_html(meta=None, records=[], only_wrong=only_wrong), None, [], _empty_eval_loader_state(), False

    task_id = meta.get("task_id")
    if not isinstance(task_id, int):
        return _render_eval_records_html(meta=meta, records=[], only_wrong=only_wrong), cell_id, [], _empty_eval_loader_state(), False

    try:
        first_batch = EvalDbService().list_eval_records_for_space(
            task_id=str(task_id),
            only_wrong=bool(only_wrong),
            limit=EVAL_PRELOAD_ROWS + EVAL_OVERSCAN_ROWS,
            offset=0,
            include_context=False,
        )
    except Exception as exc:  # noqa: BLE001
        error_html = f'<div class="space-table-empty">读取 Eval 记录失败：{_html(exc)}</div>'
        return error_html, cell_id, [], _empty_eval_loader_state(), False

    visible_rows = first_batch[:EVAL_PRELOAD_ROWS]
    has_more = len(first_batch) > EVAL_PRELOAD_ROWS
    loader_state = {
        "cell_id": cell_id,
        "task_id": task_id,
        "only_wrong": bool(only_wrong),
        "next_offset": len(visible_rows),
        "page_size": EVAL_FETCH_ROWS,
        "has_more": has_more,
        "meta": meta,
    }
    html = _render_eval_records_html(
        meta=meta,
        records=visible_rows,
        only_wrong=bool(only_wrong),
        is_loading=has_more,
    )
    return html, cell_id, visible_rows, loader_state, has_more


def _continue_eval_records_load(
    *,
    loader_state: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any], bool]:
    state = dict(loader_state) if isinstance(loader_state, dict) else _empty_eval_loader_state()
    safe_records = list(records) if isinstance(records, list) else []
    meta = state.get("meta") if isinstance(state.get("meta"), dict) else None
    only_wrong = bool(state.get("only_wrong", False))

    if not state.get("has_more"):
        html = _render_eval_records_html(meta=meta, records=safe_records, only_wrong=only_wrong, is_loading=False)
        return html, safe_records, state, False

    task_id = state.get("task_id")
    if not isinstance(task_id, int):
        state["has_more"] = False
        html = _render_eval_records_html(meta=meta, records=safe_records, only_wrong=only_wrong, is_loading=False)
        return html, safe_records, state, False

    try:
        next_offset = int(state.get("next_offset", len(safe_records)))
    except (TypeError, ValueError):
        next_offset = len(safe_records)
    next_offset = max(0, next_offset)

    try:
        page_size = int(state.get("page_size", EVAL_FETCH_ROWS))
    except (TypeError, ValueError):
        page_size = EVAL_FETCH_ROWS
    page_size = max(1, page_size)

    try:
        next_batch = EvalDbService().list_eval_records_for_space(
            task_id=str(task_id),
            only_wrong=only_wrong,
            limit=page_size + EVAL_OVERSCAN_ROWS,
            offset=next_offset,
            include_context=False,
        )
    except Exception as exc:  # noqa: BLE001
        state["has_more"] = False
        base_html = _render_eval_records_html(meta=meta, records=safe_records, only_wrong=only_wrong, is_loading=False)
        error_html = base_html + f'<div class="space-table-empty">加载下一页失败：{_html(exc)}</div>'
        return error_html, safe_records, state, False

    append_rows = next_batch[:page_size]
    merged_records = safe_records + append_rows
    state["next_offset"] = next_offset + len(append_rows)
    state["has_more"] = len(next_batch) > page_size

    html = _render_eval_records_html(
        meta=meta,
        records=merged_records,
        only_wrong=only_wrong,
        is_loading=bool(state["has_more"]),
    )
    return html, merged_records, state, bool(state["has_more"])


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
        if not isinstance(stage_name, str):
            continue
        if not isinstance(stage_cfg, dict):
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
            rows.append(
                {
                    "id": tid,
                    "token": token_id_to_display(tid),
                }
            )
        mapping[stage_name] = rows

    return mapping


def _build_context_event_payload(
    *,
    payload: str,
    interaction_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base_event: dict[str, Any] = {
        "kind": "eval_context",
        "ts": datetime.now().timestamp(),
        "view": "text",
        "raw_text": "",
        "context": None,
        "stop_tokens": {},
        "errors": [],
    }

    cell_id, sample_index, repeat_index = _parse_context_payload(payload)
    if not cell_id:
        base_event["errors"].append("未解析到上下文定位信息。")
        return base_event

    meta = interaction_meta.get(cell_id) if isinstance(interaction_meta, dict) else None
    if not isinstance(meta, dict):
        base_event["raw_text"] = "未找到对应分数单元格，请重新点击分数后再试。"
        base_event["errors"].append("missing_cell_meta")
        return base_event

    task_id = meta.get("task_id")
    if not isinstance(task_id, int):
        base_event["raw_text"] = "该分数没有可查询的 task_id。"
        base_event["errors"].append("missing_task_id")
        return base_event

    try:
        context_value = EvalDbService().get_eval_context_for_space(
            task_id=str(task_id),
            sample_index=sample_index,
            repeat_index=repeat_index,
        )
    except Exception as exc:  # noqa: BLE001
        base_event["raw_text"] = f"读取 context 失败：{exc}"
        base_event["errors"].append("read_context_failed")
        return base_event

    if context_value is None:
        base_event["raw_text"] = "当前样本没有 context 内容。"
        return base_event

    if isinstance(context_value, str):
        base_event["raw_text"] = context_value
    else:
        try:
            base_event["raw_text"] = json.dumps(context_value, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            base_event["raw_text"] = str(context_value)

    context_obj = _extract_context_object(context_value)
    if context_obj is None:
        return base_event
    if isinstance(context_value, str):
        try:
            base_event["raw_text"] = json.dumps(context_obj, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass

    stages = context_obj.get("stages")
    events = context_obj.get("events")
    sampling_config = context_obj.get("sampling_config")
    if isinstance(stages, list) or isinstance(events, list) or isinstance(sampling_config, dict):
        base_event["view"] = "structured"
        base_event["context"] = context_obj
        base_event["stop_tokens"] = _build_stop_tokens_mapping(sampling_config)

    return base_event
