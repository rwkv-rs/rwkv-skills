"""JSON serialization of leaderboard pivots and chart series for the FastAPI API.

This module reuses the pure data/selection/metrics logic from
``src.dashboard.core``, but emits plain JSON-serialisable structures instead of
HTML/Plotly figures so a React frontend can render them.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..core.constants import (
    AUTO_MODEL_LABEL,
    DOMAIN_GROUPS,
    TABLE_VIEW_LABELS,
    DetailPoint,
    ParamLineage,
    SelectionState,
    TableCellMeta,
    _normalize_table_view,
)
from ..core.boards import BOARD_NAIVE, BOARD_NORMAL, filter_entries_by_board
from ..core.data import ScoreEntry
from ..core.metrics import (
    _detail_sort_key,
    _field_primary_score,
    _format_param,
    _numeric_value,
    _score_to_percent,
)
from ..core.selection import (
    _build_detail_point_map,
    _build_model_entries_cache,
    _entries_for_model_in_domains,
    _model_data_param_label,
    _resolve_param_lineages,
    _series_sort_key,
)
from ..core.tables import _make_cell_id, _tooltip_for_entry

# Rows whose best score is below this percentage are hidden (matches the old
# Gradio behaviour where near-zero rows were dropped to reduce noise).
MIN_VISIBLE_PERCENT = 10.0


# ---------------------------------------------------------------------------
# Cell + meta helpers
# ---------------------------------------------------------------------------

def _meta_dict(
    *,
    prefix: str,
    row_key: tuple[str, str, str],
    param: str,
    model: str,
    entry: ScoreEntry,
    column_label: str,
) -> dict[str, Any]:
    meta = TableCellMeta(
        cell_id=_make_cell_id(prefix, row_key[0], row_key[1], row_key[2], param, model),
        task_id=entry.task_id,
        benchmark_name=row_key[0],
        eval_method=row_key[1],
        k_metric=row_key[2],
        column_label=column_label,
        model=entry.model,
        tooltip=_tooltip_for_entry(entry),
        clickable=entry.task_id is not None,
    )
    return meta.as_dict()


def _param_columns(lineages: Iterable[ParamLineage]) -> list[dict[str, Any]]:
    columns: list[dict[str, Any]] = []
    for item in lineages:
        columns.append(
            {
                "param": item.param,
                "param_label": _format_param(item.param).lower(),
                "latest_model": item.latest_model,
                "latest_label": _model_data_param_label(item.latest_model, include_params=False),
                "prev_model": item.prev_model,
                "prev_label": (
                    _model_data_param_label(item.prev_model, include_params=False)
                    if item.prev_model
                    else None
                ),
            }
        )
    return columns


# ---------------------------------------------------------------------------
# Detail views (latest / delta)
# ---------------------------------------------------------------------------

def _serialize_detail_latest(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
    domains: set[str],
    interaction_meta: dict[str, dict[str, Any]],
    cell_prefix: str,
) -> list[dict[str, Any]]:
    row_values: dict[tuple[str, str, str], dict[str, DetailPoint]] = {}
    for item in lineages:
        model_entries = _entries_for_model_in_domains(model_cache, item.latest_model, domains)
        for row_key, point in _build_detail_point_map(model_entries).items():
            row_values.setdefault(row_key, {})[item.param] = point

    rows: list[dict[str, Any]] = []
    for row_key in sorted(row_values, key=_detail_sort_key):
        points = row_values[row_key]
        percents = [
            _score_to_percent(points[item.param].score)
            for item in lineages
            if points.get(item.param) is not None
        ]
        best = max((p for p in percents if p is not None), default=None)
        if best is None or best < MIN_VISIBLE_PERCENT:
            continue

        sample_counts = [p.entry.samples for p in points.values() if p and p.entry.samples]
        cells: list[dict[str, Any]] = []
        for item in lineages:
            point = points.get(item.param)
            if point is None:
                cells.append({"percent": None, "meta": None})
                continue
            meta = _meta_dict(
                prefix=f"{cell_prefix}_latest",
                row_key=row_key,
                param=item.param,
                model=item.latest_model,
                entry=point.entry,
                column_label=f"{_format_param(item.param).lower()} · {item.latest_label}",
            )
            interaction_meta[meta["cell_id"]] = meta
            cells.append({"percent": _score_to_percent(point.score), "meta": meta})

        rows.append(
            {
                "benchmark_name": row_key[0],
                "num_samples": max(sample_counts) if sample_counts else None,
                "eval_method": row_key[1],
                "k_metric": row_key[2],
                "cells": cells,
            }
        )
    return rows


def _serialize_detail_delta(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
    domains: set[str],
    interaction_meta: dict[str, dict[str, Any]],
    cell_prefix: str,
) -> list[dict[str, Any]]:
    latest_by_param: dict[str, dict[tuple[str, str, str], DetailPoint]] = {}
    prev_by_param: dict[str, dict[tuple[str, str, str], DetailPoint]] = {}
    row_keys: set[tuple[str, str, str]] = set()
    for item in lineages:
        latest_map = _build_detail_point_map(
            _entries_for_model_in_domains(model_cache, item.latest_model, domains)
        )
        prev_map = _build_detail_point_map(
            _entries_for_model_in_domains(model_cache, item.prev_model, domains)
        )
        latest_by_param[item.param] = latest_map
        prev_by_param[item.param] = prev_map
        row_keys.update(latest_map.keys())
        row_keys.update(prev_map.keys())

    scored_rows: list[tuple[float, tuple[Any, ...], dict[str, Any]]] = []
    for row_key in row_keys:
        cells: list[dict[str, Any]] = []
        deltas: list[float] = []
        all_percents: list[float] = []
        sample_counts: list[int] = []
        for item in lineages:
            latest_point = latest_by_param.get(item.param, {}).get(row_key)
            prev_point = prev_by_param.get(item.param, {}).get(row_key)
            latest_pct = _score_to_percent(latest_point.score) if latest_point else None
            prev_pct = _score_to_percent(prev_point.score) if prev_point else None
            delta = latest_pct - prev_pct if latest_pct is not None and prev_pct is not None else None
            if delta is not None:
                deltas.append(delta)
            for pct in (latest_pct, prev_pct):
                if pct is not None:
                    all_percents.append(pct)

            prev_meta = None
            if prev_point is not None and item.prev_model:
                prev_meta = _meta_dict(
                    prefix=f"{cell_prefix}_delta_prev",
                    row_key=row_key,
                    param=item.param,
                    model=item.prev_model,
                    entry=prev_point.entry,
                    column_label=f"{_format_param(item.param).lower()} prev ({item.prev_label})",
                )
                interaction_meta[prev_meta["cell_id"]] = prev_meta
                if prev_point.entry.samples:
                    sample_counts.append(prev_point.entry.samples)
            latest_meta = None
            if latest_point is not None:
                latest_meta = _meta_dict(
                    prefix=f"{cell_prefix}_delta_latest",
                    row_key=row_key,
                    param=item.param,
                    model=item.latest_model,
                    entry=latest_point.entry,
                    column_label=f"{_format_param(item.param).lower()} latest ({item.latest_label})",
                )
                interaction_meta[latest_meta["cell_id"]] = latest_meta
                if latest_point.entry.samples:
                    sample_counts.append(latest_point.entry.samples)

            cells.append(
                {
                    "prev": prev_pct,
                    "latest": latest_pct,
                    "delta": delta,
                    "prev_meta": prev_meta,
                    "latest_meta": latest_meta,
                }
            )

        best = max(all_percents, default=None)
        if best is None or best < MIN_VISIBLE_PERCENT:
            continue
        avg_delta = sum(deltas) / len(deltas) if deltas else float("-inf")
        scored_rows.append(
            (
                avg_delta,
                _detail_sort_key(row_key),
                {
                    "benchmark_name": row_key[0],
                    "num_samples": max(sample_counts) if sample_counts else None,
                    "eval_method": row_key[1],
                    "k_metric": row_key[2],
                    "cells": cells,
                },
            )
        )

    scored_rows.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored_rows]


# ---------------------------------------------------------------------------
# Field-average views (overview across domains)
# ---------------------------------------------------------------------------

def _field_average_percent(
    entries: list[ScoreEntry],
) -> float | None:
    percents = [
        _score_to_percent(score)
        for score in (_field_primary_score(entry) for entry in entries)
        if score is not None
    ]
    percents = [p for p in percents if p is not None]
    if not percents:
        return None
    return sum(percents) / len(percents)


def _serialize_field_avg(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
    delta: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in DOMAIN_GROUPS:
        domains = set(group["domains"])
        cells: list[dict[str, Any]] = []
        any_value = False
        for item in lineages:
            latest_entries = _entries_for_model_in_domains(model_cache, item.latest_model, domains)
            latest_pct = _field_average_percent(latest_entries)
            if delta:
                prev_entries = _entries_for_model_in_domains(model_cache, item.prev_model, domains)
                prev_pct = _field_average_percent(prev_entries)
                d = latest_pct - prev_pct if latest_pct is not None and prev_pct is not None else None
                cells.append({"prev": prev_pct, "latest": latest_pct, "delta": d})
                any_value = any_value or latest_pct is not None or prev_pct is not None
            else:
                cells.append({"percent": latest_pct})
                any_value = any_value or latest_pct is not None
        if any_value:
            rows.append({"domain_key": group["key"], "domain_title": group["title"], "cells": cells})
    return rows


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def serialize_leaderboard(
    selection: SelectionState,
    *,
    all_entries: list[ScoreEntry],
    view_mode: str,
) -> dict[str, Any]:
    """Build the full leaderboard payload for a given table view."""

    mode = _normalize_table_view(view_mode)
    source_entries = all_entries if all_entries else selection.entries

    # Keep the two boards fully isolated. Domain tabs are 正式榜 only; the
    # separate 朴素榜 tab is a flat filtered view over naive scores only.
    domain_entries = filter_entries_by_board(source_entries, BOARD_NORMAL)
    naive_entries = filter_entries_by_board(source_entries, BOARD_NAIVE)

    # Resolve columns independently so a newer naive-only run cannot pull the
    # normal tabs onto the wrong model lineage (or vice versa).
    domain_lineages = _resolve_param_lineages(domain_entries, selection)
    naive_lineages = _resolve_param_lineages(naive_entries, selection)
    domain_columns = _param_columns(domain_lineages)
    naive_columns = _param_columns(naive_lineages)
    domain_cache = _build_model_entries_cache(domain_entries)
    naive_cache = _build_model_entries_cache(naive_entries)
    interaction_meta: dict[str, dict[str, Any]] = {}

    is_delta = mode.endswith("_delta")
    payload: dict[str, Any] = {
        "view": mode,
        "view_label": TABLE_VIEW_LABELS.get(mode, mode),
        "is_delta": is_delta,
        "is_field_avg": mode.startswith("field_avg"),
        "param_columns": domain_columns,
        "interaction_meta": interaction_meta,
    }

    if mode in ("field_avg_latest", "field_avg_delta"):
        payload["overview"] = _serialize_field_avg(
            lineages=domain_lineages,
            model_cache=domain_cache,
            delta=(mode == "field_avg_delta"),
        )
        payload["domains"] = []
    else:
        domains_payload: list[dict[str, Any]] = []
        for group in DOMAIN_GROUPS:
            domains = set(group["domains"])
            if is_delta:
                rows = _serialize_detail_delta(
                    lineages=domain_lineages,
                    model_cache=domain_cache,
                    domains=domains,
                    interaction_meta=interaction_meta,
                    cell_prefix=f"normal_{group['key']}",
                )
            else:
                rows = _serialize_detail_latest(
                    lineages=domain_lineages,
                    model_cache=domain_cache,
                    domains=domains,
                    interaction_meta=interaction_meta,
                    cell_prefix=f"normal_{group['key']}",
                )
            domains_payload.append(
                {
                    "key": group["key"],
                    "title": group["title"],
                    "label": group["label"],
                    "param_columns": domain_columns,
                    "rows": rows,
                }
            )
        payload["domains"] = domains_payload

    # 朴素榜: a single flat detail table over ALL naive scores (no domain split),
    # following the latest/delta of the current view. Always present as its own tab.
    all_domains: set[str] = set()
    for group in DOMAIN_GROUPS:
        all_domains.update(group["domains"])
    naive_serializer = _serialize_detail_delta if is_delta else _serialize_detail_latest
    payload["naive_board"] = {
        "key": BOARD_NAIVE,
        "title": "朴素榜",
        "label": "朴素榜",
        "is_delta": is_delta,
        "param_columns": naive_columns,
        "rows": naive_serializer(
            lineages=naive_lineages,
            model_cache=naive_cache,
            domains=all_domains,
            interaction_meta=interaction_meta,
            cell_prefix="naive",
        ),
    }
    return payload


def serialize_selection(selection: SelectionState) -> dict[str, Any]:
    """Serialise the resolved selection state for the control panel."""

    return {
        "dropdown_value": selection.dropdown_value,
        "selected_label": selection.selected_label,
        "auto_selected": selection.auto_selected,
        "model_sequence": list(selection.model_sequence),
        "skipped_small_params": selection.skipped_small_params,
        "auto_label": AUTO_MODEL_LABEL,
    }


__all__ = [
    "serialize_leaderboard",
    "serialize_selection",
    "MIN_VISIBLE_PERCENT",
]
