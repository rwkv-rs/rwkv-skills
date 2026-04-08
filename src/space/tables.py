from __future__ import annotations

"""Table building, tooltip formatting, HTML rendering, and CSV export helpers."""

import csv
import hashlib
import html
import io
import os
import re
import tempfile
from typing import Any, Iterable

from .constants import (
    DEFAULT_TABLE_VIEW,
    DOMAIN_GROUPS,
    SUBDOMAIN_ORDER,
    TABLE_VIEW_LABELS,
    DetailPoint,
    ParamLineage,
    SelectionState,
    TableCellMeta,
    _normalize_table_view,
)
from .data import ScoreEntry, parse_model_signature
from .domains import is_instruction_following_domain, is_multi_choice_domain
from .score_index import resolve_score_index_path
from .metrics import (
    _cell_metric_value,
    _cell_numeric_value,
    _dataset_base,
    _detail_sort_key,
    _entry_method_tag,
    _field_average_score,
    _format_param,
    _map_subject_to_subdomain,
    _max_percent,
    _metric_score,
    _numeric_value,
    _parse_display_number,
    _primary_metric,
    _score_to_percent,
    _styled_delta_cell,
    _styled_score_cell,
)
from .selection import (
    _build_detail_point_map,
    _build_model_entries_cache,
    _entries_for_model_in_domains,
    _model_data_param_label,
    _resolve_param_lineages,
    _summarise_snapshots,
)


def _html(text: Any) -> str:
    """Escape text for safe HTML rendering."""
    return html.escape(str(text), quote=True)


_PAREN_RE = re.compile(r"\s*\(.*?\)")


def _col_display_name(header_name: str) -> str:
    """Shorten a data column header to show version only.

    Examples:
      "0.4b prev (g1d)"  → "g1d"
      "0.4b latest (g1d)" → "g1d"
      "0.4b delta"        → "delta"
    Fixed meta columns are returned unchanged.
    """
    name = str(header_name).strip()
    if name in {"benchmark_name", "num_samples", "eval_method", "k_metric", "field_name", "metric_rule"}:
        return name

    # Extract version from parentheses
    match = _PAREN_RE.search(name)
    if match:
        version = match.group(0).strip("() ")
        if version:
            return version

    # Fallback: extract last token
    clean = _PAREN_RE.sub("", name).strip()
    parts = clean.split()
    return parts[-1] if parts else name


def _render_summary(
    *,
    all_entries: list[ScoreEntry],
    visible: list[ScoreEntry],
    selection: SelectionState,
    view_mode: str = DEFAULT_TABLE_VIEW,
    warnings: Iterable[str] | None = None,
) -> str:
    score_index_path = resolve_score_index_path()
    pg_host = os.environ.get("PG_HOST", "localhost")
    pg_port = os.environ.get("PG_PORT", "5432")
    pg_name = os.environ.get("PG_DBNAME", "rwkv-eval")

    if not all_entries:
        return (
            "当前没有可展示分数。"
            f"（score index：`{score_index_path}`，明细回看仍使用 PostgreSQL `"
            f"{pg_host}:{pg_port}/{pg_name}`）"
        )

    normalized_view = _normalize_table_view(view_mode)
    benchmark_count = len({(_dataset_base(entry.dataset), _entry_method_tag(entry)) for entry in visible})
    lines = [
        f"- 数据源：score_index (`{score_index_path}`)",
        "- 数据范围：仅正式评测（已过滤 param-search）",
        f"- 明细回看：PostgreSQL (`{pg_host}:{pg_port}/{pg_name}`)",
        f"- 当前策略：`{selection.selected_label}`" + ("（按排序规则自动选择）" if selection.auto_selected else ""),
        f"- 表格视图：{TABLE_VIEW_LABELS.get(normalized_view, TABLE_VIEW_LABELS[DEFAULT_TABLE_VIEW])}",
        "- 领域分块：knowledge / math / coding / instruction_following / function_call",
        f"- 模型列数：{len(selection.model_sequence)}",
        f"- 基准行数：{benchmark_count}",
        f"- 可见数据集：{len(visible)} / 总分数记录：{len(all_entries)}",
        "- 排序：架构 > 参数量 > data_version（G0→…→G1d）> domain > dataset / task",
    ]
    if selection.aggregated_models:
        lines.extend(_summarise_snapshots(selection.aggregated_models))
    for warn in warnings or ():
        lines.append(f"⚠️ {warn}")
    return "\n".join(lines)


def _build_pivot_table(selection: SelectionState, entries: Iterable[ScoreEntry] | None = None) -> tuple[list[str], list[list[Any]]]:
    headers = ["Benchmark"] + [_model_data_param_label(model, include_params=True) for model in selection.model_sequence]
    target_entries = list(entries) if entries is not None else selection.entries
    if not target_entries:
        return headers, []

    row_meta: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], ScoreEntry] = {}

    for entry in target_entries:
        base = _dataset_base(entry.dataset)
        method = _entry_method_tag(entry)
        row_key = (base, method)
        meta = row_meta.get(row_key)
        if meta is None:
            primary = _primary_metric(entry.metrics)
            metric_name = primary[0] if primary else None
            row_meta[row_key] = {
                "base": base,
                "method": method,
                "metric": metric_name or "metric",
            }
        elif meta["metric"] == "metric":
            primary = _primary_metric(entry.metrics)
            metric_name = primary[0] if primary else None
            if metric_name:
                meta["metric"] = metric_name

        group_key = (entry.model, base, method)
        current = grouped.get(group_key)
        cur_score = _metric_score(current)
        new_score = _metric_score(entry)
        if current is None or (new_score is not None and (cur_score is None or new_score > cur_score)) or (
            new_score == cur_score and entry.created_at > current.created_at
        ):
            grouped[group_key] = entry

    ordered_rows = sorted(
        row_meta.values(),
        key=lambda meta: (meta["base"], 0 if meta["method"] == "cot" else 1),
    )

    rows: list[list[Any]] = []
    for meta in ordered_rows:
        row_label = f"{meta['base']}_{meta['method']}"
        row: list[Any] = [row_label]
        numeric_values: list[float | None] = []
        for model in selection.model_sequence:
            entry = grouped.get((model, meta["base"], meta["method"]))
            numeric_values.append(_cell_numeric_value(entry, dataset_base=meta["base"]))
            row.append(_cell_metric_value(entry, dataset_base=meta["base"]))
        max_score = _max_percent(numeric_values)
        if max_score is None or max_score < 10.0:
            continue
        rows.append(row)
    return headers, rows


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
        score = _score_to_percent(value)
        if score is None:
            continue
        lines.append(f"{key}: {score:.1f}%")
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


def _build_field_avg_latest_table(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
    domains: set[str],
    field_label: str,
) -> tuple[list[str], list[list[Any]]]:
    headers = ["field_name", "metric_rule"] + [
        f"{_format_param(item.param).lower()} · {_model_data_param_label(item.latest_model, include_params=False)}"
        for item in lineages
    ]
    if not lineages:
        return headers, []

    row: list[Any] = [field_label, "benchmark_equal_weight"]
    score_values: list[float | None] = []
    for item in lineages:
        entries = _entries_for_model_in_domains(model_cache, item.latest_model, domains)
        score = _field_average_score(entries)
        score_values.append(score)
        row.append(_styled_score_cell(score))
    max_score = _max_percent(score_values)
    if max_score is None or max_score < 10.0:
        return headers, []
    return headers, [row]


def _build_field_avg_delta_table(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
    domains: set[str],
    field_label: str,
) -> tuple[list[str], list[list[Any]]]:
    headers = ["field_name", "metric_rule"]
    for item in lineages:
        param_label = _format_param(item.param).lower()
        latest_label = _model_data_param_label(item.latest_model, include_params=False)
        prev_label = _model_data_param_label(item.prev_model, include_params=False) if item.prev_model else "—"
        headers.extend(
            [
                f"{param_label} prev ({prev_label})",
                f"{param_label} latest ({latest_label})",
                f"{param_label} delta",
            ]
        )
    if not lineages:
        return headers, []

    row: list[Any] = [field_label, "benchmark_equal_weight"]
    score_values: list[float | None] = []
    for item in lineages:
        latest_entries = _entries_for_model_in_domains(model_cache, item.latest_model, domains)
        prev_entries = _entries_for_model_in_domains(model_cache, item.prev_model, domains)
        latest_score = _field_average_score(latest_entries)
        prev_score = _field_average_score(prev_entries)
        score_values.extend([prev_score, latest_score])

        delta_value = None
        latest_n = _score_to_percent(latest_score)
        prev_n = _score_to_percent(prev_score)
        if latest_n is not None and prev_n is not None:
            delta_value = latest_n - prev_n

        row.extend(
            [
                _styled_score_cell(prev_score),
                _styled_score_cell(latest_score),
                _styled_delta_cell(delta_value),
            ]
        )

    max_score = _max_percent(score_values)
    if max_score is None or max_score < 10.0:
        return headers, []
    return headers, [row]


def _build_benchmark_detail_latest_table(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
    domains: set[str],
) -> tuple[list[str], list[list[Any]], dict[tuple[int, int], TableCellMeta]]:
    headers = ["benchmark_name", "num_samples", "eval_method", "k_metric"] + [
        f"{_format_param(item.param).lower()} · {_model_data_param_label(item.latest_model, include_params=False)}"
        for item in lineages
    ]
    if not lineages:
        return headers, [], {}

    row_values: dict[tuple[str, str, str], dict[str, DetailPoint]] = {}
    for item in lineages:
        model_entries = _entries_for_model_in_domains(model_cache, item.latest_model, domains)
        point_map = _build_detail_point_map(model_entries)
        for row_key, point in point_map.items():
            row_values.setdefault(row_key, {})[item.param] = point

    rows: list[list[Any]] = []
    cell_meta: dict[tuple[int, int], TableCellMeta] = {}
    row_idx = 0
    for row_key in sorted(row_values, key=_detail_sort_key):
        points = row_values[row_key]
        all_samples = [
            p.entry.samples for p in points.values()
            if p is not None and p.entry.samples
        ]
        sample_count = max(all_samples) if all_samples else 0
        row = [row_key[0], str(sample_count) if sample_count else "—", row_key[1], row_key[2]]
        max_score = _max_percent(points.get(item.param).score if points.get(item.param) else None for item in lineages)
        if max_score is None or max_score < 10.0:
            continue

        for col_offset, item in enumerate(lineages, start=4):
            point = points.get(item.param)
            row.append(_styled_score_cell(point.score if point else None))
            if point is None:
                continue
            entry = point.entry
            cell_id = _make_cell_id(
                "latest",
                row_key[0],
                row_key[1],
                row_key[2],
                item.param,
                item.latest_model,
            )
            cell_meta[(row_idx, col_offset)] = TableCellMeta(
                cell_id=cell_id,
                task_id=entry.task_id,
                benchmark_name=row_key[0],
                eval_method=row_key[1],
                k_metric=row_key[2],
                column_label=headers[col_offset],
                model=entry.model,
                tooltip=_tooltip_for_entry(entry),
                clickable=entry.task_id is not None,
            )

        rows.append(row)
        row_idx += 1

    return headers, rows, cell_meta


def _build_benchmark_detail_delta_table(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
    domains: set[str],
) -> tuple[list[str], list[list[Any]], dict[tuple[int, int], TableCellMeta]]:
    headers = ["benchmark_name", "num_samples", "eval_method", "k_metric"]
    for item in lineages:
        param_label = _format_param(item.param).lower()
        latest_label = _model_data_param_label(item.latest_model, include_params=False)
        prev_label = _model_data_param_label(item.prev_model, include_params=False) if item.prev_model else "—"
        headers.extend(
            [
                f"{param_label} prev ({prev_label})",
                f"{param_label} latest ({latest_label})",
                f"{param_label} delta",
            ]
        )
    if not lineages:
        return headers, [], {}

    latest_by_param: dict[str, dict[tuple[str, str, str], DetailPoint]] = {}
    prev_by_param: dict[str, dict[tuple[str, str, str], DetailPoint]] = {}
    row_keys: set[tuple[str, str, str]] = set()
    for item in lineages:
        latest_map = _build_detail_point_map(_entries_for_model_in_domains(model_cache, item.latest_model, domains))
        prev_map = _build_detail_point_map(_entries_for_model_in_domains(model_cache, item.prev_model, domains))
        latest_by_param[item.param] = latest_map
        prev_by_param[item.param] = prev_map
        row_keys.update(latest_map.keys())
        row_keys.update(prev_map.keys())

    rows_with_meta: list[tuple[float, tuple[Any, ...], list[Any], dict[int, TableCellMeta]]] = []
    for row_key in row_keys:
        max_score = _max_percent(
            [
                score
                for item in lineages
                for score in (
                    latest_by_param.get(item.param, {}).get(row_key).score
                    if latest_by_param.get(item.param, {}).get(row_key)
                    else None,
                    prev_by_param.get(item.param, {}).get(row_key).score
                    if prev_by_param.get(item.param, {}).get(row_key)
                    else None,
                )
            ]
        )
        if max_score is None or max_score < 10.0:
            continue

        all_samples = [
            p.entry.samples
            for param_map in (latest_by_param, prev_by_param)
            for p in [param_map.get(item.param, {}).get(row_key) for item in lineages]
            if p is not None and p.entry.samples
        ]
        sample_count = max(all_samples) if all_samples else 0
        row = [row_key[0], str(sample_count) if sample_count else "—", row_key[1], row_key[2]]
        delta_values: list[float] = []
        row_cell_meta: dict[int, TableCellMeta] = {}
        for item in lineages:
            latest_point = latest_by_param.get(item.param, {}).get(row_key)
            prev_point = prev_by_param.get(item.param, {}).get(row_key)
            latest_score = latest_point.score if latest_point else None
            prev_score = prev_point.score if prev_point else None
            delta_value = None
            latest_n = _score_to_percent(latest_score)
            prev_n = _score_to_percent(prev_score)
            if latest_n is not None and prev_n is not None:
                delta_value = latest_n - prev_n
                delta_values.append(delta_value)

            prev_col_idx = len(row)
            latest_col_idx = prev_col_idx + 1
            row.extend(
                [
                    _styled_score_cell(prev_score),
                    _styled_score_cell(latest_score),
                    _styled_delta_cell(delta_value),
                ]
            )

            if prev_point is not None and item.prev_model:
                prev_entry = prev_point.entry
                row_cell_meta[prev_col_idx] = TableCellMeta(
                    cell_id=_make_cell_id(
                        "delta_prev",
                        row_key[0],
                        row_key[1],
                        row_key[2],
                        item.param,
                        item.prev_model,
                    ),
                    task_id=prev_entry.task_id,
                    benchmark_name=row_key[0],
                    eval_method=row_key[1],
                    k_metric=row_key[2],
                    column_label=headers[prev_col_idx],
                    model=prev_entry.model,
                    tooltip=_tooltip_for_entry(prev_entry),
                    clickable=prev_entry.task_id is not None,
                )

            if latest_point is not None:
                latest_entry = latest_point.entry
                row_cell_meta[latest_col_idx] = TableCellMeta(
                    cell_id=_make_cell_id(
                        "delta_latest",
                        row_key[0],
                        row_key[1],
                        row_key[2],
                        item.param,
                        item.latest_model,
                    ),
                    task_id=latest_entry.task_id,
                    benchmark_name=row_key[0],
                    eval_method=row_key[1],
                    k_metric=row_key[2],
                    column_label=headers[latest_col_idx],
                    model=latest_entry.model,
                    tooltip=_tooltip_for_entry(latest_entry),
                    clickable=latest_entry.task_id is not None,
                )

        avg_delta = sum(delta_values) / len(delta_values) if delta_values else float("-inf")
        rows_with_meta.append((avg_delta, _detail_sort_key(row_key), row, row_cell_meta))

    ordered = sorted(rows_with_meta, key=lambda item: (-item[0], item[1]))
    rows = [row for _, _, row, _ in ordered]
    cell_meta: dict[tuple[int, int], TableCellMeta] = {}
    for row_idx, (_, _, _, row_cells) in enumerate(ordered):
        for col_idx, meta in row_cells.items():
            cell_meta[(row_idx, col_idx)] = meta

    return headers, rows, cell_meta


def _build_all_field_avg_latest_table(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
) -> tuple[list[str], list[list[Any]]]:
    headers: list[str] | None = None
    rows: list[list[Any]] = []
    for group in DOMAIN_GROUPS:
        table_headers, table_rows = _build_field_avg_latest_table(
            lineages=lineages,
            model_cache=model_cache,
            domains=set(group["domains"]),
            field_label=group["label"],
        )
        if headers is None:
            headers = table_headers
        rows.extend(table_rows)
    return headers or ["field_name", "metric_rule"], rows


def _build_all_field_avg_delta_table(
    *,
    lineages: list[ParamLineage],
    model_cache: dict[str, list[ScoreEntry]],
) -> tuple[list[str], list[list[Any]]]:
    headers: list[str] | None = None
    rows_with_meta: list[tuple[float, int, list[Any]]] = []
    delta_indices: list[int] | None = None

    for idx, group in enumerate(DOMAIN_GROUPS):
        table_headers, table_rows = _build_field_avg_delta_table(
            lineages=lineages,
            model_cache=model_cache,
            domains=set(group["domains"]),
            field_label=group["label"],
        )
        if headers is None:
            headers = table_headers
            delta_indices = [i for i, name in enumerate(headers) if str(name).strip().endswith("delta")]
        if not table_rows:
            continue

        row = table_rows[0]
        values: list[float] = []
        for i in delta_indices or []:
            if i >= len(row):
                continue
            num = _parse_display_number(row[i])
            if num is not None:
                values.append(num)
        avg_delta = sum(values) / len(values) if values else float("-inf")
        rows_with_meta.append((avg_delta, idx, row))

    if headers is None:
        return ["field_name", "metric_rule"], []

    rows = [row for _, _, row in sorted(rows_with_meta, key=lambda item: (-item[0], item[1]))]
    return headers, rows


def _build_domain_tables(
    selection: SelectionState,
    *,
    all_entries: list[ScoreEntry],
    view_mode: str,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    mode = _normalize_table_view(view_mode)
    source_entries = all_entries if all_entries else selection.entries
    model_cache = _build_model_entries_cache(source_entries)
    lineages = _resolve_param_lineages(source_entries, selection)
    interaction_meta: dict[str, dict[str, Any]] = {}

    if mode == "field_avg_latest":
        headers, rows = _build_all_field_avg_latest_table(
            lineages=lineages,
            model_cache=model_cache,
        )
        table_html = _render_pivot_html(headers, rows, title=f"全领域 · {TABLE_VIEW_LABELS[mode]}")
        return {group["key"]: table_html for group in DOMAIN_GROUPS}, interaction_meta

    if mode == "field_avg_delta":
        headers, rows = _build_all_field_avg_delta_table(
            lineages=lineages,
            model_cache=model_cache,
        )
        table_html = _render_pivot_html(headers, rows, title=f"全领域 · {TABLE_VIEW_LABELS[mode]}")
        return {group["key"]: table_html for group in DOMAIN_GROUPS}, interaction_meta

    tables: dict[str, str] = {}
    for group in DOMAIN_GROUPS:
        domains = set(group["domains"])
        if mode == "benchmark_detail_delta":
            headers, rows, cell_meta = _build_benchmark_detail_delta_table(
                lineages=lineages,
                model_cache=model_cache,
                domains=domains,
            )
        else:
            headers, rows, cell_meta = _build_benchmark_detail_latest_table(
                lineages=lineages,
                model_cache=model_cache,
                domains=domains,
            )

        title = f"{group['title']} · {TABLE_VIEW_LABELS[mode]}"
        tables[group["key"]] = _render_pivot_html(headers, rows, title=title, cell_meta=cell_meta)
        for meta in cell_meta.values():
            interaction_meta[meta.cell_id] = meta.as_dict()

    return tables, interaction_meta


def _pivot_to_csv(headers: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([cell[0] if isinstance(cell, tuple) and len(cell) == 2 else cell for cell in row])
    return buffer.getvalue()


def _export_selection_csv(selection: SelectionState) -> str:
    """Build a temporary CSV file for the current selection."""
    headers, rows = _build_pivot_table(selection)
    csv_text = _pivot_to_csv(headers, rows)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=".csv",
        delete=False,
        prefix="rwkv_space_",
    ) as temp_file:
        temp_file.write(csv_text)
        return temp_file.name


_PARAM_TOKEN_PATTERN = re.compile(r"(\d+(?:[._]\d+)?[bB])")


def _header_param_token(header: Any) -> str | None:
    text = str(header).strip()
    if not text:
        return None

    signature = parse_model_signature(text)
    if signature.params:
        return signature.params.lower()

    match = _PARAM_TOKEN_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).lower().replace(".", "_")


def _build_group_row(headers: list[str]) -> str:
    tokens = [_header_param_token(name) for name in headers]
    first_param_index = next((idx for idx, token in enumerate(tokens) if token), len(headers))

    cells: list[str] = []
    if first_param_index > 0:
        cells.append(f"<th colspan=\"{first_param_index}\"></th>")

    idx = first_param_index
    while idx < len(headers):
        token = tokens[idx]
        if not token:
            cells.append("<th class=\"param-group\" colspan=\"1\"></th>")
            idx += 1
            continue

        end = idx + 1
        while end < len(headers) and tokens[end] == token:
            end += 1
        colspan = end - idx
        cls = f"param-group param-{token}"
        label = _format_param(token).upper()
        cells.append(f"<th class=\"{_html(cls)}\" colspan=\"{colspan}\">{_html(label)}</th>")
        idx = end

    if not cells:
        cells.append(f"<th colspan=\"{len(headers)}\"></th>")
    return "".join(cells)


def _header_cell_classes(col_idx: int, header_name: str) -> list[str]:
    classes: list[str] = []
    name = header_name.strip().lower()
    if col_idx == 0:
        classes.append("col-name")
    if name in {"eval_method", "k_metric", "num_samples"}:
        classes.append("col-meta")
    return classes


def _render_pivot_html(
    headers: list[str],
    rows: list[list[Any]],
    *,
    title: str = "明细",
    cell_meta: dict[tuple[int, int], TableCellMeta] | None = None,
) -> str:
    """Render a pivot table with two-row thead and CSS-class-based cell styling."""
    if not headers:
        return '<div class="space-table-empty">当前筛选条件下没有数据。</div>'

    group_row = _build_group_row(headers)
    col_header_cells: list[str] = []
    for col_idx, header_name in enumerate(headers):
        class_names = _header_cell_classes(col_idx, str(header_name))
        class_attr = f' class="{" ".join(class_names)}"' if class_names else ""
        display_name = _col_display_name(str(header_name))
        col_header_cells.append(f"<th{class_attr}>{_html(display_name)}</th>")

    body_rows: list[str] = []
    for row_idx, row in enumerate(rows):
        cells: list[str] = []
        for col_idx, header_name in enumerate(headers):
            cell = row[col_idx] if col_idx < len(row) else "—"

            class_names = _header_cell_classes(col_idx, str(header_name))
            data_attrs: list[str] = []
            display_value: Any = cell

            if isinstance(cell, tuple) and len(cell) == 2:
                display_value = cell[0]
                css_class = str(cell[1]).strip()
                if css_class:
                    class_names.append(css_class)
            elif str(display_value).strip() == "—":
                class_names.append("cell-na")

            meta = cell_meta.get((row_idx, col_idx)) if cell_meta else None
            if meta and meta.tooltip:
                data_attrs.append(f'data-tooltip="{_html(meta.tooltip)}"')
            if meta:
                data_attrs.append(f'data-cell-id="{_html(meta.cell_id)}"')
                if meta.clickable:
                    class_names.append("space-clickable-score")
                    data_attrs.append('data-clickable="1"')

            deduped_classes: list[str] = []
            for cls in class_names:
                if cls and cls not in deduped_classes:
                    deduped_classes.append(cls)

            class_attr = f' class="{" ".join(deduped_classes)}"' if deduped_classes else ""
            data_attr = f" {' '.join(data_attrs)}" if data_attrs else ""
            cell_html = _html(display_value)

            if meta and meta.clickable:
                inner_html = f'<button type="button" class="space-score-button">{cell_html}</button>'
            else:
                inner_html = cell_html

            cells.append(f"<td{class_attr}{data_attr}>{inner_html}</td>")

        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    rows_html = (
        "".join(body_rows)
        if body_rows
        else f'<tr><td class="cell-na" colspan="{len(headers)}">当前筛选条件下没有数据。</td></tr>'
    )

    return f"""
<div class="space-section-card">
    <div class="space-section-header">
        <h3 class="space-section-title">{_html(title)}</h3>
    </div>
    <div class="space-table-wrapper">
      <table class="bench-table">
        <thead>
          <tr class="group-row">{group_row}</tr>
          <tr class="col-row">{"".join(col_header_cells)}</tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
</div>
""".strip()
