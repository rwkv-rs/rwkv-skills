from __future__ import annotations

"""Gradio space to visualise evaluation scores."""

import html
import json
from pathlib import Path
from typing import Any, Iterable

import gradio as gr

from src.eval.results.layout import SCORES_ROOT, ensure_results_structure
from .data import (
    ScoreEntry,
    latest_entries_for_model,
    list_domains,
    list_models,
    load_scores,
    pick_latest_model,
)


AUTO_MODEL_LABEL = "最新 (按 data_version)"
DEFAULT_DOMAIN = "全部"
TABLE_HEADERS = [
    "Dataset",
    "Domain",
    "Model Spec",
    "Task",
    "CoT",
    "Samples",
    "Primary",
    "Metrics",
    "Created At",
    "Log",
]


def _load_css() -> tuple[str, str | None]:
    style_path = Path(__file__).parent / "styles" / "space.css"
    if not style_path.exists():
        warning = f"未找到样式文件：{style_path}"
        print(f"[space] {warning}")
        return "", warning
    try:
        return style_path.read_text(encoding="utf-8"), None
    except Exception as exc:  # noqa: BLE001
        warning = f"未加载样式：读取 {style_path.name} 失败 ({exc})"
        print(f"[space] {warning}")
        return "", warning


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
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return key, _format_metric_value(value)
    for key, value in metrics.items():
        if value is not None:
            return key, _format_metric_value(value)
    return None


def _summarise_metrics(metrics: dict[str, Any], limit: int = 4) -> str:
    pairs: list[str] = []
    for idx, (key, value) in enumerate(metrics.items()):
        if idx >= limit:
            break
        pairs.append(f"{key}: {_format_metric_value(value)}")
    if len(metrics) > limit:
        pairs.append(f"... {len(metrics) - limit} more")
    return " · ".join(pairs) if pairs else "—"


def _render_cards(entries: list[ScoreEntry]) -> str:
    if not entries:
        return '<div class="empty-state">暂未找到符合条件的分数，请先运行评测脚本。</div>'
    cards: list[str] = []
    for entry in entries:
        primary = _primary_metric(entry.metrics)
        primary_html = (
            f'<div class="primary-metric">{html.escape(primary[1])}</div>'
            f'<div class="metric-key">{html.escape(primary[0])}</div>'
            if primary
            else '<div class="metric-key">暂无指标</div>'
        )
        metric_lines = "".join(
            f'<div class="metric-line"><span>{html.escape(str(k))}</span><span>{html.escape(_format_metric_value(v))}</span></div>'
            for k, v in entry.metrics.items()
        )
        cards.append(
            f"""
<article class="score-card">
  <div class="card-head">
    <div class="dataset">{html.escape(entry.dataset)}</div>
    <div class="badge">{html.escape(entry.domain)}</div>
  </div>
  {primary_html}
  <div class="metric-list">{metric_lines}</div>
  <div class="meta">
    <span><span class="dot"></span>{'CoT' if entry.cot else 'Direct'}</span>
    <span>Task: {html.escape(entry.task or "—")}</span>
    <span>Samples: {entry.samples}</span>
    <span>Spec: {html.escape(entry.arch_version or "?")} / {html.escape(entry.data_version or "?")} / {html.escape(entry.num_params or "?")}</span>
    <span>Created: {entry.created_at.isoformat(sep=" ", timespec="seconds")}</span>
  </div>
</article>
"""
        )
    return f'<div class="score-grid">{"".join(cards)}</div>'


def _render_summary(
    *,
    all_entries: list[ScoreEntry],
    visible: list[ScoreEntry],
    model_choice: str | None,
    auto_selected: bool,
    domain_choice: str,
    warnings: Iterable[str] | None = None,
) -> str:
    if not all_entries:
        ensure_results_structure()
        return f"未找到任何分数文件，期待路径：`{SCORES_ROOT}`。运行评测脚本后再刷新即可。"

    lines = [
        f"- 分数根目录：`{SCORES_ROOT}`",
        f"- 当前模型：`{model_choice}`" + ("（按排序规则自动选择）" if auto_selected else ""),
        f"- 已选大领域：{domain_choice}",
        f"- 可见数据集：{len(visible)} / 总分数文件：{len(all_entries)}",
        "- 自动排序：仅 data_version（枚举 G0→…→G1b）> created_at；其余字段仅展示",
    ]
    for warn in warnings or ():
        lines.append(f"⚠️ {warn}")
    return "\n".join(lines)


def _filter_by_domain(entries: Iterable[ScoreEntry], domain: str) -> list[ScoreEntry]:
    if domain == DEFAULT_DOMAIN:
        return list(entries)
    return [entry for entry in entries if entry.domain == domain]


def _resolve_model(selection: str | None, entries: list[ScoreEntry]) -> tuple[str | None, bool]:
    models = set(list_models(entries))
    if not models:
        return None, False
    if selection == AUTO_MODEL_LABEL or selection is None or selection not in models:
        return pick_latest_model(entries), True
    return selection, False


def _build_table(entries: list[ScoreEntry]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for entry in entries:
        primary = _primary_metric(entry.metrics)
        rows.append(
            [
                entry.dataset,
                entry.domain,
                f"{entry.arch_version or '?'} / {entry.data_version or '?'} / {entry.num_params or '?'}",
                entry.task or "—",
                "✓" if entry.cot else "—",
                entry.samples,
                f"{primary[0]}: {primary[1]}" if primary else "—",
                _summarise_metrics(entry.metrics),
                entry.created_at.isoformat(sep=" ", timespec="seconds"),
                entry.log_path or entry.path.name,
            ]
        )
    return rows


def _compute_choices(entries: list[ScoreEntry]) -> tuple[list[str], list[str]]:
    models = [AUTO_MODEL_LABEL] + list_models(entries)
    domains = [DEFAULT_DOMAIN] + list_domains(entries)
    return models, domains


def _initial_payload() -> tuple[list[ScoreEntry], list[ScoreEntry], str, bool, str, list[str]]:
    errors: list[str] = []
    entries = load_scores(errors=errors)
    model, auto_selected = _resolve_model(AUTO_MODEL_LABEL, entries)
    visible = _filter_by_domain(latest_entries_for_model(entries, model), DEFAULT_DOMAIN)
    return entries, visible, model or "未检测到模型", auto_selected, DEFAULT_DOMAIN, errors


def _build_dashboard() -> gr.Blocks:
    css, style_warning = _load_css()
    entries, visible, model, auto_selected, domain, load_errors = _initial_payload()
    model_choices, domain_choices = _compute_choices(entries)
    warnings = load_errors + ([style_warning] if style_warning else [])

    summary = _render_summary(
        all_entries=entries,
        visible=visible,
        model_choice=model,
        auto_selected=auto_selected,
        domain_choice=domain,
        warnings=warnings,
    )
    cards_html = _render_cards(visible)
    table_rows = _build_table(visible)

    with gr.Blocks(css=css, elem_id="space-app", theme=gr.themes.Soft()) as demo:
        gr.HTML(
            """
<div class="space-card space-hero">
  <div>
    <h1>RWKV Skills · Space</h1>
    <div class="hero-subtitle">以最新分数为基准，快速浏览各评测领域。</div>
  </div>
</div>
"""
        )

        with gr.Row(elem_classes="space-card space-controls"):
            model_dropdown = gr.Dropdown(
                label="模型选择",
                info="默认按照 data_version（G0→…→G1b）> created_at 选择最新分数文件，其余字段仅展示",
                choices=model_choices,
                value=AUTO_MODEL_LABEL,
            )
            domain_dropdown = gr.Dropdown(
                label="大领域",
                choices=domain_choices,
                value=domain,
            )
            refresh_btn = gr.Button("刷新分数", variant="primary")

        summary_md = gr.Markdown(summary, elem_classes="space-card")
        cards = gr.HTML(cards_html, elem_classes="space-card")
        table = gr.Dataframe(
            headers=TABLE_HEADERS,
            value=table_rows,
            interactive=False,
            label="明细",
            elem_classes="space-card",
        )

        def update_dashboard(selected_model: str, selected_domain: str):
            load_errors: list[str] = []
            entries = load_scores(errors=load_errors)
            model_choices, domain_choices = _compute_choices(entries)
            domain_value = selected_domain if selected_domain in domain_choices else DEFAULT_DOMAIN
            dropdown_value = selected_model if selected_model in model_choices else AUTO_MODEL_LABEL

            model, auto_selected = _resolve_model(dropdown_value, entries)
            visible_entries = _filter_by_domain(latest_entries_for_model(entries, model), domain_value)
            warnings = load_errors + ([style_warning] if style_warning else [])

            summary_value = _render_summary(
                all_entries=entries,
                visible=visible_entries,
                model_choice=model or "未检测到模型",
                auto_selected=auto_selected,
                domain_choice=domain_value,
                warnings=warnings,
            )
            return (
                gr.update(choices=model_choices, value=dropdown_value),
                gr.update(choices=domain_choices, value=domain_value),
                gr.update(value=summary_value),
                gr.update(value=_render_cards(visible_entries)),
                gr.update(value=_build_table(visible_entries)),
            )

        model_dropdown.change(
            update_dashboard,
            inputs=[model_dropdown, domain_dropdown],
            outputs=[model_dropdown, domain_dropdown, summary_md, cards, table],
        )
        domain_dropdown.change(
            update_dashboard,
            inputs=[model_dropdown, domain_dropdown],
            outputs=[model_dropdown, domain_dropdown, summary_md, cards, table],
        )
        refresh_btn.click(
            update_dashboard,
            inputs=[model_dropdown, domain_dropdown],
            outputs=[model_dropdown, domain_dropdown, summary_md, cards, table],
        )

    return demo


def main() -> None:
    demo = _build_dashboard()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
