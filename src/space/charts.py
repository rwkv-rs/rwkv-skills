from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import ScoreEntry
from .constants import (
    AIME_BASES,
    CODING_FALLBACK_SAMPLE,
    INSTRUCTION_DOMAIN_ORDER,
    SelectionState,
)
from .domains import (
    is_coding_domain,
    is_instruction_following_domain,
    is_knowledge_group_domain,
)
from .metrics import (
    _dataset_base,
    _normalize_subject_label,
    _numeric_value,
    _best_numeric_metric,
    _extract_pass_curve,
    _format_param,
)
from .selection import _model_display_name, _series_sort_key


def _update_chart_layout(fig: go.Figure) -> None:
    """Apply consistent styling to all charts (No internal titles)."""
    fig.update_layout(
        title=None, # Remove internal title to prevent overlap/squeeze
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#94a3b8"),
        # Standardize margins now that title is gone
        margin=dict(t=20, l=20, r=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="#161b22",
            font_size=13,
            font_family="Inter, sans-serif",
            bordercolor="#30363d",
        ),
        # autosize=True # Removed to allow strict Python-side height control
    )


def _build_knowledge_bar(selection: SelectionState) -> go.Figure | None:
    entries = [entry for entry in selection.entries if is_knowledge_group_domain(entry.domain)]
    if not entries:
        return None

    subjects: set[str] = set()
    subject_scores: dict[str, dict[str, float]] = {}

    for entry in entries:
        details = entry.task_details or {}
        acc_map = details.get("accuracy_by_subject")
        if not isinstance(acc_map, dict):
            continue
        model_key = entry.model
        for raw_name, raw_val in acc_map.items():
            canonical = _normalize_subject_label(str(raw_name))
            score = _numeric_value(raw_val)
            if canonical and score is not None:
                subjects.add(canonical)
                subject_scores.setdefault(model_key, {})
                current = subject_scores[model_key].get(canonical)
                if current is None or score > current:
                    subject_scores[model_key][canonical] = score


    # --- VISUALIZATION CHANGE: ROTATE TO HORIZONTAL BAR CHART ---
    # Radar charts with >10 axes are unreadable. Horizontal bars are the professional standard.

    # We need to flatten the data for a bar chart
    # x: score, y: subject + model (grouped)

    # 1. Flatten data
    # We want to show comparison.
    # Option A: Grouped Bar Chart by Subject (X=Score, Y=Subject, Color=Model)

    bar_data: list[dict[str, Any]] = []

    # Collect all unique subjects and models
    for model, domain_scores in subject_scores.items():
        for subject, score in domain_scores.items():
            bar_data.append({
                "model": model,
                "subject": subject.replace("_", " ").title(), # Clean label
                "score": score
            })

    if not bar_data:
        return None

    df_bar = pd.DataFrame(bar_data)

    # Sort subjects by average score across models to make the chart readable
    subject_avg = df_bar.groupby("subject")["score"].mean().sort_values(ascending=True) # Ascending for Bottom-to-Top in Plotly
    subject_order = subject_avg.index.tolist()

    # Model order (consistent coloring)
    model_order = sorted(df_bar["model"].unique(), key=lambda m: _series_sort_key(m))

    fig = px.bar(
        df_bar,
        x="score",
        y="subject",
        color="model",
        orientation="h", # HORIZONTAL
        barmode="group",
        category_orders={"subject": subject_order, "model": model_order},
        text_auto=".1%", # Show values explicitly
    )

    # Dynamic Height Calculation
    # ~30px per subject * number of models? No, grouped bars.
    # ~40px per subject group.
    # Min height 500.
    n_subjects = len(subject_order)
    dynamic_height = max(600, n_subjects * 40 + 100)

    _update_chart_layout(fig)
    fig.update_layout(
        height=dynamic_height,
        xaxis=dict(
            title="Accuracy",
            tickformat=".0%",
            range=[0, 1.05],
            gridcolor="#30363d",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=13, family="Inter"),
            automargin=True,
        ),
        margin=dict(t=40, l=20, r=20, b=20),
        bargap=0.2,
        bargroupgap=0.1,
    )
    return fig


def _build_aime_plot(selection: SelectionState) -> go.Figure | None:
    rows: list[dict[str, Any]] = []
    series_order: dict[str, tuple[Any, ...]] = {}
    for entry in selection.entries:
        base = _dataset_base(entry.dataset).lower()
        if base not in AIME_BASES:
            continue
        curve = _extract_pass_curve(entry)
        if not curve:
            continue
        model_label = _model_display_name(entry.model)
        bench_label = base.upper()
        series = f"{bench_label} · {model_label}"
        series_order.setdefault(series, (bench_label, *_series_sort_key(entry.model)))
        for k, acc in curve.items():
            rows.append(
                {
                    "series": series,
                    "pass_k": int(k),
                    "acc": float(acc),
                }
            )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.sort_values(["series", "pass_k"], inplace=True)
    series_ordered = sorted(series_order.keys(), key=lambda key: series_order[key])

    fig = px.line(
        df,
        x="pass_k",
        y="acc",
        color="series",
        markers=True,
        category_orders={"series": series_ordered},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    _update_chart_layout(fig)
    fig.update_layout(
        height=400, # Reduced from 500 to tighten layout
        margin=dict(t=20, l=30, r=20, b=30), # Tighter margins
    )
    fig.update_yaxes(
        title_text="Accuracy",
        tickformat=".0%",
        rangemode="tozero",
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.1)"
    )
    unique_ks = sorted(df["pass_k"].unique().tolist())
    fig.update_xaxes(
        title_text="pass@k",
        tickmode="array",
        tickvals=unique_ks,
        gridcolor="rgba(255,255,255,0.05)",
        type="log" if len(unique_ks) > 4 and max(unique_ks) > 100 else "linear"
    )
    return fig


def _build_instruction_bar(selection: SelectionState) -> go.Figure | None:
    entries = [entry for entry in selection.entries if is_instruction_following_domain(entry.domain)]
    if not entries:
        return None

    rows: list[dict[str, Any]] = []
    label_to_model = {_model_display_name(e.model): e.model for e in entries}
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
            rows.append(
                {
                    "domain": domain.replace("_", " "),
                    "model": _model_display_name(entry.model),
                    "score": sum(scores) / len(scores),
                }
            )
    if not rows:
        return None

    df = pd.DataFrame(rows)
    domain_order = [d.replace("_", " ") for d in INSTRUCTION_DOMAIN_ORDER if d.replace("_", " ") in df["domain"].unique()]
    series_order = sorted(df["model"].unique(), key=lambda label: _series_sort_key(label_to_model.get(label, label)))
    df.sort_values(["domain", "model"], inplace=True)

    fig = px.bar(
        df,
        x="domain",
        y="score",
        color="model",
        barmode="group",
        category_orders={"domain": domain_order, "model": series_order},
        hover_data=None,
    )
    _update_chart_layout(fig)
    fig.update_layout(
        height=400, # Reduced
        margin=dict(t=20, l=40, r=20, b=60),
    )
    fig.update_yaxes(
        title_text="Accuracy",
        tickformat=".0%",
        rangemode="tozero",
        gridcolor="rgba(255,255,255,0.05)"
    )
    fig.update_xaxes(
        title_text=None,
        tickangle=-45
    )
    return fig


def _build_coding_bar(selection: SelectionState) -> go.Figure | None:
    entries = [entry for entry in selection.entries if is_coding_domain(entry.domain)]
    if not entries:
        return None

    rows: list[dict[str, Any]] = []
    for entry in entries:
        base = _dataset_base(entry.dataset)
        metric_key, metric_value = _best_numeric_metric(entry, dataset_base=base)
        if metric_value is None:
            continue
        rows.append(
            {
                "dataset": base.upper(),
                "model": _model_display_name(entry.model),
                "score": metric_value,
                "metric": metric_key or "score",
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.sort_values(["dataset", "model"], inplace=True)
    label_to_model = {_model_display_name(e.model): e.model for e in entries}
    series_order = sorted(df["model"].unique(), key=lambda label: _series_sort_key(label_to_model.get(label, label)))

    fig = px.bar(
        df,
        x="dataset",
        y="score",
        color="model",
        barmode="group",
        hover_data=["metric"],
        text_auto=".1%",
        category_orders={"model": series_order},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    _update_chart_layout(fig)
    fig.update_layout(height=400) # Reduced
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(
        title_text="Accuracy",
        tickformat=".0%",
        rangemode="tozero",
        gridcolor="rgba(255,255,255,0.05)"
    )
    fig.update_xaxes(title_text=None)
    return fig


def _truncate(text: str, limit: int = 900) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _load_coding_example(selection: SelectionState) -> str | None:
    entries = [entry for entry in selection.entries if is_coding_domain(entry.domain)]
    if not entries:
        return CODING_FALLBACK_SAMPLE

    # Pick the smallest parameter model first to keep prompts concise.
    entries = sorted(entries, key=lambda e: _series_sort_key(e.model))
    for entry in entries:
        path = Path(entry.log_path) if entry.log_path else None
        if not path or not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                line = fh.readline()
        except OSError:
            continue
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue

        prompt = payload.get("prompt1") or payload.get("prompt_raw") or payload.get("prompt") or ""
        completion = payload.get("completion") or payload.get("output1") or payload.get("response") or ""
        result = payload.get("result") or payload.get("passed")
        task_id = payload.get("task_id") or payload.get("entry_point") or ""

        parts = [
            f"**模型**：{_model_display_name(entry.model)}",
            f"**数据集**：{_dataset_base(entry.dataset).upper()}  ·  **样例**：{task_id or 'N/A'}  ·  **结果**：{result}",
            "**Prompt**:",
            f"```python\n{_truncate(str(prompt).strip())}\n```",
            "**Completion**:",
            f"```python\n{_truncate(str(completion).strip())}\n```",
        ]
        return "\n\n".join(parts)

    return CODING_FALLBACK_SAMPLE
