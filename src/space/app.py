from __future__ import annotations

"""Gradio space to visualise evaluation scores."""

import csv
import html
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.eval.results.layout import ensure_results_structure
from .data import (
    ARCH_VERSIONS,
    DATA_VERSIONS,
    NUM_PARAMS,
    SPACE_SCORES_ROOT,
    ScoreEntry,
    parse_model_signature,
    latest_entries_for_model,
    list_models,
    load_scores,
    pick_latest_model,
)


AUTO_MODEL_LABEL = "每档最新（调度策略）"
AUTO_EXCLUDED_PARAMS = {"0_1B", "0_4B"}
PRIMARY_KEYS = (
    "judge_accuracy",
    "exact_accuracy",
    "accuracy",
    "prompt_accuracy",
    "instruction_accuracy",
    "pass@1",
    "pass@2",
    "pass@5",
    "pass@10",
)
DOMAIN_GROUPS = (
    {
        "key": "knowledge",
        "label": "Knowledge",
        "domains": {"mmlu系列", "multi-choice系列", "其他"},
        "title": "知识类（MMLU / Multi-choice）",
    },
    {
        "key": "math",
        "label": "Math",
        "domains": {"math reasoning系列"},
        "title": "数学推理（AIME / Math-500 等）",
    },
    {
        "key": "coding",
        "label": "Coding",
        "domains": {"coding系列"},
        "title": "代码",
    },
    {
        "key": "instruction_following",
        "label": "Instruction Following",
        "domains": {"instruction following系列"},
        "title": "指令遵循（IFEval 等）",
    },
    {
        "key": "function_call",
        "label": "Function Call",
        "domains": {"function_call系列", "function_call"},
        "title": "函数调用",
    },
)
AIME_BASES = {"aime24", "aime25"}
MATH500_BASES = {"math_500"}
IFEVAL_BASES = {"ifeval"}
CODING_FALLBACK_SAMPLE = """
**模型**：示例
**数据集**：HUMANEVAL · 样例：HumanEval/0 · 结果：passed

**Prompt**:
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
```

**Completion**:
```python
    numbers.sort()
    for i in range(len(numbers) - 1):
        if numbers[i + 1] - numbers[i] < threshold:
            return True
    return False
```
""".strip()
SUBDOMAIN_ORDER = [
    "business",
    "economics",
    "law",
    "politics",
    "history",
    "philosophy",
    "psychology",
    "sociology",
    "education",
    "language",
    "literature",
    "religion",
    "biology",
    "medicine",
    "chemistry",
    "physics",
    "math",
    "computer_science",
    "engineering",
    "security",
    "other",
]
SUBDOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "business": ("business", "management", "marketing", "finance", "accounting", "commerce", "administration"),
    "economics": ("economics", "econometrics", "macroeconomics", "microeconomics"),
    "law": ("law", "legal", "jurisprudence"),
    "politics": ("politics", "policy", "government", "marxism", "mao", "us_foreign_policy"),
    "history": ("history", "prehistory"),
    "philosophy": ("philosophy", "logic", "ethics", "moral", "world_religions", "religions"),
    "psychology": ("psychology",),
    "sociology": ("sociology", "anthropology"),
    "education": ("education", "teacher"),
    "language": ("language", "linguistics", "chinese_language", "chinese", "literature"),
    "literature": ("literature", "reading"),
    "religion": ("religion", "theology"),
    "biology": ("biology", "anatomy", "genetics", "virology", "neuroscience", "molecular", "organismal"),
    "medicine": ("medicine", "clinical", "pharmacy", "medical"),
    "chemistry": ("chemistry", "organic_chemistry"),
    "physics": ("physics", "astronomy"),
    "math": ("math", "mathematics", "algebra", "statistics", "probability", "geometry"),
    "computer_science": ("computer", "programming", "machine_learning", "operating_system", "network", "architecture"),
    "engineering": ("engineering", "electrical", "civil", "mechanical", "metrology"),
    "security": ("security", "cybersecurity", "computer_security"),
}
INSTRUCTION_DOMAIN_ORDER = [
    "change_case",
    "combination",
    "detectable_content",
    "detectable_format",
    "keywords",
    "language",
    "length_constraints",
    "punctuation",
    "startend",
    "other",
]


@dataclass(slots=True)
class SelectionState:
    entries: list[ScoreEntry]
    dropdown_value: str
    selected_label: str
    auto_selected: bool
    model_sequence: list[str]
    aggregated_models: list[dict[str, Any]] | None = None
    skipped_small_params: int = 0


def _rank_token(candidates: Sequence[str], value: str | None) -> int | None:
    if not value:
        return None
    low = value.lower()
    try:
        return candidates.index(low)
    except ValueError:
        return None


def _sort_entries(entries: Iterable[ScoreEntry]) -> list[ScoreEntry]:
    def sort_key(item: ScoreEntry) -> tuple[Any, ...]:
        arch_rank = _rank_token(ARCH_VERSIONS, item.arch_version)
        param_rank = _rank_token(NUM_PARAMS, item.num_params)
        data_rank = _rank_token(DATA_VERSIONS, item.data_version)
        return (
            arch_rank if arch_rank is not None else len(ARCH_VERSIONS),
            param_rank if param_rank is not None else len(NUM_PARAMS),
            -(data_rank if data_rank is not None else -1),
            item.domain,
            item.dataset,
            item.task or "",
            item.model,
        )

    return sorted(entries, key=sort_key)


def _snapshot_entry(entry: ScoreEntry) -> dict[str, Any]:
    arch_rank = _rank_token(ARCH_VERSIONS, entry.arch_version)
    data_rank = _rank_token(DATA_VERSIONS, entry.data_version)
    param_rank = _rank_token(NUM_PARAMS, entry.num_params)
    return {
        "model": entry.model,
        "arch": entry.arch_version,
        "data": entry.data_version,
        "params": entry.num_params,
        "arch_rank": arch_rank,
        "data_rank": data_rank,
        "param_rank": param_rank,
        "created": entry.created_at,
        "has_signature": bool(entry.arch_version and entry.num_params),
    }


def _model_snapshots(entries: Iterable[ScoreEntry]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for entry in entries:
        current = snapshots.get(entry.model)
        if current is None or entry.created_at > current["created"]:
            snapshots[entry.model] = _snapshot_entry(entry)
    return snapshots


def _select_signature_snapshots(entries: Iterable[ScoreEntry]) -> list[dict[str, Any]]:
    snapshots = _model_snapshots(entries)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    fallback: list[dict[str, Any]] = []
    for snap in snapshots.values():
        if snap["has_signature"]:
            key = (snap["arch"], snap["params"])
            grouped.setdefault(key, []).append(snap)
        else:
            fallback.append(snap)

    selected: list[dict[str, Any]] = []
    for items in grouped.values():
        best = max(
            items,
            key=lambda snap: (
                snap["data_rank"] if snap["data_rank"] is not None else -1,
                snap["created"].timestamp(),
                snap["model"],
            ),
        )
        selected.append(best)

    selected.sort(
        key=lambda snap: (
            snap["arch_rank"] if snap["arch_rank"] is not None else len(ARCH_VERSIONS),
            snap["param_rank"] if snap["param_rank"] is not None else len(NUM_PARAMS),
            -(snap["data_rank"] if snap["data_rank"] is not None else -1),
            snap["model"],
        )
    )
    fallback.sort(key=lambda snap: (-snap["created"].timestamp(), snap["model"]))
    return selected + fallback


def _latest_entries_for_signatures(entries: Iterable[ScoreEntry]) -> tuple[list[ScoreEntry], list[dict[str, Any]], list[str]]:
    snapshots = _select_signature_snapshots(entries)
    if not snapshots:
        return [], [], []
    ordered_models = []
    seen: set[str] = set()
    for snap in snapshots:
        model = snap["model"]
        if model in seen:
            continue
        seen.add(model)
        ordered_models.append(model)
    combined: list[ScoreEntry] = []
    entry_list = list(entries)
    for model in ordered_models:
        combined.extend(latest_entries_for_model(entry_list, model))
    return _sort_entries(combined), snapshots, ordered_models


def _format_param(token: str | None) -> str:
    if not token:
        return "?"
    return token.replace("_", ".")


def _dataset_base(name: str) -> str:
    for suffix in ("_test", "_eval", "_val"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _normalize_token(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() or ch == "_" else "_" for ch in text)


def _normalize_subject_label(text: str) -> str:
    return text.replace("_", " ").strip().lower()


def _series_sort_key(model: str) -> tuple[int, int, int, str]:
    sig = parse_model_signature(model)
    param_rank = sig.param_rank if sig.param_rank is not None else len(NUM_PARAMS)
    arch_rank = sig.arch_rank if sig.arch_rank is not None else len(ARCH_VERSIONS)
    data_rank = sig.data_rank if sig.data_rank is not None else len(DATA_VERSIONS)
    return (param_rank, arch_rank, data_rank, model)


def _method_tag(is_cot: bool) -> str:
    return "cot" if is_cot else "nocot"


def _model_display_name(model: str) -> str:
    sig = parse_model_signature(model)
    arch = (sig.arch or "").lower()
    data = (sig.data or "").lower()
    params = sig.params
    param_label = _format_param(params).lower() if params else ""
    if arch and data and param_label:
        return f"{arch}-{data}-{param_label}"
    parts = model.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return model


def _format_combo_label(snapshot: dict[str, Any]) -> str:
    if snapshot.get("has_signature"):
        arch = snapshot.get("arch") or "未知架构"
        params = _format_param(snapshot.get("params"))
        data = snapshot.get("data") or "?"
        return f"{arch} · {params} → {data}"
    return snapshot.get("model") or "未知模型"


def _summarise_snapshots(snapshots: Sequence[dict[str, Any]]) -> list[str]:
    combo_labels = [_format_combo_label(snap) for snap in snapshots if snap.get("has_signature")]
    extra_labels = [_format_combo_label(snap) for snap in snapshots if not snap.get("has_signature")]

    lines: list[str] = []
    if combo_labels:
        preview = " / ".join(combo_labels[:4])
        if len(combo_labels) > 4:
            preview += f" 等 {len(combo_labels)} 个组合"
        lines.append(f"- 覆盖组合：{preview}")
    if extra_labels:
        preview = " / ".join(extra_labels[:4])
        if len(extra_labels) > 4:
            preview += f" 等 {len(extra_labels)} 个模型"
        lines.append(f"- 其他未解析模型：{preview}")
    return lines


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
    # Prefer common accuracy keys in a stable order so judge results surface first.
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


def _html(text: Any) -> str:
    """Escape text for safe HTML rendering inside our custom table."""
    return html.escape(str(text), quote=True)


def _render_summary(
    *,
    all_entries: list[ScoreEntry],
    visible: list[ScoreEntry],
    selection: SelectionState,
    warnings: Iterable[str] | None = None,
) -> str:
    if not all_entries:
        ensure_results_structure()
        try:
            SPACE_SCORES_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return f"未找到任何分数文件，期待路径：`{SPACE_SCORES_ROOT}`。运行评测脚本后再刷新即可。"

    benchmark_count = len(
        {(_dataset_base(entry.dataset), _method_tag(entry.cot)) for entry in visible}
    )
    lines = [
        f"- 分数根目录：`{SPACE_SCORES_ROOT}`",
        f"- 当前策略：`{selection.selected_label}`" + ("（按排序规则自动选择）" if selection.auto_selected else ""),
        "- 领域分块：knowledge / math / coding / instruction_following / function_call",
        f"- 模型列数：{len(selection.model_sequence)}",
        f"- 基准行数：{benchmark_count}",
        f"- 可见数据集：{len(visible)} / 总分数文件：{len(all_entries)}",
        "- 排序：架构 > 参数量 > data_version（G0→…→G1d）> domain > dataset / task",
    ]
    if selection.selected_label == AUTO_MODEL_LABEL and selection.skipped_small_params:
        lines.append(f"- 已忽略 {selection.skipped_small_params} 个 0.1B / 0.4B 组合（调度策略）")
    if selection.aggregated_models:
        lines.extend(_summarise_snapshots(selection.aggregated_models))
    for warn in warnings or ():
        lines.append(f"⚠️ {warn}")
    return "\n".join(lines)


def _prepare_selection(entries: list[ScoreEntry], selection_value: str | None) -> SelectionState:
    if not entries:
        return SelectionState(
            entries=[],
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label="未检测到模型",
            auto_selected=True,
            model_sequence=[],
            aggregated_models=None,
            skipped_small_params=0,
        )

    if selection_value == AUTO_MODEL_LABEL or selection_value is None:
        combined_entries, snapshots, ordered_models = _latest_entries_for_signatures(entries)
        allowed_models: list[str] = []
        filtered_snapshots: list[dict[str, Any]] = []
        skipped_small = 0
        for snap in snapshots:
            params = snap.get("params")
            if snap.get("has_signature") and params in AUTO_EXCLUDED_PARAMS:
                skipped_small += 1
                continue
            allowed_models.append(snap["model"])
            filtered_snapshots.append(snap)
        if allowed_models:
            snapshots = filtered_snapshots
        else:
            allowed_models = ordered_models
        filtered_entries = [entry for entry in combined_entries if entry.model in allowed_models]
        return SelectionState(
            entries=_sort_entries(filtered_entries),
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label=AUTO_MODEL_LABEL,
            auto_selected=False,
            model_sequence=allowed_models,
            aggregated_models=snapshots,
            skipped_small_params=skipped_small,
        )

    models = set(list_models(entries))
    target_model = selection_value if selection_value in models else pick_latest_model(entries)
    auto_selected = target_model != selection_value
    if not target_model:
        return SelectionState(
            entries=[],
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label="未检测到模型",
            auto_selected=True,
            model_sequence=[],
            aggregated_models=None,
            skipped_small_params=0,
        )

    latest = latest_entries_for_model(entries, target_model)
    return SelectionState(
        entries=_sort_entries(latest),
        dropdown_value=target_model,
        selected_label=target_model,
        auto_selected=auto_selected,
        model_sequence=[target_model],
        aggregated_models=None,
        skipped_small_params=0,
    )


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
        # Prefer pass@1, then the lowest available k, else any numeric.
        for k in (1, 2, 4, 8, 16):
            key, value = _preferred_numeric(metrics, (f"pass@{k}",))
            if key:
                return key, value
        # If no pass@k found, fall back to highest numeric metric.
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


def _build_pivot_table(selection: SelectionState, entries: Iterable[ScoreEntry] | None = None) -> tuple[list[str], list[list[Any]]]:
    headers = ["Benchmark"] + [_model_display_name(model) for model in selection.model_sequence]
    target_entries = list(entries) if entries is not None else selection.entries
    if not target_entries:
        return headers, []

    row_meta: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], ScoreEntry] = {}

    for entry in target_entries:
        base = _dataset_base(entry.dataset)
        method = _method_tag(entry.cot)
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
        def _metric_score(item: ScoreEntry | None) -> float | None:
            if item is None:
                return None
            for val in item.metrics.values():
                if isinstance(val, (int, float)):
                    return float(val)
            return None

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
        for model in selection.model_sequence:
            entry = grouped.get((model, meta["base"], meta["method"]))
            row.append(_cell_metric_value(entry, dataset_base=meta["base"]))
        rows.append(row)
    return headers, rows


def _parse_pass_suffix(key: str) -> int | None:
    token = str(key).strip().lower()
    if not token.startswith("pass"):
        return None
    token = token[len("pass") :]
    if token.startswith("@"):
        token = token[1:]
    if token.startswith("at"):
        token = token[2:]
    try:
        return int(token)
    except ValueError:
        return None


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _build_knowledge_radar(selection: SelectionState) -> go.Figure | None:
    entries = [entry for entry in selection.entries if entry.domain in {"mmlu系列", "multi-choice系列", "其他"}]
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


def _build_domain_tables(selection: SelectionState) -> dict[str, str]:
    tables: dict[str, str] = {}
    for group in DOMAIN_GROUPS:
        filtered = [entry for entry in selection.entries if entry.domain in group["domains"]]
        headers, rows = _build_pivot_table(selection, entries=filtered)
        tables[group["key"]] = _render_pivot_html(headers, rows, title=group["title"])
    return tables





def _build_instruction_bar(selection: SelectionState) -> go.Figure | None:
    entries = [entry for entry in selection.entries if entry.domain == "instruction following系列"]
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
    entries = [entry for entry in selection.entries if entry.domain == "coding系列"]
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
    entries = [entry for entry in selection.entries if entry.domain == "coding系列"]
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


def _pivot_to_csv(headers: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def _export_selection_csv(selection: SelectionState) -> str:
    """Build a temporary CSV file for the current selection."""
    headers, rows = _build_pivot_table(selection)
    csv_text = _pivot_to_csv(headers, rows)
    temp_dir = Path(tempfile.mkdtemp(prefix="rwkv_space_"))
    path = temp_dir / "rwkv_scores.csv"
    path.write_text(csv_text, encoding="utf-8")
    return str(path)


def _render_pivot_html(headers: list[str], rows: list[list[Any]], *, title: str = "明细") -> str:
    """Render pivot table into a HTML table with predictable column widths."""
    if not headers:
        return '<div class="space-table-empty">当前筛选条件下没有数据。</div>'

    header_cells = "".join(f'<th title="{_html(title)}">{_html(title)}</th>' for title in headers)
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for idx, cell in enumerate(row):
            # First cell is model name, others are metrics
            # Add TITLE attribute for hover tooltips
            cell_html = _html(cell)
            cells.append(f'<td title="{cell_html}">{cell_html}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    rows_html = "".join(body_rows) if body_rows else '<tr><td colspan="999">当前筛选条件下没有数据。</td></tr>'

    return f"""
<div class="space-section-card">
    <div class="space-header" style="padding:0; margin-bottom:16px;">
        <h3 style="font-size:18px; color:var(--space-text-primary); margin:0;">{_html(title)}</h3>
    </div>
    <div class="space-table-wrapper">
      <table>
        <thead>
          <tr>{header_cells}</tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
</div>
""".strip()


def _compute_choices(entries: list[ScoreEntry]) -> list[str]:
    return [AUTO_MODEL_LABEL] + list_models(entries)


def _initial_payload() -> tuple[list[ScoreEntry], SelectionState, list[str]]:
    errors: list[str] = []
    entries = load_scores(errors=errors)
    selection = _prepare_selection(entries, AUTO_MODEL_LABEL)
    return entries, selection, errors


def _build_dashboard() -> gr.Blocks:
    css, style_warning = _load_css()
    entries, selection, load_errors = _initial_payload()
    model_choices = _compute_choices(entries)
    warnings = load_errors + ([style_warning] if style_warning else [])

    summary = _render_summary(
        all_entries=entries,
        visible=selection.entries,
        selection=selection,
        warnings=warnings,
    )
    domain_tables = _build_domain_tables(selection)
    aime_plot_value = _build_aime_plot(selection)
    knowledge_radar_value = _build_knowledge_radar(selection)
    instruction_bar_value = _build_instruction_bar(selection)
    coding_bar_value = _build_coding_bar(selection)
    coding_example_value = _load_coding_example(selection)
    csv_export_path = _export_selection_csv(selection)

    # Inject JS to force dark mode
    js_func = """
    () => {
        document.body.classList.add('dark');
        document.querySelector('gradio-app').style.backgroundColor = 'var(--space-bg-color)';
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Base(), js=js_func) as demo:
        with gr.Column(elem_classes="space-root"):
            
            # Header
            gr.HTML(
                """
<div class="space-header">
    <h1>RWKV Skills · Space</h1>
    <div class="subtitle">以最新分数为基准，快速浏览各评测领域。</div>
</div>
"""
            )

            # Controls
            with gr.Group(elem_classes="space-controls-card"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="模型选择",
                        info="默认项会对每个架构 + 参数量组合选取 data_version（G0→…→G1d）最新的模型；手动选择时展示单个模型的最新分数文件。",
                        choices=model_choices,
                        value=AUTO_MODEL_LABEL,
                        scale=3,
                        elem_classes="space-dropdown"
                    )
                    refresh_btn = gr.Button("刷新分数", variant="primary", scale=1)
                    download_btn = gr.DownloadButton("导出为 CSV", scale=1, value=csv_export_path)

                summary_md = gr.Markdown(summary, elem_classes="space-info-card")

            tables: dict[str, gr.HTML] = {}
            plot_aime: gr.Plot | None = None
            plot_knowledge: gr.Plot | None = None
            plot_instruction: gr.Plot | None = None
            plot_coding: gr.Plot | None = None
            coding_example: gr.Markdown | None = None

            # Main Content Tabs
            with gr.Tabs(elem_classes="tabs"):
                for group in DOMAIN_GROUPS:
                    with gr.Tab(group["label"]):
                        
                        # Use a spacer
                        gr.HTML('<div class="space-spacer"></div>')
                        
                        # Charts Section
                        if group["key"] == "knowledge":
                            with gr.Column(elem_classes="space-section-card"):
                                gr.Markdown("### Knowledge Accuracy by Subject (Horizontal Bar)", elem_classes="chart-title")
                                plot_knowledge = gr.Plot(
                                    value=knowledge_radar_value,
                                    show_label=False,
                                    elem_classes="space-chart-container",
                                )
                        
                        if group["key"] == "math":
                            with gr.Column(elem_classes="space-section-card"):
                                gr.Markdown("### AIME pass@k Curve", elem_classes="chart-title")
                                plot_aime = gr.Plot(
                                    value=aime_plot_value,
                                    show_label=False,
                                    elem_classes="space-chart-container",
                                )

                        if group["key"] == "coding":
                            with gr.Column(elem_classes="space-section-card"):
                                gr.Markdown("### Coding Benchmark (pass@1 priority)", elem_classes="chart-title")
                                plot_coding = gr.Plot(
                                    value=coding_bar_value,
                                    show_label=False,
                                    elem_classes="space-chart-container",
                                )
                            with gr.Column(elem_classes="space-section-card"):
                                coding_example = gr.Markdown(
                                    value=coding_example_value or "暂未找到 Coding 示例。",
                                    elem_classes="prose",
                                )

                        if group["key"] == "instruction_following":
                            with gr.Column(elem_classes="space-section-card"):
                                plot_instruction = gr.Plot(
                                    value=instruction_bar_value,
                                    show_label=False,
                                    elem_classes="space-chart-container",
                                )

                        # Table Section
                        # Tables already rendered with .space-section-card in _render_pivot_html wrapper if we want consistent look,
                        # BUT _render_pivot_html returns a string HTML.
                        # I updated _render_pivot_html to include the card wrapper div.
                        tables[group["key"]] = gr.HTML(
                            domain_tables.get(group["key"], _render_pivot_html([], [])),
                        )
            
            def update_dashboard(selected_model: str):
                load_errors: list[str] = []
                entries = load_scores(errors=load_errors)
                model_choices = _compute_choices(entries)
                dropdown_value = selected_model if selected_model in model_choices else AUTO_MODEL_LABEL
                
                selection_state = _prepare_selection(entries, dropdown_value)
                warnings = load_errors + ([style_warning] if style_warning else [])
                csv_path = _export_selection_csv(selection_state)
                
                summary_value = _render_summary(
                    all_entries=entries,
                    visible=selection_state.entries,
                    selection=selection_state,
                    warnings=warnings,
                )
                domain_table_values = _build_domain_tables(selection_state)
                aime_plot_fig = _build_aime_plot(selection_state)
                knowledge_radar_fig = _build_knowledge_radar(selection_state)
                instruction_bar_fig = _build_instruction_bar(selection_state)
                coding_bar_fig = _build_coding_bar(selection_state)
                coding_example_md = _load_coding_example(selection_state) or "暂未找到 Coding 示例。"

                outputs: list[Any] = [
                    gr.update(choices=model_choices, value=selection_state.dropdown_value),
                    gr.update(value=csv_path),
                    gr.update(value=summary_value),
                ]
                for group in DOMAIN_GROUPS:
                    outputs.append(gr.update(value=domain_table_values.get(group["key"])))
                    if group["key"] == "knowledge" and plot_knowledge is not None:
                        outputs.append(gr.update(value=knowledge_radar_fig))
                    if group["key"] == "math" and plot_aime is not None:
                        outputs.append(gr.update(value=aime_plot_fig))
                    if group["key"] == "coding" and plot_coding is not None:
                        outputs.append(gr.update(value=coding_bar_fig))
                        if coding_example is not None:
                            outputs.append(gr.update(value=coding_example_md))
                    if group["key"] == "instruction_following" and plot_instruction is not None:
                        outputs.append(gr.update(value=instruction_bar_fig))
                return outputs

            model_dropdown.change(
                update_dashboard,
                inputs=[model_dropdown],
                outputs=[c for c in [
                    model_dropdown,
                    download_btn,
                    summary_md,
                    tables["knowledge"],
                    plot_knowledge,
                    tables["math"],
                    plot_aime,
                    tables["coding"],
                    plot_coding,
                    coding_example,
                    tables["instruction_following"],
                    plot_instruction,
                    tables["function_call"],
                ] if c is not None],
            )
            refresh_btn.click(
                update_dashboard,
                inputs=[model_dropdown],
                outputs=[c for c in [
                    model_dropdown,
                    download_btn,
                    summary_md,
                    tables["knowledge"],
                    plot_knowledge,
                    tables["math"],
                    plot_aime,
                    tables["coding"],
                    plot_coding,
                    coding_example,
                    tables["instruction_following"],
                    plot_instruction,
                    tables["function_call"],
                ] if c is not None],
            )

            def export_csv(selected_model: str):
                entries = load_scores()
                selection_state = _prepare_selection(entries, selected_model)
                return _export_selection_csv(selection_state)

            download_btn.click(
                export_csv,
                inputs=[model_dropdown],
                outputs=download_btn,
                queue=False,
            )

    return demo


def main() -> None:
    demo = _build_dashboard()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
