from __future__ import annotations

"""Gradio space to visualise evaluation scores."""

import csv
from datetime import datetime
import hashlib
import html
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.db.eval_db_service import EvalDbService
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from .data import (
    ARCH_VERSIONS,
    DATA_VERSIONS,
    NUM_PARAMS,
    ScoreEntry,
    parse_model_signature,
    latest_entries_for_model,
    list_models,
    load_scores,
    pick_latest_model,
)
from .vocab import token_id_to_display


AUTO_MODEL_LABEL = "每档最新（调度策略）"
TABLE_VIEW_LABELS: dict[str, str] = {
    "benchmark_detail_latest": "明细（最新）",
    "field_avg_latest": "领域均分（最新）",
    "benchmark_detail_delta": "明细（上一代 vs 最新）",
    "field_avg_delta": "领域均分（上一代 vs 最新）",
}
DEFAULT_TABLE_VIEW = "benchmark_detail_latest"


def _normalize_table_view(raw_value: Any) -> str:
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if value in TABLE_VIEW_LABELS:
            return value
        for key, label in TABLE_VIEW_LABELS.items():
            if value == label:
                return key
    return DEFAULT_TABLE_VIEW


EVAL_PAGE_SIZE = 15
EVAL_PRELOAD_PAGES = 2
EVAL_PRELOAD_ROWS = EVAL_PAGE_SIZE * EVAL_PRELOAD_PAGES
EVAL_FETCH_ROWS = EVAL_PAGE_SIZE
EVAL_OVERSCAN_ROWS = 1
EVAL_CONTEXT_PREVIEW_LIMIT = 20


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


@dataclass(slots=True, frozen=True)
class ParamLineage:
    param: str
    latest_model: str
    latest_label: str
    prev_model: str | None
    prev_label: str


@dataclass(slots=True, frozen=True)
class DetailPoint:
    score: float
    entry: ScoreEntry


@dataclass(slots=True, frozen=True)
class TableCellMeta:
    cell_id: str
    task_id: int | None
    benchmark_name: str
    eval_method: str
    k_metric: str
    column_label: str
    model: str | None
    tooltip: str | None
    clickable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "task_id": self.task_id,
            "benchmark_name": self.benchmark_name,
            "eval_method": self.eval_method,
            "k_metric": self.k_metric,
            "column_label": self.column_label,
            "model": self.model,
            "tooltip": self.tooltip,
            "clickable": self.clickable,
        }


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


def _model_data_param_label(model: str, *, include_params: bool = True) -> str:
    sig = parse_model_signature(model)
    data = (sig.data or "").lower()
    params = _format_param(sig.params).lower() if sig.params else ""
    if include_params:
        if data and params:
            return f"{data}-{params}"
        if params:
            return params
        if data:
            return data
    else:
        if data:
            return data
        if params:
            return params
    return _model_display_name(model)


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


def _load_vendor_css() -> tuple[str, str | None]:
    """Load vendor CSS for context modal rendering (markdown, code, math).

    Note: Gradio content lives inside a Shadow DOM. CSS must be injected via the
    Blocks `css=` parameter to apply inside that tree.
    """
    vendor_dir = Path(__file__).parent / "assets" / "vendor"
    if not vendor_dir.exists():
        warning = f"未找到前端依赖目录：{vendor_dir}"
        print(f"[space] {warning}")
        return "", warning

    # KaTeX CSS references fonts/KaTeX_*.woff2. We ship the .woff2 files and rewrite URLs
    # to Gradio's /file= endpoint.
    fonts_dir = vendor_dir / "fonts"
    fonts_url_prefix = f"/file={fonts_dir.as_posix()}/"

    def read_text(name: str) -> str:
        return (vendor_dir / name).read_text(encoding="utf-8")

    try:
        katex_css = read_text("katex.min.css")
        # Drop .woff/.ttf fallbacks to avoid extra 404 requests (we only vendor .woff2).
        katex_css = re.sub(
            r',url\(fonts/[^)]+?\.woff\) format\("woff"\),url\(fonts/[^)]+?\.ttf\) format\("truetype"\)',
            "",
            katex_css,
        )
        katex_css = katex_css.replace("url(fonts/", f"url({fonts_url_prefix}")
        hljs_css = read_text("github-dark.min.css")

        warning = None
        if not fonts_dir.exists():
            warning = f"未找到 KaTeX 字体目录：{fonts_dir}（数学公式可能无法正常显示）"
            print(f"[space] {warning}")

        css = "\n".join(
            [
                "/* Vendor: highlight.js theme */",
                hljs_css,
                "/* Vendor: KaTeX */",
                katex_css,
            ]
        )
        return css, warning
    except Exception as exc:  # noqa: BLE001
        warning = f"未加载前端渲染依赖：{exc}"
        print(f"[space] {warning}")
        return "", warning


def _load_vendor_head() -> tuple[str, str | None]:
    """Load vendor JS for context modal rendering (markdown, code, math)."""
    vendor_dir = Path(__file__).parent / "assets" / "vendor"
    if not vendor_dir.exists():
        warning = f"未找到前端依赖目录：{vendor_dir}"
        print(f"[space] {warning}")
        return "", warning

    def read_js(name: str) -> str:
        return (vendor_dir / name).read_text(encoding="utf-8")

    try:
        head_parts = [
            f"<script id=\"space-vendor-katex-js\">{read_js('katex.min.js')}</script>",
            f"<script id=\"space-vendor-katex-auto-render\">{read_js('auto-render.min.js')}</script>",
            f"<script id=\"space-vendor-markdown-it\">{read_js('markdown-it.min.js')}</script>",
            f"<script id=\"space-vendor-highlight-js\">{read_js('highlight.min.js')}</script>",
        ]
        return "\n".join(head_parts), None
    except Exception as exc:  # noqa: BLE001
        warning = f"未加载前端渲染依赖：{exc}"
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
    view_mode: str = DEFAULT_TABLE_VIEW,
    warnings: Iterable[str] | None = None,
) -> str:
    db_host = os.environ.get("PG_HOST", DEFAULT_DB_CONFIG.host)
    db_port = os.environ.get("PG_PORT", str(DEFAULT_DB_CONFIG.port))
    db_name = os.environ.get("PG_DBNAME", DEFAULT_DB_CONFIG.dbname)

    if not all_entries:
        return (
            "数据库暂无可展示分数。"
            f"（连接目标：`{db_host}:{db_port}/{db_name}`，仅统计非 param-search 结果）"
        )

    benchmark_count = len(
        {(_dataset_base(entry.dataset), _method_tag(entry.cot)) for entry in visible}
    )
    lines = [
        f"- 数据源：PostgreSQL (`{db_host}:{db_port}/{db_name}`)",
        "- 数据范围：仅正式评测（已过滤 param-search）",
        f"- 当前策略：`{selection.selected_label}`" + ("（按排序规则自动选择）" if selection.auto_selected else ""),
        f"- 表格视图：{TABLE_VIEW_LABELS.get(view_mode, TABLE_VIEW_LABELS[DEFAULT_TABLE_VIEW])}",
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
        allowed_models = ordered_models
        filtered_entries = [entry for entry in combined_entries if entry.model in allowed_models]
        return SelectionState(
            entries=_sort_entries(filtered_entries),
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label=AUTO_MODEL_LABEL,
            auto_selected=False,
            model_sequence=allowed_models,
            aggregated_models=snapshots,
            skipped_small_params=0,
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


def _cell_numeric_value(entry: ScoreEntry | None, *, dataset_base: str) -> float | None:
    if entry is None:
        return None

    base = dataset_base.lower()
    metrics = entry.metrics

    def _numeric_specific(key: str) -> float | None:
        return _numeric_value(metrics.get(key))

    if base.startswith("aime"):
        candidates: list[float] = []
        for key in ("avg@16", "pass@8"):
            value = _numeric_specific(key)
            if value is not None:
                candidates.append(value)
        if candidates:
            return max(candidates)

    if base in MATH500_BASES:
        value = _numeric_specific("avg@4")
        if value is not None:
            return value

    if base.startswith("ifeval"):
        value = _numeric_specific("avg@4")
        if value is not None:
            return value

    _, value = _primary_numeric_metric(metrics)
    return value


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
    headers = ["Benchmark"] + [_model_data_param_label(model, include_params=True) for model in selection.model_sequence]
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


def _score_to_percent(value: float | None) -> float | None:
    if value is None:
        return None
    if -1.0 <= value <= 1.0:
        return value * 100.0
    return value


def _max_percent(values: Iterable[float | None]) -> float | None:
    best: float | None = None
    for value in values:
        percent = _score_to_percent(value)
        if percent is None:
            continue
        if best is None or percent > best:
            best = percent
    return best


def _format_score_1dp(value: float | None) -> str:
    normalized = _score_to_percent(value)
    if normalized is None:
        return "—"
    return f"{normalized:.1f}"


def _format_delta_1dp(latest: float | None, previous: float | None) -> str:
    latest_n = _score_to_percent(latest)
    prev_n = _score_to_percent(previous)
    if latest_n is None or prev_n is None:
        return "—"
    delta = latest_n - prev_n
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"



def _format_delta_value(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}"


def _score_cell_style(value: float | None, *, mode: str) -> str | None:
    if value is None:
        return None

    if mode == "score":
        percent = _score_to_percent(value)
        if percent is None:
            return None
        delta = percent - 50.0
        if abs(delta) < 1e-6:
            return None
        intensity = min(abs(delta) / 50.0, 1.0)
    else:
        delta_raw = float(value)
        if abs(delta_raw) < 1e-6:
            return None
        # Clamp delta intensity to [-10, 10] for consistent colouring.
        delta = max(min(delta_raw, 10.0), -10.0)
        intensity = min(abs(delta) / 10.0, 1.0)

    base_alpha = 0.06
    alpha = base_alpha + 0.24 * intensity
    if delta > 0:
        color = (34, 197, 94)
    else:
        color = (239, 68, 68)
    return f"background-color: rgba({color[0]}, {color[1]}, {color[2]}, {alpha:.2f});"



def _score_cell_style_norm(value: float | None, min_v: float | None, max_v: float | None) -> str | None:
    if value is None or min_v is None or max_v is None:
        return None
    if max_v <= min_v:
        return None
    percent = _score_to_percent(value)
    if percent is None:
        return None
    mid = (min_v + max_v) / 2.0
    delta = percent - mid
    if abs(delta) < 1e-6:
        return None
    intensity = min(abs(delta) / (max_v - min_v), 1.0)

    base_alpha = 0.06
    alpha = base_alpha + 0.24 * intensity
    if delta > 0:
        color = (34, 197, 94)
    else:
        color = (239, 68, 68)
    return f"background-color: rgba({color[0]}, {color[1]}, {color[2]}, {alpha:.2f});"

def _styled_score_cell(value: float | None) -> tuple[str, str | None]:
    return _format_score_1dp(value), _score_cell_style(value, mode="score")



def _styled_score_cell_norm(value: float | None, min_v: float | None, max_v: float | None) -> tuple[str, str | None]:
    return _format_score_1dp(value), _score_cell_style_norm(value, min_v, max_v)

def _styled_delta_cell(delta_value: float | None) -> tuple[str, str | None]:
    return _format_delta_value(delta_value), _score_cell_style(delta_value, mode="delta")



def _parse_display_number(value: Any) -> float | None:
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text == "—":
        return None
    text = text.replace("%", "").replace(",", "")
    if text.startswith("+"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return None
def _benchmark_name(entry: ScoreEntry) -> str:
    return f"{_dataset_base(entry.dataset)}_{_method_tag(entry.cot)}"


def _is_multi_choice_entry(entry: ScoreEntry) -> bool:
    task = (entry.task or "").lower()
    if "multi" in task and "choice" in task:
        return True
    job_hint = entry.domain in {"mmlu系列", "multi-choice系列"}
    return job_hint and _numeric_value(entry.metrics.get("accuracy")) is not None


def _parse_k_metric(key: str) -> tuple[str, int] | None:
    token = str(key).strip().lower()
    if token.startswith("pass@"):
        try:
            return "pass", int(token.split("@", 1)[1])
        except ValueError:
            return None
    if token.startswith("avg@"):
        try:
            return "avg", int(token.split("@", 1)[1])
        except ValueError:
            return None
    return None


def _preferred_k_metric(metrics: dict[str, Any]) -> str:
    avg_candidates: list[tuple[int, str]] = []
    pass_candidates: list[tuple[int, str]] = []
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


def _score_for_eval_method(entry: ScoreEntry, method: str, k_metric: str) -> float | None:
    metrics = entry.metrics
    if method == "logits":
        acc = _numeric_value(metrics.get("accuracy"))
        if acc is not None:
            return acc
        _, fallback = _best_numeric_metric(entry, dataset_base=_dataset_base(entry.dataset))
        return fallback
    if method == "llm_judge":
        return _numeric_value(metrics.get("judge_accuracy"))

    # exact_match
    for key in ("exact_accuracy", "instruction_accuracy", "prompt_accuracy"):
        number = _numeric_value(metrics.get(key))
        if number is not None:
            return number
    if not _is_multi_choice_entry(entry):
        acc = _numeric_value(metrics.get("accuracy"))
        if acc is not None:
            return acc
    k_value = _numeric_value(metrics.get(k_metric))
    if k_value is not None:
        return k_value
    _, fallback = _best_numeric_metric(entry, dataset_base=_dataset_base(entry.dataset))
    return fallback


def _detail_rows_for_entry(entry: ScoreEntry) -> list[tuple[str, str, str, float]]:
    benchmark = _benchmark_name(entry)
    k_metric = _preferred_k_metric(entry.metrics)
    methods: list[str] = []
    if _is_multi_choice_entry(entry):
        methods.append("logits")
    else:
        if _numeric_value(entry.metrics.get("judge_accuracy")) is not None:
            methods.append("llm_judge")
        methods.append("exact_match")

    rows: list[tuple[str, str, str, float]] = []
    for method in methods:
        score = _score_for_eval_method(entry, method, k_metric)
        if score is None:
            continue
        rows.append((benchmark, method, k_metric, score))
    return rows


def _field_primary_score(entry: ScoreEntry) -> float | None:
    if _is_multi_choice_entry(entry):
        return _score_for_eval_method(entry, "logits", _preferred_k_metric(entry.metrics))
    if _numeric_value(entry.metrics.get("judge_accuracy")) is not None:
        return _score_for_eval_method(entry, "llm_judge", _preferred_k_metric(entry.metrics))
    return _score_for_eval_method(entry, "exact_match", _preferred_k_metric(entry.metrics))


def _detail_sort_key(row_key: tuple[str, str, str]) -> tuple[Any, ...]:
    benchmark, method, k_metric = row_key
    method_rank = {"llm_judge": 0, "exact_match": 1, "logits": 2}.get(method, 9)
    parsed = _parse_k_metric(k_metric)
    if parsed is None:
        k_rank = (9, 0)
    else:
        kind, k = parsed
        kind_rank = 0 if kind == "avg" else 1
        k_rank = (kind_rank, k)
    return benchmark, method_rank, k_rank, k_metric


def _field_average_score(entries: Iterable[ScoreEntry]) -> float | None:
    values: list[float] = []
    for entry in entries:
        score = _field_primary_score(entry)
        if score is not None:
            values.append(score)
    if not values:
        return None
    return sum(values) / len(values)


def _build_model_entries_cache(entries: list[ScoreEntry]) -> dict[str, list[ScoreEntry]]:
    cache: dict[str, list[ScoreEntry]] = {}
    for model in list_models(entries):
        cache[model] = latest_entries_for_model(entries, model)
    return cache


def _resolve_param_lineages(entries: list[ScoreEntry], selection: SelectionState) -> list[ParamLineage]:
    snapshots = [snap for snap in _model_snapshots(entries).values() if snap.get("has_signature")]
    if not snapshots:
        return []

    by_arch_param: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for snap in snapshots:
        arch = snap.get("arch")
        param = snap.get("params")
        if not arch or not param:
            continue
        by_arch_param.setdefault((arch, param), []).append(snap)

    for items in by_arch_param.values():
        items.sort(
            key=lambda snap: (
                snap.get("data_rank") if snap.get("data_rank") is not None else -1,
                snap["created"].timestamp(),
                snap.get("model", ""),
            )
        )

    lineages: list[ParamLineage] = []

    if selection.dropdown_value != AUTO_MODEL_LABEL:
        target_model = selection.dropdown_value
        sig = parse_model_signature(target_model)
        if sig.arch and sig.params:
            items = by_arch_param.get((sig.arch, sig.params), [])
            prev_model: str | None = None
            for idx, snap in enumerate(items):
                if snap.get("model") != target_model:
                    continue
                if idx > 0:
                    prev_model = str(items[idx - 1]["model"])
                break
            lineages.append(
                ParamLineage(
                    param=sig.params,
                    latest_model=target_model,
                    latest_label=_model_display_name(target_model),
                    prev_model=prev_model,
                    prev_label=_model_display_name(prev_model) if prev_model else "—",
                )
            )
        else:
            lineages.append(
                ParamLineage(
                    param="custom",
                    latest_model=target_model,
                    latest_label=_model_display_name(target_model),
                    prev_model=None,
                    prev_label="—",
                )
            )
        return lineages

    by_param: dict[str, list[dict[str, Any]]] = {}
    for (_, param), items in by_arch_param.items():
        if not items:
            continue
        by_param.setdefault(param, []).append(items[-1])

    for param, candidates in by_param.items():
        latest = max(
            candidates,
            key=lambda snap: (
                snap.get("data_rank") if snap.get("data_rank") is not None else -1,
                snap["created"].timestamp(),
                snap.get("arch_rank") if snap.get("arch_rank") is not None else -1,
                snap.get("model", ""),
            ),
        )
        latest_model = str(latest["model"])
        arch = str(latest["arch"])
        chain = by_arch_param.get((arch, param), [])
        prev_model: str | None = None
        for idx, snap in enumerate(chain):
            if snap.get("model") != latest_model:
                continue
            if idx > 0:
                prev_model = str(chain[idx - 1]["model"])
            break
        lineages.append(
            ParamLineage(
                param=param,
                latest_model=latest_model,
                latest_label=_model_display_name(latest_model),
                prev_model=prev_model,
                prev_label=_model_display_name(prev_model) if prev_model else "—",
            )
        )

    lineages.sort(
        key=lambda item: (
            _rank_token(NUM_PARAMS, item.param) if _rank_token(NUM_PARAMS, item.param) is not None else len(NUM_PARAMS),
            item.param,
        )
    )
    return lineages


def _entries_for_model_in_domains(
    model_cache: dict[str, list[ScoreEntry]],
    model: str | None,
    domains: set[str],
) -> list[ScoreEntry]:
    if not model:
        return []
    return [entry for entry in model_cache.get(model, []) if entry.domain in domains]


def _build_detail_point_map(entries: Iterable[ScoreEntry]) -> dict[tuple[str, str, str], DetailPoint]:
    mapped: dict[tuple[str, str, str], DetailPoint] = {}
    for entry in entries:
        for benchmark, method, k_metric, score in _detail_rows_for_entry(entry):
            key = (benchmark, method, k_metric)
            previous = mapped.get(key)
            if previous is None or score > previous.score or (
                score == previous.score and entry.created_at > previous.entry.created_at
            ):
                mapped[key] = DetailPoint(score=score, entry=entry)
    return mapped


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
    if dataset.startswith("mmlu") or entry.domain in {"mmlu系列", "multi-choice系列"}:
        tooltip = _mmlu_tooltip(entry)
        if tooltip:
            return tooltip
    if dataset.startswith("ifeval") or entry.domain == "instruction following系列":
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
        row.append(_format_score_1dp(score))
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
        if latest_score is not None and prev_score is not None:
            delta_value = _score_to_percent(latest_score) - _score_to_percent(prev_score)
        row.extend(
            [
                _format_score_1dp(prev_score),
                _format_score_1dp(latest_score),
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
    headers = ["benchmark_name", "eval_method", "k_metric"] + [
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
        row = [row_key[0], row_key[1], row_key[2]]
        points = row_values[row_key]
        max_score = _max_percent(points.get(item.param).score if points.get(item.param) else None for item in lineages)
        if max_score is None or max_score < 10.0:
            continue
        for col_offset, item in enumerate(lineages, start=3):
            point = points.get(item.param)
            row.append(_format_score_1dp(point.score if point else None))
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
    headers = ["benchmark_name", "eval_method", "k_metric"]
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
        row = [row_key[0], row_key[1], row_key[2]]
        delta_values: list[float] = []
        row_cell_meta: dict[int, TableCellMeta] = {}
        for item in lineages:
            latest_point = latest_by_param.get(item.param, {}).get(row_key)
            prev_point = prev_by_param.get(item.param, {}).get(row_key)
            latest_score = latest_point.score if latest_point else None
            prev_score = prev_point.score if prev_point else None
            delta_value = None
            if latest_score is not None and prev_score is not None:
                delta_value = _score_to_percent(latest_score) - _score_to_percent(prev_score)
                delta_values.append(delta_value)
            prev_col_idx = len(row)
            latest_col_idx = prev_col_idx + 1
            row.extend(
                [
                    _format_score_1dp(prev_score),
                    _format_score_1dp(latest_score),
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
            delta_indices = [
                i for i, name in enumerate(headers)
                if str(name).strip().endswith("delta")
            ]
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


def _render_pivot_html(
    headers: list[str],
    rows: list[list[Any]],
    *,
    title: str = "明细",
    cell_meta: dict[tuple[int, int], TableCellMeta] | None = None,
) -> str:
    """Render pivot table into a HTML table with predictable column widths."""
    if not headers:
        return '<div class="space-table-empty">当前筛选条件下没有数据。</div>'

    header_cells = "".join(f"<th>{_html(title)}</th>" for title in headers)
    body_rows: list[str] = []
    for row_idx, row in enumerate(rows):
        cells: list[str] = []
        for col_idx, cell in enumerate(row):
            style_attr = ""
            class_names: list[str] = []
            data_attrs: list[str] = []
            display_value = cell
            if isinstance(cell, tuple) and len(cell) == 2:
                display_value, style = cell
                if style:
                    style_attr = f' style="{_html(style)}"'

            numeric_value = _parse_display_number(display_value)
            if numeric_value is not None:
                class_names.append("score-cell")

            meta = cell_meta.get((row_idx, col_idx)) if cell_meta else None
            if meta and meta.tooltip:
                data_attrs.append(f'data-tooltip="{_html(meta.tooltip)}"')
            if meta:
                data_attrs.append(f'data-cell-id="{_html(meta.cell_id)}"')
                if meta.clickable:
                    class_names.append("space-clickable-score")
                    data_attrs.append('data-clickable="1"')

            class_attr = f' class="{" ".join(class_names)}"' if class_names else ""
            data_attr = f" {' '.join(data_attrs)}" if data_attrs else ""
            cell_html = _html(display_value)

            if meta and meta.clickable:
                inner_html = f'<button type="button" class="space-score-button">{cell_html}</button>'
            else:
                inner_html = cell_html
            cells.append(f"<td{class_attr}{style_attr}{data_attr}>{inner_html}</td>")
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
  <div class="space-header" style="padding:0; margin-bottom:12px;">
    <h3 style="font-size:18px; color:var(--space-text-primary); margin:0;">Eval 记录</h3>
    <div class="subtitle">点击上方分数后，这里会展示对应样本记录。</div>
  </div>
</div>
""".strip()

    benchmark = _html(meta.get("benchmark_name") or "N/A")
    method = _html(meta.get("eval_method") or "N/A")
    model = _html(meta.get("model") or "N/A")
    filter_tag = "仅错题" if only_wrong else "全部样本"

    if not records:
        loading_text = " · 还有更多，请点击“下一页”" if is_loading else ""
        return f"""
<div class="space-section-card space-eval-panel">
  <div class="space-header" style="padding:0; margin-bottom:12px; text-align:left;">
    <h3 style="font-size:18px; color:var(--space-text-primary); margin:0;">Eval 记录</h3>
    <div class="subtitle">{benchmark} · {method} · {model} · {filter_tag}{loading_text}</div>
  </div>
  <div class="space-table-empty">当前筛选下没有样本记录。</div>
</div>
""".strip()

    header_cells = "".join(
        f"<th>{name}</th>"
        for name in ("sample_index", "repeat_index", "is_passed", "answer", "ref_answer", "fail_reason", "context")
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

        values = [
            _html(sample_index),
            _html(repeat_index),
            "✓" if bool(row.get("is_passed")) else "✕",
            _html(row.get("answer") or ""),
            _html(row.get("ref_answer") or ""),
            _html(row.get("fail_reason") or ""),
            context_button,
        ]
        cells = "".join(f"<td>{value}</td>" for value in values)
        body_rows.append(f"<tr>{cells}</tr>")

    rows_html = "".join(body_rows)
    load_status = (
        f"已加载 {len(sorted_rows)} 条（每页 {EVAL_PAGE_SIZE} 条，点击“下一页”继续加载）"
        if is_loading
        else f"已加载 {len(sorted_rows)} 条（每页 {EVAL_PAGE_SIZE} 条，已全部加载）"
    )
    return f"""
<div class="space-section-card space-eval-panel">
  <div class="space-header" style="padding:0; margin-bottom:12px; text-align:left;">
    <h3 style="font-size:18px; color:var(--space-text-primary); margin:0;">Eval 记录</h3>
    <div class="subtitle">{benchmark} · {method} · {model} · {filter_tag}</div>
    <div class="subtitle">{load_status}</div>
  </div>
  <div class="space-table-wrapper">
    <table>
      <thead><tr>{header_cells}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
""".strip()


def _resolve_eval_records(
    *,
    cell_id: str | None,
    only_wrong: bool,
    interaction_meta: dict[str, dict[str, Any]],
) -> tuple[str, str | None]:
    if not cell_id:
        return _render_eval_records_html(meta=None, records=[], only_wrong=only_wrong), None

    meta = interaction_meta.get(cell_id)
    if not meta:
        return _render_eval_records_html(meta=None, records=[], only_wrong=only_wrong), None

    task_id = meta.get("task_id")
    if not isinstance(task_id, int):
        return _render_eval_records_html(meta=meta, records=[], only_wrong=only_wrong), cell_id

    try:
        records = EvalDbService().list_eval_records_for_space(task_id=str(task_id), only_wrong=only_wrong)
    except Exception as exc:  # noqa: BLE001
        error_html = f'<div class="space-table-empty">读取 Eval 记录失败：{_html(exc)}</div>'
        return error_html, cell_id

    return _render_eval_records_html(meta=meta, records=records, only_wrong=only_wrong), cell_id


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
    sampling_config = context_obj.get("sampling_config")
    if isinstance(stages, list) or isinstance(sampling_config, dict):
        base_event["view"] = "structured"
        base_event["context"] = context_obj
        base_event["stop_tokens"] = _build_stop_tokens_mapping(sampling_config)

    return base_event


def _compute_choices(entries: list[ScoreEntry]) -> list[str]:
    return [AUTO_MODEL_LABEL] + list_models(entries)


def _initial_payload() -> tuple[list[ScoreEntry], SelectionState, list[str]]:
    errors: list[str] = []
    entries = load_scores(errors=errors)
    selection = _prepare_selection(entries, AUTO_MODEL_LABEL)
    return entries, selection, errors


def _build_dashboard() -> gr.Blocks:
    base_css, style_warning = _load_css()
    vendor_css, vendor_css_warning = _load_vendor_css()
    css = base_css + ("\n\n" + vendor_css if vendor_css else "")
    head, head_warning = _load_vendor_head()
    entries, selection, load_errors = _initial_payload()
    model_choices = _compute_choices(entries)
    warnings = (
        load_errors
        + ([style_warning] if style_warning else [])
        + ([vendor_css_warning] if vendor_css_warning else [])
        + ([head_warning] if head_warning else [])
    )
    initial_view_mode = DEFAULT_TABLE_VIEW

    summary = _render_summary(
        all_entries=entries,
        visible=selection.entries,
        selection=selection,
        view_mode=initial_view_mode,
        warnings=warnings,
    )
    domain_tables, interaction_meta = _build_domain_tables(
        selection,
        all_entries=entries,
        view_mode=initial_view_mode,
    )
    initial_eval_html = _render_eval_records_html(meta=None, records=[], only_wrong=False)
    initial_eval_rows: list[dict[str, Any]] = []
    initial_eval_loader = _empty_eval_loader_state()
    csv_export_path = _export_selection_csv(selection)

    modal_html = """
<div id="space-context-modal" class="space-modal" aria-hidden="true">
  <div class="space-modal-content">
    <div class="space-modal-topbar">
      <div class="space-modal-title">Context 详情</div>
      <button type="button" class="space-modal-close" data-close-modal="1">关闭</button>
    </div>
    <div class="space-context-layout">
      <div id="space-context-left" class="space-context-left"></div>
      <div id="space-context-right" class="space-context-right"></div>
    </div>
  </div>
</div>
"""

    js_func = """
    () => {
        const app = document.querySelector('gradio-app');
        const root = (app && app.shadowRoot) ? app.shadowRoot : document;
        if (app) {
            app.style.backgroundColor = 'var(--space-bg-color)';
        }
        document.body.classList.add('dark');

        if (window.__rwkvSpaceHandlersBound) {
            return;
        }
        window.__rwkvSpaceHandlersBound = true;

        const escapeId = (id) => {
            if (window.CSS && typeof CSS.escape === 'function') {
                return CSS.escape(id);
            }
            return id.replace(/([ #;?%&,.+*~':"!^$\\[\\]()=>|/@])/g, '\\$1');
        };

        const queryById = (id) => root.querySelector(`#${escapeId(id)}`);

        const clearNode = (node) => {
            if (!node) {
                return;
            }
            while (node.firstChild) {
                node.removeChild(node.firstChild);
            }
        };

        const toDisplayString = (value) => {
            if (value === null || value === undefined) {
                return '';
            }
            if (typeof value === 'string') {
                return value;
            }
            if (typeof value === 'number' || typeof value === 'boolean') {
                return String(value);
            }
            try {
                return JSON.stringify(value);
            } catch (error) {
                return String(value);
            }
        };

        const ensureMarkdown = () => {
            if (window.__rwkvSpaceMarkdown) {
                return window.__rwkvSpaceMarkdown;
            }
            if (typeof window.markdownit !== 'function') {
                window.__rwkvSpaceMarkdown = null;
                return null;
            }
            const md = window.markdownit({
                html: false,
                linkify: true,
                breaks: true,
                typographer: true,
            });
            window.__rwkvSpaceMarkdown = md;
            return md;
        };

        const applyHighlight = (container) => {
            if (!container || !window.hljs) {
                return;
            }
            container.querySelectorAll('pre code').forEach((block) => {
                try {
                    window.hljs.highlightElement(block);
                } catch (error) {
                    // ignore highlight failures
                }
            });
        };

        const applyMath = (container) => {
            if (!container || typeof window.renderMathInElement !== 'function') {
                return;
            }
            try {
                window.renderMathInElement(container, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                        { left: '\\\\(', right: '\\\\)', display: false },
                        { left: '\\\\[', right: '\\\\]', display: true },
                    ],
                    ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                });
            } catch (error) {
                // ignore math failures
            }
        };

        const renderMarkdown = (container, text) => {
            if (!container) {
                return;
            }
            const md = ensureMarkdown();
            if (!md) {
                container.textContent = text || '';
                return;
            }
            container.innerHTML = md.render(text || '');
            applyHighlight(container);
            applyMath(container);
        };

        const closeModal = () => {
            const modal = queryById('space-context-modal');
            if (!modal) {
                return;
            }
            modal.classList.remove('open');
            modal.setAttribute('aria-hidden', 'true');
        };

        const openModalEvent = (evt) => {
            const modal = queryById('space-context-modal');
            const left = queryById('space-context-left');
            const right = queryById('space-context-right');
            if (!modal || !left || !right) {
                return;
            }
            clearNode(left);
            clearNode(right);

            const eventObj = evt && typeof evt === 'object' ? evt : null;
            const view = eventObj && typeof eventObj.view === 'string' ? eventObj.view : 'text';
            const rawText = eventObj && typeof eventObj.raw_text === 'string' ? eventObj.raw_text : '';

            const addCard = (parent, title, bodyBuilder) => {
                const card = document.createElement('div');
                card.className = 'space-context-card';

                const header = document.createElement('div');
                header.className = 'space-context-card-header';

                const h = document.createElement('div');
                h.className = 'space-context-card-title';
                h.textContent = title || '';
                header.appendChild(h);
                card.appendChild(header);

                if (typeof bodyBuilder === 'function') {
                    bodyBuilder(card);
                }
                parent.appendChild(card);
            };

            const addSectionPre = (parent, label, text) => {
                const labelEl = document.createElement('div');
                labelEl.className = 'space-context-label';
                labelEl.textContent = label || '';
                parent.appendChild(labelEl);

                const pre = document.createElement('pre');
                pre.className = 'space-context-pre';
                pre.textContent = text || '';
                parent.appendChild(pre);
            };

            const addSectionMarkdown = (parent, label, text) => {
                const labelEl = document.createElement('div');
                labelEl.className = 'space-context-label';
                labelEl.textContent = label || '';
                parent.appendChild(labelEl);

                const box = document.createElement('div');
                box.className = 'space-context-md';
                renderMarkdown(box, text || '');
                parent.appendChild(box);
            };

            const renderRaw = (text) => {
                addCard(left, 'Raw Context', (card) => {
                    addSectionPre(card, 'raw_text', text || '');
                });
            };

            const renderSampling = (samplingConfig, stopTokens) => {
                if (!samplingConfig || typeof samplingConfig !== 'object') {
                    addCard(right, 'sampling_config', (card) => {
                        addSectionPre(card, 'sampling_config', '—');
                    });
                    return;
                }

                const stageNames = Object.keys(samplingConfig).sort();
                stageNames.forEach((stageName) => {
                    const cfg = samplingConfig[stageName];
                    if (!cfg || typeof cfg !== 'object') {
                        return;
                    }
                    addCard(right, `sampling_config · ${stageName}`, (card) => {
                        const kv = document.createElement('div');
                        kv.className = 'space-kv-list';

                        Object.keys(cfg)
                            .sort()
                            .forEach((key) => {
                                if (key === 'stop_tokens') {
                                    return;
                                }
                                const row = document.createElement('div');
                                row.className = 'space-kv';
                                const k = document.createElement('div');
                                k.className = 'space-kv-key';
                                k.textContent = key;
                                const v = document.createElement('div');
                                v.className = 'space-kv-value';
                                v.textContent = toDisplayString(cfg[key]);
                                row.appendChild(k);
                                row.appendChild(v);
                                kv.appendChild(row);
                            });

                        card.appendChild(kv);

                        const tokens = stopTokens && typeof stopTokens === 'object' ? stopTokens[stageName] : null;
                        const rows = Array.isArray(tokens) ? tokens : [];
                        const label = document.createElement('div');
                        label.className = 'space-context-label';
                        label.textContent = 'stop_tokens';
                        card.appendChild(label);

                        if (!rows.length) {
                            const empty = document.createElement('div');
                            empty.className = 'space-stop-empty';
                            empty.textContent = '—';
                            card.appendChild(empty);
                            return;
                        }

                        const list = document.createElement('div');
                        list.className = 'space-stop-list';
                        rows.forEach((item) => {
                            const line = document.createElement('div');
                            line.className = 'space-stop-token';
                            const id = document.createElement('div');
                            id.className = 'space-stop-id';
                            id.textContent = String(item && typeof item.id === 'number' ? item.id : item && item.id ? item.id : '');
                            const tok = document.createElement('div');
                            tok.className = 'space-stop-text';
                            tok.textContent = item && typeof item.token === 'string' ? item.token : '';
                            line.appendChild(id);
                            line.appendChild(tok);
                            list.appendChild(line);
                        });
                        card.appendChild(list);
                    });
                });
            };

            const errors = eventObj && Array.isArray(eventObj.errors) ? eventObj.errors : [];
            if (errors.length) {
                addCard(right, 'errors', (card) => {
                    addSectionPre(card, 'errors', errors.map((e) => String(e)).join('\\n'));
                });
            }

            if (
                view === 'structured' &&
                eventObj &&
                eventObj.context &&
                typeof eventObj.context === 'object' &&
                eventObj.context !== null
            ) {
                const ctx = eventObj.context;
                const stages = Array.isArray(ctx.stages) ? ctx.stages : [];
                const samplingConfig = ctx.sampling_config && typeof ctx.sampling_config === 'object' ? ctx.sampling_config : null;
                const stopTokens = eventObj.stop_tokens && typeof eventObj.stop_tokens === 'object' ? eventObj.stop_tokens : {};

                if (stages.length) {
                    stages.forEach((stage, idx) => {
                        const title = `Stage ${idx + 1}`;
                        const stopReason =
                            stage && typeof stage.stop_reason === 'string' && stage.stop_reason ? stage.stop_reason : '';
                        addCard(left, stopReason ? `${title} · stop_reason: ${stopReason}` : title, (card) => {
                            const prompt = stage && typeof stage.prompt === 'string' ? stage.prompt : '';
                            const completion = stage && typeof stage.completion === 'string' ? stage.completion : '';
                            addSectionPre(card, 'prompt', prompt);
                            addSectionMarkdown(card, 'completion', completion);
                        });
                    });
                } else {
                    renderRaw(rawText);
                }

                renderSampling(samplingConfig, stopTokens);
            } else {
                renderRaw(rawText);
            }

            modal.classList.add('open');
            modal.setAttribute('aria-hidden', 'false');
        };

        const getTextboxInput = (elemId) => {
            const rootNode = queryById(elemId);
            if (!rootNode) {
                return null;
            }
            return rootNode.matches('textarea, input')
                ? rootNode
                : rootNode.querySelector('textarea, input');
        };

        const setTextboxValue = (elemId, value) => {
            const input = getTextboxInput(elemId);
            if (!input) {
                return;
            }
            if (input.value === value) {
                return;
            }
            input.value = value;
            input.dispatchEvent(new Event('input', { bubbles: true, composed: true }));
            input.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
        };

        const parseContextEvent = (rawValue) => {
            if (!rawValue) {
                return { view: 'text', raw_text: '' };
            }
            try {
                const parsed = JSON.parse(rawValue);
                if (parsed && typeof parsed === 'object') {
                    if (typeof parsed.raw_text !== 'string' && typeof parsed.text === 'string') {
                        return { ...parsed, raw_text: parsed.text };
                    }
                    return parsed;
                }
            } catch (error) {
                // Keep raw string payload as fallback.
            }
            return { view: 'text', raw_text: String(rawValue) };
        };

        let contextWatchToken = 0;

        const watchContextResult = (previousValue) => {
            const currentToken = ++contextWatchToken;
            const deadline = Date.now() + 30000;

            const poll = () => {
                if (currentToken !== contextWatchToken) {
                    return;
                }

                const input = getTextboxInput('space-context-result');
                const rawValue = input ? (input.value || '') : '';
                if (rawValue && rawValue !== previousValue) {
                    openModalEvent(parseContextEvent(rawValue));
                    return;
                }

                if (Date.now() >= deadline) {
                    if (rawValue) {
                        openModalEvent(parseContextEvent(rawValue));
                    } else {
                        openModalEvent({ view: 'text', raw_text: '读取完整 context 超时，请重试。' });
                    }
                    return;
                }

                window.setTimeout(poll, 120);
            };

            poll();
        };
        window.rwkvSpaceOpenContext = (event) => {
            openModalEvent(event || { view: 'text', raw_text: '' });
        };

        window.rwkvSpaceRequestContext = (cellId, sampleIndex, repeatIndex) => {
            if (!cellId) {
                return;
            }
            const contextResultInput = getTextboxInput('space-context-result');
            const previousValue = contextResultInput ? contextResultInput.value : '';
            openModalEvent({ view: 'text', raw_text: '正在加载完整 context...' });
            setTextboxValue(
                'space-context-payload',
                JSON.stringify({
                    cell_id: cellId,
                    sample_index: Number(sampleIndex),
                    repeat_index: Number(repeatIndex),
                    ts: Date.now(),
                }),
            );
            watchContextResult(previousValue);
        };

        window.rwkvSpaceCellClick = (cellId) => {
            if (!cellId) {
                return;
            }
            setTextboxValue(
                'space-click-payload',
                JSON.stringify({ cell_id: cellId, ts: Date.now() }),
            );
        };


        let tooltip = document.getElementById('space-hover-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'space-hover-tooltip';
            tooltip.className = 'space-hover-tooltip';
            document.body.appendChild(tooltip);
        }

        const hideTooltip = () => {
            if (!tooltip) {
                return;
            }
            tooltip.classList.remove('open');
            tooltip.innerHTML = '';
        };

        const showTooltip = (text, x, y) => {
            if (!tooltip || !text) {
                hideTooltip();
                return;
            }
            tooltip.innerHTML = String(text).replace(/\\n/g, '<br>');
            tooltip.style.left = `${x + 14}px`;
            tooltip.style.top = `${y + 14}px`;
            tooltip.classList.add('open');
        };

        const firstElementFromEvent = (event) => {
            if (typeof event.composedPath === 'function') {
                const path = event.composedPath();
                for (const node of path) {
                    if (node instanceof Element) {
                        return node;
                    }
                }
            }
            return event.target instanceof Element ? event.target : null;
        };

        root.addEventListener('click', (event) => {
            const target = firstElementFromEvent(event);
            if (!target) {
                return;
            }

            const contextButton = target.closest('button.space-context-open[data-context-cell-id]');
            if (contextButton) {
                event.preventDefault();
                const cellId = contextButton.getAttribute('data-context-cell-id');
                const sampleIndex = contextButton.getAttribute('data-sample-index');
                const repeatIndex = contextButton.getAttribute('data-repeat-index');
                if (cellId) {
                    window.rwkvSpaceRequestContext(cellId, sampleIndex, repeatIndex);
                }
                return;
            }

            const scoreCell = target.closest('td[data-clickable="1"][data-cell-id]');
            if (scoreCell) {
                event.preventDefault();
                const cellId = scoreCell.getAttribute('data-cell-id');
                if (cellId) {
                    window.rwkvSpaceCellClick(cellId);
                }
                return;
            }

            if (target.id === 'space-context-modal' || target.getAttribute('data-close-modal') === '1') {
                closeModal();
            }
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeModal();
            }
        });

        root.addEventListener('mousemove', (event) => {
            const target = firstElementFromEvent(event);
            if (!target) {
                hideTooltip();
                return;
            }
            const cell = target.closest('td[data-tooltip]');
            if (!cell) {
                hideTooltip();
                return;
            }
            const tooltipText = cell.getAttribute('data-tooltip');
            if (!tooltipText) {
                hideTooltip();
                return;
            }
            showTooltip(tooltipText, event.clientX, event.clientY);
        });

        root.addEventListener('scroll', hideTooltip, true);
        window.addEventListener('scroll', hideTooltip, true);
    }
    """


    with gr.Blocks(css=css, head=head, theme=gr.themes.Base(), js=js_func) as demo:
        with gr.Column(elem_classes="space-root"):

            gr.HTML(
                """
<div class="space-header">
    <h1>RWKV Skills · Space</h1>
    <div class="subtitle">以最新分数为基准，快速浏览各评测领域。</div>
</div>
"""
            )

            with gr.Group(elem_classes="space-controls-card"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        label="模型选择",
                        info="默认项会对每个架构 + 参数量组合选取 data_version（G0→…→G1d）最新的模型；手动选择时展示单个模型的最新数据库分数。",
                        choices=model_choices,
                        value=AUTO_MODEL_LABEL,
                        scale=3,
                        elem_classes="space-dropdown",
                    )
                    refresh_btn = gr.Button("刷新分数", variant="primary", scale=1)
                    download_btn = gr.DownloadButton("导出为 CSV", scale=1, value=csv_export_path)

                table_view = gr.Radio(
                    label="表格视图",
                    choices=[(label, key) for key, label in TABLE_VIEW_LABELS.items()],
                    value=initial_view_mode,
                    info="在旧表位置切换：最新明细 / 最新均分 / 与上一代对比",
                )

                summary_md = gr.Markdown(summary, elem_classes="space-info-card")

            interaction_state = gr.State(interaction_meta)
            selected_cell_state = gr.State(None)
            eval_records_state = gr.State(initial_eval_rows)
            eval_loader_state = gr.State(initial_eval_loader)
            tables: dict[str, gr.HTML] = {}

            with gr.Tabs(elem_classes="tabs"):
                for group in DOMAIN_GROUPS:
                    with gr.Tab(group["label"]):
                        gr.HTML('<div class="space-spacer"></div>')
                        tables[group["key"]] = gr.HTML(
                            domain_tables.get(group["key"], _render_pivot_html([], [])),
                        )

            with gr.Row(elem_classes="space-eval-toggle-row"):
                wrong_only_toggle = gr.Checkbox(
                    label="仅展示错题",
                    value=False,
                    elem_id="space-wrong-only-toggle",
                    container=False,
                )

            eval_records_html = gr.HTML(initial_eval_html, elem_id="space-eval-records-panel")

            with gr.Row(elem_classes="space-eval-pagination-row"):
                load_next_page_btn = gr.Button(
                    f"下一页（每次 {EVAL_FETCH_ROWS} 条）",
                    variant="secondary",
                    interactive=False,
                    elem_id="space-load-next-page-btn",
                )

            gr.HTML(modal_html)

            click_payload = gr.Textbox(
                value="",
                visible=True,
                container=False,
                show_label=False,
                elem_id="space-click-payload",
                elem_classes="space-hidden-input",
            )
            context_payload = gr.Textbox(
                value="",
                visible=True,
                container=False,
                show_label=False,
                elem_id="space-context-payload",
                elem_classes="space-hidden-input",
            )
            context_result = gr.Textbox(
                value="",
                visible=True,
                container=False,
                show_label=False,
                elem_id="space-context-result",
                elem_classes="space-hidden-input",
            )

            def update_dashboard(
                selected_model: str,
                selected_view_mode: str,
                only_wrong: bool,
                selected_cell: str | None,
            ):
                load_errors: list[str] = []
                entries = load_scores(errors=load_errors)
                model_choices = _compute_choices(entries)
                dropdown_value = selected_model if selected_model in model_choices else AUTO_MODEL_LABEL
                view_mode = _normalize_table_view(selected_view_mode)

                selection_state = _prepare_selection(entries, dropdown_value)
                warnings = load_errors + ([style_warning] if style_warning else [])
                csv_path = _export_selection_csv(selection_state)

                summary_value = _render_summary(
                    all_entries=entries,
                    visible=selection_state.entries,
                    selection=selection_state,
                    view_mode=view_mode,
                    warnings=warnings,
                )
                domain_table_values, interaction_map = _build_domain_tables(
                    selection_state,
                    all_entries=entries,
                    view_mode=view_mode,
                )

                selected_cell_id = selected_cell if isinstance(selected_cell, str) else None
                if selected_cell_id and selected_cell_id not in interaction_map:
                    selected_cell_id = None
                eval_html, resolved_cell, eval_rows, eval_loader, should_poll = _start_eval_records_load(
                    cell_id=selected_cell_id,
                    only_wrong=bool(only_wrong),
                    interaction_meta=interaction_map,
                )

                outputs: list[Any] = [
                    gr.update(choices=model_choices, value=selection_state.dropdown_value),
                    gr.update(value=csv_path),
                    gr.update(value=summary_value),
                ]
                for group in DOMAIN_GROUPS:
                    outputs.append(gr.update(value=domain_table_values.get(group["key"])))
                outputs.extend(
                    [
                        interaction_map,
                        resolved_cell,
                        gr.update(value=eval_html),
                        eval_rows,
                        eval_loader,
                        gr.update(interactive=should_poll),
                    ]
                )
                return outputs

            dashboard_outputs: list[Any] = [
                model_dropdown,
                download_btn,
                summary_md,
                tables["knowledge"],
                tables["math"],
                tables["coding"],
                tables["instruction_following"],
                tables["function_call"],
                interaction_state,
                selected_cell_state,
                eval_records_html,
                eval_records_state,
                eval_loader_state,
                load_next_page_btn,
            ]

            model_dropdown.change(
                update_dashboard,
                inputs=[model_dropdown, table_view, wrong_only_toggle, selected_cell_state],
                outputs=dashboard_outputs,
            )
            refresh_btn.click(
                update_dashboard,
                inputs=[model_dropdown, table_view, wrong_only_toggle, selected_cell_state],
                outputs=dashboard_outputs,
            )
            table_view.change(
                update_dashboard,
                inputs=[model_dropdown, table_view, wrong_only_toggle, selected_cell_state],
                outputs=dashboard_outputs,
            )

            def on_cell_click(
                payload: str,
                only_wrong: bool,
                interaction_map: dict[str, dict[str, Any]],
                selected_cell: str | None,
            ):
                parsed_cell = _parse_click_payload(payload)
                if parsed_cell is None:
                    parsed_cell = selected_cell if isinstance(selected_cell, str) else None
                eval_html, resolved_cell, eval_rows, eval_loader, should_poll = _start_eval_records_load(
                    cell_id=parsed_cell,
                    only_wrong=bool(only_wrong),
                    interaction_meta=interaction_map if isinstance(interaction_map, dict) else {},
                )
                return (
                    gr.update(value=eval_html),
                    resolved_cell,
                    eval_rows,
                    eval_loader,
                    gr.update(interactive=should_poll),
                )

            click_payload.input(
                on_cell_click,
                inputs=[click_payload, wrong_only_toggle, interaction_state, selected_cell_state],
                outputs=[
                    eval_records_html,
                    selected_cell_state,
                    eval_records_state,
                    eval_loader_state,
                    load_next_page_btn,
                ],
                queue=False,
            )

            def on_context_click(payload: str, interaction_map: dict[str, dict[str, Any]]):
                context_event = _build_context_event_payload(
                    payload=payload,
                    interaction_meta=interaction_map if isinstance(interaction_map, dict) else {},
                )
                return gr.update(value=json.dumps(context_event, ensure_ascii=False))

            context_payload.input(
                on_context_click,
                inputs=[context_payload, interaction_state],
                outputs=[context_result],
                queue=False,
            )

            def on_wrong_toggle(
                only_wrong: bool,
                interaction_map: dict[str, dict[str, Any]],
                selected_cell: str | None,
            ):
                cell_id = selected_cell if isinstance(selected_cell, str) else None
                eval_html, resolved_cell, eval_rows, eval_loader, should_poll = _start_eval_records_load(
                    cell_id=cell_id,
                    only_wrong=bool(only_wrong),
                    interaction_meta=interaction_map if isinstance(interaction_map, dict) else {},
                )
                return (
                    gr.update(value=eval_html),
                    resolved_cell,
                    eval_rows,
                    eval_loader,
                    gr.update(interactive=should_poll),
                )

            wrong_only_toggle.change(
                on_wrong_toggle,
                inputs=[wrong_only_toggle, interaction_state, selected_cell_state],
                outputs=[
                    eval_records_html,
                    selected_cell_state,
                    eval_records_state,
                    eval_loader_state,
                    load_next_page_btn,
                ],
                queue=False,
            )

            def on_load_next_page(
                loader_state: dict[str, Any],
                current_rows: list[dict[str, Any]],
            ):
                eval_html, merged_rows, next_loader, has_more = _continue_eval_records_load(
                    loader_state=loader_state if isinstance(loader_state, dict) else _empty_eval_loader_state(),
                    records=current_rows if isinstance(current_rows, list) else [],
                )
                return (
                    gr.update(value=eval_html),
                    merged_rows,
                    next_loader,
                    gr.update(interactive=has_more),
                )

            load_next_page_btn.click(
                on_load_next_page,
                inputs=[eval_loader_state, eval_records_state],
                outputs=[eval_records_html, eval_records_state, eval_loader_state, load_next_page_btn],
                queue=False,
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
    # Allow KaTeX font files to be served via Gradio's /file= endpoint.
    fonts_dir = Path(__file__).parent / "assets" / "vendor" / "fonts"
    allowed_paths = [str(fonts_dir)] if fonts_dir.exists() else None
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=allowed_paths)


if __name__ == "__main__":  # pragma: no cover
    main()
