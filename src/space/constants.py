"""Constants, dataclasses, and tiny helpers shared across the space package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domains import (
    CODING_DOMAINS,
    FUNCTION_CALL_DOMAINS,
    INSTRUCTION_FOLLOWING_DOMAINS,
    KNOWLEDGE_GROUP_DOMAINS,
    MATH_DOMAINS,
)
from .data import ScoreEntry


# ---------------------------------------------------------------------------
# UI labels & defaults
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Eval record pagination
# ---------------------------------------------------------------------------

EVAL_PAGE_SIZE = 15
EVAL_PRELOAD_PAGES = 2
EVAL_PRELOAD_ROWS = EVAL_PAGE_SIZE * EVAL_PRELOAD_PAGES
EVAL_FETCH_ROWS = EVAL_PAGE_SIZE
EVAL_OVERSCAN_ROWS = 1
EVAL_CONTEXT_PREVIEW_LIMIT = 20


# ---------------------------------------------------------------------------
# Metric keys & domain grouping
# ---------------------------------------------------------------------------

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
        "domains": KNOWLEDGE_GROUP_DOMAINS,
        "title": "知识类（MMLU / Multi-choice）",
    },
    {
        "key": "math",
        "label": "Math",
        "domains": MATH_DOMAINS,
        "title": "数学推理（AIME / Math-500 等）",
    },
    {
        "key": "coding",
        "label": "Coding",
        "domains": CODING_DOMAINS,
        "title": "代码",
    },
    {
        "key": "instruction_following",
        "label": "Instruction Following",
        "domains": INSTRUCTION_FOLLOWING_DOMAINS,
        "title": "指令遵循（IFEval 等）",
    },
    {
        "key": "function_call",
        "label": "Function Call",
        "domains": FUNCTION_CALL_DOMAINS,
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


# ---------------------------------------------------------------------------
# Subdomain / subject mapping
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model-size colour palette (thead group row border + text)
# ---------------------------------------------------------------------------

PARAM_SIZE_COLORS: dict[str, tuple[str, str]] = {
    # param_token: (border_color, text_color)
    "0_1b": ("#ec4899", "#f472b6"),
    "0_4b": ("#6366f1", "#818cf8"),
    "1_5b": ("#0ea5e9", "#38bdf8"),
    "2_9b": ("#10b981", "#34d399"),
    "7_2b": ("#f59e0b", "#fbbf24"),
    "13_3b": ("#ef4444", "#f87171"),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
