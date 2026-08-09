from __future__ import annotations

"""Dataclasses that capture EvalPlus/代码生成样本的完整信息。

`CodeGenerationRecord` 包含 prompt、starter code、entry point、单元测试等，
并保留 `metadata`（通过 `RecordBase`）。评估器将 JSONL 转成这些结构后，
可统一访问这些字段，无需关心具体数据集的原始 schema。
"""

from dataclasses import dataclass
from typing import Any

from .base import JsonlDataset, RecordBase

@dataclass(slots=True)
class CodeGenerationRecord(RecordBase):
    """All信息 needed for EvalPlus: prompt/stubs/tests/答案。"""

    task_id: str
    prompt: str
    starter_code: str | None = None
    entry_point: str | None = None
    canonical_solution: str | None = None
    test_cases: Any | None = None


class CodeGenerationDataset(JsonlDataset[CodeGenerationRecord]):
    """Container used by EvalPlus evaluator to stream samples。"""


__all__ = [
    "CodeGenerationRecord",
    "CodeGenerationDataset",
]
