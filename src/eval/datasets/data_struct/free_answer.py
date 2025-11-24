from __future__ import annotations

"""Shared record definitions for自由回答 / 推理类任务。

很多评估脚本会把各种数学、翻译、开放式问答数据统一转换成“问题 + 答案 +
subject + metadata”这种 JSONL 结构。本模块提供了
`FreeAnswerRecord`/`FreeAnswerDataset`，让上层逻辑在读取任何 free-form
数据集时都能获得一致的字段，并且可以通过 `subject`/`metadata` 扩展统计信息。
"""

from dataclasses import dataclass

from .base import JsonlDataset, SubjectRecordBase


@dataclass(slots=True)
class FreeAnswerRecord(SubjectRecordBase):
    """Basic schema for“问题-答案”样本，可选 subject + metadata。"""

    question: str
    answer: str


class FreeAnswerDataset(JsonlDataset[FreeAnswerRecord]):
    """Lightweight container used by free-form评估器和指标。"""


__all__ = [
    "FreeAnswerRecord",
    "FreeAnswerDataset",
]
