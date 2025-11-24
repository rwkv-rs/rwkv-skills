from __future__ import annotations

"""Canonical in-memory表示，用于多选评估流程。

所有多选 JSONL（CEVAL、MMLU 等）在 loader 中都会被解析为
`MultipleChoiceRecord`，字段包括题干、选项列表、标准答案索引以及 subject
和 metadata。`MultipleChoiceDataset` 只是 `JsonlDataset` 的轻量封装，
便于评估器批量访问和做 subject 分组统计。
"""

from dataclasses import dataclass
from collections.abc import Iterable, Iterator

from .base import JsonlDataset, SubjectRecordBase


@dataclass(slots=True)
class MultipleChoiceRecord(SubjectRecordBase):
    """Canonical representation of a multiple-choice QA sample."""

    question: str
    choices: list[str]
    answer_index: int

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("choices 不能为空")
        if not (0 <= self.answer_index < len(self.choices)):
            raise ValueError("answer_index 超出 choices 范围")


class MultipleChoiceDataset(JsonlDataset[MultipleChoiceRecord]):
    """In-memory dataset wrapper to enable random access and batching."""


def iter_batches(
    records: Iterable[MultipleChoiceRecord], batch_size: int
) -> Iterator[list[MultipleChoiceRecord]]:
    dataset = MultipleChoiceDataset(list(records))
    yield from dataset.iter_batches(batch_size)


__all__ = [
    "MultipleChoiceRecord",
    "MultipleChoiceDataset",
    "iter_batches",
]
