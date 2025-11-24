from __future__ import annotations

"""Common primitives for dataset record/dataset representations.

`data_struct` 目录下的其他模块都依赖这里定义的抽象来实现：

- `RecordBase` / `SubjectRecordBase`：提供 metadata、subject 等共用字段。
- `JsonlDataset`：以 Sequence 形式暴露记录、提供批量迭代能力。

借助这些基类，不同任务（多选、自由问答、代码生成等）只需要描述
自身特有的字段即可。
"""

from dataclasses import dataclass, field
from typing import Any, Generic, Iterable, Iterator, TypeVar
from collections.abc import Sequence

MetadataDict = dict[str, Any]
T = TypeVar("T")


class JsonlDataset(Sequence[T], Generic[T]):
    """Generic immutable dataset wrapper with shared batching helpers."""

    def __init__(self, records: Iterable[T]):
        self._records = list(records)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._records)

    def __getitem__(self, idx: int) -> T:  # type: ignore[override]
        return self._records[idx]

    def iter_batches(self, batch_size: int) -> Iterator[list[T]]:
        """Yield contiguous batches; used by evaluators做 mini-batching。"""
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        batch: list[T] = []
        for record in self._records:
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


@dataclass(slots=True)
class RecordBase:
    """Base class for JSONL records that carry optional metadata."""

    metadata: MetadataDict = field(default_factory=dict)


@dataclass(slots=True)
class SubjectRecordBase(RecordBase):
    """Record type that optionally carries subject/category info."""

    subject: str | None = None
