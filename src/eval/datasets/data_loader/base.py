from __future__ import annotations

"""Shared streaming loader infrastructure.

所有 loader 都遵循“JSONL 文件 -> dataclass -> dataset”这一流程。这个模块
把文件存在性检查、逐行解析、`load()` 构造等公共能力抽出来，保证具体 loader
只需关心如何把一行 JSON dict 映射成 `Record`。
"""

import json
from pathlib import Path
from typing import Generic, TypeVar
from collections.abc import Iterator

from src.eval.datasets.data_struct.base import JsonlDataset

RecordT = TypeVar("RecordT")
DatasetT = TypeVar("DatasetT", bound=JsonlDataset[RecordT])


class JsonlDatasetLoader(Generic[RecordT, DatasetT]):
    """Base class for loaders that stream JSONL into datasets.

    Subclasses需要：
    - 指定 `dataset_cls`（通常是某个 JsonlDataset 子类）
    - 重写 `_parse_record`，描述如何将单行 JSON 转为 dataclass
    """

    dataset_cls: type[DatasetT]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"dataset 文件不存在: {self.path}")

    def __iter__(self) -> Iterator[RecordT]:
        with self.path.open("r", encoding="utf-8") as stream:
            for line in stream:
                payload = line.strip()
                if not payload:
                    continue
                record = json.loads(payload)
                yield self._parse_record(record)

    def load(self) -> DatasetT:
        return self.dataset_cls(list(iter(self)))

    def _parse_record(self, payload: dict) -> RecordT:  # pragma: no cover - abstract
        """Turn a single JSON object into `RecordT`. Must be implemented by subclasses."""
        raise NotImplementedError
