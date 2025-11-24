from __future__ import annotations

"""Registry objects that map dataset slugs to their preparer functions."""

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

DatasetPreparer: TypeAlias = Callable[[Path, str], list[Path]]


class DatasetRegistry:
    """Registry helper used by each dataset family."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._preparers: dict[str, DatasetPreparer] = {}

    def register(self, dataset_name: str) -> Callable[[DatasetPreparer], DatasetPreparer]:
        def decorator(func: DatasetPreparer) -> DatasetPreparer:
            key = dataset_name.lower()
            if key in self._preparers:
                raise ValueError(f"{self._name} 数据集 {dataset_name} 已存在注册函数")
            self._preparers[key] = func
            return func

        return decorator

    def get(self, dataset_name: str) -> DatasetPreparer | None:
        return self._preparers.get(dataset_name.lower())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._preparers.keys()))

    def items(self) -> dict[str, DatasetPreparer]:
        return dict(self._preparers)


MULTIPLE_CHOICE_REGISTRY = DatasetRegistry("multiple_choice")
FREE_ANSWER_REGISTRY = DatasetRegistry("free_answer")
INSTRUCTION_FOLLOWING_REGISTRY = DatasetRegistry("instruction_following")
CODE_GENERATION_REGISTRY = DatasetRegistry("code_generation")

__all__ = [
    "DatasetRegistry",
    "DatasetPreparer",
    "MULTIPLE_CHOICE_REGISTRY",
    "FREE_ANSWER_REGISTRY",
    "INSTRUCTION_FOLLOWING_REGISTRY",
    "CODE_GENERATION_REGISTRY",
]
