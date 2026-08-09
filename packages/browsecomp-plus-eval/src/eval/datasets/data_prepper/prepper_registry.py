from __future__ import annotations

"""Registry objects that map dataset slugs to their preparer functions."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

DatasetSpecFactory: TypeAlias = Callable[[Path, str], "DatasetSpec"]

if TYPE_CHECKING:
    from src.eval.datasets.runtime import DatasetSpec


class DatasetRegistry:
    """Registry helper used by each dataset family."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._spec_factories: dict[str, DatasetSpecFactory] = {}

    def _ensure_available(self, dataset_name: str) -> str:
        key = dataset_name.lower()
        if key in self._spec_factories:
            raise ValueError(f"{self._name} 数据集 {dataset_name} 已存在注册函数")
        return key

    def register_spec(self, dataset_name: str) -> Callable[[DatasetSpecFactory], DatasetSpecFactory]:
        def decorator(factory: DatasetSpecFactory) -> DatasetSpecFactory:
            key = self._ensure_available(dataset_name)
            self._spec_factories[key] = factory
            return factory

        return decorator

    def get_spec_factory(self, dataset_name: str) -> DatasetSpecFactory | None:
        return self._spec_factories.get(dataset_name.lower())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._spec_factories))


MULTIPLE_CHOICE_REGISTRY = DatasetRegistry("multiple_choice")
FREE_ANSWER_REGISTRY = DatasetRegistry("free_answer")
INSTRUCTION_FOLLOWING_REGISTRY = DatasetRegistry("instruction_following")
CODE_GENERATION_REGISTRY = DatasetRegistry("code_generation")
FUNCTION_CALLING_REGISTRY = DatasetRegistry("function_calling")

__all__ = [
    "DatasetRegistry",
    "DatasetSpecFactory",
    "MULTIPLE_CHOICE_REGISTRY",
    "FREE_ANSWER_REGISTRY",
    "INSTRUCTION_FOLLOWING_REGISTRY",
    "CODE_GENERATION_REGISTRY",
    "FUNCTION_CALLING_REGISTRY",
]
