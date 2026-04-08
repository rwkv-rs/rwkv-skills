from __future__ import annotations

"""Prepare HumanEval+ dataset (EvalPlus release with additional test cases)."""

import os
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_REQUIRED_FIELDS = ("task_id", "prompt")


def _load_human_eval_plus_problems(cache_root: Path) -> dict[str, dict[str, Any]]:
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    try:
        from evalplus.data import get_human_eval_plus  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `evalplus` 才能准备 human_eval_plus 数据集，请运行 `pip install evalplus`"
        ) from exc
    return get_human_eval_plus()


def _iter_records(split: str, cache_root: Path) -> Iterable[dict[str, Any]]:
    if split != "test":
        raise ValueError("human_eval_plus 仅提供 test split")

    problems = _load_human_eval_plus_problems(cache_root)
    for task_id, problem in problems.items():
        yield {
            "task_id": task_id,
            "prompt": problem.get("prompt", ""),
            "canonical_solution": problem.get("canonical_solution"),
            "entry_point": problem.get("entry_point"),
            "test": problem.get("test"),
            "contract": problem.get("contract"),
            "plus_input": problem.get("plus_input"),
            "atol": problem.get("atol"),
        }


class HumanEvalPlusDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str) -> None:
        super().__init__(
            "human_eval_plus",
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="evalplus",
        )

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(_iter_records(self.split, self.context.cache_root))

    def manifest_extra(self) -> dict[str, Any]:
        return {"source_split": self.split, "cache_root": str(self.context.cache_root)}


@CODE_GENERATION_REGISTRY.register_spec("human_eval_plus")
def prepare_human_eval_plus_spec(output_root: Path, split: str = "test") -> HumanEvalPlusDatasetSpec:
    return HumanEvalPlusDatasetSpec(output_root, split)


__all__ = ["prepare_human_eval_plus_spec"]
