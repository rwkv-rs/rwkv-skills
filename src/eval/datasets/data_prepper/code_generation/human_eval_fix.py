from __future__ import annotations

"""Prepare HumanEvalFix (HumanEvalPack, Python split) as JSONL."""

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_DATASET_ID = "bigcode/humanevalpack"
_DATASET_CONFIG = "python"
_REQUIRED_FIELDS = ("task_id", "prompt")


def _format_bugfix_prompt(
    prompt: str | None,
    buggy_solution: str | None,
    entry_point: str | None,
) -> str:
    prompt_text = (prompt or "").rstrip()
    buggy = (buggy_solution or "").rstrip()
    entry = (entry_point or "").strip()

    parts: list[str] = []
    if prompt_text:
        parts.append(prompt_text)
    if buggy:
        parts.append("# Buggy implementation:")
        parts.append(buggy)
    if entry:
        parts.append(f"# Fix the function `{entry}` so it passes all tests.")
    return "\n".join(parts).strip()


def _load_human_eval_fix_rows(split: str) -> Iterable[Mapping[str, Any]]:
    configure_hf_home()
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 human_eval_fix 数据集，请运行 `pip install datasets`"
        ) from exc
    return load_dataset(_DATASET_ID, _DATASET_CONFIG, split=split)


def _iter_records(split: str) -> Iterable[dict[str, Any]]:
    dataset = _load_human_eval_fix_rows(split)
    for row in dataset:
        prompt = row.get("prompt") or row.get("instruction") or ""
        buggy_solution = row.get("buggy_solution") or ""
        if not isinstance(prompt, str):
            prompt = ""
        if not isinstance(buggy_solution, str):
            buggy_solution = str(buggy_solution)
        entry_point = row.get("entry_point") or ""
        if not isinstance(entry_point, str):
            entry_point = str(entry_point)

        prompt = _format_bugfix_prompt(prompt, buggy_solution, entry_point)
        yield {
            "task_id": row.get("task_id"),
            "prompt": prompt,
            "canonical_solution": row.get("canonical_solution"),
            "buggy_solution": row.get("buggy_solution"),
            "entry_point": row.get("entry_point"),
            "test": row.get("test"),
            "example_test": row.get("example_test"),
        }


class HumanEvalFixDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str) -> None:
        super().__init__(
            "human_eval_fix",
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="hf_load_dataset",
        )

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(_iter_records(self.split))

    def manifest_extra(self) -> dict[str, Any]:
        return {"dataset_id": _DATASET_ID, "config": _DATASET_CONFIG, "source_split": self.split}


@CODE_GENERATION_REGISTRY.register_spec("human_eval_fix")
def prepare_human_eval_fix_spec(output_root: Path, split: str = "test") -> HumanEvalFixDatasetSpec:
    return HumanEvalFixDatasetSpec(output_root, split)


__all__ = ["prepare_human_eval_fix_spec"]
