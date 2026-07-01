from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.tasks.function_calling.toolalpaca_source import load_toolalpaca_rows_from_source
from src.eval.scheduler.config import REPO_ROOT

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_tool_calls")

_TOOLALPACA_FILES = {
    "toolalpaca_eval_simulated": "eval_simulated.json",
    "toolalpaca_eval_real": "eval_real.json",
}


def toolalpaca_source_root() -> Path:
    override = os.environ.get("RWKV_TOOLALPACA_SOURCE_ROOT") or os.environ.get("TOOLALPACA_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    reference_root = REPO_ROOT / "references" / "ToolAlpaca" / "data"
    if reference_root.exists():
        return reference_root.resolve()
    return (REPO_ROOT.parent / "ToolAlpaca" / "data").resolve()


def toolalpaca_source_path(dataset_name: str) -> Path:
    if dataset_name not in _TOOLALPACA_FILES:
        raise ValueError(f"unknown ToolAlpaca dataset: {dataset_name}")
    return (toolalpaca_source_root() / _TOOLALPACA_FILES[dataset_name]).resolve()


def _prepare_toolalpaca_spec(dataset_name: str, output_root: Path, split: str) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    def _path() -> tuple[Path, ...]:
        return (toolalpaca_source_path(dataset_name),)

    def _load() -> list[dict[str, Any]]:
        return load_toolalpaca_rows_from_source(toolalpaca_source_path(dataset_name), dataset_name=dataset_name)

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="local_toolalpaca_source",
        required_paths=_path,
        load_local_records=_load,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("toolalpaca_eval_simulated")
def prepare_toolalpaca_eval_simulated_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_toolalpaca_spec("toolalpaca_eval_simulated", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("toolalpaca_eval_real")
def prepare_toolalpaca_eval_real_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_toolalpaca_spec("toolalpaca_eval_real", output_root, split)


__all__ = [
    "prepare_toolalpaca_eval_real_spec",
    "prepare_toolalpaca_eval_simulated_spec",
    "toolalpaca_source_path",
    "toolalpaca_source_root",
]
