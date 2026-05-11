from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling.simple_tool_call import load_bfcl_ast_rows_from_sources
from src.eval.scheduler.config import REPO_ROOT

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_tool_calls")

_BFCL_SMALL_CATEGORY_PATHS = {
    "bfcl_simple_python": (
        "simple_python",
        ("BFCL_v4_simple_python.json",),
        ("possible_answer", "BFCL_v4_simple_python.json"),
    ),
    "bfcl_multiple": (
        "multiple",
        ("BFCL_v4_multiple.json",),
        ("possible_answer", "BFCL_v4_multiple.json"),
    ),
    "bfcl_exec_simple": (
        "exec_simple",
        ("unused_datasets", "question", "BFCL_v4_exec_simple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_simple.json"),
    ),
    "bfcl_exec_multiple": (
        "exec_multiple",
        ("unused_datasets", "question", "BFCL_v4_exec_multiple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_multiple.json"),
    ),
}


def bfcl_small_source_root() -> Path:
    override = (
        os.environ.get("RWKV_BFCL_SMALL_SOURCE_ROOT")
        or os.environ.get("RWKV_BFCL_V4_SOURCE_ROOT")
        or os.environ.get("BFCL_V4_SOURCE_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    reference_root = REPO_ROOT / "references" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data"
    if reference_root.exists():
        return reference_root.resolve()
    return (REPO_ROOT.parent / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data").resolve()


def bfcl_small_source_paths(dataset_name: str) -> tuple[Path, Path]:
    if dataset_name not in _BFCL_SMALL_CATEGORY_PATHS:
        raise ValueError(f"unknown BFCL small dataset: {dataset_name}")
    _category, question_parts, answer_parts = _BFCL_SMALL_CATEGORY_PATHS[dataset_name]
    root = bfcl_small_source_root()
    return (root.joinpath(*question_parts).resolve(), root.joinpath(*answer_parts).resolve())


def _prepare_bfcl_small_spec(dataset_name: str, output_root: Path, split: str) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    category, _question_parts, _answer_parts = _BFCL_SMALL_CATEGORY_PATHS[dataset_name]

    def _paths() -> tuple[Path, Path]:
        return bfcl_small_source_paths(dataset_name)

    def _load() -> list[dict[str, Any]]:
        question_path, answer_path = _paths()
        return load_bfcl_ast_rows_from_sources(question_path, answer_path, category=category)

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="local_bfcl_v4_small_source",
        required_paths=_paths,
        load_local_records=_load,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_simple_python")
def prepare_bfcl_simple_python_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_simple_python", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_simple")
def prepare_bfcl_exec_simple_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_exec_simple", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_multiple")
def prepare_bfcl_multiple_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_multiple", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_multiple")
def prepare_bfcl_exec_multiple_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_exec_multiple", output_root, split)


__all__ = [
    "bfcl_small_source_paths",
    "bfcl_small_source_root",
    "prepare_bfcl_exec_multiple_spec",
    "prepare_bfcl_exec_simple_spec",
    "prepare_bfcl_multiple_spec",
    "prepare_bfcl_simple_python_spec",
]
