from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import download_git_repo
from src.eval.tasks.function_calling.bfcl_exec import load_bfcl_exec_rows_from_sources
from src.eval.tasks.function_calling.simple_tool_call import load_bfcl_ast_rows_from_sources
from src.eval.scheduler.config import REPO_ROOT

from .common import OfficialRowsDatasetSpec, first_complete_source_root

_AST_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_tool_calls")
_EXEC_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_executable_calls", "execution_result_type")
_BFCL_REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
_BFCL_REPO_REVISION = "main"
_BFCL_REPO_ROOT_NAME = "gorilla"
_BFCL_SOURCE_SUBDIR = ("berkeley-function-call-leaderboard", "bfcl_eval", "data")

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
    "bfcl_exec_simple_ast": (
        "exec_simple",
        ("unused_datasets", "question", "BFCL_v4_exec_simple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_simple.json"),
    ),
    "bfcl_exec_multiple_ast": (
        "exec_multiple",
        ("unused_datasets", "question", "BFCL_v4_exec_multiple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_multiple.json"),
    ),
}

_BFCL_EXEC_CATEGORY_PATHS = {
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
    "bfcl_exec_parallel": (
        "exec_parallel",
        ("unused_datasets", "question", "BFCL_v4_exec_parallel.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_parallel.json"),
    ),
    "bfcl_exec_parallel_multiple": (
        "exec_parallel_multiple",
        ("unused_datasets", "question", "BFCL_v4_exec_parallel_multiple.json"),
        ("unused_datasets", "possible_answer", "BFCL_v4_exec_parallel_multiple.json"),
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
    paths = _BFCL_SMALL_CATEGORY_PATHS.get(dataset_name) or _BFCL_EXEC_CATEGORY_PATHS.get(dataset_name)
    if paths is None:
        raise ValueError(f"unknown BFCL small dataset: {dataset_name}")
    _category, question_parts, answer_parts = paths
    root = bfcl_small_source_root()
    return (root.joinpath(*question_parts).resolve(), root.joinpath(*answer_parts).resolve())


def _bfcl_source_candidates() -> tuple[Path, ...]:
    candidates = [
        bfcl_small_source_root(),
        REPO_ROOT / "references" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
        REPO_ROOT.parent / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
        Path("/tmp/rwkv-official-refs/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
        Path("/tmp/gorilla-official/berkeley-function-call-leaderboard/bfcl_eval/data"),
    ]
    return tuple(dict.fromkeys(candidates))


def _bfcl_source_paths_from_root(dataset_name: str, source_root: Path) -> tuple[Path, Path]:
    paths = _BFCL_SMALL_CATEGORY_PATHS.get(dataset_name) or _BFCL_EXEC_CATEGORY_PATHS.get(dataset_name)
    if paths is None:
        raise ValueError(f"unknown BFCL small dataset: {dataset_name}")
    _category, question_parts, answer_parts = paths
    return (
        source_root.joinpath(*question_parts).expanduser().resolve(),
        source_root.joinpath(*answer_parts).expanduser().resolve(),
    )


def _bfcl_required_paths(dataset_name: str) -> Callable[[Path], tuple[Path, Path]]:
    def _required(source_root: Path) -> tuple[Path, Path]:
        return _bfcl_source_paths_from_root(dataset_name, source_root)

    return _required


def _bfcl_downloaded_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return spec.cache_dir.joinpath(_BFCL_REPO_ROOT_NAME, *_BFCL_SOURCE_SUBDIR)


def _resolve_bfcl_source_root(
    spec: OfficialRowsDatasetSpec,
    *,
    dataset_name: str,
) -> Path:
    return first_complete_source_root(_bfcl_source_candidates, _bfcl_required_paths(dataset_name)) or _bfcl_downloaded_source_root(spec)


def _download_bfcl_source(spec: OfficialRowsDatasetSpec) -> None:
    download_git_repo(
        spec.cache_dir,
        _BFCL_REPO_URL,
        revision=_BFCL_REPO_REVISION,
        root_name=_BFCL_REPO_ROOT_NAME,
    )


def bfcl_official_root_from_source_root(source_root: Path | None = None) -> Path | None:
    root = (source_root or bfcl_small_source_root()).expanduser().resolve()
    if root.name == "data" and root.parent.name == "bfcl_eval":
        candidate = root.parent.parent
        if (candidate / "bfcl_eval").is_dir():
            return candidate.resolve()
    for parent in (root, *root.parents):
        if (parent / "bfcl_eval" / "eval_checker").is_dir():
            return parent.resolve()
    return None


def _with_bfcl_official_metadata(rows: list[dict[str, Any]], *, source_root: Path) -> list[dict[str, Any]]:
    official_root = bfcl_official_root_from_source_root(source_root)
    if official_root is None:
        return rows
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            row["metadata"] = metadata
        metadata.setdefault("official_root", str(official_root))
        metadata.setdefault("official_source", "gorilla/berkeley-function-call-leaderboard")
    return rows


def _prepare_bfcl_small_spec(dataset_name: str, output_root: Path, split: str) -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    category, _question_parts, _answer_parts = _BFCL_SMALL_CATEGORY_PATHS[dataset_name]

    def _load(source_root: Path) -> list[dict[str, Any]]:
        question_path, answer_path = _bfcl_source_paths_from_root(dataset_name, source_root)
        return _with_bfcl_official_metadata(
            load_bfcl_ast_rows_from_sources(question_path, answer_path, category=category),
            source_root=source_root,
        )

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_AST_REQUIRED_FIELDS,
        source_kind="official_bfcl_v4_ast_git",
        official_source="ShishirPatil/gorilla/berkeley-function-call-leaderboard",
        resolve_source_root=lambda spec: _resolve_bfcl_source_root(spec, dataset_name=dataset_name),
        required_paths=_bfcl_required_paths(dataset_name),
        load_official_records=_load,
        download_source=_download_bfcl_source,
        extra={
            "source_repo_url": _BFCL_REPO_URL,
            "source_revision": _BFCL_REPO_REVISION,
            "category": category,
        },
    )


def _prepare_bfcl_exec_spec(dataset_name: str, output_root: Path, split: str) -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    category, _question_parts, _answer_parts = _BFCL_EXEC_CATEGORY_PATHS[dataset_name]

    def _load(source_root: Path) -> list[dict[str, Any]]:
        question_path, answer_path = _bfcl_source_paths_from_root(dataset_name, source_root)
        return _with_bfcl_official_metadata(
            load_bfcl_exec_rows_from_sources(question_path, answer_path, category=category),
            source_root=source_root,
        )

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_EXEC_REQUIRED_FIELDS,
        source_kind="official_bfcl_v4_exec_git",
        official_source="ShishirPatil/gorilla/berkeley-function-call-leaderboard",
        resolve_source_root=lambda spec: _resolve_bfcl_source_root(spec, dataset_name=dataset_name),
        required_paths=_bfcl_required_paths(dataset_name),
        load_official_records=_load,
        download_source=_download_bfcl_source,
        extra={
            "source_repo_url": _BFCL_REPO_URL,
            "source_revision": _BFCL_REPO_REVISION,
            "category": category,
        },
    )


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_simple_python")
def prepare_bfcl_simple_python_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_simple_python", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_simple_ast")
def prepare_bfcl_exec_simple_ast_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_exec_simple_ast", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_multiple")
def prepare_bfcl_multiple_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_multiple", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_multiple_ast")
def prepare_bfcl_exec_multiple_ast_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_small_spec("bfcl_exec_multiple_ast", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_simple")
def prepare_bfcl_exec_simple_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_exec_spec("bfcl_exec_simple", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_multiple")
def prepare_bfcl_exec_multiple_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_exec_spec("bfcl_exec_multiple", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_parallel")
def prepare_bfcl_exec_parallel_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_exec_spec("bfcl_exec_parallel", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_exec_parallel_multiple")
def prepare_bfcl_exec_parallel_multiple_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_bfcl_exec_spec("bfcl_exec_parallel_multiple", output_root, split)


__all__ = [
    "bfcl_small_source_paths",
    "bfcl_small_source_root",
    "bfcl_official_root_from_source_root",
    "prepare_bfcl_exec_multiple_spec",
    "prepare_bfcl_exec_multiple_ast_spec",
    "prepare_bfcl_exec_parallel_multiple_spec",
    "prepare_bfcl_exec_parallel_spec",
    "prepare_bfcl_exec_simple_spec",
    "prepare_bfcl_exec_simple_ast_spec",
    "prepare_bfcl_multiple_spec",
    "prepare_bfcl_simple_python_spec",
]
