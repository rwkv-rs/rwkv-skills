from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import download_git_repo
from src.eval.tasks.function_calling.toolalpaca_source import load_toolalpaca_rows_from_source
from src.eval.scheduler.config import REPO_ROOT

from .common import OfficialRowsDatasetSpec, first_complete_source_root

_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_tool_calls")
_TOOLALPACA_REPO_URL = "https://github.com/tangqiaoyu/ToolAlpaca.git"
_TOOLALPACA_REPO_REVISION = "main"
_TOOLALPACA_REPO_ROOT_NAME = "ToolAlpaca"

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


def _toolalpaca_source_candidates() -> tuple[Path, ...]:
    candidates = [
        toolalpaca_source_root(),
        REPO_ROOT / "references" / "ToolAlpaca" / "data",
        REPO_ROOT / "references" / "ToolAlpaca",
        REPO_ROOT.parent / "ToolAlpaca" / "data",
        REPO_ROOT.parent / "ToolAlpaca",
        Path("/tmp/rwkv-official-refs/ToolAlpaca/data"),
        Path("/tmp/rwkv-official-refs/ToolAlpaca"),
    ]
    return tuple(dict.fromkeys(candidates))


def _toolalpaca_data_root(root: Path) -> Path:
    return root / "data" if (root / "data").is_dir() else root


def _toolalpaca_source_path_from_root(dataset_name: str, root: Path) -> Path:
    if dataset_name not in _TOOLALPACA_FILES:
        raise ValueError(f"unknown ToolAlpaca dataset: {dataset_name}")
    return (_toolalpaca_data_root(root) / _TOOLALPACA_FILES[dataset_name]).resolve()


def toolalpaca_source_path(dataset_name: str) -> Path:
    return _toolalpaca_source_path_from_root(dataset_name, toolalpaca_source_root())


def _toolalpaca_required_paths(dataset_name: str):
    def _required(source_root: Path) -> tuple[Path, ...]:
        return (_toolalpaca_source_path_from_root(dataset_name, source_root),)

    return _required


def _toolalpaca_downloaded_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return spec.cache_dir / _TOOLALPACA_REPO_ROOT_NAME


def _resolve_toolalpaca_source_root(spec: OfficialRowsDatasetSpec, *, dataset_name: str) -> Path:
    return first_complete_source_root(_toolalpaca_source_candidates, _toolalpaca_required_paths(dataset_name)) or _toolalpaca_downloaded_source_root(spec)


def _download_toolalpaca_source(spec: OfficialRowsDatasetSpec) -> None:
    download_git_repo(
        spec.cache_dir,
        _TOOLALPACA_REPO_URL,
        revision=_TOOLALPACA_REPO_REVISION,
        root_name=_TOOLALPACA_REPO_ROOT_NAME,
    )


def _prepare_toolalpaca_spec(dataset_name: str, output_root: Path, split: str) -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return load_toolalpaca_rows_from_source(
            _toolalpaca_source_path_from_root(dataset_name, source_root),
            dataset_name=dataset_name,
        )

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_toolalpaca_git",
        official_source="tangqiaoyu/ToolAlpaca",
        resolve_source_root=lambda spec: _resolve_toolalpaca_source_root(spec, dataset_name=dataset_name),
        required_paths=_toolalpaca_required_paths(dataset_name),
        load_official_records=_load,
        download_source=_download_toolalpaca_source,
        extra={
            "source_repo_url": _TOOLALPACA_REPO_URL,
            "source_revision": _TOOLALPACA_REPO_REVISION,
        },
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
