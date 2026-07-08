from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import download_git_repo
from src.eval.tasks.function_calling.agentbench import load_agentbench_rows_from_source
from src.eval.scheduler.config import REPO_ROOT

from .common import OfficialRowsDatasetSpec, first_complete_source_root

_REQUIRED_FIELDS = ("task_id", "task_name", "index")
_AGENTBENCH_REPO_URL = "https://github.com/THUDM/AgentBench.git"
_AGENTBENCH_REPO_REVISION = "main"
_AGENTBENCH_REPO_ROOT_NAME = "AgentBench"


def agentbench_source_root() -> Path:
    override = os.environ.get("AGENTBENCH_SOURCE_ROOT") or os.environ.get("RWKV_AGENTBENCH_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    reference_root = REPO_ROOT / "references" / "AgentBench"
    if reference_root.exists():
        return reference_root.resolve()
    return (REPO_ROOT.parent / "AgentBench").resolve()


def _agentbench_source_candidates() -> tuple[Path, ...]:
    candidates = [
        agentbench_source_root(),
        REPO_ROOT / "references" / "AgentBench",
        REPO_ROOT.parent / "AgentBench",
        Path("/tmp/rwkv-official-refs/AgentBench"),
    ]
    return tuple(dict.fromkeys(candidates))


def _agentbench_data_file_from_root(dataset_name: str, root: Path) -> Path:
    if dataset_name == "agentbench_db":
        return root / "data" / "dbbench" / "standard.jsonl"
    if dataset_name == "agentbench_kg":
        return root / "data" / "knowledgegraph" / "std.json"
    raise ValueError(f"unknown AgentBench dataset: {dataset_name}")


def agentbench_data_file(dataset_name: str) -> Path:
    return _agentbench_data_file_from_root(dataset_name, agentbench_source_root())


def _task_name(dataset_name: str) -> str:
    if dataset_name == "agentbench_db":
        return "dbbench-std"
    if dataset_name == "agentbench_kg":
        return "kg-std"
    raise ValueError(f"unknown AgentBench dataset: {dataset_name}")


def _agentbench_required_paths(dataset_name: str):
    def _required(source_root: Path) -> tuple[Path, ...]:
        return (_agentbench_effective_data_file(dataset_name, source_root),)

    return _required


def _agentbench_effective_data_file(dataset_name: str, source_root: Path) -> Path:
    try:
        configured = agentbench_data_file(dataset_name).expanduser().resolve()
    except Exception:
        configured = None
    if configured is not None and configured.exists():
        return configured
    return _agentbench_data_file_from_root(dataset_name, source_root)


def _agentbench_downloaded_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return spec.cache_dir / _AGENTBENCH_REPO_ROOT_NAME


def _resolve_agentbench_source_root(spec: OfficialRowsDatasetSpec, *, dataset_name: str) -> Path:
    try:
        data_file = agentbench_data_file(dataset_name).expanduser().resolve()
    except Exception:
        data_file = None
    if data_file is not None and data_file.exists():
        return data_file.parent.resolve()
    return first_complete_source_root(_agentbench_source_candidates, _agentbench_required_paths(dataset_name)) or _agentbench_downloaded_source_root(spec)


def _download_agentbench_source(spec: OfficialRowsDatasetSpec) -> None:
    download_git_repo(
        spec.cache_dir,
        _AGENTBENCH_REPO_URL,
        revision=_AGENTBENCH_REPO_REVISION,
        root_name=_AGENTBENCH_REPO_ROOT_NAME,
    )


def _prepare_agentbench_spec(dataset_name: str, output_root: Path, split: str) -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} only provides test split")

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return load_agentbench_rows_from_source(
            _agentbench_effective_data_file(dataset_name, source_root),
            dataset_name=dataset_name,
            task_name=_task_name(dataset_name),
        )

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_agentbench_git",
        official_source="THUDM/AgentBench",
        resolve_source_root=lambda spec: _resolve_agentbench_source_root(spec, dataset_name=dataset_name),
        required_paths=_agentbench_required_paths(dataset_name),
        load_official_records=_load,
        download_source=_download_agentbench_source,
        extra={
            "source_repo_url": _AGENTBENCH_REPO_URL,
            "source_revision": _AGENTBENCH_REPO_REVISION,
            "controller_preflight_required": True,
        },
    )


@FUNCTION_CALLING_REGISTRY.register_spec("agentbench_db")
def prepare_agentbench_db_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_agentbench_spec("agentbench_db", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("agentbench_kg")
def prepare_agentbench_kg_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_agentbench_spec("agentbench_kg", output_root, split)


__all__ = [
    "agentbench_data_file",
    "agentbench_source_root",
    "prepare_agentbench_db_spec",
    "prepare_agentbench_kg_spec",
]
