from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling.agentbench import load_agentbench_rows_from_source
from src.eval.scheduler.config import REPO_ROOT

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "task_name", "index")


def agentbench_source_root() -> Path:
    override = os.environ.get("AGENTBENCH_SOURCE_ROOT") or os.environ.get("RWKV_AGENTBENCH_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    reference_root = REPO_ROOT / "references" / "AgentBench"
    if reference_root.exists():
        return reference_root.resolve()
    return (REPO_ROOT.parent / "AgentBench").resolve()


def agentbench_data_file(dataset_name: str) -> Path:
    root = agentbench_source_root()
    if dataset_name == "agentbench_db":
        return root / "data" / "dbbench" / "standard.jsonl"
    if dataset_name == "agentbench_kg":
        return root / "data" / "knowledgegraph" / "std.json"
    raise ValueError(f"unknown AgentBench dataset: {dataset_name}")


def _task_name(dataset_name: str) -> str:
    if dataset_name == "agentbench_db":
        return "dbbench-std"
    if dataset_name == "agentbench_kg":
        return "kg-std"
    raise ValueError(f"unknown AgentBench dataset: {dataset_name}")


def _prepare_agentbench_spec(dataset_name: str, output_root: Path, split: str) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} only provides test split")

    def _paths() -> tuple[Path, ...]:
        return (agentbench_data_file(dataset_name),)

    def _load() -> list[dict[str, Any]]:
        return load_agentbench_rows_from_source(
            agentbench_data_file(dataset_name),
            dataset_name=dataset_name,
            task_name=_task_name(dataset_name),
        )

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="local_official_agentbench",
        required_paths=_paths,
        load_local_records=_load,
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
