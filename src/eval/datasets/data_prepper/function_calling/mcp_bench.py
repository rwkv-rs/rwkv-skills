from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling import McpBenchItem, load_mcp_bench_task_items

from .common import LocalRowsDatasetSpec, rwkv_rs_datasets_root

_REQUIRED_FIELDS = ("task_id", "instruction", "task")


def _rows_from_items(
    items: list[McpBenchItem],
    *,
    tasks_root: Path,
    runtime_root: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": item.task.task_id,
            "instruction": item.task.fuzzy_description or item.task.task_description,
            "task_file": item.task_file,
            "server_name": item.server_name,
            "combination_name": item.combination_name,
            "combination_type": item.combination_type,
            "servers": list(item.servers),
            "task": {
                "task_id": item.task.task_id,
                "task_description": item.task.task_description,
                "fuzzy_description": item.task.fuzzy_description,
                "dependency_analysis": item.task.dependency_analysis,
                "distraction_servers": list(item.task.distraction_servers),
            },
            "runtime_root": str(runtime_root),
            "tasks_root": str(tasks_root),
            "task_assets_commit_hint": "local_rwkv_rs_snapshot",
        }
        for item in items
    ]


@FUNCTION_CALLING_REGISTRY.register_spec("mcp_bench")
def prepare_mcp_bench_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError("mcp_bench 仅提供 test split")

    datasets_root = rwkv_rs_datasets_root()
    tasks_root = datasets_root / "mcp_bench" / "tasks"
    runtime_root = datasets_root / "mcp_bench" / "runtime"

    def _load() -> list[dict[str, Any]]:
        return _rows_from_items(load_mcp_bench_task_items(tasks_root, runtime_root), tasks_root=tasks_root, runtime_root=runtime_root)

    return LocalRowsDatasetSpec(
        "mcp_bench",
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="rwkv_rs_local_manifest",
        required_paths=(tasks_root, runtime_root),
        load_local_records=_load,
    )


__all__ = ["prepare_mcp_bench_spec"]
