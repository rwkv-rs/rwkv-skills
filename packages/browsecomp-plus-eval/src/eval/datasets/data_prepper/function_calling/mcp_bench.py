from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import download_git_repo
from src.eval.tasks.function_calling import McpBenchItem, load_mcp_bench_task_items
from src.eval.scheduler.config import REPO_ROOT

from .common import OfficialRowsDatasetSpec, first_complete_source_root, rwkv_rs_datasets_root

_REQUIRED_FIELDS = ("task_id", "instruction", "task")
_MCP_BENCH_REPO_URL = "https://github.com/Accenture/mcp-bench.git"
_MCP_BENCH_REPO_REVISION = "main"
_MCP_BENCH_REPO_ROOT_NAME = "mcp-bench"
_MCP_BENCH_DATASET_FILES: dict[str, tuple[str, ...] | None] = {
    "mcp_bench": None,
    "mcp_bench_single": ("mcpbench_tasks_single_runner_format.json",),
    "mcp_bench_multi_2server": ("mcpbench_tasks_multi_2server_runner_format.json",),
    "mcp_bench_multi_3server": ("mcpbench_tasks_multi_3server_runner_format.json",),
}


def _mcp_bench_source_candidates() -> tuple[Path, ...]:
    datasets_root = rwkv_rs_datasets_root()
    candidates = [
        datasets_root / "mcp_bench",
        REPO_ROOT / "references" / "mcp-bench",
        REPO_ROOT.parent / "mcp-bench",
        Path("/tmp/rwkv-official-refs/mcp-bench"),
        Path("/tmp/ref-mcp-bench"),
    ]
    return tuple(dict.fromkeys(candidates))


def _mcp_bench_tasks_root(source_root: Path) -> Path:
    legacy_tasks_root = source_root / "tasks"
    runtime_tasks_root = source_root / "runtime" / "tasks"
    tasks_root = legacy_tasks_root if legacy_tasks_root.exists() else runtime_tasks_root
    return tasks_root


def _mcp_bench_runtime_root(source_root: Path) -> Path:
    runtime_root = source_root / "runtime"
    return runtime_root if runtime_root.exists() else source_root


def _mcp_bench_required_paths(dataset_name: str) -> Callable[[Path], tuple[Path, ...]]:
    file_names = _MCP_BENCH_DATASET_FILES[dataset_name] or tuple(
        file_name for files in _MCP_BENCH_DATASET_FILES.values() if files for file_name in files
    )

    def _required(source_root: Path) -> tuple[Path, ...]:
        tasks_root = _mcp_bench_tasks_root(source_root)
        return tuple((tasks_root / file_name).resolve() for file_name in file_names)

    return _required


def _resolve_mcp_bench_source_root(spec: OfficialRowsDatasetSpec, *, dataset_name: str) -> Path:
    return first_complete_source_root(_mcp_bench_source_candidates, _mcp_bench_required_paths(dataset_name)) or (
        spec.cache_dir / _MCP_BENCH_REPO_ROOT_NAME
    )


def _download_mcp_bench_source(spec: OfficialRowsDatasetSpec) -> None:
    download_git_repo(
        spec.cache_dir,
        _MCP_BENCH_REPO_URL,
        revision=_MCP_BENCH_REPO_REVISION,
        root_name=_MCP_BENCH_REPO_ROOT_NAME,
    )


def _resolve_mcp_bench_roots(source_root: Path) -> tuple[Path, Path]:
    tasks_root = _mcp_bench_tasks_root(source_root)
    runtime_root = _mcp_bench_runtime_root(source_root)
    return tasks_root, runtime_root


def _rows_from_items(
    items: list[McpBenchItem],
    *,
    tasks_root: Path,
    runtime_root: Path,
    source_root: Path,
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
            "task_assets_commit_hint": "official_accenture_mcp_bench",
            "official_source_root": str(source_root),
        }
        for item in items
    ]


def _prepare_mcp_bench_spec(dataset_name: str, output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    file_names = _MCP_BENCH_DATASET_FILES[dataset_name]

    def _load(source_root: Path) -> list[dict[str, Any]]:
        tasks_root, runtime_root = _resolve_mcp_bench_roots(source_root)
        if file_names is None:
            items = load_mcp_bench_task_items(tasks_root, runtime_root)
        else:
            items = load_mcp_bench_task_items(tasks_root, runtime_root, file_names=file_names)
        return _rows_from_items(
            items,
            tasks_root=tasks_root,
            runtime_root=runtime_root,
            source_root=source_root,
        )

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_mcp_bench_git",
        official_source="Accenture/mcp-bench",
        resolve_source_root=lambda spec: _resolve_mcp_bench_source_root(spec, dataset_name=dataset_name),
        required_paths=_mcp_bench_required_paths(dataset_name),
        load_official_records=_load,
        download_source=_download_mcp_bench_source,
        extra=lambda root: {
            "source_repo_url": _MCP_BENCH_REPO_URL,
            "source_revision": _MCP_BENCH_REPO_REVISION,
            "tasks_root": str(_mcp_bench_tasks_root(root)),
            "runtime_root": str(_mcp_bench_runtime_root(root)),
            "runtime_preflight_required": True,
            "task_files": list(file_names or ()),
        },
    )


@FUNCTION_CALLING_REGISTRY.register_spec("mcp_bench")
def prepare_mcp_bench_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_mcp_bench_spec("mcp_bench", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("mcp_bench_single")
def prepare_mcp_bench_single_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_mcp_bench_spec("mcp_bench_single", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("mcp_bench_multi_2server")
def prepare_mcp_bench_multi_2server_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_mcp_bench_spec("mcp_bench_multi_2server", output_root, split)


@FUNCTION_CALLING_REGISTRY.register_spec("mcp_bench_multi_3server")
def prepare_mcp_bench_multi_3server_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_mcp_bench_spec("mcp_bench_multi_3server", output_root, split)


__all__ = [
    "prepare_mcp_bench_multi_2server_spec",
    "prepare_mcp_bench_multi_3server_spec",
    "prepare_mcp_bench_single_spec",
    "prepare_mcp_bench_spec",
]
