from __future__ import annotations

import json
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.data_prepper.data_utils import write_jsonl
from src.eval.function_calling import (
    load_browsecomp_rows_from_csv,
    load_browsecomp_zh_rows_from_xlsx,
    load_mcp_bench_task_items,
)
from src.eval.scheduler.config import REPO_ROOT


def _rwkv_rs_datasets_root() -> Path:
    override = (
        __import__("os").environ.get("RWKV_RS_DATASETS_ROOT")
        or __import__("os").environ.get("RWKV_RS_ROOT")
    )
    if override:
        root = Path(override).expanduser().resolve()
        if root.name == "rwkv-rs":
            return root / "examples" / "rwkv-lm-eval" / "datasets"
        return root
    return (REPO_ROOT.parent / "rwkv-rs" / "examples" / "rwkv-lm-eval" / "datasets").resolve()


def _write_dataset(output_root: Path, dataset_name: str, split: str, rows: list[dict]) -> list[Path]:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    dataset_dir = (output_root / dataset_name).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, rows)
    return [target]


@CODE_GENERATION_REGISTRY.register("browsecomp")
def prepare_browsecomp(output_root: Path, split: str = "test") -> list[Path]:
    datasets_root = _rwkv_rs_datasets_root()
    source = datasets_root / "browsecomp" / "browse_comp_test_set.csv"
    if not source.is_file():
        raise FileNotFoundError(f"missing local browsecomp source: {source}")
    records = load_browsecomp_rows_from_csv(source)
    rows = [
        {
            "task_id": record.task_id,
            "question": record.question,
            "answer": record.answer,
            "topic": record.topic or "",
            "locale": record.locale,
            "source_path": str(source),
        }
        for record in records
    ]
    return _write_dataset(output_root, "browsecomp", split, rows)


@CODE_GENERATION_REGISTRY.register("browsecomp_zh")
def prepare_browsecomp_zh(output_root: Path, split: str = "test") -> list[Path]:
    datasets_root = _rwkv_rs_datasets_root()
    source = datasets_root / "browsecomp_zh" / "browsecomp-zh-encrypted.xlsx"
    if not source.is_file():
        raise FileNotFoundError(f"missing local browsecomp_zh source: {source}")
    records = load_browsecomp_zh_rows_from_xlsx(source)
    rows = [
        {
            "task_id": record.task_id,
            "question": record.question,
            "answer": record.answer,
            "locale": record.locale,
            "source_path": str(source),
        }
        for record in records
    ]
    return _write_dataset(output_root, "browsecomp_zh", split, rows)


@CODE_GENERATION_REGISTRY.register("mcp_bench")
def prepare_mcp_bench(output_root: Path, split: str = "test") -> list[Path]:
    datasets_root = _rwkv_rs_datasets_root()
    tasks_root = datasets_root / "mcp_bench" / "tasks"
    runtime_root = datasets_root / "mcp_bench" / "runtime"
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"missing local mcp_bench tasks root: {tasks_root}")
    if not runtime_root.is_dir():
        raise FileNotFoundError(f"missing local mcp_bench runtime root: {runtime_root}")
    items = load_mcp_bench_task_items(tasks_root, runtime_root)
    rows = [
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
    return _write_dataset(output_root, "mcp_bench", split, rows)


__all__ = [
    "prepare_browsecomp",
    "prepare_browsecomp_zh",
    "prepare_mcp_bench",
]
