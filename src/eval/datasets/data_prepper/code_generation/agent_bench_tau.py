from __future__ import annotations

"""Prepare vendored tau-bench/tau2-bench manifests for scheduler consumption."""

from pathlib import Path

from src.eval.agent_bench.tasks import iter_task_rows
from src.eval.datasets.data_prepper.data_utils import write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY


def _prepare_dataset(output_root: Path, *, dataset_name: str, split: str) -> list[Path]:
    output_dir = (output_root / dataset_name).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{dataset_name}_{split}.jsonl"
    rows = list(iter_task_rows(dataset_name, split))
    write_jsonl(target, rows)
    return [target]


@CODE_GENERATION_REGISTRY.register("tau_bench_retail")
def prepare_tau_bench_retail(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_dataset(output_root, dataset_name="tau_bench_retail", split=split)


@CODE_GENERATION_REGISTRY.register("tau_bench_airline")
def prepare_tau_bench_airline(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_dataset(output_root, dataset_name="tau_bench_airline", split=split)


@CODE_GENERATION_REGISTRY.register("tau2_bench_retail")
def prepare_tau2_bench_retail(output_root: Path, split: str = "base") -> list[Path]:
    return _prepare_dataset(output_root, dataset_name="tau2_bench_retail", split=split)


@CODE_GENERATION_REGISTRY.register("tau2_bench_airline")
def prepare_tau2_bench_airline(output_root: Path, split: str = "base") -> list[Path]:
    return _prepare_dataset(output_root, dataset_name="tau2_bench_airline", split=split)


@CODE_GENERATION_REGISTRY.register("tau2_bench_telecom")
def prepare_tau2_bench_telecom(output_root: Path, split: str = "base") -> list[Path]:
    return _prepare_dataset(output_root, dataset_name="tau2_bench_telecom", split=split)


__all__ = [
    "prepare_tau_bench_retail",
    "prepare_tau_bench_airline",
    "prepare_tau2_bench_retail",
    "prepare_tau2_bench_airline",
    "prepare_tau2_bench_telecom",
]
