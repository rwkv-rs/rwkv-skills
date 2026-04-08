from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.agent_bench.tasks import TAU_V2_DATA_ROOT, load_tau_v2_tasks
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling import load_rwkv_rs_tau_bench_rows

from .common import LocalRowsDatasetSpec, rwkv_rs_datasets_root

_REQUIRED_FIELDS = ("task_id", "instruction", "task", "benchmark_version")


def _tau_v1_source_paths(domain: str) -> tuple[Path, Path]:
    base = rwkv_rs_datasets_root() / "tau_bench" / domain
    return base / "tasks.json", base / "split_tasks.json"


def _tau_v1_spec(output_root: Path, *, dataset_name: str, domain: str, split: str) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    tasks_path, split_path = _tau_v1_source_paths(domain)

    def _load() -> list[dict[str, Any]]:
        return load_rwkv_rs_tau_bench_rows(datasets_root=rwkv_rs_datasets_root(), domain=domain)

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="rwkv_rs_local_manifest",
        required_paths=(tasks_path, split_path),
        load_local_records=_load,
        extra={"domain": domain, "benchmark_version": "tau_bench"},
    )


def _tau_v2_spec(output_root: Path, *, dataset_name: str, domain: str, split: str) -> LocalRowsDatasetSpec:
    def _load() -> list[dict[str, Any]]:
        return load_tau_v2_tasks(domain=domain, split=split)

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="tau_v2_vendor_manifest",
        required_paths=(TAU_V2_DATA_ROOT,),
        load_local_records=_load,
        extra={"domain": domain, "benchmark_version": "tau_v2"},
    )


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_retail")
def prepare_tau_bench_retail_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_v1_spec(output_root, dataset_name="tau_bench_retail", domain="retail", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_airline")
def prepare_tau_bench_airline_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_v1_spec(output_root, dataset_name="tau_bench_airline", domain="airline", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_telecom")
def prepare_tau_bench_telecom_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_v1_spec(output_root, dataset_name="tau_bench_telecom", domain="telecom", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau2_bench_retail")
def prepare_tau2_bench_retail_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v2_spec(output_root, dataset_name="tau2_bench_retail", domain="retail", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau2_bench_airline")
def prepare_tau2_bench_airline_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v2_spec(output_root, dataset_name="tau2_bench_airline", domain="airline", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau2_bench_telecom")
def prepare_tau2_bench_telecom_spec(output_root: Path, split: str = "base") -> LocalRowsDatasetSpec:
    return _tau_v2_spec(output_root, dataset_name="tau2_bench_telecom", domain="telecom", split=split)


__all__ = [
    "prepare_tau_bench_retail_spec",
    "prepare_tau_bench_airline_spec",
    "prepare_tau_bench_telecom_spec",
    "prepare_tau2_bench_retail_spec",
    "prepare_tau2_bench_airline_spec",
    "prepare_tau2_bench_telecom_spec",
]
