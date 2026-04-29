from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.agent_bench.tasks import TAU_V2_DATA_ROOT, load_tau_v2_tasks
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "instruction", "task", "benchmark_version")


def _tau_bench_spec(output_root: Path, *, dataset_name: str, domain: str, split: str) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    def _load() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in load_tau_v2_tasks(domain=domain, split=split):
            payload = dict(row)
            payload["benchmark_version"] = "tau_bench"
            rows.append(payload)
        return rows

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="tau_v2_vendor_manifest",
        required_paths=(TAU_V2_DATA_ROOT,),
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
    return _tau_bench_spec(output_root, dataset_name="tau_bench_retail", domain="retail", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_airline")
def prepare_tau_bench_airline_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_bench_spec(output_root, dataset_name="tau_bench_airline", domain="airline", split=split)


@FUNCTION_CALLING_REGISTRY.register_spec("tau_bench_telecom")
def prepare_tau_bench_telecom_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _tau_bench_spec(output_root, dataset_name="tau_bench_telecom", domain="telecom", split=split)


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
