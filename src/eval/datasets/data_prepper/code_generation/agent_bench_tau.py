from __future__ import annotations

"""Prepare tau-bench/tau2-bench manifests for scheduler consumption."""

import os
from pathlib import Path

from src.eval.agent_bench.tasks import load_tau_v2_tasks
from src.eval.datasets.data_prepper.data_utils import write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.function_calling import load_rwkv_rs_tau_bench_rows
from src.eval.scheduler.config import REPO_ROOT


def _rwkv_rs_datasets_root() -> Path:
    override = os.environ.get("RWKV_RS_DATASETS_ROOT") or os.environ.get("RWKV_RS_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if root.name == "rwkv-rs":
            return root / "examples" / "rwkv-lm-eval" / "datasets"
        return root
    return (REPO_ROOT.parent / "rwkv-rs" / "examples" / "rwkv-lm-eval" / "datasets").resolve()


def _prepare_dataset(output_root: Path, *, dataset_name: str, rows: list[dict], split: str) -> list[Path]:
    output_dir = (output_root / dataset_name).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{dataset_name}_{split}.jsonl"
    write_jsonl(target, rows)
    return [target]


def _prepare_tau_bench_domain(output_root: Path, *, domain: str, split: str) -> list[Path]:
    if split != "test":
        raise ValueError(f"tau_bench_{domain} 仅支持 test split")
    rows = load_rwkv_rs_tau_bench_rows(
        datasets_root=_rwkv_rs_datasets_root(),
        domain=domain,
    )
    return _prepare_dataset(output_root, dataset_name=f"tau_bench_{domain}", rows=rows, split=split)


def _prepare_tau2_bench_domain(output_root: Path, *, domain: str, split: str) -> list[Path]:
    rows = load_tau_v2_tasks(domain=domain, split=split)
    return _prepare_dataset(output_root, dataset_name=f"tau2_bench_{domain}", rows=rows, split=split)


@CODE_GENERATION_REGISTRY.register("tau_bench_retail")
def prepare_tau_bench_retail(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_tau_bench_domain(output_root, domain="retail", split=split)


@CODE_GENERATION_REGISTRY.register("tau_bench_airline")
def prepare_tau_bench_airline(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_tau_bench_domain(output_root, domain="airline", split=split)


@CODE_GENERATION_REGISTRY.register("tau_bench_telecom")
def prepare_tau_bench_telecom(output_root: Path, split: str = "test") -> list[Path]:
    return _prepare_tau_bench_domain(output_root, domain="telecom", split=split)


@CODE_GENERATION_REGISTRY.register("tau2_bench_retail")
def prepare_tau2_bench_retail(output_root: Path, split: str = "base") -> list[Path]:
    return _prepare_tau2_bench_domain(output_root, domain="retail", split=split)


@CODE_GENERATION_REGISTRY.register("tau2_bench_airline")
def prepare_tau2_bench_airline(output_root: Path, split: str = "base") -> list[Path]:
    return _prepare_tau2_bench_domain(output_root, domain="airline", split=split)


@CODE_GENERATION_REGISTRY.register("tau2_bench_telecom")
def prepare_tau2_bench_telecom(output_root: Path, split: str = "base") -> list[Path]:
    return _prepare_tau2_bench_domain(output_root, domain="telecom", split=split)


__all__ = [
    "prepare_tau_bench_retail",
    "prepare_tau_bench_airline",
    "prepare_tau_bench_telecom",
    "prepare_tau2_bench_retail",
    "prepare_tau2_bench_airline",
    "prepare_tau2_bench_telecom",
]
