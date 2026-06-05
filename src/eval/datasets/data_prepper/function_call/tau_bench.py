from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.agent_bench.tau_specs import TAU_BENCH_SPECS, TauBenchSpec
from src.eval.agent_bench.tasks import load_tau_tasks, require_tau_v3_source, tau_v2_data_root
from src.eval.agent_bench.tau_official import tau_domain_info
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALL_REGISTRY

from ..data_utils import write_jsonl


def _prepare_tau_dataset(
    output_root: Path,
    *,
    spec: TauBenchSpec,
    split: str,
) -> list[Path]:
    task_split = _resolve_tau_task_split(spec, split)
    if spec.benchmark_version == "tau_v3":
        require_tau_v3_source(spec.dataset_name)
    env_kwargs = _tau_env_kwargs(spec)
    domain_info = tau_domain_info(spec.domain, env_kwargs=env_kwargs or None)
    rows = [
        _row_from_tau_task(
            row,
            spec=spec,
            tools=domain_info.tools,
            policy=domain_info.policy,
            env_kwargs=env_kwargs,
        )
        for row in load_tau_tasks(
            task_set=spec.task_set_name,
            domain=spec.domain,
            split=task_split,
            benchmark_version=spec.benchmark_version,
        )
    ]
    if not rows:
        raise ValueError(f"{spec.dataset_name} did not yield any TAU task rows from {tau_v2_data_root()}")
    target = output_root / f"{spec.dataset_name}.jsonl"
    write_jsonl(target, rows)
    return [target]


def _row_from_tau_task(
    row: dict[str, Any],
    *,
    spec: TauBenchSpec,
    tools: list[dict[str, Any]],
    policy: str,
    env_kwargs: dict[str, Any],
) -> dict[str, Any]:
    task_payload = dict(row.get("task") or {})
    task_id = str(row.get("task_id") or task_payload.get("id") or f"{spec.dataset_name}_{row.get('index', 0)}")
    instruction = str(row.get("instruction") or task_payload.get("ticket") or "")
    metadata = {
        "source_format": "official_tau_manifest",
        "dataset_name": spec.dataset_name,
        "task_id": task_id,
        "task_set": spec.task_set_name,
        "task_split": row.get("task_split"),
        "domain": spec.domain,
        "index": int(row.get("index") or 0),
        "task": task_payload,
        "benchmark_version": spec.benchmark_version,
        "tau_policy": policy,
        "env_kwargs": env_kwargs,
    }
    if spec.retrieval_config:
        metadata["retrieval_config"] = spec.retrieval_config
    prepared_row = {
        "task_id": task_id,
        "instruction": instruction,
        "messages": [],
        "tools": [dict(tool) for tool in tools],
        "expected_tool_calls": [],
        "env": {
            "type": "tau_official",
            "domain": spec.domain,
            "task_set": spec.task_set_name,
            "task_split": row.get("task_split"),
            "benchmark_version": spec.benchmark_version,
            "policy": policy,
            "env_kwargs": env_kwargs,
        },
        "scorer": {
            "type": "tau_official",
            "benchmark_version": spec.benchmark_version,
        },
        "metadata": metadata,
        "domain": spec.domain,
        "task_set": spec.task_set_name,
        "task_split": row.get("task_split"),
        "index": int(row.get("index") or 0),
        "task": task_payload,
        "benchmark_version": spec.benchmark_version,
        "tau_policy": policy,
        "env_kwargs": env_kwargs,
    }
    if spec.retrieval_config:
        prepared_row["retrieval_config"] = spec.retrieval_config
        prepared_row["env"]["retrieval_config"] = spec.retrieval_config
    return prepared_row


def _register_tau_bench_preparer(spec: TauBenchSpec):
    def prepare_tau_bench(output_root: Path, split: str = spec.prep_split) -> list[Path]:
        return _prepare_tau_dataset(output_root, spec=spec, split=split)

    prepare_tau_bench.__name__ = f"prepare_{spec.dataset_name}"
    prepare_tau_bench.__qualname__ = prepare_tau_bench.__name__
    split_label = spec.task_split or "all"
    prepare_tau_bench.__doc__ = f"Prepare {spec.dataset_name} official {split_label} task set."
    return FUNCTION_CALL_REGISTRY.register(spec.dataset_name)(prepare_tau_bench)


def _resolve_tau_task_split(spec: TauBenchSpec, requested_split: str | None) -> str | None:
    requested = str(requested_split or "").strip()
    if spec.task_split is None:
        if requested in {"", "base", "all"}:
            return None
        raise ValueError(f"{spec.dataset_name} uses an official task set without splits")
    if requested in {"", spec.task_split}:
        return spec.task_split
    raise ValueError(f"{spec.dataset_name} only provides official task split {spec.task_split!r}")


def _tau_env_kwargs(spec: TauBenchSpec) -> dict[str, Any]:
    if not spec.retrieval_config:
        return {}
    return {"retrieval_variant": spec.retrieval_config}


for _spec in TAU_BENCH_SPECS:
    globals()[f"prepare_{_spec.dataset_name}"] = _register_tau_bench_preparer(_spec)


__all__ = [
    *(f"prepare_{spec.dataset_name}" for spec in TAU_BENCH_SPECS),
]
