from __future__ import annotations

"""Shared helpers for field-oriented benchmark runners."""

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.datasets.snapshot import bind_resume_identity
from src.eval.execution_plan import AvgKExecutionPlan, avg_k_metric_key, build_avg_k_execution_plan
from src.eval.k_values import NumericK, max_generation_k
from src.eval.metrics.at_k import compute_avg_at_k


def set_task_env(task_id: str) -> None:
    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id


def build_avg_k_metrics(
    rows: list[tuple[int, int, bool]],
    *,
    avg_k: float,
    primary_name: str,
    primary_value: float,
) -> dict[str, float]:
    avg_metric_name = avg_k_metric_key(avg_k)
    avg_metrics = compute_avg_at_k(rows, (avg_k,))
    return {
        primary_name: primary_value,
        avg_metric_name: avg_metrics.get(avg_metric_name, primary_value),
    }


@dataclass(frozen=True, slots=True)
class ConfiguredKPlan:
    pass_k: tuple[int, ...]
    avg_k: tuple[NumericK, ...]
    report_pass_k: tuple[int, ...]
    report_avg_k: tuple[NumericK, ...]
    sample_limit: int | None
    samples_per_task: int
    plan: AvgKExecutionPlan


def resolve_configured_k_plan(
    *,
    slug: str,
    model_name: str,
    dataset_len: int,
    args: Any,
    default_pass_k: tuple[int, ...] = (),
    default_avg_k: tuple[NumericK, ...] = (),
) -> ConfiguredKPlan:
    from src.eval.benchmark_config import resolve_benchmark_model_config

    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    raw_pass_k = getattr(args, "pass_k", None)
    pass_k = tuple(int(item) for item in raw_pass_k) if raw_pass_k else (
        config.pass_k if config is not None and config.pass_k is not None else default_pass_k
    )
    raw_avg_k = getattr(args, "avg_k", None)
    avg_k = tuple(raw_avg_k) if raw_avg_k else (
        config.avg_k if config is not None and config.avg_k is not None else default_avg_k
    )
    report_pass_k = (
        config.report_pass_k
        if config is not None and config.report_pass_k is not None
        else pass_k
    )
    report_avg_k = (
        config.report_avg_k
        if config is not None and config.report_avg_k is not None
        else avg_k
    )
    cli_sample_limit = getattr(args, "max_samples", None)
    sample_limit = (
        int(cli_sample_limit)
        if cli_sample_limit is not None
        else (config.max_samples if config is not None else None)
    )
    sample_count = dataset_len
    if sample_limit is not None and sample_limit > 0:
        sample_count = min(dataset_len, sample_limit)
    sample_indices = tuple(range(max(0, sample_count)))
    samples_per_task = max(max_generation_k(pass_k), max_generation_k(avg_k), 1)
    plan_avg_k = _primary_avg_k(avg_k, samples_per_task)
    if avg_k:
        plan = build_avg_k_execution_plan(str(slug), sample_count, avg_k=float(plan_avg_k))
    else:
        plan = AvgKExecutionPlan(
            avg_k=float(plan_avg_k),
            repeat_count=samples_per_task,
            sample_indices=sample_indices,
        )
    return ConfiguredKPlan(
        pass_k=tuple(pass_k),
        avg_k=tuple(avg_k),
        report_pass_k=tuple(report_pass_k),
        report_avg_k=tuple(report_avg_k),
        sample_limit=sample_limit,
        samples_per_task=samples_per_task,
        plan=plan,
    )


def _primary_avg_k(avg_k: tuple[NumericK, ...], fallback: int) -> NumericK:
    integer_ks = [
        item
        for item in avg_k
        if isinstance(item, int) and not isinstance(item, bool) and item > 0
    ]
    if integer_ks:
        return max(integer_ks)
    if avg_k:
        return avg_k[-1]
    return max(1, int(fallback))


def build_plan_task_details(
    plan: AvgKExecutionPlan,
    *,
    cot_mode: str,
    prompt_profile: str = "normal",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "cot_mode": cot_mode,
        "avg_k": plan.avg_k,
        "sample_size": plan.sample_size,
        "avg_repeat_count": plan.repeat_count,
        "effective_sample_count": plan.effective_sample_count,
    }
    if prompt_profile != "normal":
        payload["prompt_profile"] = prompt_profile
    return payload


def rwkv_rs_cot_mode_name(cot_mode: CoTMode | str) -> str:
    if isinstance(cot_mode, CoTMode):
        resolved = cot_mode
    else:
        resolved = CoTMode(str(cot_mode))
    if resolved is CoTMode.NO_COT:
        return "NoCoT"
    return "CoT"


def build_task_sampling_config(
    *,
    cot_mode: CoTMode | str,
    avg_k: float,
    sampling_config: Mapping[str, Any] | None = None,
    pass_ks: Sequence[int] | None = None,
    n_shot: int = 0,
    sample_limit: int | None = None,
    effective_sample_count: int,
    judger_model_name: str | None = None,
    checker_model_name: str | None = None,
    prompt_profile: str = "normal",
    dataset_snapshot: Mapping[str, Any] | None = None,
    protocol_bundle: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    normalized_pass_ks = sorted(
        {
            int(item)
            for item in (pass_ks or ())
            if int(item) > 0
        }
    )
    payload: dict[str, object] = {
        "cot_mode": rwkv_rs_cot_mode_name(cot_mode),
        "n_shot": int(n_shot),
        "avg_k": float(avg_k),
        "sample_limit": int(sample_limit) if sample_limit is not None else None,
        "effective_sample_count": int(effective_sample_count),
        "pass_ks": normalized_pass_ks,
        "sampling_config": dict(sampling_config or {}),
        "judger_model_name": judger_model_name,
        "checker_model_name": checker_model_name,
    }
    if prompt_profile != "normal":
        payload["prompt_profile"] = str(prompt_profile or "normal")
    if dataset_snapshot is not None:
        payload["dataset_snapshot"] = dict(dataset_snapshot)
    if protocol_bundle is not None:
        payload["protocol_bundle"] = dict(protocol_bundle)
    return bind_resume_identity(payload)


__all__ = [
    "build_avg_k_metrics",
    "build_plan_task_details",
    "build_task_sampling_config",
    "ConfiguredKPlan",
    "resolve_configured_k_plan",
    "rwkv_rs_cot_mode_name",
    "set_task_env",
]
