from __future__ import annotations

"""Shared helpers for field-oriented benchmark runners."""

import os
from typing import Any, Mapping, Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.execution_plan import AvgKExecutionPlan, avg_k_metric_key
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


def build_plan_task_details(plan: AvgKExecutionPlan, *, cot_mode: str) -> dict[str, object]:
    return {
        "cot_mode": cot_mode,
        "avg_k": plan.avg_k,
        "sample_size": plan.sample_size,
        "avg_repeat_count": plan.repeat_count,
        "effective_sample_count": plan.effective_sample_count,
    }


def rwkv_rs_cot_mode_name(cot_mode: CoTMode | str) -> str:
    if isinstance(cot_mode, CoTMode):
        resolved = cot_mode
    else:
        resolved = CoTMode(str(cot_mode))
    if resolved is CoTMode.NO_COT:
        return "NoCoT"
    if resolved is CoTMode.FAKE_COT:
        return "FakeCoT"
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
) -> dict[str, object]:
    normalized_pass_ks = sorted(
        {
            int(item)
            for item in (pass_ks or ())
            if int(item) > 0
        }
    )
    return {
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


__all__ = [
    "build_avg_k_metrics",
    "build_plan_task_details",
    "build_task_sampling_config",
    "rwkv_rs_cot_mode_name",
    "set_task_env",
]
