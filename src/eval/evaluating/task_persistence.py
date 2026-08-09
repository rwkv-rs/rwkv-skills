"""Task persistence helpers aligned with rwkv-rs evaluating run modes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from src.db.eval_db_service import EvalDbService, ResumeContext
from src.eval.datasets.snapshot import (
    bind_resume_identity,
    validate_runtime_attestation_provenance,
)


STRICT_RUNTIME_PROVENANCE_ENV = "RWKV_STRICT_RUNTIME_PROVENANCE_JSON"


class RunMode(str, Enum):
    """Scheduler/runner task execution policy."""

    AUTO = "auto"
    NEW = "new"
    RESUME = "resume"
    RERUN = "rerun"
    FRESH = "fresh"

    @classmethod
    def parse(cls, value: str | None) -> "RunMode":
        raw = (value or cls.AUTO.value).strip().lower()
        for mode in cls:
            if mode.value == raw:
                return mode
        supported = ", ".join(mode.value for mode in cls)
        raise ValueError(f"unsupported run mode `{value}`; expected one of: {supported}")


@dataclass(frozen=True, slots=True)
class TaskExecutionState:
    task_id: str
    run_mode: RunMode
    resume_context: ResumeContext

    @property
    def skip_keys(self) -> set[tuple[int, int, int]]:
        return set(self.resume_context.completed_keys)


def current_run_mode(env: Mapping[str, str] | None = None) -> RunMode:
    source = env if env is not None else os.environ
    explicit = source.get("RWKV_EVAL_RUN_MODE")
    if explicit:
        return RunMode.parse(explicit)
    if source.get("RWKV_SCHEDULER_OVERWRITE") == "1":
        return RunMode.RERUN
    return RunMode.AUTO


def prepare_task_execution(
    *,
    service: EvalDbService,
    dataset: str,
    model: str,
    is_param_search: bool,
    job_name: str | None,
    sampling_config: dict[str, Any] | None = None,
    run_mode: RunMode | str | None = None,
) -> TaskExecutionState:
    requested_mode = run_mode if isinstance(run_mode, RunMode) else RunMode.parse(run_mode)
    if run_mode is None:
        requested_mode = current_run_mode()
    runtime_provenance = _runtime_attestation_provenance_for_model(model)
    if runtime_provenance is not None:
        if sampling_config is None:
            raise ValueError(
                "strict G1i runtime provenance requires a protocol-bound sampling config"
            )
        sampling_config = dict(sampling_config)
        existing = sampling_config.get("runtime_attestation_provenance")
        if existing is not None and existing != runtime_provenance:
            raise ValueError("task config contains conflicting runtime attestation provenance")
        sampling_config["runtime_attestation_provenance"] = runtime_provenance
    bound_sampling_config = (
        bind_resume_identity(sampling_config) if sampling_config is not None else None
    )

    if requested_mode in {RunMode.RERUN, RunMode.FRESH}:
        ctx = service.get_resume_context(
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            job_name=job_name,
            sampling_config=bound_sampling_config,
            force_new_task=True,
        )
        task_id = service.create_task_from_context(
            ctx=ctx,
            job_name=job_name,
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            sampling_config=bound_sampling_config,
        )
        return TaskExecutionState(task_id=task_id, run_mode=requested_mode, resume_context=ctx)

    ctx = service.get_resume_context(
        dataset=dataset,
        model=model,
        is_param_search=is_param_search,
        job_name=job_name,
        sampling_config=bound_sampling_config,
        force_new_task=False,
    )

    if requested_mode is RunMode.NEW:
        if ctx.matching_tasks:
            raise ValueError(
                "run_mode=new refused because matching task(s) already exist: "
                f"{_render_task_match(ctx)}"
            )
    elif requested_mode is RunMode.RESUME:
        if ctx.completed_task_ids:
            raise ValueError(
                "run_mode=resume refused because a matching completed task already exists: "
                f"{_render_task_match(ctx)}"
            )
        if not ctx.resumable_task_ids:
            raise ValueError("run_mode=resume could not find a matching running/failed task")
        if len(ctx.resumable_task_ids) != 1:
            raise ValueError(
                "run_mode=resume is ambiguous because multiple matching running/failed tasks exist: "
                f"{_render_task_match(ctx)}"
            )

    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name=job_name,
        dataset=dataset,
        model=model,
        is_param_search=is_param_search,
        sampling_config=bound_sampling_config,
    )
    effective_mode = requested_mode
    if requested_mode is RunMode.AUTO:
        effective_mode = _auto_effective_mode(ctx)
    return TaskExecutionState(task_id=task_id, run_mode=effective_mode, resume_context=ctx)


def _auto_effective_mode(ctx: ResumeContext) -> RunMode:
    if ctx.task_id is None:
        return RunMode.NEW
    if ctx.can_resume:
        return RunMode.RESUME
    return RunMode.RERUN


def _runtime_attestation_provenance_for_model(
    model: str,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    source = env if env is not None else os.environ
    raw = source.get(STRICT_RUNTIME_PROVENANCE_ENV, "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("strict G1i runtime provenance is not valid JSON") from exc
    validated = validate_runtime_attestation_provenance(payload)
    if validated["model"] != model:
        raise ValueError(
            "strict G1i runtime provenance model does not match task model: "
            f"attested={validated['model']!r} task={model!r}"
        )
    return validated


def _render_task_match(ctx: ResumeContext) -> str:
    if not ctx.matching_tasks:
        return "no matching task"
    return ", ".join(
        f"task_id={task.task_id} status={task.status}"
        for task in ctx.matching_tasks
    )


__all__ = [
    "RunMode",
    "TaskExecutionState",
    "current_run_mode",
    "prepare_task_execution",
    "STRICT_RUNTIME_PROVENANCE_ENV",
]
