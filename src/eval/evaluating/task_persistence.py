from __future__ import annotations

"""Task persistence helpers aligned with rwkv-rs evaluating run modes."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from src.db.eval_db_service import EvalDbService, ResumeContext


class RunMode(str, Enum):
    """Scheduler/runner task execution policy."""

    AUTO = "auto"
    NEW = "new"
    RESUME = "resume"
    RERUN = "rerun"

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

    if requested_mode is RunMode.RERUN:
        ctx = service.get_resume_context(
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            job_name=job_name,
            sampling_config=sampling_config,
            force_new_task=True,
        )
        task_id = service.create_task_from_context(
            ctx=ctx,
            job_name=job_name,
            dataset=dataset,
            model=model,
            is_param_search=is_param_search,
            sampling_config=sampling_config,
        )
        return TaskExecutionState(task_id=task_id, run_mode=RunMode.RERUN, resume_context=ctx)

    ctx = service.get_resume_context(
        dataset=dataset,
        model=model,
        is_param_search=is_param_search,
        job_name=job_name,
        sampling_config=sampling_config,
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
        sampling_config=sampling_config,
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
]
