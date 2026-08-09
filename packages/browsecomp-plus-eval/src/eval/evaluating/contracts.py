from __future__ import annotations

"""Explicit runtime contracts for unified evaluation execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .task_persistence import RunMode


@dataclass(frozen=True, slots=True)
class TaskSpec:
    run_kind: str
    runner_name: str
    dataset_slug: str
    dataset_path: Path
    benchmark_name: str
    benchmark_split: str
    model_name: str
    model_path: str | None = None
    config_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RunContext:
    job_name: str
    run_mode: RunMode
    run_id: str | None = None
    task_id: str | None = None
    version_id: str | None = None

    def with_task(self, task_id: str, *, version_id: str | None = None) -> "RunContext":
        resolved_version = version_id if version_id is not None else task_id
        return RunContext(
            job_name=self.job_name,
            run_mode=self.run_mode,
            run_id=self.run_id,
            task_id=str(task_id),
            version_id=str(resolved_version),
        )

    def env_overrides(self, *, dataset_slug: str | None = None) -> dict[str, str]:
        env = {
            "RWKV_SKILLS_JOB_NAME": self.job_name,
            "RWKV_EVAL_RUN_MODE": self.run_mode.value,
            "RWKV_SCHEDULER_OVERWRITE": "1" if self.run_mode is RunMode.RERUN else "0",
        }
        if dataset_slug:
            env["RWKV_SKILLS_DATASET_SLUG"] = str(dataset_slug)
        if self.run_id:
            env["RWKV_MAIN_RUN_ID"] = self.run_id
        if self.task_id:
            env["RWKV_SKILLS_TASK_ID"] = self.task_id
        if self.version_id:
            env["RWKV_SKILLS_VERSION_ID"] = self.version_id
        return env


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    kind: str
    task_id: str
    payload: Mapping[str, Any]


__all__ = ["ResultEnvelope", "RunContext", "TaskSpec"]
