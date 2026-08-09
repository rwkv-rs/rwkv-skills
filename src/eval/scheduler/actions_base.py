from __future__ import annotations

"""Shared dataclasses, helpers, and library re-exports for scheduler actions."""

import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from .config import (
    DEFAULT_DB_CONFIG,
    DEFAULT_DISPATCH_POLL_SECONDS,
    DEFAULT_GPU_IDLE_MAX_MEM,
    DEFAULT_INFER_MAX_WORKERS,
    DEFAULT_MODEL_GLOBS,
    DEFAULT_PYTHON,
    DEFAULT_RUN_LOG_DIR,
    REPO_ROOT,
)
from .control import DesiredState, ObservedStatus, SchedulerProgressSnapshot, SchedulerRuntimeControl
from .datasets import DATASET_ROOTS, DATA_OUTPUT_ROOT
from .dataset_utils import safe_slug, split_benchmark_and_split
from .jobs import JOB_CATALOGUE, JobSpec, locate_dataset
from .lease import SchedulerLeaseManager
from .naming import build_run_log_name
from .process import FAILURE_MONITOR, JobFailure, handle_job_failure, launch_job, list_idle_gpus, log_job_event
from .profiler import BatchProfiler
from .queue import QueueItem, build_queue, sort_queue_items
from .question_counts import derive_question_counts
from .remote_slots import parse_remote_model_slots, remote_slot_map
from .backpressure import (
    RemoteBackpressureError,
    RemoteConcurrencyBudget,
    compute_remote_concurrency_budgets,
    fetch_remote_backpressure,
    static_remote_concurrency_budgets,
)
from .state import (
    CompletedKey,
    CompletedRecord,
    RunningEntry,
    ensure_dirs,
    load_running,
    scan_completed_jobs,
    stop_all_jobs,
    stop_job,
    tail_file,
    write_pid_file,
)
from src.eval.benchmark_config import config_path_for_benchmark
from src.eval.evaluating import RunMode
from src.eval.runner_registry import RunnerGroup

_SAMPLE_WORKER_JOB_NAMES = frozenset({"function_bfcl_v3", "function_browsecomp_plus"})
_NO_GENERATION_SLOT_RELEASE_JOBS = frozenset(
    {
        # Math A/B/C uses the same remote endpoint after strategy A reaches
        # ``effective_sample_count``. Releasing the slot at that intermediate
        # point launches a second runner into the same endpoint.
        "free_response",
        "free_response_naive",
        "free_response_judge",
        "free_response_judge_naive",
        "code_human_eval_naive",
        "code_mbpp_naive",
        "code_livecodebench_naive",
    }
)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    base_url: str | None = None
    models: tuple[str, ...] = ()
    api_key: str = ""
    timeout_s: float = 600.0
    max_workers: int = DEFAULT_INFER_MAX_WORKERS
    worker_profile: str = "fixed"
    protocol: str = "openai"
    seed_policy: str = "preserve"
    remote_batch_size: int | None = None
    plain_choice_batch_size: int | None = None
    plain_choice_timeout_s: float | None = None
    sample_workers: int | None = None
    backpressure: bool = True
    backpressure_timeout_s: float = 2.0
    backpressure_pending_high_watermark: int = 0
    budget_min_workers: int = 1


@dataclass(frozen=True, slots=True)
class FunctionCallingConfig:
    prompt_style: str | None = None
    tool_catalog_format: str | None = None
    cot_max_tokens: int | None = None
    decision_max_tokens: int | None = None
    planning_max_tokens: int | None = None
    final_max_tokens: int | None = None
    answer_max_tokens: int | None = None
    judge_max_workers: int | None = None
    history_max_chars: int | None = None
    prompt_max_chars: int | None = None
    long_doc_mode: str | None = None
    tool_router_mode: str | None = None
    tool_router_max_tools: int | None = None
    tool_router_trigger_tool_count: int | None = None
    tool_router_trigger_catalog_chars: int | None = None
    candidate_router_mode: str | None = None
    candidate_router_chunk_tools: int | None = None
    candidate_router_batch_size: int | None = None
    candidate_router_prompt_max_chars: int | None = None
    candidate_router_context_chars: int | None = None
    candidate_router_candidate_max_tokens: int | None = None
    candidate_router_aggregate_max_tokens: int | None = None
    candidate_router_max_candidates: int | None = None
    candidate_router_tool_schema_mode: str | None = None
    candidate_router_evidence_chars: int | None = None
    candidate_router_policy_chars: int | None = None
    max_rounds: int | None = None
    max_steps: int | None = None
    max_tool_errors: int | None = None
    complexfuncbench_disable_response_eval: bool = False
    complexfuncbench_offline_compare: bool = False


@dataclass(frozen=True, slots=True)
class CodingConfig:
    eval_workers: int | None = None
    max_active_runners: int | None = None


@dataclass(frozen=True, slots=True)
class MathConfig:
    judge_max_workers: int | None = None
    prompt_max_chars: int | None = None
    long_doc_mode: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeConfig:
    prompt_max_chars: int | None = None
    long_doc_mode: str | None = None


@dataclass(slots=True)
class QueueOptions:
    log_dir: Path
    pid_dir: Path
    job_order: tuple[str, ...]
    job_priority: tuple[str, ...] | None = None
    model_select: str = "all"
    min_param_b: float | None = None
    max_param_b: float | None = None
    skip_dataset_slugs: tuple[str, ...] = ()
    model_globs: tuple[str, ...] = DEFAULT_MODEL_GLOBS
    only_dataset_slugs: tuple[str, ...] = ()
    model_name_patterns: tuple[re.Pattern[str], ...] = ()
    enable_param_search: bool = False
    run_mode: RunMode = RunMode.AUTO
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    functions: FunctionCallingConfig = field(default_factory=FunctionCallingConfig)
    coding: CodingConfig = field(default_factory=CodingConfig)
    math: MathConfig = field(default_factory=MathConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    distributed_claims: bool = False
    scheduler_node_id: str | None = None
    lease_duration_s: int = 900


@dataclass(slots=True)
class DispatchOptions(QueueOptions):
    run_log_dir: Path = DEFAULT_RUN_LOG_DIR
    dispatch_poll_seconds: int = DEFAULT_DISPATCH_POLL_SECONDS
    gpu_idle_max_mem: int = DEFAULT_GPU_IDLE_MAX_MEM
    skip_missing_dataset: bool = False
    clean_param_swap: bool = False
    batch_cache_path: Path | None = None
    disable_checker: bool = False
    benchmark_config_root: Path | None = None


@dataclass(slots=True)
class StatusOptions:
    pid_dir: Path


@dataclass(slots=True)
class StopOptions:
    pid_dir: Path
    job_ids: tuple[str, ...] = ()
    stop_all: bool = False


@dataclass(slots=True)
class LogsOptions:
    run_log_dir: Path
    pid_dir: Path
    tail_lines: int = 60
    rotate_seconds: int = 15


@dataclass(slots=True)
class DispatcherState:
    """Mutable per-job bookkeeping for a single ``action_dispatch`` loop.

    Consolidates the per-job tracking dicts (plus the cancel flag) that the
    dispatch loop and its helpers thread around, with :meth:`forget` collapsing
    the scattered ``pop`` calls used when a job leaves the pipeline.
    """

    pending_since: dict[str, float] = field(default_factory=dict)
    launch_times: dict[str, float] = field(default_factory=dict)
    job_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    completed_versions: dict[str, str | None] = field(default_factory=dict)
    # ``completed_versions`` can legitimately be empty on the first poll.  Keep
    # that empty snapshot distinct from an uninitialized state so a score that
    # appears after an empty first poll is treated as a current-session result.
    completed_versions_initialized: bool = False
    cooldown_until: dict[str, float] = field(default_factory=dict)
    cancel_requested: bool = False

    def forget(self, job_id: str) -> tuple[dict[str, object], float | None]:
        """Drop all per-job tracking for ``job_id`` once it leaves the pipeline.

        Returns the job's metadata and recorded launch time (``None`` when it
        was never launched) so callers can still build their completion events.
        """
        meta = self.job_metadata.pop(job_id, {})
        start = self.launch_times.pop(job_id, None)
        self.pending_since.pop(job_id, None)
        self.cooldown_until.pop(job_id, None)
        return meta, start


def _read_scheduler_state(
    *,
    pid_dir: Path,
) -> tuple[set[CompletedKey], dict[str, CompletedRecord], dict[str, RunningEntry], dict[str, int]]:
    completed, score_records = scan_completed_jobs()
    running_entries = load_running(pid_dir)
    question_counts = derive_question_counts(score_records)
    return completed, score_records, running_entries, question_counts


def _completed_for_queue(
    *,
    run_mode: RunMode,
    completed: Sequence[CompletedKey] | set[CompletedKey],
    session_completed: Sequence[CompletedKey] | set[CompletedKey] = (),
) -> set[CompletedKey]:
    # ``fresh`` is also a deliberate replacement run.  Existing score rows
    # are historical evidence and must not suppress the requested queue.
    if run_mode in {RunMode.RERUN, RunMode.FRESH}:
        return set(session_completed)
    return set(completed) | set(session_completed)


def _build_pending_queue(
    opts: QueueOptions,
    *,
    completed: set[CompletedKey],
    failed: set[CompletedKey],
    running: Sequence[str] | set[str],
    question_counts: Mapping[str, int],
    job_priority: Mapping[str, int],
) -> list[QueueItem]:
    pending = build_queue(
        model_globs=opts.model_globs,
        job_order=opts.job_order,
        completed=completed,
        failed=failed,
        running=running,
        skip_dataset_slugs=opts.skip_dataset_slugs,
        only_dataset_slugs=opts.only_dataset_slugs,
        model_select=opts.model_select,
        min_param_b=opts.min_param_b,
        max_param_b=opts.max_param_b,
        enable_param_search=opts.enable_param_search,
        model_name_patterns=opts.model_name_patterns,
        infer_base_url=opts.inference.base_url,
        infer_models=opts.inference.models,
    )
    return sort_queue_items(pending, question_counts=question_counts, job_priority=job_priority)


def _reconcile_completed_versions(
    *,
    completed_records: Mapping[str, CompletedRecord],
    state: DispatcherState,
    session_completed: set[CompletedKey],
    now: float,
) -> set[str]:
    current_versions = {job_id: info.version_id for job_id, info in completed_records.items()}
    if not state.completed_versions_initialized:
        state.completed_versions.update(current_versions)
        state.completed_versions_initialized = True
        return set()

    new_completed = {
        job_id for job_id, version_id in current_versions.items() if state.completed_versions.get(job_id) != version_id
    }
    if new_completed:
        for job_id in sorted(new_completed):
            info = completed_records[job_id]
            meta, start = state.forget(job_id)
            runtime = now - start if start else None
            session_completed.add(info.key)
            payload: dict[str, object] = {
                "job": info.key.job,
                "dataset_slug": info.key.dataset_slug,
                "model_slug": info.key.model_slug,
                "model_name": info.model_name,
                "runtime_s": runtime,
                "is_cot": info.key.is_cot,
            }
            if info.version_id:
                payload["version_id"] = info.version_id
            payload.update(meta)
            log_job_event("job_done", job_id, **payload)
    state.completed_versions.clear()
    state.completed_versions.update(current_versions)
    return new_completed


def _update_cooldown_jobs(
    *,
    previous_running: set[str],
    running_entries: Mapping[str, RunningEntry],
    completed_records: Mapping[str, CompletedRecord],
    state: DispatcherState,
    now: float,
    dispatch_poll_seconds: int,
) -> set[str]:
    ended_jobs = previous_running - set(running_entries.keys())
    for job_id in ended_jobs:
        if job_id not in completed_records:
            state.cooldown_until[job_id] = max(state.cooldown_until.get(job_id, 0.0), now + 2 * dispatch_poll_seconds)
    return {job_id for job_id, until in state.cooldown_until.items() if until > now}


def _mark_pending_jobs(
    *,
    queue: Sequence[QueueItem],
    state: DispatcherState,
    now: float,
) -> None:
    for position, item in enumerate(queue):
        if item.job_id not in state.pending_since:
            state.pending_since[item.job_id] = now
            meta = state.job_metadata.setdefault(item.job_id, {})
            meta.setdefault("job", item.job_name)
            meta.setdefault("dataset_slug", item.dataset_slug)
            meta.setdefault("model_name", item.model_name)
            if item.model_path is not None:
                meta.setdefault("model_path", str(item.model_path))
            if item.is_remote:
                meta.setdefault("infer_base_url", str(item.infer_base_url))
            meta.setdefault("model_slug", item.model_slug)
            payload: dict[str, object] = {
                "job": item.job_name,
                "dataset_slug": item.dataset_slug,
                "model_name": item.model_name or item.model_slug,
                "queue_pos": position,
                "pending": len(queue),
            }
            if item.model_path is not None:
                payload["model_path"] = str(item.model_path)
            if item.is_remote:
                payload["infer_base_url"] = str(item.infer_base_url)
                payload["infer_model"] = str(item.infer_model or item.model_name)
            log_job_event("job_pending", item.job_id, **payload)


def _dispatch_uses_remote_inference(opts: QueueOptions) -> bool:
    return bool(str(opts.inference.base_url or "").strip() and opts.inference.models)


def _distributed_claims_enabled(opts: QueueOptions) -> bool:
    return bool(getattr(opts, "distributed_claims", False))


def _build_lease_manager(opts: QueueOptions) -> SchedulerLeaseManager | None:
    if not _distributed_claims_enabled(opts):
        return None
    return SchedulerLeaseManager(
        node_id=opts.scheduler_node_id,
        lease_duration_s=opts.lease_duration_s,
    )


def _resolve_remote_concurrency_budgets(opts: QueueOptions) -> dict[str, RemoteConcurrencyBudget]:
    if not _dispatch_uses_remote_inference(opts):
        return {}
    if not opts.inference.backpressure:
        budgets = static_remote_concurrency_budgets(
            infer_models=opts.inference.models,
            default_infer_max_workers=opts.inference.max_workers,
            default_remote_batch_size=opts.inference.remote_batch_size,
            reason="static_backpressure_disabled",
            infer_worker_profile=opts.inference.worker_profile,
        )
        _log_remote_budgets(budgets)
        return budgets
    try:
        signals = fetch_remote_backpressure(
            base_url=str(opts.inference.base_url or ""),
            api_key=opts.inference.api_key,
            timeout_s=float(opts.inference.backpressure_timeout_s),
        )
    except RemoteBackpressureError as exc:
        budgets = static_remote_concurrency_budgets(
            infer_models=opts.inference.models,
            default_infer_max_workers=opts.inference.max_workers,
            default_remote_batch_size=opts.inference.remote_batch_size,
            reason="static_backpressure_unavailable",
            infer_worker_profile=opts.inference.worker_profile,
        )
        for budget in budgets.values():
            budget.error = str(exc)
        log_job_event("remote_backpressure_unavailable", "_dispatcher", error=str(exc))
        _log_remote_budgets(budgets)
        return budgets
    budgets = compute_remote_concurrency_budgets(
        infer_models=opts.inference.models,
        backpressure=signals,
        default_infer_max_workers=opts.inference.max_workers,
        default_remote_batch_size=opts.inference.remote_batch_size,
        pending_high_watermark=opts.inference.backpressure_pending_high_watermark,
        min_infer_max_workers=opts.inference.budget_min_workers,
        infer_worker_profile=opts.inference.worker_profile,
    )
    _log_remote_budgets(budgets)
    return budgets


def _log_remote_budgets(budgets: Mapping[str, RemoteConcurrencyBudget]) -> None:
    for budget in budgets.values():
        log_job_event("remote_budget", budget.model_slug, **budget.to_event_payload())


def _lease_meta_for_item(item: QueueItem) -> dict[str, object]:
    payload: dict[str, object] = {
        "job": item.job_name,
        "dataset_slug": item.dataset_slug,
        "model_name": item.model_name or item.model_slug,
        "model_slug": item.model_slug,
    }
    if item.model_path is not None:
        payload["model_path"] = str(item.model_path)
    if item.is_remote:
        payload["infer_base_url"] = str(item.infer_base_url or "")
        payload["infer_model"] = str(item.infer_model or item.model_name or "")
    return payload


def _resolve_available_dispatch_resources(
    opts: DispatchOptions,
    running_entries: Mapping[str, RunningEntry],
    generated_job_ids: Sequence[str] | set[str] = (),
    remote_budgets: Mapping[str, RemoteConcurrencyBudget] | None = None,
) -> list[str]:
    if _dispatch_uses_remote_inference(opts):
        occupied_slot_slugs = _running_remote_slot_slugs(
            running_entries,
            opts.inference.models,
            generated_job_ids=generated_job_ids,
        )
        budgets = remote_budgets or {}
        return [
            f"model:{slot.slot_slug}"
            for slot in parse_remote_model_slots(opts.inference.models)
            if slot.slot_slug not in occupied_slot_slugs
            and (slot.slot_slug not in budgets or budgets[slot.slot_slug].launch_allowed)
        ]

    idle_gpus = list_idle_gpus(opts.gpu_idle_max_mem)
    running_gpus = {entry.gpu for entry in running_entries.values() if entry.gpu}
    return [gpu for gpu in idle_gpus if gpu not in running_gpus]


def _running_remote_slot_slugs(
    running_entries: Mapping[str, RunningEntry],
    infer_models: Sequence[str],
    generated_job_ids: Sequence[str] | set[str] = (),
) -> set[str]:
    slots = parse_remote_model_slots(infer_models)
    slot_slugs = {slot.slot_slug for slot in slots}
    model_to_slots: dict[str, list[str]] = {}
    for slot in slots:
        model_to_slots.setdefault(slot.model_slug, []).append(slot.slot_slug)
    if not slot_slugs:
        return set()
    generated = set(generated_job_ids)
    occupied: set[str] = set()
    for job_id, entry in running_entries.items():
        if job_id in generated:
            continue
        resource_slot = _remote_resource_model_slug(entry.gpu or "")
        if resource_slot in slot_slugs:
            occupied.add(resource_slot)
            continue
        if resource_slot is not None:
            continue
        for model_slug, matching_slots in model_to_slots.items():
            if job_id.endswith(f"_{model_slug}"):
                for slot_slug in matching_slots:
                    if slot_slug not in occupied:
                        occupied.add(slot_slug)
                        break
                break
    return occupied


def _generated_running_job_ids(
    *,
    running_entries: Mapping[str, RunningEntry],
    job_metadata: Mapping[str, Mapping[str, object]],
) -> set[str]:
    generated: set[str] = set()
    for job_id in running_entries:
        meta = job_metadata.get(job_id)
        if not meta:
            continue
        if bool(meta.get("generation_slot_released")):
            generated.add(job_id)
            continue
        job_name = str(meta.get("job") or "").strip()
        model_name = str(meta.get("model_name") or "").strip()
        dataset_slug = str(meta.get("dataset_slug") or "").strip()
        if not job_name or not model_name or not dataset_slug:
            continue
        if job_name in _NO_GENERATION_SLOT_RELEASE_JOBS:
            continue
        try:
            if _job_generation_is_complete(
                job_name=job_name,
                model_name=model_name,
                dataset_slug=dataset_slug,
            ):
                generated.add(job_id)
                if isinstance(meta, dict):
                    meta["generation_slot_released"] = True
        except Exception as exc:
            log_job_event(
                "generation_progress_probe_failed",
                job_id,
                job=job_name,
                dataset_slug=dataset_slug,
                model_name=model_name,
                error=type(exc).__name__,
                message=str(exc),
            )
    return generated


def _expected_completion_count_from_sampling(sampling_config: object) -> int | None:
    if not isinstance(sampling_config, Mapping):
        return None
    raw = sampling_config.get("effective_sample_count")
    if raw is None:
        return None
    try:
        expected = int(raw)
    except (TypeError, ValueError):
        return None
    return expected if expected > 0 else None


def _job_generation_is_complete(
    *,
    job_name: str,
    model_name: str,
    dataset_slug: str,
) -> bool:
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService

    benchmark_name, benchmark_split = split_benchmark_and_split(dataset_slug)
    init_db(DEFAULT_DB_CONFIG)
    progress = EvalDbService().get_latest_task_generation_progress(
        evaluator=job_name,
        model_name=model_name,
        benchmark_name=benchmark_name,
        benchmark_split=benchmark_split,
    )
    if not progress or bool(progress.get("has_score")):
        return False
    expected = _expected_completion_count_from_sampling(progress.get("sampling_config"))
    if expected is None:
        return False
    try:
        completed = int(progress.get("completed_completions") or 0)
    except (TypeError, ValueError):
        return False
    if completed < expected:
        return False
    log_job_event(
        "generation_complete_release_slot",
        f"{job_name}__{dataset_slug}",
        task_id=str(progress.get("task_id") or ""),
        job=job_name,
        dataset_slug=dataset_slug,
        model_name=model_name,
        completed_completions=completed,
        expected_completions=expected,
    )
    return True


def _remote_resource_model_slug(resource: str) -> str | None:
    if not resource.startswith("model:"):
        return None
    model_slug = resource.removeprefix("model:").strip()
    return model_slug or None


def _launch_target_label(item: QueueItem, resource: str) -> str:
    if item.is_remote:
        return resource
    return f"cuda:{resource}"


def _job_priority_map(job_order: Sequence[str] | None) -> dict[str, int]:
    if not job_order:
        return {}
    return {name: idx for idx, name in enumerate(job_order)}


def _print_queue_summary(pending: Sequence[QueueItem], running: Mapping[str, RunningEntry]) -> None:
    if not pending:
        print("🟢 没有需要调度的任务")
        if running:
            print(f"ℹ️  当前运行 {len(running)} 个任务")
        return
    print(f"待调度任务：{len(pending)}")
    for idx, item in enumerate(pending, start=1):
        model_label = item.model_name or (item.model_path.name if item.model_path is not None else item.model_slug)
        print(f"[{idx:02d}] {item.job_id} | {model_label} | {item.dataset_slug}")
    if running:
        print(f"ℹ️  当前运行 {len(running)} 个任务")


def _write_stdout(text: str) -> bool:
    return sys.stdout.write(text) >= 0


__all__ = [
    "QueueOptions",
    "InferenceConfig",
    "FunctionCallingConfig",
    "CodingConfig",
    "MathConfig",
    "KnowledgeConfig",
    "DispatchOptions",
    "StatusOptions",
    "StopOptions",
    "LogsOptions",
    "DispatcherState",
    "RunMode",
    "RunnerGroup",
    "QueueItem",
    "JobSpec",
    "JobFailure",
    "CompletedKey",
    "CompletedRecord",
    "RunningEntry",
    "SchedulerLeaseManager",
    "SchedulerProgressSnapshot",
    "SchedulerRuntimeControl",
    "DesiredState",
    "ObservedStatus",
    "RemoteBackpressureError",
    "RemoteConcurrencyBudget",
    "BatchProfiler",
    "FAILURE_MONITOR",
    "DATASET_ROOTS",
    "DATA_OUTPUT_ROOT",
    "REPO_ROOT",
    "DEFAULT_DB_CONFIG",
    "DEFAULT_DISPATCH_POLL_SECONDS",
    "DEFAULT_GPU_IDLE_MAX_MEM",
    "DEFAULT_INFER_MAX_WORKERS",
    "DEFAULT_MODEL_GLOBS",
    "DEFAULT_PYTHON",
    "DEFAULT_RUN_LOG_DIR",
    "_SAMPLE_WORKER_JOB_NAMES",
    "_NO_GENERATION_SLOT_RELEASE_JOBS",
    "scan_completed_jobs",
    "load_running",
    "derive_question_counts",
    "sort_queue_items",
    "build_queue",
    "build_run_log_name",
    "config_path_for_benchmark",
    "ensure_dirs",
    "stop_all_jobs",
    "stop_job",
    "tail_file",
    "write_pid_file",
    "handle_job_failure",
    "launch_job",
    "list_idle_gpus",
    "log_job_event",
    "locate_dataset",
    "safe_slug",
    "split_benchmark_and_split",
    "parse_remote_model_slots",
    "remote_slot_map",
    "compute_remote_concurrency_budgets",
    "fetch_remote_backpressure",
    "static_remote_concurrency_budgets",
    "_read_scheduler_state",
    "_completed_for_queue",
    "_build_pending_queue",
    "_reconcile_completed_versions",
    "_update_cooldown_jobs",
    "_mark_pending_jobs",
    "_dispatch_uses_remote_inference",
    "_distributed_claims_enabled",
    "_build_lease_manager",
    "_resolve_remote_concurrency_budgets",
    "_log_remote_budgets",
    "_lease_meta_for_item",
    "_resolve_available_dispatch_resources",
    "_running_remote_slot_slugs",
    "_generated_running_job_ids",
    "_expected_completion_count_from_sampling",
    "_job_generation_is_complete",
    "_remote_resource_model_slug",
    "_launch_target_label",
    "_job_priority_map",
    "_print_queue_summary",
    "_write_stdout",
]
