from __future__ import annotations

"""User-facing actions backed by the scheduler library."""

import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from .config import (
    DEFAULT_DISPATCH_POLL_SECONDS,
    DEFAULT_GPU_IDLE_MAX_MEM,
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
    infer_base_url: str | None = None
    infer_models: tuple[str, ...] = ()
    infer_api_key: str = ""
    infer_timeout_s: float = 600.0
    infer_max_workers: int = 32
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
    max_concurrent_jobs: int | None = None


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
    if run_mode is RunMode.RERUN:
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
        infer_base_url=opts.infer_base_url,
        infer_models=opts.infer_models,
    )
    return sort_queue_items(pending, question_counts=question_counts, job_priority=job_priority)


def _reconcile_completed_versions(
    *,
    completed_records: Mapping[str, CompletedRecord],
    completed_versions: dict[str, str | None],
    job_metadata: dict[str, dict[str, object]],
    launch_times: dict[str, float],
    pending_since: dict[str, float],
    session_completed: set[CompletedKey],
    cooldown_until: dict[str, float],
    now: float,
) -> set[str]:
    current_versions = {job_id: info.version_id for job_id, info in completed_records.items()}
    if not completed_versions:
        completed_versions.update(current_versions)
        return set()

    new_completed = {
        job_id for job_id, version_id in current_versions.items() if completed_versions.get(job_id) != version_id
    }
    if new_completed:
        for job_id in sorted(new_completed):
            info = completed_records[job_id]
            meta = job_metadata.pop(job_id, {})
            start = launch_times.pop(job_id, None)
            runtime = now - start if start else None
            pending_since.pop(job_id, None)
            session_completed.add(info.key)
            cooldown_until.pop(job_id, None)
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
    completed_versions.clear()
    completed_versions.update(current_versions)
    return new_completed


def _update_cooldown_jobs(
    *,
    previous_running: set[str],
    running_entries: Mapping[str, RunningEntry],
    completed_records: Mapping[str, CompletedRecord],
    cooldown_until: dict[str, float],
    now: float,
    dispatch_poll_seconds: int,
) -> set[str]:
    ended_jobs = previous_running - set(running_entries.keys())
    for job_id in ended_jobs:
        if job_id not in completed_records:
            cooldown_until[job_id] = max(cooldown_until.get(job_id, 0.0), now + 2 * dispatch_poll_seconds)
    return {job_id for job_id, until in cooldown_until.items() if until > now}


def _mark_pending_jobs(
    *,
    queue: Sequence[QueueItem],
    pending_since: dict[str, float],
    job_metadata: dict[str, dict[str, object]],
    now: float,
) -> None:
    for position, item in enumerate(queue):
        if item.job_id not in pending_since:
            pending_since[item.job_id] = now
            meta = job_metadata.setdefault(item.job_id, {})
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
    return bool(str(opts.infer_base_url or "").strip() and opts.infer_models)


def _distributed_claims_enabled(opts: QueueOptions) -> bool:
    return bool(getattr(opts, "distributed_claims", False))


def _build_lease_manager(opts: QueueOptions) -> SchedulerLeaseManager | None:
    if not _distributed_claims_enabled(opts):
        return None
    return SchedulerLeaseManager(
        node_id=opts.scheduler_node_id,
        lease_duration_s=opts.lease_duration_s,
    )


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
) -> list[str]:
    running_count = len(running_entries)
    if _dispatch_uses_remote_inference(opts):
        limit = opts.max_concurrent_jobs if opts.max_concurrent_jobs is not None else 1
        available = max(0, int(limit) - running_count)
        return [f"slot-{index + 1}" for index in range(available)]

    idle_gpus = list_idle_gpus(opts.gpu_idle_max_mem)
    running_gpus = {entry.gpu for entry in running_entries.values() if entry.gpu}
    available = [gpu for gpu in idle_gpus if gpu not in running_gpus]
    if opts.max_concurrent_jobs is not None:
        remaining = max(0, int(opts.max_concurrent_jobs) - running_count)
        available = available[:remaining]
    return available


def _launch_target_label(item: QueueItem, resource: str) -> str:
    if item.is_remote:
        return resource
    return f"cuda:{resource}"


def _launch_queue_items(
    *,
    opts: DispatchOptions,
    queue: Sequence[QueueItem],
    available_resources: Sequence[str],
    question_counts: Mapping[str, int],
    batch_profiler: BatchProfiler,
    pending_since: dict[str, float],
    launch_times: dict[str, float],
    job_metadata: dict[str, dict[str, object]],
    lease_manager: SchedulerLeaseManager | None,
    claimed_job_ids: set[str],
) -> None:
    remote_mode = _dispatch_uses_remote_inference(opts)
    resource_label = "Free slots" if remote_mode else "Idle GPUs"
    print(f"🧮 Pending={len(queue)} | {resource_label}={', '.join(available_resources)}")

    queue_index = 0
    for resource in available_resources:
        item: QueueItem | None = None
        while queue_index < len(queue):
            candidate = queue[queue_index]
            queue_index += 1
            if lease_manager is not None and not lease_manager.claim(
                candidate.job_id,
                lease_meta=_lease_meta_for_item(candidate),
            ):
                log_job_event("job_claim_conflict", candidate.job_id, worker_slot=resource, **_lease_meta_for_item(candidate))
                continue
            item = candidate
            break
        if item is None:
            break

        job = JOB_CATALOGUE[item.job_name]
        dataset_slug = item.dataset_slug
        try:
            dataset_path = locate_dataset(dataset_slug, search=DATASET_ROOTS, output_root=DATA_OUTPUT_ROOT)
        except FileNotFoundError as exc:
            if opts.skip_missing_dataset:
                print(f"⚠️  {item.job_id} 缺少数据集：{exc}. 已跳过。")
                if lease_manager is not None:
                    lease_manager.release((item.job_id,))
                log_job_event(
                    "job_skip",
                    item.job_id,
                    reason="missing_dataset",
                    dataset_slug=dataset_slug,
                )
                continue
            log_job_event(
                "job_error",
                item.job_id,
                reason="missing_dataset",
                dataset_slug=dataset_slug,
            )
            if lease_manager is not None:
                lease_manager.release((item.job_id,))
            raise

        log_relpath = build_run_log_name(item.model_name or item.model_slug, dataset_slug, is_cot=job.is_cot)
        console_log_path = _allocate_console_log_path(opts.run_log_dir, log_relpath)
        pid_path = opts.pid_dir / f"{item.job_id}.pid"
        item.dataset_path = dataset_path

        if pid_path.exists():
            lines = pid_path.read_text().splitlines()
            if lines:
                try:
                    existing_pid = int(lines[0])
                except ValueError:
                    existing_pid = None
                else:
                    if existing_pid and existing_pid > 0:
                        print(f"ℹ️  {item.job_id} 已有运行中的 PID({existing_pid})，跳过")
                        log_job_event(
                            "job_skip",
                            item.job_id,
                            reason="already_running",
                            pid=existing_pid,
                        )
                        if lease_manager is not None:
                            lease_manager.release((item.job_id,))
                        continue
            pid_path.unlink(missing_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "RWKV_SKILLS_JOB_ID": item.job_id,
                "RWKV_SKILLS_JOB_NAME": item.job_name,
                "RWKV_SKILLS_MODEL_NAME": str(item.model_name or item.model_slug),
                "RWKV_SKILLS_DATASET": str(dataset_path),
                "RWKV_SKILLS_DATASET_SLUG": dataset_slug,
                "RWKV_TASK_DESC": f"job={item.job_name}, dataset={dataset_slug}",
                "RUN_LOG_DIR": str(opts.log_dir),
                "RUN_RUN_LOG_DIR": str(opts.run_log_dir),
                "RWKV_EVAL_RUN_MODE": opts.run_mode.value,
                "RWKV_SCHEDULER_OVERWRITE": "1" if opts.run_mode is RunMode.RERUN else "0",
            }
        )
        if item.model_path is not None:
            env["RWKV_SKILLS_MODEL_PATH"] = str(item.model_path)
        if item.is_remote:
            env["RWKV_SKILLS_INFER_BASE_URL"] = str(item.infer_base_url or "")
            env["RWKV_SKILLS_INFER_MODEL"] = str(item.infer_model or item.model_name or "")
            if opts.infer_api_key:
                env["RWKV_SKILLS_INFER_API_KEY"] = opts.infer_api_key
        if opts.disable_checker:
            env["RWKV_SKILLS_DISABLE_CHECKER"] = "1"

        questions = question_counts.get(dataset_slug)

        batch_size = None
        if not item.is_remote and item.model_path is not None:
            batch_size = batch_profiler.determine_batch_size(
                job=job,
                job_id=item.job_id,
                gpu=resource,
                dataset_path=dataset_path,
                model_path=item.model_path,
                model_slug=item.model_slug,
                env=env,
                dataset_questions=questions,
            )

        extra_args = item.extra_args
        if opts.run_mode is RunMode.RERUN and item.job_name == "param_search_select" and "--overwrite" not in extra_args:
            extra_args = extra_args + ("--overwrite",)

        command = build_command(
            job,
            item,
            dataset_path,
            None if item.is_remote else f"cuda:{resource}",
            batch_size=batch_size,
            extra_args=extra_args,
            infer_api_key=opts.infer_api_key,
            infer_timeout_s=opts.infer_timeout_s,
            infer_max_workers=opts.infer_max_workers,
        )
        _backup_run_config(
            model_name=item.model_name or item.model_slug,
            model_path=item.model_path,
            infer_base_url=item.infer_base_url,
            infer_model=item.infer_model,
            dataset_slug=dataset_slug,
            dataset_path=dataset_path,
            job_name=item.job_name,
            job_id=item.job_id,
            batch_size=batch_size,
            gpu=(None if item.is_remote else f"cuda:{resource}"),
            log_path=console_log_path,
        )
        print(f"🚀 Launch {item.job_id} -> {_launch_target_label(item, resource)}")
        print(f"    Dataset: {dataset_path}")
        print(f"    Console: {console_log_path}")
        print(f"    Cmd: {' '.join(command)}")
        meta = job_metadata.setdefault(item.job_id, {})
        meta.update(
            job=item.job_name,
            dataset_slug=dataset_slug,
            dataset_path=str(dataset_path),
            model_name=item.model_name or item.model_slug,
            model_slug=item.model_slug,
            console_log_path=str(console_log_path),
        )
        if item.model_path is not None:
            meta["model_path"] = str(item.model_path)
        if item.is_remote:
            meta["infer_base_url"] = str(item.infer_base_url)
            meta["infer_model"] = str(item.infer_model or item.model_name)
        else:
            meta["gpu"] = resource

        try:
            process = launch_job(
                item.job_id,
                command,
                cwd=REPO_ROOT,
                log_path=console_log_path,
                env=env,
            )
        except Exception:
            if lease_manager is not None:
                lease_manager.release((item.job_id,))
            raise
        claimed_job_ids.add(item.job_id)
        try:
            log_reference = str(console_log_path.relative_to(opts.run_log_dir))
        except ValueError:
            log_reference = str(console_log_path)
        write_pid_file(opts.pid_dir, item.job_id, process.pid, (None if item.is_remote else resource), log_reference)
        launch_times[item.job_id] = time.time()
        pending_start = pending_since.pop(item.job_id, None)
        wait_s = time.time() - pending_start if pending_start else None
        payload: dict[str, object] = {
            "job": item.job_name,
            "dataset_slug": dataset_slug,
            "dataset_path": str(dataset_path),
            "model_name": item.model_name or item.model_slug,
            "pid": process.pid,
            "wait_s": wait_s,
        }
        if item.model_path is not None:
            payload["model_path"] = str(item.model_path)
            payload["gpu"] = f"cuda:{resource}"
        if item.is_remote:
            payload["infer_base_url"] = str(item.infer_base_url)
            payload["infer_model"] = str(item.infer_model or item.model_name)
            payload["worker_slot"] = resource
        log_job_event("job_launch", item.job_id, **payload)


def action_queue(opts: QueueOptions) -> list[QueueItem]:
    completed, score_records, running_entries, question_counts = _read_scheduler_state(pid_dir=opts.pid_dir)
    failed = {
        record.key for record in score_records.values() if getattr(record, "missing_artifacts", False)
    }
    lease_manager = _build_lease_manager(opts)
    cluster_claimed_job_ids = lease_manager.active_foreign_job_ids() if lease_manager is not None else set()
    job_priority_map = _job_priority_map(opts.job_priority)
    pending = _build_pending_queue(
        opts,
        completed=_completed_for_queue(run_mode=opts.run_mode, completed=completed),
        failed=failed,
        running=tuple(set(running_entries.keys()) | cluster_claimed_job_ids),
        question_counts=question_counts,
        job_priority=job_priority_map,
    )
    _print_queue_summary(pending, running_entries)
    return pending


def action_dispatch(
    opts: DispatchOptions,
    *,
    runtime_control: SchedulerRuntimeControl | None = None,
) -> None:
    ensure_dirs(opts.log_dir, opts.pid_dir, opts.run_log_dir)
    if opts.clean_param_swap:
        _clean_param_swap_records(opts.log_dir)

    batch_cache = opts.batch_cache_path or (opts.log_dir / "batch_cache.json")
    batch_profiler = BatchProfiler(batch_cache)
    job_priority = _job_priority_map(opts.job_priority)

    FAILURE_MONITOR.reset()
    pending_since: dict[str, float] = {}
    launch_times: dict[str, float] = {}
    job_metadata: dict[str, dict[str, object]] = {}
    completed_versions: dict[str, str | None] = {}
    session_completed: set[CompletedKey] = set()
    cooldown_until: dict[str, float] = {}
    previous_running: set[str] = set()
    claimed_job_ids: set[str] = set()
    pending_notice_printed = False
    cancel_requested = False
    lease_manager = _build_lease_manager(opts)

    if runtime_control is not None:
        runtime_control.write_status(ObservedStatus.STARTING)

    while True:
        failure = FAILURE_MONITOR.wait_failure(timeout=0)
        if failure is not None:
            failure_meta = job_metadata.get(failure.job_id, {}).copy()
            handle_job_failure(failure, opts.pid_dir, job_metadata, launch_times)
            _handle_batch_failure(batch_profiler, failure, failure_meta)
            if runtime_control is not None:
                runtime_control.write_status(
                    ObservedStatus.FAILED,
                    error=f"{failure.job_id} exited with returncode={failure.returncode}",
                    progress=_build_progress_snapshot(
                        queue=(),
                        running_entries=load_running(opts.pid_dir),
                        completed_count=len(completed_versions),
                        available_gpus=(),
                    ),
                )
            print("❗️ 调度因异常退出而终止。")
            return

        completed, completed_records, running_entries, question_counts = _read_scheduler_state(pid_dir=opts.pid_dir)
        failed_keys: set[CompletedKey] = set()
        now = time.time()

        completed_job_ids = _reconcile_completed_versions(
            completed_records=completed_records,
            completed_versions=completed_versions,
            job_metadata=job_metadata,
            launch_times=launch_times,
            pending_since=pending_since,
            session_completed=session_completed,
            cooldown_until=cooldown_until,
            now=now,
        )
        if lease_manager is not None and completed_job_ids:
            lease_manager.release(tuple(completed_job_ids))
            claimed_job_ids.difference_update(completed_job_ids)

        # If a job stops without a new score, briefly avoid re-queueing to allow DB writes to land.
        cooldown_jobs = _update_cooldown_jobs(
            previous_running=previous_running,
            running_entries=running_entries,
            completed_records=completed_records,
            cooldown_until=cooldown_until,
            now=now,
            dispatch_poll_seconds=opts.dispatch_poll_seconds,
        )
        previous_running = set(running_entries.keys())
        foreign_claimed_job_ids: set[str] = set()
        if lease_manager is not None:
            owned_running_jobs = {job_id for job_id in running_entries.keys() if job_id in claimed_job_ids}
            renewed_job_ids = lease_manager.renew(tuple(sorted(owned_running_jobs)))
            lost_job_ids = owned_running_jobs - renewed_job_ids
            if lost_job_ids:
                claimed_job_ids.difference_update(lost_job_ids)
                print(f"⚠️  已失去 lease：{', '.join(sorted(lost_job_ids))}")
                log_job_event(
                    "dispatcher_lease_lost",
                    "_dispatcher",
                    jobs=",".join(sorted(lost_job_ids)),
                )
            foreign_claimed_job_ids = lease_manager.active_foreign_job_ids()

        queue = _build_pending_queue(
            opts,
            completed=_completed_for_queue(
                run_mode=opts.run_mode,
                completed=completed,
                session_completed=session_completed,
            ),
            failed=failed_keys,
            running=tuple(set(running_entries.keys()) | cooldown_jobs | foreign_claimed_job_ids),
            question_counts=question_counts,
            job_priority=job_priority,
        )
        _mark_pending_jobs(
            queue=queue,
            pending_since=pending_since,
            job_metadata=job_metadata,
            now=now,
        )
        available_resources = _resolve_available_dispatch_resources(opts, running_entries)
        progress = _build_progress_snapshot(
            queue=queue,
            running_entries=running_entries,
            completed_count=len(completed_records),
            available_gpus=available_resources,
        )
        desired_state = runtime_control.desired_state() if runtime_control is not None else DesiredState.RUNNING

        if desired_state is DesiredState.CANCELLED:
            if runtime_control is not None:
                runtime_control.write_status(ObservedStatus.CANCELLING, progress=progress)
            if not cancel_requested:
                cancel_requested = True
                FAILURE_MONITOR.mark_aborting()
                stop_all_jobs(opts.pid_dir)
            if running_entries:
                time.sleep(1)
                continue
            if lease_manager is not None and claimed_job_ids:
                lease_manager.release(tuple(sorted(claimed_job_ids)))
                claimed_job_ids.clear()
            if runtime_control is not None:
                runtime_control.write_status(ObservedStatus.CANCELLED, progress=progress)
            print("🛑 调度已取消")
            log_job_event("dispatcher_cancelled", "_dispatcher", completed=len(completed_records))
            return

        if not queue:
            running_count = len(running_entries)
            if running_count > 0:
                if runtime_control is not None:
                    status = ObservedStatus.PAUSING if desired_state is DesiredState.PAUSED else ObservedStatus.RUNNING
                    runtime_control.write_status(status, progress=progress)
                if not pending_notice_printed:
                    print(f"⏳ 所有任务已调度，等待 {running_count} 个任务完成…")
                    pending_notice_printed = True
                log_job_event(
                    "dispatcher_wait",
                    "_dispatcher",
                    reason="running",
                    running=running_count,
                    pending=0,
                )
                time.sleep(opts.dispatch_poll_seconds)
                continue
            if foreign_claimed_job_ids:
                if runtime_control is not None:
                    runtime_control.write_status(ObservedStatus.RUNNING, progress=progress)
                if not pending_notice_printed:
                    print(f"⏳ 当前节点无可启动任务，等待集群中 {len(foreign_claimed_job_ids)} 个 lease 任务完成…")
                    pending_notice_printed = True
                log_job_event(
                    "dispatcher_wait",
                    "_dispatcher",
                    reason="cluster_running",
                    foreign_claims=len(foreign_claimed_job_ids),
                    pending=0,
                    running=0,
                )
                time.sleep(opts.dispatch_poll_seconds)
                continue
            print("🎉 所有任务调度完成")
            log_job_event("dispatcher_done", "_dispatcher", completed=len(completed_records))
            if lease_manager is not None and claimed_job_ids:
                lease_manager.release(tuple(sorted(claimed_job_ids)))
                claimed_job_ids.clear()
            if runtime_control is not None:
                runtime_control.write_status(ObservedStatus.COMPLETED, progress=progress)
            break

        pending_notice_printed = False
        if desired_state is DesiredState.PAUSED:
            status = ObservedStatus.PAUSING if running_entries else ObservedStatus.PAUSED
            if runtime_control is not None:
                runtime_control.write_status(status, progress=progress)
            time.sleep(opts.dispatch_poll_seconds)
            continue
        if runtime_control is not None:
            runtime_control.write_status(ObservedStatus.RUNNING, progress=progress)
        if not available_resources:
            running_count = len(running_entries)
            suffix = f"（当前运行 {running_count} 个任务）" if running_count else ""
            if _dispatch_uses_remote_inference(opts):
                print(f"⏳ 远端推理 worker 已达到并发上限，{opts.dispatch_poll_seconds} 秒后重试{suffix}")
                wait_reason = "remote_slots_exhausted"
            else:
                print(f"⏳ 未检测到空闲 GPU，{opts.dispatch_poll_seconds} 秒后重试{suffix}")
                wait_reason = "no_gpu"
            log_job_event(
                "dispatcher_wait",
                "_dispatcher",
                reason=wait_reason,
                pending=len(queue),
                running=running_count,
            )
            time.sleep(opts.dispatch_poll_seconds)
            continue

        _launch_queue_items(
            opts=opts,
            queue=queue,
            available_resources=available_resources,
            question_counts=question_counts,
            batch_profiler=batch_profiler,
            pending_since=pending_since,
            launch_times=launch_times,
            job_metadata=job_metadata,
            lease_manager=lease_manager,
            claimed_job_ids=claimed_job_ids,
        )

        time.sleep(1)


def _build_progress_snapshot(
    *,
    queue: Sequence[QueueItem],
    running_entries: Mapping[str, RunningEntry],
    completed_count: int,
    available_gpus: Sequence[str],
) -> SchedulerProgressSnapshot:
    return SchedulerProgressSnapshot(
        pending_jobs=len(queue),
        running_jobs=len(running_entries),
        completed_jobs=completed_count,
        failed_jobs=0,
        queue_head=tuple(item.job_id for item in queue[:8]),
        active_jobs=tuple(sorted(running_entries.keys())),
        available_gpus=tuple(available_gpus),
    )


def build_command(
    job: JobSpec,
    item: QueueItem,
    dataset_path: Path,
    device: str | None,
    *,
    batch_size: int | None = None,
    extra_args: Sequence[str] = (),
    infer_api_key: str = "",
    infer_timeout_s: float = 600.0,
    infer_max_workers: int = 32,
) -> list[str]:
    base = [DEFAULT_PYTHON, "-m", job.module]
    args = ["--dataset", str(dataset_path)]
    if item.is_remote:
        args.extend(
            [
                "--infer-base-url",
                str(item.infer_base_url or ""),
                "--infer-model",
                str(item.infer_model or item.model_name or ""),
            ]
        )
        if infer_api_key:
            args.extend(["--infer-api-key", infer_api_key])
        args.extend(["--infer-timeout-s", str(float(infer_timeout_s))])
        args.extend(["--infer-max-workers", str(int(infer_max_workers))])
    else:
        if item.model_path is None:
            raise ValueError("local scheduler launch requires model_path")
        args.extend(["--model-path", str(item.model_path)])
        if device:
            args.extend(["--device", device])
    if batch_size is not None and job.batch_flag:
        args.extend([job.batch_flag, str(batch_size)])
    if job.extra_args:
        args.extend(job.extra_args)
    if extra_args:
        args.extend(extra_args)
    return base + args


def action_status(opts: StatusOptions) -> dict[str, RunningEntry]:
    running = load_running(opts.pid_dir)
    if not running:
        print("🟡 无运行任务")
        return running
    header = f"{'Job ID':<32} {'GPU':<6} PID"
    print(header)
    print("-" * len(header))
    for job_id, entry in sorted(running.items()):
        gpu = entry.gpu or "?"
        print(f"{job_id:<32} {gpu:<6} {entry.pid}")
    return running


def action_stop(opts: StopOptions) -> None:
    pid_dir = opts.pid_dir
    if opts.stop_all:
        running = load_running(pid_dir)
        if not running:
            print("ℹ️  无运行任务")
            return
        for job_id in sorted(running.keys()):
            stop_job(job_id, pid_dir)
        return

    if not opts.job_ids:
        print("请指定 job id，或使用 --all")
        return
    for job_id in opts.job_ids:
        stop_job(job_id, pid_dir)


def action_logs(opts: LogsOptions) -> None:
    run_log_dir = opts.run_log_dir
    pid_dir = opts.pid_dir
    if not run_log_dir.exists():
        print(f"Log 目录 {run_log_dir} 不存在")
        return
    running = load_running(pid_dir)
    if not running:
        print("当前没有运行中的任务；logs 仅展示活跃任务。")
        return

    display_items: list[tuple[str, Path]] = []
    for job_id in sorted(running.keys()):
        entry = running[job_id]
        log_path = entry.log_path
        if log_path is None:
            log_path = run_log_dir / f"{job_id}.log"
        elif not log_path.is_absolute():
            log_path = run_log_dir / log_path
        display_items.append((job_id, log_path))

    if not display_items:
        print("当前没有运行中的任务；logs 仅展示活跃任务。")
        return

    rotate_seconds = max(1, opts.rotate_seconds)
    alt_screen = "\033[?1049h"
    restore_screen = "\033[?1049l"
    clear_screen = "\033[2J\033[H"
    hide_cursor = "\033[?25l"
    show_cursor = "\033[?25h"

    try:
        if not _write_stdout(alt_screen + hide_cursor):
            raise RuntimeError("stdout write failed")
        sys.stdout.flush()
        while True:
            for job_id, log_path in display_items:
                if not _write_stdout(clear_screen):
                    raise RuntimeError("stdout write failed")
                sys.stdout.flush()
                lines = tail_file(log_path, opts.tail_lines)
                print(f"===> {job_id} | {log_path}")
                print("\n".join(lines))
                time.sleep(rotate_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        _write_stdout(restore_screen + show_cursor)


def _clean_param_swap_records(log_dir: Path) -> None:
    target = (log_dir / "param_swap").resolve()
    if not target.exists():
        return
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    print(f"🧹 已清理参数搜索记录: {target}")


def _allocate_console_log_path(base_dir: Path, rel: Path) -> Path:
    target_dir = base_dir / rel.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / f"{rel.name}.log"
    if not candidate.exists():
        return candidate
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    candidate = target_dir / f"{rel.name}--{timestamp}.log"
    if not candidate.exists():
        return candidate
    attempt = 1
    while True:
        numbered = target_dir / f"{rel.name}--{timestamp}-{attempt}.log"
        if not numbered.exists():
            return numbered
        attempt += 1


def _handle_batch_failure(batch_profiler: BatchProfiler, failure: JobFailure, metadata: Mapping[str, object]) -> None:
    if not metadata:
        return
    job_name = metadata.get("job")
    model_slug = metadata.get("model_slug")
    gpu = metadata.get("gpu")
    if not job_name or not model_slug or gpu is None:
        return
    log_path = failure.log_path
    if not log_path.exists():
        return
    if not _log_contains_oom(log_path):
        return
    reason = f"runtime oom ({failure.job_id})"
    batch_profiler.invalidate_cache(str(job_name), str(model_slug), str(gpu), reason=reason)
    print(f"⚠️  {failure.job_id} 日志包含 OOM，已清理 {job_name}/{model_slug} 在 GPU {gpu} 的批量缓存。")


def _log_contains_oom(log_path: Path, *, tail_bytes: int = 65536) -> bool:
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - tail_bytes), os.SEEK_SET)
            chunk = fh.read()
    except OSError:
        return False
    text = chunk.decode("utf-8", errors="ignore").lower()
    keywords = ("out of memory", "cuda oom", "cuda out of memory", "torch.outofmemoryerror")
    return any(keyword in text for keyword in keywords)


def _backup_run_config(
    *,
    model_name: str,
    model_path: Path | None,
    infer_base_url: str | None,
    infer_model: str | None,
    dataset_slug: str,
    dataset_path: Path,
    job_name: str,
    job_id: str,
    batch_size: int | None,
    gpu: str | None,
    log_path: Path,
) -> Path:
    benchmark, _ = split_benchmark_and_split(dataset_slug)
    config_path = config_path_for_benchmark(benchmark, model_name)
    model_dir = safe_slug(model_name)
    benchmark_dir = safe_slug(benchmark)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    target = REPO_ROOT / "config_backup" / model_dir / benchmark_dir / f"{timestamp}.toml"
    target.parent.mkdir(parents=True, exist_ok=True)

    base_text = ""
    if config_path.exists():
        base_text = config_path.read_text(encoding="utf-8")

    run_block = _render_run_block(
        benchmark=benchmark,
        dataset_slug=dataset_slug,
        model_name=model_name,
        model_path=model_path,
        infer_base_url=infer_base_url,
        infer_model=infer_model,
        config_path=config_path,
        job_name=job_name,
        job_id=job_id,
        batch_size=batch_size,
        gpu=gpu,
        dataset_path=dataset_path,
        log_path=log_path,
    )
    separator = "\n\n" if base_text.strip() else ""
    target.write_text(f"{base_text.rstrip()}{separator}{run_block}", encoding="utf-8")
    return target


def _render_run_block(
    *,
    benchmark: str,
    dataset_slug: str,
    model_name: str,
    model_path: Path | None,
    infer_base_url: str | None,
    infer_model: str | None,
    config_path: Path,
    job_name: str,
    job_id: str,
    batch_size: int | None,
    gpu: str | None,
    dataset_path: Path,
    log_path: Path,
) -> str:
    created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "[run]",
        f"created_at = {_toml_quote(created_at)}",
        f"benchmark = {_toml_quote(benchmark)}",
        f"dataset_slug = {_toml_quote(dataset_slug)}",
        f"model_name = {_toml_quote(model_name)}",
        f"config_path = {_toml_quote(str(config_path))}",
        f"job_name = {_toml_quote(job_name)}",
        f"job_id = {_toml_quote(job_id)}",
        f"dataset_path = {_toml_quote(str(dataset_path))}",
        f"log_path = {_toml_quote(str(log_path))}",
    ]
    if model_path is not None:
        lines.append(f"model_path = {_toml_quote(str(model_path))}")
    if infer_base_url:
        lines.append(f"infer_base_url = {_toml_quote(str(infer_base_url))}")
    if infer_model:
        lines.append(f"infer_model = {_toml_quote(str(infer_model))}")
    if gpu:
        lines.append(f"gpu = {_toml_quote(gpu)}")
    if batch_size is not None:
        lines.append(f"batch_size = {int(batch_size)}")
    return "\n".join(lines) + "\n"


def _toml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


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
    "DispatchOptions",
    "QueueOptions",
    "StatusOptions",
    "StopOptions",
    "LogsOptions",
    "action_dispatch",
    "action_queue",
    "action_status",
    "action_stop",
    "action_logs",
    "MODEL_SELECT_CHOICES",
]
