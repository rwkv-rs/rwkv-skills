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
    DEFAULT_COMPLETION_DIR,
    DEFAULT_DISPATCH_POLL_SECONDS,
    DEFAULT_EVAL_RESULT_DIR,
    DEFAULT_GPU_IDLE_MAX_MEM,
    DEFAULT_MODEL_GLOBS,
    DEFAULT_PYTHON,
    DEFAULT_RUN_LOG_DIR,
    REPO_ROOT,
)
from .datasets import DATASET_ROOTS, DATA_OUTPUT_ROOT
from .jobs import JOB_CATALOGUE, JobSpec, detect_job_from_dataset, locate_dataset
from .naming import build_run_log_name
from .process import FAILURE_MONITOR, JobFailure, handle_job_failure, launch_job, list_idle_gpus, log_job_event
from .profiler import BatchProfiler
from .queue import _PARAM_SEARCH_BENCHMARKS, QueueItem, build_queue, sort_queue_items
from .question_counts import derive_question_counts
from .state import (
    CompletedKey,
    RunningEntry,
    ensure_dirs,
    load_running,
    scan_completed_jobs,
    stop_job,
    tail_file,
    write_pid_file,
)


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
    param_search_scan_mode: str = "both"


@dataclass(slots=True)
class DispatchOptions(QueueOptions):
    run_log_dir: Path = DEFAULT_RUN_LOG_DIR
    completion_dir: Path = DEFAULT_COMPLETION_DIR
    eval_result_dir: Path = DEFAULT_EVAL_RESULT_DIR
    dispatch_poll_seconds: int = DEFAULT_DISPATCH_POLL_SECONDS
    gpu_idle_max_mem: int = DEFAULT_GPU_IDLE_MAX_MEM
    skip_missing_dataset: bool = False
    clean_param_swap: bool = False
    batch_cache_path: Path | None = None
    overwrite: bool = False


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


def _purge_previous_outputs(completion_path: Path, score_path: Path, eval_path: Path) -> None:
    targets = [
        completion_path,
        completion_path.with_suffix(completion_path.suffix + ".tmp"),
        score_path,
        score_path.with_suffix(score_path.suffix + ".tmp"),
        eval_path,
        eval_path.with_suffix(eval_path.suffix + ".tmp"),
    ]
    for path in targets:
        path.unlink(missing_ok=True)


def _artifact_path(root: Path, rel: Path, suffix: str) -> Path:
    target = root / rel.parent / f"{rel.name}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def action_queue(opts: QueueOptions) -> list[QueueItem]:
    completed, score_records = scan_completed_jobs(opts.log_dir)
    failed = {record.key for record in score_records.values() if record.missing_artifacts}
    running_entries = load_running(opts.pid_dir)
    job_priority_map = _job_priority_map(opts.job_priority)
    pending = build_queue(
        model_globs=opts.model_globs,
        job_order=opts.job_order,
        completed=completed,
        failed=failed,
        running=running_entries.keys(),
        skip_dataset_slugs=opts.skip_dataset_slugs,
        only_dataset_slugs=opts.only_dataset_slugs,
        model_select=opts.model_select,
        min_param_b=opts.min_param_b,
        max_param_b=opts.max_param_b,
        param_search_scan_mode=opts.param_search_scan_mode,
        model_name_patterns=opts.model_name_patterns,
    )
    question_counts = derive_question_counts(score_records)
    pending = sort_queue_items(pending, question_counts=question_counts, job_priority=job_priority_map)
    _print_queue_summary(pending, running_entries)
    return pending


def action_dispatch(opts: DispatchOptions) -> None:
    ensure_dirs(opts.log_dir, opts.pid_dir, opts.run_log_dir, opts.completion_dir, opts.eval_result_dir)
    if opts.clean_param_swap:
        _clean_param_swap_records(opts.log_dir)

    batch_cache = opts.batch_cache_path or (opts.log_dir / "batch_cache.json")
    batch_profiler = BatchProfiler(batch_cache)
    job_priority = _job_priority_map(opts.job_priority)

    FAILURE_MONITOR.reset()
    pending_since: dict[str, float] = {}
    launch_times: dict[str, float] = {}
    job_metadata: dict[str, dict[str, object]] = {}
    overwritten_keys: set[CompletedKey] = set()
    completed_log_ids: set[str] | None = None
    pending_notice_printed = False
    failed_announced: set[str] = set()

    while True:
        failure = FAILURE_MONITOR.wait_failure(timeout=0)
        if failure is not None:
            failure_meta = job_metadata.get(failure.job_id, {}).copy()
            handle_job_failure(failure, opts.pid_dir, job_metadata, launch_times)
            _handle_batch_failure(batch_profiler, failure, failure_meta)
            print("❗️ 调度因异常退出而终止。")
            return

        completed, completed_records = scan_completed_jobs(opts.log_dir)
        failed_keys = {record.key for record in completed_records.values() if record.missing_artifacts}
        for job_id, record in completed_records.items():
            if not record.missing_artifacts or job_id in failed_announced:
                continue
            missing_desc = "、".join(record.missing_artifacts)
            print(f"❌ {job_id} 先前运行缺少 {missing_desc}，已标记失败（score: {record.score_path})")
            log_job_event(
                "job_fail",
                job_id,
                reason="missing_artifacts",
                missing=record.missing_artifacts,
                score_path=str(record.score_path),
                log_path=str(record.completion_path),
            )
            failed_announced.add(job_id)
        running_entries = load_running(opts.pid_dir)
        completed_for_queue = completed
        if opts.overwrite:
            completed_for_queue = {key for key in completed if key in overwritten_keys}

        queue = build_queue(
            model_globs=opts.model_globs,
            job_order=opts.job_order,
            completed=completed_for_queue,
            failed=failed_keys,
            running=running_entries.keys(),
            skip_dataset_slugs=opts.skip_dataset_slugs,
            only_dataset_slugs=opts.only_dataset_slugs,
            model_select=opts.model_select,
            min_param_b=opts.min_param_b,
            max_param_b=opts.max_param_b,
            param_search_scan_mode=opts.param_search_scan_mode,
            model_name_patterns=opts.model_name_patterns,
        )
        question_counts = derive_question_counts(completed_records)
        queue = sort_queue_items(queue, question_counts=question_counts, job_priority=job_priority)
        now = time.time()

        completed_ids = {job_id for job_id, record in completed_records.items() if not record.missing_artifacts}
        new_completed = set() if completed_log_ids is None else completed_ids - completed_log_ids
        if new_completed:
            for job_id in sorted(new_completed):
                info = completed_records[job_id]
                meta = job_metadata.pop(job_id, {})
                start = launch_times.pop(job_id, None)
                runtime = now - start if start else None
                pending_since.pop(job_id, None)
                payload: dict[str, object] = {
                    "job": info.key.job,
                    "dataset_slug": info.key.dataset_slug,
                    "model_slug": info.key.model_slug,
                    "dataset_path": info.dataset_path,
                    "model_name": info.model_name,
                    "log_path": str(info.completion_path),
                    "runtime_s": runtime,
                    "is_cot": info.key.is_cot,
                }
                if info.eval_result_path:
                    payload["eval_details_path"] = str(info.eval_result_path)
                payload.update(meta)
                log_job_event("job_done", job_id, **payload)
        completed_log_ids = completed_ids

        for position, item in enumerate(queue):
            if item.job_id not in pending_since:
                pending_since[item.job_id] = now
                meta = job_metadata.setdefault(item.job_id, {})
                meta.setdefault("job", item.job_name)
                meta.setdefault("dataset_slug", item.dataset_slug)
                meta.setdefault("model_path", str(item.model_path))
                meta.setdefault("model_slug", item.model_slug)
                log_job_event(
                    "job_pending",
                    item.job_id,
                    job=item.job_name,
                    dataset_slug=item.dataset_slug,
                    model_path=str(item.model_path),
                    queue_pos=position,
                    pending=len(queue),
                )

        if not queue:
            running_count = len(running_entries)
            if running_count > 0:
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
            print("🎉 所有任务调度完成")
            log_job_event("dispatcher_done", "_dispatcher", completed=len(completed_records))
            break

        pending_notice_printed = False
        idle_gpus = list_idle_gpus(opts.gpu_idle_max_mem)
        running_gpus = {entry.gpu for entry in running_entries.values() if entry.gpu}
        available_gpus = [gpu for gpu in idle_gpus if gpu not in running_gpus]
        if not available_gpus:
            running_count = len(running_entries)
            suffix = f"（当前运行 {running_count} 个任务）" if running_count else ""
            print(f"⏳ 未检测到空闲 GPU，{opts.dispatch_poll_seconds} 秒后重试{suffix}")
            log_job_event(
                "dispatcher_wait",
                "_dispatcher",
                reason="no_gpu",
                pending=len(queue),
                running=running_count,
            )
            time.sleep(opts.dispatch_poll_seconds)
            continue

        jobs_to_launch = min(len(queue), len(available_gpus))
        print(f"🧮 Pending={len(queue)} | Idle GPUs={', '.join(available_gpus)}")

        for item, gpu in zip(queue[:jobs_to_launch], available_gpus):
            job = JOB_CATALOGUE[item.job_name]
            dataset_slug = item.dataset_slug
            try:
                dataset_path = locate_dataset(dataset_slug, search=DATASET_ROOTS, output_root=DATA_OUTPUT_ROOT)
            except FileNotFoundError as exc:
                if opts.skip_missing_dataset:
                    print(f"⚠️  {item.job_id} 缺少数据集：{exc}. 已跳过。")
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
                raise

            log_relpath = build_run_log_name(item.model_path, dataset_slug, is_cot=job.is_cot)
            completion_path = _artifact_path(opts.completion_dir, log_relpath, ".jsonl")
            score_path = _artifact_path(opts.log_dir, log_relpath, ".json")
            eval_path = _artifact_path(opts.eval_result_dir, log_relpath, "_results.jsonl")
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
                            continue
                pid_path.unlink(missing_ok=True)

            if opts.overwrite:
                _purge_previous_outputs(completion_path, score_path, eval_path)
                print(f"    ↻ overwrite: cleared previous outputs for {log_relpath}")

            completed_key = CompletedKey(
                job=item.job_name,
                model_slug=item.model_slug,
                dataset_slug=dataset_slug,
                is_cot=job.is_cot,
            )
            if opts.overwrite:
                overwritten_keys.add(completed_key)
                if item.job_name == "param_search_select":
                    # param_search_select promotes trial artifacts into the canonical eval layout, so the
                    # "completed" keys that appear under log_dir are the downstream eval jobs (e.g.
                    # free_response_judge), not param_search_select itself. Register them so overwrite mode
                    # can converge instead of re-queuing selection indefinitely.
                    for benchmark_slug in _PARAM_SEARCH_BENCHMARKS:
                        promoted_job = detect_job_from_dataset(benchmark_slug, True) or "free_response_judge"
                        overwritten_keys.add(
                            CompletedKey(
                                job=promoted_job,
                                model_slug=item.model_slug,
                                dataset_slug=benchmark_slug,
                                is_cot=True,
                            )
                        )

            env = os.environ.copy()
            env.update(
                {
                    "RWKV_SKILLS_JOB_ID": item.job_id,
                    "RWKV_SKILLS_JOB_NAME": item.job_name,
                    "RWKV_SKILLS_MODEL_PATH": str(item.model_path),
                    "RWKV_SKILLS_DATASET": str(dataset_path),
                    "RWKV_SKILLS_DATASET_SLUG": dataset_slug,
                    "RWKV_SKILLS_LOG_PATH": str(completion_path),
                    "RUN_LOG_DIR": str(opts.log_dir),
                    "RUN_COMPLETION_DIR": str(opts.completion_dir),
                    "RUN_EVAL_RESULT_DIR": str(opts.eval_result_dir),
                    "RUN_RUN_LOG_DIR": str(opts.run_log_dir),
                }
            )

            questions = question_counts.get(dataset_slug)

            batch_size = batch_profiler.determine_batch_size(
                job=job,
                job_id=item.job_id,
                gpu=gpu,
                dataset_path=dataset_path,
                model_path=item.model_path,
                model_slug=item.model_slug,
                env=env,
                dataset_questions=questions,
            )

            extra_args = item.extra_args
            if opts.overwrite and item.job_name == "param_search_select" and "--overwrite" not in extra_args:
                extra_args = extra_args + ("--overwrite",)

            command = build_command(
                job,
                item.model_path,
                dataset_path,
                f"cuda:{gpu}",
                batch_size=batch_size,
                output_path=completion_path,
                extra_args=extra_args,
            )
            print(f"🚀 Launch {item.job_id} -> cuda:{gpu}")
            print(f"    Dataset: {dataset_path}")
            print(f"    Completion: {completion_path}")
            print(f"    Console: {console_log_path}")
            print(f"    Cmd: {' '.join(command)}")
            meta = job_metadata.setdefault(item.job_id, {})
            meta.update(
                job=item.job_name,
                dataset_slug=dataset_slug,
                dataset_path=str(dataset_path),
                model_path=str(item.model_path),
                model_slug=item.model_slug,
                log_path=str(completion_path),
                console_log_path=str(console_log_path),
                gpu=gpu,
            )

            process = launch_job(
                item.job_id,
                command,
                cwd=REPO_ROOT,
                log_path=console_log_path,
                env=env,
            )
            try:
                log_reference = str(console_log_path.relative_to(opts.run_log_dir))
            except ValueError:
                log_reference = str(console_log_path)
            write_pid_file(opts.pid_dir, item.job_id, process.pid, gpu, log_reference)
            launch_times[item.job_id] = time.time()
            pending_start = pending_since.pop(item.job_id, None)
            wait_s = time.time() - pending_start if pending_start else None
            log_job_event(
                "job_launch",
                item.job_id,
                job=item.job_name,
                dataset_slug=dataset_slug,
                dataset_path=str(dataset_path),
                model_path=str(item.model_path),
                log_path=str(completion_path),
                gpu=f"cuda:{gpu}",
                pid=process.pid,
                wait_s=wait_s,
            )

        time.sleep(1)


def build_command(
    job: JobSpec,
    model_path: Path,
    dataset_path: Path,
    device: str,
    *,
    batch_size: int | None = None,
    output_path: Path | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    base = [DEFAULT_PYTHON, "-m", job.module]
    args = [
        "--model-path",
        str(model_path),
        "--dataset",
        str(dataset_path),
        "--device",
        device,
    ]
    if batch_size is not None and job.batch_flag:
        args.extend([job.batch_flag, str(batch_size)])
    if output_path is not None:
        args.extend(["--output", str(output_path)])
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
        print(f"[{idx:02d}] {item.job_id} | {item.model_path.name} | {item.dataset_slug}")
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
