from __future__ import annotations

"""Argparse-based CLI that exposes the scheduler actions."""

import argparse
from pathlib import Path
from typing import Sequence

from .actions import (
    DispatchOptions,
    LogsOptions,
    QueueOptions,
    StatusOptions,
    StopOptions,
    action_dispatch,
    action_logs,
    action_queue,
    action_status,
    action_stop,
)
from .config import DEFAULT_LOG_DIR, DEFAULT_MODEL_GLOBS, DEFAULT_PID_DIR, DEFAULT_RUN_LOG_DIR
from .dataset_utils import canonical_slug
from .jobs import JOB_CATALOGUE, JOB_ORDER
from .models import MODEL_SELECT_CHOICES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWKV 调度器 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    queue_parser = sub.add_parser("queue", help="查看待调度队列")
    _add_job_filters(queue_parser)

    dispatch_parser = sub.add_parser("dispatch", help="根据 GPU 空闲情况调度任务")
    _add_job_filters(dispatch_parser)
    dispatch_parser.add_argument("--run-log-dir", default=str(DEFAULT_RUN_LOG_DIR), help="运行日志目录")
    dispatch_parser.add_argument(
        "--dispatch-poll-seconds",
        type=int,
        default=30,
        help="空闲 GPU 轮询间隔",
    )
    dispatch_parser.add_argument(
        "--gpu-idle-max-mem",
        type=int,
        default=1000,
        help="将 GPU 视为空闲的显存占用阈值 (MB)",
    )
    dispatch_parser.add_argument(
        "--skip-missing-dataset",
        action="store_true",
        help="缺少数据集时跳过该任务",
    )
    dispatch_parser.add_argument(
        "--clean-param-swap",
        action="store_true",
        help="启动前清理 log_dir/param_swap",
    )
    dispatch_parser.add_argument(
        "--batch-cache",
        help="自定义 batch profiler 缓存路径 (默认为 log_dir/batch_cache.json)",
    )

    status_parser = sub.add_parser("status", help="查看正在运行的任务")
    status_parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")

    stop_parser = sub.add_parser("stop", help="停止任务")
    stop_parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    stop_parser.add_argument("--all", action="store_true", help="停止全部任务")
    stop_parser.add_argument("job_ids", nargs="*", help="待停止的 job id")

    logs_parser = sub.add_parser("logs", help="轮询输出运行日志")
    logs_parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    logs_parser.add_argument("--run-log-dir", default=str(DEFAULT_RUN_LOG_DIR), help="运行日志目录")
    logs_parser.add_argument("--tail-lines", type=int, default=60, help="每次展示的尾行数")
    logs_parser.add_argument("--rotate-seconds", type=int, default=15, help="轮播间隔秒数")

    return parser


def _add_job_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="评估 JSON 结果目录")
    parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_GLOBS),
        help="模型文件 glob（可多次指定）",
    )
    parser.add_argument(
        "--model-select",
        choices=MODEL_SELECT_CHOICES,
        default="latest-data",
        help="模型筛选策略（默认 latest-data：每档参数取 data_version 最新，忽略 0.1b/0.4b）",
    )
    parser.add_argument("--min-param-b", type=float, help="仅保留参数量 >= 阈值 (B)")
    parser.add_argument("--max-param-b", type=float, help="仅保留参数量 <= 阈值 (B)")
    parser.add_argument(
        "--only-jobs",
        nargs="+",
        choices=sorted(JOB_CATALOGUE.keys()),
        help="仅运行指定 job",
    )
    parser.add_argument(
        "--skip-jobs",
        nargs="+",
        choices=sorted(JOB_CATALOGUE.keys()),
        help="跳过指定 job",
    )
    parser.add_argument(
        "--skip-datasets",
        nargs="+",
        help="跳过指定数据集 slug（使用 canonical 名称）",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    job_order = _resolve_job_order(getattr(args, "only_jobs", None), getattr(args, "skip_jobs", None))
    if not job_order:
        print("⚠️ 未剩余可调度的 job，请检查 --only-jobs / --skip-jobs 参数设置")
        return 1

    model_globs = tuple(getattr(args, "models", list(DEFAULT_MODEL_GLOBS)))
    skip_dataset_slugs = _canonicalize_slugs(getattr(args, "skip_datasets", None))
    min_param_b = getattr(args, "min_param_b", None)
    max_param_b = getattr(args, "max_param_b", None)
    model_select = getattr(args, "model_select", "all")

    if command == "queue":
        opts = QueueOptions(
            log_dir=Path(args.log_dir),
            pid_dir=Path(args.pid_dir),
            job_order=job_order,
            model_select=model_select,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            skip_dataset_slugs=skip_dataset_slugs,
            model_globs=model_globs,
        )
        action_queue(opts)
    elif command == "dispatch":
        batch_cache = Path(args.batch_cache) if getattr(args, "batch_cache", None) else None
        opts = DispatchOptions(
            log_dir=Path(args.log_dir),
            pid_dir=Path(args.pid_dir),
            run_log_dir=Path(args.run_log_dir),
            job_order=job_order,
            model_select=model_select,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            skip_dataset_slugs=skip_dataset_slugs,
            model_globs=model_globs,
            dispatch_poll_seconds=int(args.dispatch_poll_seconds),
            gpu_idle_max_mem=int(args.gpu_idle_max_mem),
            skip_missing_dataset=bool(args.skip_missing_dataset),
            clean_param_swap=bool(args.clean_param_swap),
            batch_cache_path=batch_cache,
        )
        action_dispatch(opts)
    elif command == "status":
        action_status(StatusOptions(pid_dir=Path(args.pid_dir)))
    elif command == "stop":
        job_ids = tuple(str(job) for job in args.job_ids)
        action_stop(StopOptions(pid_dir=Path(args.pid_dir), job_ids=job_ids, stop_all=bool(args.all)))
    elif command == "logs":
        action_logs(
            LogsOptions(
                pid_dir=Path(args.pid_dir),
                run_log_dir=Path(args.run_log_dir),
                tail_lines=int(args.tail_lines),
                rotate_seconds=int(args.rotate_seconds),
            )
        )
    else:
        parser.print_help()
        return 1
    return 0


def _resolve_job_order(include: Sequence[str] | None, exclude: Sequence[str] | None) -> tuple[str, ...]:
    order = list(JOB_ORDER)
    if include:
        allowed = {job for job in include}
        order = [job for job in order if job in allowed]
    if exclude:
        blocked = {job for job in exclude}
        order = [job for job in order if job not in blocked]
    return tuple(order)


def _canonicalize_slugs(slugs: Sequence[str] | None) -> tuple[str, ...]:
    if not slugs:
        return tuple()
    return tuple(sorted({canonical_slug(slug) for slug in slugs}))


__all__ = ["build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
