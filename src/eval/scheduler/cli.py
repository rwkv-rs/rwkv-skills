from __future__ import annotations

"""Argparse-based CLI that exposes the scheduler actions."""

import argparse
import re
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
from .config import (
    DEFAULT_COMPLETION_DIR,
    DEFAULT_EVAL_RESULT_DIR,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_GLOBS,
    DEFAULT_PID_DIR,
    DEFAULT_RUN_LOG_DIR,
)
from .dataset_utils import canonical_slug, canonicalize_benchmark_list
from .jobs import JOB_CATALOGUE, JOB_ORDER
from .models import MODEL_SELECT_CHOICES


_KNOWN_DATASET_SLUGS: tuple[str, ...] = tuple(
    sorted({canonical_slug(slug) for spec in JOB_CATALOGUE.values() for slug in spec.dataset_slugs})
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWKV 调度器 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    queue_parser = sub.add_parser("queue", help="查看待调度队列")
    _add_job_filters(queue_parser)

    dispatch_parser = sub.add_parser("dispatch", help="根据 GPU 空闲情况调度任务")
    _add_job_filters(dispatch_parser)
    dispatch_parser.add_argument("--run-log-dir", default=str(DEFAULT_RUN_LOG_DIR), help="运行日志目录")
    dispatch_parser.add_argument("--completion-dir", default=str(DEFAULT_COMPLETION_DIR), help="completion JSONL 目录")
    dispatch_parser.add_argument("--eval-result-dir", default=str(DEFAULT_EVAL_RESULT_DIR), help="评测器结果目录")
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
    dispatch_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="忽略 log_dir 中已存在的结果，重新评测并覆盖",
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
    domain_choices = sorted({spec.domain for spec in JOB_CATALOGUE.values() if spec.domain})
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="评估 JSON 结果目录")
    parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_GLOBS),
        help="模型文件 glob（用于定位权重，可多次指定；也可配合 --model-regex 过滤文件名）",
    )
    parser.add_argument(
        "--model-regex",
        nargs="+",
        help="仅保留文件名（不含路径）匹配任一正则的模型，例如 --model-regex '^rwkv7-.*7\\.2b'",
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
        "--job-order",
        nargs="+",
        choices=sorted(JOB_CATALOGUE.keys()),
        help="自定义 job 优先级（按给定顺序优先），未指定时按题量/CoT 自动排序",
    )
    if domain_choices:
        parser.add_argument(
            "--domains",
            nargs="+",
            choices=domain_choices,
            help=f"按任务域筛选 job，例如 --domains {'/'.join(domain_choices)}",
        )
    parser.add_argument(
        "--only-datasets",
        nargs="+",
        help="仅运行指定 benchmark（使用数据集名称即可，如 aime24 或 gpqa）",
    )
    parser.add_argument(
        "--skip-datasets",
        nargs="+",
        help="跳过指定 benchmark（名称即可，无需 *_test 后缀）",
    )
    parser.add_argument(
        "--param-search-scan-mode",
        choices=("both", "normal", "simple"),
        default="both",
        help="param-search 扫描模式：both/normal/simple（默认 both）",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command

    job_list = _resolve_job_list(
        getattr(args, "only_jobs", None),
        getattr(args, "skip_jobs", None),
        getattr(args, "domains", None),
    )
    if not job_list:
        print("⚠️ 未剩余可调度的 job，请检查 --domains / --only-jobs / --skip-jobs 参数设置")
        return 1
    job_priority = _resolve_job_priority(getattr(args, "job_order", None), job_list)

    model_globs = tuple(getattr(args, "models", list(DEFAULT_MODEL_GLOBS)))
    skip_dataset_slugs = _canonicalize_slugs(parser, getattr(args, "skip_datasets", None))
    only_dataset_slugs = _canonicalize_slugs(parser, getattr(args, "only_datasets", None))
    model_name_patterns = _compile_model_patterns(parser, getattr(args, "model_regex", None))
    min_param_b = getattr(args, "min_param_b", None)
    max_param_b = getattr(args, "max_param_b", None)
    model_select = getattr(args, "model_select", "all")

    if command == "queue":
        opts = QueueOptions(
            log_dir=Path(args.log_dir),
            pid_dir=Path(args.pid_dir),
            job_order=job_list,
            job_priority=job_priority,
            model_select=model_select,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            skip_dataset_slugs=skip_dataset_slugs,
            model_globs=model_globs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_name_patterns,
            param_search_scan_mode=str(args.param_search_scan_mode),
        )
        action_queue(opts)
    elif command == "dispatch":
        batch_cache = Path(args.batch_cache) if getattr(args, "batch_cache", None) else None
        opts = DispatchOptions(
            log_dir=Path(args.log_dir),
            pid_dir=Path(args.pid_dir),
            run_log_dir=Path(args.run_log_dir),
            completion_dir=Path(args.completion_dir),
            eval_result_dir=Path(args.eval_result_dir),
            job_order=job_list,
            job_priority=job_priority,
            model_select=model_select,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            skip_dataset_slugs=skip_dataset_slugs,
            model_globs=model_globs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_name_patterns,
            param_search_scan_mode=str(args.param_search_scan_mode),
            dispatch_poll_seconds=int(args.dispatch_poll_seconds),
            gpu_idle_max_mem=int(args.gpu_idle_max_mem),
            skip_missing_dataset=bool(args.skip_missing_dataset),
            clean_param_swap=bool(args.clean_param_swap),
            batch_cache_path=batch_cache,
            overwrite=bool(args.overwrite),
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


def _resolve_job_list(
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    domains: Sequence[str] | None,
) -> tuple[str, ...]:
    order = list(JOB_ORDER)

    if domains:
        allowed_domains = set(domains)
        order = [job for job in order if JOB_CATALOGUE[job].domain in allowed_domains]

    if include:
        allowed = {job for job in include}
        order = [job for job in order if job in allowed]
    if exclude:
        blocked = {job for job in exclude}
        order = [job for job in order if job not in blocked]
    return tuple(order)


def _canonicalize_slugs(
    parser: argparse.ArgumentParser,
    slugs: Sequence[str] | None,
) -> tuple[str, ...]:
    if not slugs:
        return tuple()
    try:
        return canonicalize_benchmark_list(slugs, known_slugs=_KNOWN_DATASET_SLUGS)
    except ValueError as exc:  # pragma: no cover - argparse already prints
        parser.error(str(exc))


def _compile_model_patterns(
    parser: argparse.ArgumentParser,
    patterns: Sequence[str] | None,
) -> tuple[re.Pattern[str], ...]:
    if not patterns:
        return tuple()
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw))
        except re.error as exc:  # pragma: no cover - argparse already prints
            parser.error(f"无效的模型正则 {raw!r}: {exc}")
    return tuple(compiled)


def _resolve_job_priority(priority: Sequence[str] | None, available: Sequence[str]) -> tuple[str, ...] | None:
    if not priority:
        return None
    allowed = {job for job in available}
    ordered: list[str] = []
    for job in priority:
        if job in allowed and job not in ordered:
            ordered.append(job)
    return tuple(ordered) if ordered else None


__all__ = ["build_parser", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
