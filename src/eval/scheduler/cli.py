from __future__ import annotations

"""Argparse-based CLI that exposes the scheduler actions."""

import argparse
import re
from pathlib import Path
from typing import Sequence

from src.eval.benchmark_registry import ALL_BENCHMARKS, BENCHMARK_ALIASES, BenchmarkField
from src.eval.evaluating import RunMode, collect_benchmark_dataset_slugs

from .actions import (
    DispatchOptions,
    LogsOptions,
    StatusOptions,
    StopOptions,
    action_dispatch,
    action_logs,
    action_queue,
    action_status,
    action_stop,
)
from .admin import SchedulerAdminController, serve_scheduler_admin
from .config import (
    DEFAULT_ADMIN_API_KEY,
    DEFAULT_ADMIN_HOST,
    DEFAULT_ADMIN_PORT,
    DEFAULT_ADMIN_STATE_DIR,
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
_KNOWN_BENCHMARK_NAMES: tuple[str, ...] = tuple(
    sorted({item.name for item in ALL_BENCHMARKS} | set(BENCHMARK_ALIASES))
)
_BENCHMARK_FIELD_CHOICES: tuple[str, ...] = tuple(field.value for field in BenchmarkField)
_RUN_MODE_CHOICES: tuple[str, ...] = tuple(mode.value for mode in RunMode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWKV 调度器 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    queue_parser = sub.add_parser("queue", help="查看待调度队列")
    _add_job_filters(queue_parser)
    _add_dispatch_options(queue_parser)

    dispatch_parser = sub.add_parser("dispatch", help="根据 GPU 空闲情况调度任务")
    _add_job_filters(dispatch_parser)
    _add_dispatch_options(dispatch_parser)

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

    serve_parser = sub.add_parser("serve", help="启动 HTTP / admin 控制服务")
    serve_parser.add_argument("--host", default=DEFAULT_ADMIN_HOST, help="HTTP 监听地址")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_ADMIN_PORT, help="HTTP 监听端口")
    serve_parser.add_argument("--state-dir", default=str(DEFAULT_ADMIN_STATE_DIR), help="scheduler admin 状态目录")
    serve_parser.add_argument(
        "--admin-api-key",
        default=DEFAULT_ADMIN_API_KEY,
        help="Bearer token；为空时不鉴权",
    )

    return parser


def _add_job_filters(parser: argparse.ArgumentParser) -> None:
    domain_choices = sorted({spec.domain for spec in JOB_CATALOGUE.values() if spec.domain})
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="调度器日志/缓存目录")
    parser.add_argument("--pid-dir", default=str(DEFAULT_PID_DIR), help="PID 文件目录")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_GLOBS),
        help="本地模型文件 glob（用于定位权重；远端推理模式下忽略）",
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
        "--benchmark-fields",
        nargs="+",
        choices=_BENCHMARK_FIELD_CHOICES,
        help="按 benchmark 领域筛选，语义对齐 rwkv-rs 的 benchmark_field",
    )
    parser.add_argument(
        "--extra-benchmarks",
        nargs="+",
        choices=_KNOWN_BENCHMARK_NAMES,
        help="额外包含的 benchmark，语义对齐 rwkv-rs 的 extra_benchmark_name",
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
        "--enable-param-search",
        action="store_true",
        help="启用 param-search（默认关闭，仅对最新 2.9b 生效）",
    )


def _add_dispatch_options(parser: argparse.ArgumentParser) -> None:
    """Add dispatch-related options (also used by `queue` for dry-run parity)."""

    parser.add_argument("--run-log-dir", default=str(DEFAULT_RUN_LOG_DIR), help="运行日志目录")
    parser.add_argument("--infer-base-url", help="远端推理服务地址；设置后 scheduler 进入评测/推理分离模式")
    parser.add_argument("--infer-models", nargs="+", help="远端推理服务上的模型名列表")
    parser.add_argument("--infer-api-key", default="", help="远端推理服务 API key")
    parser.add_argument("--infer-timeout-s", type=float, default=600.0, help="远端推理请求超时")
    parser.add_argument("--infer-max-workers", type=int, default=32, help="每个评测 worker 的远端请求并发上限")
    parser.add_argument("--distributed-claims", action="store_true", help="启用 PostgreSQL claim/lease，允许多个 scheduler 节点协同")
    parser.add_argument("--scheduler-node-id", help="当前 scheduler 节点标识；默认取主机名")
    parser.add_argument("--lease-duration-s", type=int, default=900, help="claim/lease 有效期秒数")
    parser.add_argument(
        "--run-mode",
        choices=_RUN_MODE_CHOICES,
        default=RunMode.AUTO.value,
        help="任务执行语义：auto/new/resume/rerun；strict 模式对齐 rwkv-rs，默认 auto 保持当前兼容行为",
    )
    parser.add_argument(
        "--dispatch-poll-seconds",
        type=int,
        default=30,
        help="空闲 GPU 轮询间隔",
    )
    parser.add_argument(
        "--gpu-idle-max-mem",
        type=int,
        default=1000,
        help="将 GPU 视为空闲的显存占用阈值 (MB)",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        help="限制同时运行的评测 worker 数；远端推理模式下未指定时默认 1",
    )
    parser.add_argument(
        "--skip-missing-dataset",
        action="store_true",
        help="缺少数据集时跳过该任务",
    )
    parser.add_argument(
        "--clean-param-swap",
        action="store_true",
        help="启动前清理 log_dir/param_swap",
    )
    parser.add_argument(
        "--batch-cache",
        help="自定义 batch profiler 缓存路径 (默认为 log_dir/batch_cache.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="兼容旧接口，相当于 --run-mode rerun",
    )
    parser.add_argument(
        "--disable-checker",
        action="store_true",
        help="关闭 LLM wrong-answer checker（不写 checker 表，也不跑离线 checker）",
    )


def _dispatch_options_from_args(
    args: argparse.Namespace,
    *,
    job_list: tuple[str, ...],
    job_priority: tuple[str, ...] | None,
    model_globs: tuple[str, ...],
    skip_dataset_slugs: tuple[str, ...],
    only_dataset_slugs: tuple[str, ...],
    model_name_patterns: tuple[re.Pattern[str], ...],
    min_param_b: float | None,
    max_param_b: float | None,
    model_select: str,
    run_mode: RunMode,
    infer_base_url: str | None,
    infer_models: tuple[str, ...],
) -> DispatchOptions:
    batch_cache = Path(args.batch_cache) if getattr(args, "batch_cache", None) else None
    return DispatchOptions(
        log_dir=Path(args.log_dir),
        pid_dir=Path(args.pid_dir),
        run_log_dir=Path(args.run_log_dir),
        job_order=job_list,
        job_priority=job_priority,
        model_select=model_select,
        min_param_b=min_param_b,
        max_param_b=max_param_b,
        skip_dataset_slugs=skip_dataset_slugs,
        model_globs=model_globs,
        only_dataset_slugs=only_dataset_slugs,
        model_name_patterns=model_name_patterns,
        enable_param_search=bool(args.enable_param_search),
        run_mode=run_mode,
        infer_base_url=infer_base_url,
        infer_models=infer_models,
        infer_api_key=str(getattr(args, "infer_api_key", "") or ""),
        infer_timeout_s=float(getattr(args, "infer_timeout_s", 600.0)),
        infer_max_workers=int(getattr(args, "infer_max_workers", 32)),
        distributed_claims=bool(getattr(args, "distributed_claims", False)),
        scheduler_node_id=(str(getattr(args, "scheduler_node_id", "") or "").strip() or None),
        lease_duration_s=int(getattr(args, "lease_duration_s", 900)),
        dispatch_poll_seconds=int(args.dispatch_poll_seconds),
        gpu_idle_max_mem=int(args.gpu_idle_max_mem),
        skip_missing_dataset=bool(args.skip_missing_dataset),
        clean_param_swap=bool(args.clean_param_swap),
        batch_cache_path=batch_cache,
        disable_checker=bool(args.disable_checker),
        max_concurrent_jobs=(
            int(args.max_concurrent_jobs)
            if getattr(args, "max_concurrent_jobs", None) is not None
            else None
        ),
    )


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


def _parse_benchmark_fields(values: Sequence[str] | None) -> tuple[BenchmarkField, ...]:
    if not values:
        return tuple()
    return tuple(BenchmarkField(value) for value in values)


def _collect_selected_dataset_slugs(
    parser: argparse.ArgumentParser,
    *,
    benchmark_fields: Sequence[BenchmarkField],
    extra_benchmarks: Sequence[str] | None,
    only_datasets: Sequence[str] | None,
) -> tuple[str, ...]:
    selected: set[str] = set()
    if benchmark_fields or extra_benchmarks:
        try:
            selected.update(
                collect_benchmark_dataset_slugs(
                    fields=benchmark_fields,
                    extra_benchmark_names=tuple(extra_benchmarks or ()),
                )
            )
        except ValueError as exc:
            parser.error(str(exc))

    selected.update(_canonicalize_slugs(parser, only_datasets))
    return tuple(sorted(selected))


def _resolve_run_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> RunMode:
    explicit = getattr(args, "run_mode", RunMode.AUTO.value)
    overwrite = bool(getattr(args, "overwrite", False))
    if overwrite and explicit not in (RunMode.AUTO.value, RunMode.RERUN.value):
        parser.error("--overwrite 只能与 --run-mode auto/rerun 搭配使用")
    if overwrite:
        return RunMode.RERUN
    try:
        return RunMode.parse(explicit)
    except ValueError as exc:
        parser.error(str(exc))


def _resolve_scheduler_inference_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    model_globs: tuple[str, ...],
) -> tuple[tuple[str, ...], str | None, tuple[str, ...]]:
    infer_base_url = str(getattr(args, "infer_base_url", "") or "").strip() or None
    infer_models = tuple(str(item).strip() for item in (getattr(args, "infer_models", None) or []) if str(item).strip())
    remote_mode = bool(infer_base_url or infer_models)
    if remote_mode:
        if not infer_base_url:
            parser.error("远端推理模式缺少 --infer-base-url")
        if not infer_models:
            parser.error("远端推理模式缺少 --infer-models")
        return tuple(), infer_base_url, infer_models
    return model_globs, None, tuple()


__all__ = ["build_parser", "main"]

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
    model_globs, infer_base_url, infer_models = _resolve_scheduler_inference_args(
        parser,
        args,
        model_globs=model_globs,
    )
    skip_dataset_slugs = _canonicalize_slugs(parser, getattr(args, "skip_datasets", None))
    benchmark_fields = _parse_benchmark_fields(getattr(args, "benchmark_fields", None))
    only_dataset_slugs = _collect_selected_dataset_slugs(
        parser,
        benchmark_fields=benchmark_fields,
        extra_benchmarks=getattr(args, "extra_benchmarks", None),
        only_datasets=getattr(args, "only_datasets", None),
    )
    model_name_patterns = _compile_model_patterns(parser, getattr(args, "model_regex", None))
    min_param_b = getattr(args, "min_param_b", None)
    max_param_b = getattr(args, "max_param_b", None)
    model_select = getattr(args, "model_select", "all")
    run_mode = _resolve_run_mode(parser, args)

    if command == "queue":
        opts = _dispatch_options_from_args(
            args,
            job_list=job_list,
            job_priority=job_priority,
            model_globs=model_globs,
            skip_dataset_slugs=skip_dataset_slugs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_name_patterns,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            model_select=model_select,
            run_mode=run_mode,
            infer_base_url=infer_base_url,
            infer_models=infer_models,
        )
        action_queue(opts)
    elif command == "dispatch":
        opts = _dispatch_options_from_args(
            args,
            job_list=job_list,
            job_priority=job_priority,
            model_globs=model_globs,
            skip_dataset_slugs=skip_dataset_slugs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_name_patterns,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            model_select=model_select,
            run_mode=run_mode,
            infer_base_url=infer_base_url,
            infer_models=infer_models,
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
    elif command == "serve":
        controller = SchedulerAdminController(state_dir=Path(args.state_dir))
        serve_scheduler_admin(
            host=str(args.host),
            port=int(args.port),
            controller=controller,
            api_key=str(args.admin_api_key) if args.admin_api_key else None,
        )
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
