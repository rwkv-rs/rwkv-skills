from __future__ import annotations

"""Run MBPP code generation + evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence
from dataclasses import replace

from src.eval.benchmark_config import resolve_benchmark_model_config, resolve_sampling_config
from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.k_values import NumericK, filter_metrics_by_k, max_generation_k
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.job_env import ensure_job_id
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.coding import CodingPipeline
from src.eval.metrics.code_generation.evaluate import eval_rows_from_payloads, evaluate_mbpp_dataset
from src.infer.model import ModelLoadConfig


DEFAULT_PASS_K: tuple[int, ...] = ()
DEFAULT_AVG_K: tuple[NumericK, ...] = ()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV MBPP evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="MBPP JSONL path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of problems for quick runs")
    parser.add_argument("--max-tokens", type=int, help="Clamp generation length")
    parser.add_argument("--temperature", type=float, help="Override sampling temperature")
    parser.add_argument("--top-k", type=int, help="Override sampling top-k")
    parser.add_argument("--top-p", type=float, help="Override sampling top-p")
    parser.add_argument("--eval-timeout", type=float, default=3.0, help="Seconds per test execution")
    parser.add_argument("--eval-workers", type=int, default=4, help="Parallel workers for evaluation")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="只跑一批生成用于 batch 探测，不评测、不写盘",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to report (default: none)",
    )
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute from generated samples (default: none)",
    )
    return parser.parse_args(argv)


def _resolve_pass_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[int, ...]:
    if args.pass_k:
        return tuple(args.pass_k)
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.pass_k is not None:
        return config.pass_k
    return DEFAULT_PASS_K


def _resolve_avg_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[NumericK, ...]:
    if args.avg_k:
        return tuple(args.avg_k)
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.avg_k is not None:
        return config.avg_k
    return DEFAULT_AVG_K


def _report_pass_k(slug: str, model_name: str, pass_k: tuple[int, ...]) -> tuple[int, ...]:
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.report_pass_k is not None:
        return config.report_pass_k
    return pass_k


def _report_avg_k(slug: str, model_name: str, avg_k: tuple[NumericK, ...]) -> tuple[NumericK, ...]:
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.report_avg_k is not None:
        return config.report_avg_k
    return avg_k


def _resolve_max_samples(slug: str, model_name: str, args: argparse.Namespace) -> int | None:
    if args.max_samples is not None:
        return args.max_samples
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    return config.max_samples if config is not None else None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    sampling = resolve_sampling_config(
        slug,
        Path(args.model_path).stem,
        fallback_templates="code_default",
    )
    if sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({Path(args.model_path).stem})")
    if args.max_tokens:
        sampling = sampling.clamp(args.max_tokens)
    if args.temperature is not None:
        sampling = replace(sampling, temperature=args.temperature)
    if args.top_k is not None:
        sampling = replace(sampling, top_k=args.top_k)
    if args.top_p is not None:
        sampling = replace(sampling, top_p=args.top_p)

    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = CodingPipeline(config)
    batch_size = max(1, args.batch_size)
    model_name = Path(args.model_path).stem
    pass_k = (1,) if args.probe_only else _resolve_pass_k(slug, model_name, args)
    avg_k = _resolve_avg_k(slug, model_name, args)
    report_pass_k = _report_pass_k(slug, model_name, pass_k)
    report_avg_k = _report_avg_k(slug, model_name, avg_k)
    samples_per_task = max(max_generation_k(pass_k), max_generation_k(avg_k), 1)
    sample_limit = batch_size if args.probe_only else _resolve_max_samples(slug, model_name, args)
    init_orm(DEFAULT_DB_CONFIG)
    
    service = EvalDbService()
    force_new_task = os.environ.get("RWKV_SCHEDULER_OVERWRITE") == "1"

    # 三层级联检索：一次查询获取所有续跑信息
    ctx = service.get_resume_context(
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        force_new_task=force_new_task,
    )
    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name="eval_code_mbpp",
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config=sampling_config_to_dict(sampling),
    )
    skip_keys = ctx.completed_keys

    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id
    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
    records = JsonlCodeGenerationLoader(str(dataset_path)).load()
    expected_count = service.expected_completion_count(
        dataset=str(slug),
        sample_limit=sample_limit,
        repeats_per_problem=samples_per_task,
    )
    if expected_count is None:
        expected_count = (min(len(records), sample_limit) if sample_limit else len(records)) * samples_per_task
    try:
        result = pipeline.run_mbpp(
            dataset_path=str(dataset_path),
            sampling=sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            eval_timeout=args.eval_timeout,
            eval_workers=args.eval_workers,
            pass_k=pass_k,
            samples_per_task=samples_per_task,
            probe_only=args.probe_only,
            skip_keys=skip_keys,
            on_record=writer.enqueue,
        )
    except BaseException:
        try:
            writer.close()
        finally:
            actual = service.count_completions(task_id=task_id)
            status = "completed" if actual == expected_count else "failed"
            service.update_task_status(task_id=task_id, status=status)
            session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
            if session_task_id:
                try:
                    service.update_task_session_status(task_id=session_task_id, session_status="failed")
                except Exception:
                    pass
        raise
    if args.probe_only:
        writer.close()
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        return 0

    print(f"✅ MBPP 生成完成：{result.sample_count} completions")

    writer.close()
    try:
        completions_payloads = service.list_completion_payloads(task_id=task_id, status="answer")
        eval_metrics, eval_payloads = evaluate_mbpp_dataset(
            completions_payloads,
            dataset_path=str(dataset_path),
            pass_k=pass_k,
            n_workers=args.eval_workers,
            timeout=args.eval_timeout,
        )
        avg_metrics_all = compute_avg_at_k(eval_rows_from_payloads(eval_payloads), avg_k)
        print(f"MBPP 评测: {eval_metrics}")
        service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
        metrics_payload: dict[str, float] = {}
        pass_payload = filter_metrics_by_k(eval_metrics, report_pass_k, "pass@")
        if report_pass_k and not pass_payload:
            pass_payload = eval_metrics or {}
        if pass_payload:
            metrics_payload.update(pass_payload)
        avg_payload = filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@")
        if report_avg_k and not avg_payload:
            avg_payload = avg_metrics_all or {}
        if avg_payload:
            metrics_payload.update(avg_payload)
        task_details: dict[str, object] = {}
        if eval_metrics and pass_payload != eval_metrics:
            task_details["pass_curve"] = eval_metrics
        if avg_metrics_all and avg_payload != avg_metrics_all:
            task_details["avg_curve"] = avg_metrics_all
        score_payload = make_score_payload(
            slug,
            is_cot=False,
            model_name=model_name,
            metrics=metrics_payload,
            samples=len(completions_payloads),
            problems=result.problem_count,
            task="code_mbpp",
            task_details=task_details,
        )
        service.record_score_payload(
            payload=score_payload,
            task_id=task_id,
        )
        # Update session status on success
        session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
        if session_task_id:
            try:
                service.update_task_session_status(task_id=session_task_id, session_status="completed")
            except Exception:
                pass
        export_version_results(
            service,
            task_id=task_id,
        )
    except BaseException:
        if service.get_score_payload(task_id=task_id) is None:
            service.update_task_status(task_id=task_id, status="failed")
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
