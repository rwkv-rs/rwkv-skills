from __future__ import annotations

"""Run function-call evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.db.async_writer import CompletionWriteWorker
from src.db.eval_db_service import EvalDbService
from src.db.export_results import export_version_results
from src.db.orm import init_orm
from src.eval.benchmark_config import resolve_benchmark_model_config, resolve_sampling_config
from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.evaluators.function_call import FunctionCallPipeline
from src.eval.k_values import NumericK, filter_metrics_by_k, max_generation_k
from src.eval.metrics.function_call import evaluate_function_call
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.model import ModelLoadConfig

DEFAULT_AVG_K: tuple[NumericK, ...] = ()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV function-call evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="Function-call dataset JSONL path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for single-turn function-call runs")
    parser.add_argument("--max-samples", type=int, help="Limit number of tasks for quick runs")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute from generated function-call samples",
    )
    return parser.parse_args(argv)


def _resolve_avg_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[NumericK, ...]:
    if args.avg_k:
        return tuple(args.avg_k)
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.avg_k is not None:
        return config.avg_k
    return DEFAULT_AVG_K


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
    model_name = Path(args.model_path).stem
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    sampling = resolve_sampling_config(
        slug,
        model_name,
        fallback_templates=("function_call_cot_default", "agent_cot_default"),
    )
    if sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")

    pipeline = FunctionCallPipeline(ModelLoadConfig(weights_path=args.model_path, device=args.device))
    avg_k = _resolve_avg_k(slug, model_name, args)
    report_avg_k = _report_avg_k(slug, model_name, avg_k)
    sample_limit = _resolve_max_samples(slug, model_name, args)
    samples_per_task = max(max_generation_k(avg_k), 1)
    records = JsonlFunctionCallTaskLoader(str(dataset_path)).load()

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    force_new_task = os.environ.get("RWKV_SCHEDULER_OVERWRITE") == "1"
    ctx = service.get_resume_context(
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        force_new_task=force_new_task,
    )
    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name="eval_function_call",
        dataset=str(slug),
        model=model_name,
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
    expected_count = service.expected_completion_count(
        dataset=str(slug),
        sample_limit=sample_limit,
        repeats_per_problem=samples_per_task,
    )
    if expected_count is None:
        expected_count = (min(len(records), sample_limit) if sample_limit else len(records)) * samples_per_task
    try:
        result = pipeline.run(
            dataset_path=str(dataset_path),
            sampling=sampling,
            batch_size=max(1, args.batch_size),
            sample_limit=sample_limit,
            samples_per_task=samples_per_task,
            skip_keys=skip_keys,
            config=config,
            on_record=writer.enqueue,
        )
    except BaseException:
        try:
            writer.close()
        finally:
            actual = service.count_completions(task_id=task_id, status="answer")
            status = "completed" if actual == expected_count else "failed"
            service.update_task_status(task_id=task_id, status=status)
            session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
            if session_task_id:
                try:
                    service.update_task_session_status(task_id=session_task_id, session_status="failed")
                except Exception:
                    pass
        raise
    writer.close()

    completions_payloads = service.list_completion_payloads(task_id=task_id, status="answer")
    metrics = evaluate_function_call(
        completions_payloads,
        dataset_path=str(dataset_path),
        avg_k=avg_k,
    )
    service.ingest_eval_payloads(payloads=metrics.payloads or [], task_id=task_id)
    avg_payload = filter_metrics_by_k(metrics.avg_at_k, report_avg_k, "avg@") or (metrics.avg_at_k or {})
    score_payload = make_score_payload(
        slug,
        is_cot=False,
        model_name=model_name,
        metrics={
            "success_rate": metrics.success_rate,
            "avg_steps": metrics.avg_steps,
            "avg_tool_calls": metrics.avg_tool_calls,
            **avg_payload,
        },
        samples=metrics.samples,
        task="function_call",
        task_details={
            "env_breakdown": metrics.env_breakdown or {},
            **({"avg_curve": metrics.avg_at_k} if metrics.avg_at_k and avg_payload != metrics.avg_at_k else {}),
        },
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
    if session_task_id:
        try:
            service.update_task_session_status(task_id=session_task_id, session_status="completed")
        except Exception:
            pass
    export_version_results(service, task_id=task_id)
    print(f"✅ function_call done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
