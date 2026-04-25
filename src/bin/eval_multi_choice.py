from __future__ import annotations

"""Run direct multiple-choice evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.eval.benchmark_config import resolve_benchmark_model_config
from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.k_values import NumericK, filter_metrics_by_k, max_generation_k
from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k
from src.eval.metrics.multi_choice import evaluate_multiple_choice
from src.eval.results.payloads import make_score_payload
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.job_env import ensure_job_id
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.infer.model import ModelLoadConfig


DEFAULT_PASS_K: tuple[int, ...] = ()
DEFAULT_AVG_K: tuple[NumericK, ...] = ()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV multiple-choice (direct) evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for scoring")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to compute (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute (default: none; can be set in configs/<benchmark>.toml)",
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
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = MultipleChoicePipeline(config, target_token_format=args.target_token_format)
    model_name = Path(args.model_path).stem
    pass_k = _resolve_pass_k(slug, model_name, args)
    avg_k = _resolve_avg_k(slug, model_name, args)
    report_pass_k = _report_pass_k(slug, model_name, pass_k)
    report_avg_k = _report_avg_k(slug, model_name, avg_k)
    samples_per_task = max(max_generation_k(pass_k), max_generation_k(avg_k), 1)
    sample_limit = _resolve_max_samples(slug, model_name, args)

    # Quick validation of dataset readability before heavy model init
    records = JsonlMultipleChoiceLoader(str(dataset_path)).load()

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
        job_name="eval_multi_choice",
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config={"mode": "logits_only"},
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
        result = pipeline.run_direct(
            dataset_path=str(dataset_path),
            sample_limit=sample_limit,
            samples_per_task=samples_per_task,
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
            # Update session status if running in a session
            session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
            if session_task_id:
                try:
                    service.update_task_session_status(task_id=session_task_id, session_status="failed")
                except Exception:
                    pass
        raise
    writer.close()
    completions_payloads = service.list_completion_payloads(task_id=task_id, status="answer")
    metrics = evaluate_multiple_choice(
        completions_payloads,
        dataset_path=dataset_path,
    )
    pass_metrics_all = compute_pass_at_k(metrics.rows, pass_k)
    avg_metrics_all = compute_avg_at_k(metrics.rows, avg_k)
    service.ingest_eval_payloads(payloads=metrics.payloads, task_id=task_id)
    has_explicit_k_metrics = bool(report_pass_k) or bool(report_avg_k)
    metrics_payload: dict[str, float] = {}
    if not has_explicit_k_metrics:
        metrics_payload["accuracy"] = metrics.accuracy
    pass_payload = filter_metrics_by_k(pass_metrics_all, report_pass_k, "pass@")
    if report_pass_k and not pass_payload:
        pass_payload = pass_metrics_all or {}
    if pass_payload:
        metrics_payload.update(pass_payload)
    avg_payload = filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@")
    if report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    if avg_payload:
        metrics_payload.update(avg_payload)
    task_details: dict[str, object] = {
        "accuracy_by_subject": metrics.accuracy_by_subject,
    }
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all
    score_payload = make_score_payload(
        slug,
        is_cot=False,
        model_name=model_name,
        metrics=metrics_payload,
        samples=metrics.samples,
        task="multiple_choice",
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
    print(f"✅ direct multiple-choice done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
