from __future__ import annotations

"""Run chain-of-thought free-form QA evaluation for RWKV models."""

import argparse
import os
import signal
from pathlib import Path
from typing import Sequence

from src.eval.k_values import NumericK, filter_metrics_by_k, max_generation_k
from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.metrics.free_response import (
    compute_pass_at_k,
    compute_avg_at_k,
    evaluate_free_response,
)
from src.eval.benchmark_config import resolve_benchmark_model_config, resolve_sampling_config
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.evaluators.free_response import FreeResponsePipeline
from src.infer.model import ModelLoadConfig


DEFAULT_PASS_K: tuple[int, ...] = ()
DEFAULT_AVG_K: tuple[NumericK, ...] = ()


def _count_records(path: str | Path, limit: int | None) -> int:
    loader = JsonlFreeAnswerLoader(str(path))
    count = 0
    for _ in loader:
        count += 1
        if limit and count >= limit:
            break
    return count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV free-form CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--cot-max-tokens", type=int, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, help="Clamp final answer generation length")
    parser.add_argument("--db-write-queue", type=int, default=8, help="DB completion write queue max size")
    parser.add_argument(
        "--db-drain-every",
        type=int,
        default=8,
        help="Force DB writer to drain every N completion payloads (0 disables)",
    )
    parser.add_argument(
        "--db-close-timeout-s",
        type=float,
        default=30.0,
        help="Max seconds to wait for DB writer drain/close on shutdown",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe and skip scoring",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to generate for and compute (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute from generated samples (default: none; can be set in configs/<benchmark>.toml)",
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


def _report_avg_k(
    slug: str,
    model_name: str,
    avg_k: tuple[NumericK, ...],
) -> tuple[NumericK, ...]:
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
    pipeline = FreeResponsePipeline(config)

    model_name = Path(args.model_path).stem
    pass_k = _resolve_pass_k(slug, model_name, args)
    avg_k = _resolve_avg_k(slug, model_name, args)
    report_pass_k = _report_pass_k(slug, model_name, pass_k)
    report_avg_k = _report_avg_k(slug, model_name, avg_k)
    sample_limit = _resolve_max_samples(slug, model_name, args)

    cot_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    final_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="final",
        fallback_templates="free_response_final_default",
    )
    if cot_sampling is None or final_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    final_sampling = final_sampling.clamp(args.final_max_tokens)

    batch_size = max(1, args.batch_size)

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
    sampling_payload = {
        "cot": sampling_config_to_dict(cot_sampling),
        "final": sampling_config_to_dict(final_sampling),
    }
    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name="eval_free_response",
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config=sampling_payload,
    )
    skip_keys = ctx.completed_keys

    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id

    if args.probe_only:
        _ = pipeline.run(
            dataset_path=str(dataset_path),
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=batch_size,
            pad_to_batch=True,
            pass_k=(1,),
            samples_per_task=1,
            probe_only=True,
        )
        print(f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}.")
        return 0

    samples_per_task = max(max_generation_k(pass_k), max_generation_k(avg_k), 1)
    expected_count = service.expected_completion_count(
        dataset=str(slug),
        sample_limit=sample_limit,
        repeats_per_problem=samples_per_task,
    )
    if expected_count is None:
        expected_count = _count_records(dataset_path, sample_limit) * samples_per_task
    close_timeout_s = max(0.0, float(args.db_close_timeout_s))
    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
        drain_every=args.db_drain_every,
    )

    should_exit = {"active": False}
    original_signal_handlers: dict[signal.Signals, object] = {}

    def _restore_signal_handlers() -> None:
        for sig, handler in original_signal_handlers.items():
            signal.signal(sig, handler)
        original_signal_handlers.clear()

    def _handle_termination(signum: int, _frame: object) -> None:
        if should_exit["active"]:
            raise SystemExit(128 + signum)
        should_exit["active"] = True
        signame = signal.Signals(signum).name
        print(f"⚠️ Received {signame}; draining completion writer before exit...")
        try:
            writer.close(timeout_s=close_timeout_s)
        except Exception as exc:
            print(f"⚠️ Failed to close completion writer during {signame}: {exc}")
        try:
            service.update_task_status(task_id=task_id, status="failed")
        except Exception as exc:
            print(f"⚠️ Failed to mark task {task_id} failed after {signame}: {exc}")
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        original_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_termination)

    try:
        result = pipeline.run(
            dataset_path=str(dataset_path),
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            pad_to_batch=False,
            pass_k=pass_k,
            samples_per_task=samples_per_task,
            skip_keys=skip_keys,
            on_record=writer.enqueue,
        )
        writer.close(timeout_s=close_timeout_s)
    except BaseException:
        try:
            writer.close(timeout_s=close_timeout_s)
        finally:
            actual = service.count_completions(task_id=task_id, status="answer")
            status = "failed" if should_exit["active"] else ("completed" if actual == expected_count else "failed")
            service.update_task_status(task_id=task_id, status=status)
            session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
            if session_task_id:
                try:
                    service.update_task_session_status(task_id=session_task_id, session_status="failed")
                except Exception:
                    pass
        raise
    finally:
        _restore_signal_handlers()

    completions_payloads = service.list_completion_payloads(
        task_id=task_id,
        status="answer",
    )
    evaluation = evaluate_free_response(
        completions_payloads,
        dataset_path=str(dataset_path),
        judge=None,
    )
    pass_metrics_all = compute_pass_at_k(evaluation.rows, pass_k)
    avg_metrics_all = compute_avg_at_k(evaluation.rows, avg_k)
    task_details: dict[str, object] = {}
    metrics_payload = {"exact_accuracy": evaluation.exact_accuracy}
    if evaluation.judge_accuracy is not None:
        metrics_payload["judge_accuracy"] = evaluation.judge_accuracy

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
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all

    service.ingest_eval_payloads(
        payloads=evaluation.payloads,
        task_id=task_id,
    )
    score_payload = make_score_payload(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics=metrics_payload,
        samples=evaluation.samples,
        problems=result.problem_count,
        task="free_response",
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
    print(f"✅ CoT free-form done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
