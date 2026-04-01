from __future__ import annotations

"""Field-oriented maths runner aligned with rwkv-rs maths datasets."""

import argparse
import os
import signal
from pathlib import Path
from typing import Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.field_common import build_plan_task_details, build_task_sampling_config, set_task_env
from src.eval.maths.common import (
    JudgeMode,
    build_llm_judge,
    count_free_answer_records,
    default_db_drain_every,
    default_db_write_queue,
    default_job_name,
    filter_avg_metrics,
    resolve_sampling_pair,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV maths benchmark runner")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Compatibility flag for quick runs")
    parser.add_argument("--cot-max-tokens", type=int, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, help="Clamp final answer generation length")
    parser.add_argument("--db-write-queue", type=int, help="DB completion write queue max size")
    parser.add_argument(
        "--db-drain-every",
        type=int,
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
        help="Compatibility flag for legacy callers.",
    )
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="Compatibility flag for legacy callers.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=[mode.value for mode in JudgeMode],
        default=JudgeMode.EXACT.value,
        help="Maths verdict mode",
    )
    parser.add_argument("--judge-model", help="LLM judge model name (env: JUDGE_MODEL / LLM_JUDGE_MODEL)")
    parser.add_argument("--judge-api-key", help="API key for judge model (env: JUDGE_API_KEY / OPENAI_API_KEY / API_KEY)")
    parser.add_argument(
        "--judge-base-url",
        help="Optional base URL for judge model (env: JUDGE_BASE_URL / LLM_JUDGE_BASE_URL / API_BASE)",
    )
    parser.add_argument("--judge-max-workers", type=int, default=32, help="Max concurrent workers for LLM judge")
    return parser.parse_args(argv)


def _compute_avg_curve(rows: list[tuple[int, int, bool]], avg_k: float) -> dict[str, float]:
    from src.eval.metrics.at_k import compute_avg_at_k

    return compute_avg_at_k(rows, (avg_k,))


def _close_writer_and_mark_failed(
    service,
    writer,
    *,
    task_id: str,
    expected_count: int,
    close_timeout_s: float,
    interrupted: bool,
) -> None:
    try:
        writer.close(timeout_s=close_timeout_s)
    finally:
        actual = service.count_completions(task_id=task_id, status="Completed")
        status = "failed" if interrupted else ("completed" if actual == expected_count else "failed")
        service.update_task_status(task_id=task_id, status=status)


def main(argv: Sequence[str] | None = None) -> int:
    from src.eval.env_config import load_env_file
    from src.eval.evaluating import prepare_task_execution, run_checker_for_task
    from src.eval.execution_plan import build_attempt_keys, build_auto_avg_k_execution_plan, plan_attempt_count
    from src.eval.maths.pipeline import FreeResponsePipeline
    from src.eval.metrics.free_response import evaluate_free_response
    from src.eval.results.payloads import make_score_payload
    from src.eval.results.schema import normalize_sampling_config_by_stage
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
    from src.db.async_writer import CompletionWriteWorker
    from src.db.eval_db_service import EvalDbService
    from src.db.orm import init_orm
    from src.infer.model import ModelLoadConfig

    load_env_file(Path(".env"))
    args = parse_args(argv)
    judge_mode = JudgeMode(args.judge_mode)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    total_records = count_free_answer_records(dataset_path, None)
    plan = build_auto_avg_k_execution_plan(slug, total_records)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = FreeResponsePipeline(config)

    model_name = Path(args.model_path).stem
    cot_sampling, final_sampling = resolve_sampling_pair(
        slug,
        model_name,
        cot_max_tokens=args.cot_max_tokens,
        final_max_tokens=args.final_max_tokens,
    )
    batch_size = max(1, args.batch_size)

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    job_name = os.environ.get("RWKV_SKILLS_JOB_NAME", default_job_name(judge_mode))
    task_state = prepare_task_execution(
        service=service,
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        job_name=job_name,
        sampling_config=build_task_sampling_config(
            cot_mode=CoTMode.COT,
            avg_k=plan.avg_k,
            sampling_config=normalize_sampling_config_by_stage([(1, cot_sampling), (2, final_sampling)]),
            effective_sample_count=plan.effective_sample_count,
            judger_model_name=(judge.config.model if judge is not None else None),
        ),
    )
    task_id = task_state.task_id
    skip_keys = task_state.skip_keys
    set_task_env(task_id)

    if args.probe_only:
        pipeline.run(
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

    judge = None
    if judge_mode is JudgeMode.LLM:
        judge = build_llm_judge(
            judge_model=args.judge_model,
            judge_api_key=args.judge_api_key,
            judge_base_url=args.judge_base_url,
            judge_max_workers=args.judge_max_workers,
            required=True,
        )

    db_write_queue = (
        args.db_write_queue
        if args.db_write_queue is not None
        else default_db_write_queue(judge_mode)
    )
    db_drain_every = (
        args.db_drain_every
        if args.db_drain_every is not None
        else default_db_drain_every(judge_mode)
    )
    close_timeout_s = max(0.0, float(args.db_close_timeout_s))
    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=db_write_queue,
        drain_every=db_drain_every,
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

    expected_count = plan_attempt_count(plan, max_pass_k=1)
    try:
        result = pipeline.run(
            dataset_path=str(dataset_path),
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            record_indices=plan.sample_indices,
            pad_to_batch=False,
            pass_k=(),
            samples_per_task=max(plan.repeat_count, 1),
            attempt_keys=attempt_keys,
            skip_keys=skip_keys,
            on_record=writer.enqueue,
        )
        writer.close(timeout_s=close_timeout_s)
    except BaseException:
        _close_writer_and_mark_failed(
            service,
            writer,
            task_id=task_id,
            expected_count=expected_count,
            close_timeout_s=close_timeout_s,
            interrupted=should_exit["active"],
        )
        raise
    finally:
        _restore_signal_handlers()

    completions_payloads = service.list_completion_payloads(task_id=task_id, status="Completed")
    evaluation = evaluate_free_response(
        completions_payloads,
        dataset_path=str(dataset_path),
        judge=judge,
    )
    avg_curve = _compute_avg_curve(evaluation.rows, plan.avg_k)
    avg_metrics = filter_avg_metrics(avg_curve, (plan.avg_k,))
    if judge_mode is JudgeMode.LLM and evaluation.judge_accuracy is None:
        raise RuntimeError("LLM judge 未返回有效 judge_accuracy，无法写入 judge-only 分数。")

    primary_metric_name = "judge_accuracy" if judge_mode is JudgeMode.LLM else "exact_accuracy"
    primary_metric_value = (
        float(evaluation.judge_accuracy)
        if judge_mode is JudgeMode.LLM
        else float(evaluation.exact_accuracy)
    )
    metrics_payload = {primary_metric_name: primary_metric_value}
    if judge_mode is JudgeMode.EXACT and evaluation.judge_accuracy is not None:
        metrics_payload["judge_accuracy"] = evaluation.judge_accuracy
    if avg_metrics:
        metrics_payload.update(avg_metrics)

    task_details: dict[str, object] = build_plan_task_details(plan, cot_mode=CoTMode.COT.value)
    if avg_curve:
        task_details["avg_curve"] = avg_curve

    service.ingest_eval_payloads(payloads=evaluation.payloads, task_id=task_id)
    run_checker_for_task(service=service, task_id=task_id, model_name=model_name)
    score_payload = make_score_payload(
        slug,
        is_cot=True,
        model_name=model_name,
        metrics=metrics_payload,
        samples=evaluation.samples,
        problems=result.problem_count,
        task=job_name,
        task_details=task_details,
        extra={"cot_mode": CoTMode.COT.value},
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    if judge_mode is JudgeMode.LLM:
        print(f"✅ judge CoT done: {result.sample_count} samples")
    else:
        print(f"✅ CoT free-form done: {result.sample_count} samples")
    return 0


__all__ = ["main", "parse_args"]
