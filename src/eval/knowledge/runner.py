from __future__ import annotations

"""Field-oriented knowledge runner aligned with rwkv-rs knowledge datasets."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.field_common import (
    build_avg_k_metrics,
    build_plan_task_details,
    build_task_sampling_config,
    set_task_env,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV knowledge benchmark runner")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation/scoring")
    parser.add_argument("--max-samples", type=int, help="Compatibility flag for quick runs")
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
    parser.add_argument(
        "--cot-mode",
        choices=[mode.value for mode in CoTMode],
        default=CoTMode.NO_COT.value,
        help="Prompt mode for knowledge benchmarks",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-batch CoT probe and skip scoring",
    )
    parser.add_argument(
        "--no-param-search",
        action="store_true",
        help="Compatibility flag (no-op).",
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
    return parser.parse_args(argv)


def _default_job_name(cot_mode: CoTMode) -> str:
    if cot_mode is CoTMode.NO_COT:
        return "multi_choice_plain"
    if cot_mode is CoTMode.FAKE_COT:
        return "multi_choice_fake_cot"
    return "multi_choice_cot"


def _print_done_message(cot_mode: CoTMode, sample_count: int) -> None:
    if cot_mode is CoTMode.NO_COT:
        print(f"✅ direct multiple-choice done: {sample_count} samples")
        return
    if cot_mode is CoTMode.FAKE_COT:
        print(f"✅ fake-CoT multiple-choice done: {sample_count} samples")
        return
    print(f"✅ CoT multiple-choice done: {sample_count} samples")


def _task_sampling_config(
    cot_mode: CoTMode,
    *,
    avg_k: float,
    effective_sample_count: int,
    cot_sampling: object | None = None,
) -> dict[str, object]:
    from src.eval.results.schema import sampling_config_to_dict

    sampling_payload: dict[str, object] = {}
    if cot_mode is CoTMode.COT:
        sampling_payload["stage1"] = sampling_config_to_dict(cot_sampling)
    return build_task_sampling_config(
        cot_mode=cot_mode,
        avg_k=avg_k,
        sampling_config=sampling_payload,
        effective_sample_count=effective_sample_count,
    )


def main(argv: Sequence[str] | None = None) -> int:
    from src.eval.benchmark_config import resolve_sampling_config
    from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
    from src.eval.evaluating import prepare_task_execution, run_checker_for_task
    from src.eval.execution_plan import build_attempt_keys, build_auto_avg_k_execution_plan, plan_attempt_count
    from src.eval.knowledge.pipeline import MultipleChoicePipeline
    from src.eval.metrics.multi_choice import evaluate_multiple_choice
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
    from src.db.async_writer import CompletionWriteWorker
    from src.db.eval_db_service import EvalDbService
    from src.db.orm import init_orm
    from src.infer.model import ModelLoadConfig

    args = parse_args(argv)
    cot_mode = CoTMode(args.cot_mode)
    if args.probe_only and cot_mode is not CoTMode.COT:
        raise ValueError("--probe-only is only supported with --cot-mode cot")

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    model_name = Path(args.model_path).stem
    dataset_records = JsonlMultipleChoiceLoader(str(dataset_path)).load()
    plan = build_auto_avg_k_execution_plan(slug, len(dataset_records))
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = MultipleChoicePipeline(config, target_token_format=args.target_token_format)

    cot_sampling = None
    if cot_mode is CoTMode.COT:
        cot_sampling = resolve_sampling_config(
            slug,
            model_name,
            fallback_templates="multi_choice_cot_default",
        )
        if cot_sampling is None:
            raise ValueError(f"缺少采样配置: {slug} ({model_name})")

        if args.probe_only:
            batch_size = max(1, args.batch_size)
            pipeline.run_chain_of_thought(
                dataset_path=str(dataset_path),
                cot_sampling=cot_sampling,
                batch_size=batch_size,
                sample_limit=batch_size,
                min_prompt_count=batch_size,
                samples_per_task=1,
                probe_only=True,
            )
            print(
                f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}."
            )
            return 0

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    job_name = os.environ.get("RWKV_SKILLS_JOB_NAME", _default_job_name(cot_mode))
    task_state = prepare_task_execution(
        service=service,
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        job_name=job_name,
        sampling_config=_task_sampling_config(
            cot_mode,
            avg_k=plan.avg_k,
            effective_sample_count=plan.effective_sample_count,
            cot_sampling=cot_sampling,
        ),
    )
    task_id = task_state.task_id
    skip_keys = task_state.skip_keys

    set_task_env(task_id)
    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
    expected_count = plan_attempt_count(plan, max_pass_k=1)
    try:
        if cot_mode is CoTMode.COT:
            result = pipeline.run_chain_of_thought(
                dataset_path=str(dataset_path),
                cot_sampling=cot_sampling,
                batch_size=max(1, args.batch_size),
                record_indices=plan.sample_indices,
                samples_per_task=max(plan.repeat_count, 1),
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
            )
        else:
            result = pipeline.run_direct(
                dataset_path=str(dataset_path),
                cot_mode=cot_mode,
                record_indices=plan.sample_indices,
                samples_per_task=plan.repeat_count,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
            )
    except BaseException:
        try:
            writer.close()
        finally:
            actual = service.count_completions(task_id=task_id, status="Completed")
            status = "completed" if actual == expected_count else "failed"
            service.update_task_status(task_id=task_id, status=status)
        raise

    writer.close()
    completions_payloads = service.list_completion_payloads(task_id=task_id, status="Completed")
    metrics = evaluate_multiple_choice(completions_payloads, dataset_path=dataset_path)
    service.ingest_eval_payloads(payloads=metrics.payloads, task_id=task_id)
    run_checker_for_task(service=service, task_id=task_id, model_name=model_name)
    score_payload = make_score_payload(
        slug,
        is_cot=cot_mode is not CoTMode.NO_COT,
        model_name=model_name,
        metrics=build_avg_k_metrics(
            metrics.rows,
            avg_k=plan.avg_k,
            primary_name="accuracy",
            primary_value=metrics.accuracy,
        ),
        samples=metrics.samples,
        task=job_name,
        task_details={
            "accuracy_by_subject": metrics.accuracy_by_subject,
            **build_plan_task_details(plan, cot_mode=cot_mode.value),
        },
        extra={"cot_mode": cot_mode.value},
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    _print_done_message(cot_mode, result.sample_count)
    return 0


__all__ = ["main", "parse_args"]
