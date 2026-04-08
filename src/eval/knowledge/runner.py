from __future__ import annotations

"""Field-oriented knowledge runner aligned with rwkv-rs knowledge datasets."""

import argparse
import os
from typing import TYPE_CHECKING
from typing import Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.field_common import (
    build_avg_k_metrics,
    build_plan_task_details,
    build_task_sampling_config,
    set_task_env,
)
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    resolve_backend_model_name,
    validate_inference_backend_args,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext, TaskSpec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV knowledge benchmark runner")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    add_inference_backend_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation/scoring")
    parser.add_argument("--max-samples", type=int, help="Limit source questions for quick runs")
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
        help="Run a single-batch CoT probe and skip scoring",
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


def main(
    argv: Sequence[str] | None = None,
    *,
    run_context: "RunContext | None" = None,
    task_spec: "TaskSpec | None" = None,
) -> int:
    del task_spec
    args = parse_args(argv)
    validate_inference_backend_args(args)

    from src.eval.benchmark_config import resolve_sampling_config
    from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
    from src.eval.evaluating import TaskRunController, TaskRunState, prepare_task_execution
    from src.eval.execution_plan import build_attempt_keys, build_auto_avg_k_execution_plan, plan_attempt_count
    from src.eval.knowledge.pipeline import MultipleChoicePipeline
    from src.eval.metrics.multi_choice import evaluate_multiple_choice
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService

    cot_mode = CoTMode(args.cot_mode)
    if args.probe_only and cot_mode is not CoTMode.COT:
        raise ValueError("--probe-only is only supported with --cot-mode cot")

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    model_name = resolve_backend_model_name(args)
    dataset_records = JsonlMultipleChoiceLoader(str(dataset_path)).load()
    plan = build_auto_avg_k_execution_plan(slug, len(dataset_records))
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    backend = build_inference_backend_from_args(args)
    pipeline = MultipleChoicePipeline(backend, target_token_format=args.target_token_format)

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

    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    job_name = run_context.job_name if run_context is not None else os.environ.get("RWKV_SKILLS_JOB_NAME", _default_job_name(cot_mode))
    expected_count = plan_attempt_count(plan, max_pass_k=1)
    task_state = prepare_task_execution(
        service=service,
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        job_name=job_name,
        run_mode=(run_context.run_mode if run_context is not None else None),
        sampling_config=_task_sampling_config(
            cot_mode,
            avg_k=plan.avg_k,
            effective_sample_count=plan.effective_sample_count,
            cot_sampling=cot_sampling,
        ),
    )
    task_run = TaskRunState.from_task_execution(
        execution_state=task_state,
        attempt_keys=attempt_keys,
        expected_attempt_count=expected_count,
    )
    runtime = TaskRunController(service=service, state=task_run)
    task_id = task_run.task_id
    skip_keys = task_state.skip_keys

    set_task_env(task_id)
    writer = runtime.create_writer(max_queue=args.db_write_queue)
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
        runtime.handle_attempt_stage_failure(writer)
        raise

    completions_payloads = runtime.complete_attempt_stage(writer)
    try:
        metrics = evaluate_multiple_choice(completions_payloads, dataset_path=dataset_path)
        runtime.ingest_eval_payloads(metrics.payloads)
        runtime.run_checker(model_name=model_name)
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
        runtime.record_score(score_payload)
    except BaseException as exc:
        runtime.fail_task(error=str(exc))
        raise
    _print_done_message(cot_mode, result.sample_count)
    return 0


__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
