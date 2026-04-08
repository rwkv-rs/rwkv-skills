from __future__ import annotations

"""Field-oriented coding runner aligned with rwkv-rs coding datasets."""

import argparse
import os
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode, resolve_benchmark_metadata
from src.eval.field_common import build_avg_k_metrics, build_plan_task_details, build_task_sampling_config, set_task_env
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    resolve_backend_model_name,
    validate_inference_backend_args,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext, TaskSpec


class CodingBenchmarkKind(str, Enum):
    AUTO = "auto"
    HUMAN_EVAL = "human_eval"
    MBPP = "mbpp"
    LIVECODEBENCH = "livecodebench"


_HUMAN_EVAL_JOB_NAMES = frozenset({"code_human_eval"})
_MBPP_JOB_NAMES = frozenset({"code_mbpp", "code_mbpp_fake_cot", "code_mbpp_cot"})
_LIVECODEBENCH_JOB_NAMES = frozenset({"code_livecodebench"})
_DEFAULT_PASS_K = (1,)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV coding benchmark runner")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    add_inference_backend_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit source questions for quick runs")
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
        help="Run a single-batch probe and skip scoring",
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=[kind.value for kind in CodingBenchmarkKind],
        default=CodingBenchmarkKind.AUTO.value,
        help="Explicit coding benchmark family (defaults to auto-detect from dataset slug)",
    )
    parser.add_argument(
        "--cot-mode",
        choices=[mode.value for mode in CoTMode],
        help="Prompt mode for MBPP benchmarks; human_eval/livecodebench use fixed modes",
    )
    return parser.parse_args(argv)


def _apply_sampling_overrides(sampling, args: argparse.Namespace):
    if args.max_tokens:
        sampling = sampling.clamp(args.max_tokens)
    if args.temperature is not None:
        sampling = replace(sampling, temperature=args.temperature)
    if args.top_k is not None:
        sampling = replace(sampling, top_k=args.top_k)
    if args.top_p is not None:
        sampling = replace(sampling, top_p=args.top_p)
    return sampling


def _require_sampling(dataset_slug: str, model_name: str, *, stage: str | None = None, fallback_templates: str):
    sampling = resolve_sampling_config(
        dataset_slug,
        model_name,
        stage=stage,
        fallback_templates=fallback_templates,
    )
    if sampling is None:
        raise ValueError(f"缺少采样配置: {dataset_slug} ({model_name})")
    return sampling


def _infer_benchmark_kind(dataset_slug: str) -> CodingBenchmarkKind:
    job_names = frozenset(resolve_benchmark_metadata(dataset_slug).scheduler_jobs)
    if job_names & _HUMAN_EVAL_JOB_NAMES:
        return CodingBenchmarkKind.HUMAN_EVAL
    if job_names & _MBPP_JOB_NAMES:
        return CodingBenchmarkKind.MBPP
    if job_names & _LIVECODEBENCH_JOB_NAMES:
        return CodingBenchmarkKind.LIVECODEBENCH
    raise ValueError(f"dataset {dataset_slug!r} 不是 coding benchmark，无法用 coding runner 运行。")


def _resolve_benchmark_kind(dataset_slug: str, requested_kind: CodingBenchmarkKind) -> CodingBenchmarkKind:
    inferred_kind = _infer_benchmark_kind(dataset_slug)
    if requested_kind is CodingBenchmarkKind.AUTO:
        return inferred_kind
    if requested_kind is not inferred_kind:
        raise ValueError(
            f"dataset {dataset_slug!r} 推断为 {inferred_kind.value}，"
            f"与显式 --benchmark-kind {requested_kind.value} 不一致。"
        )
    return requested_kind


def _resolve_cot_mode(kind: CodingBenchmarkKind, requested_mode: str | None) -> CoTMode:
    if kind is CodingBenchmarkKind.HUMAN_EVAL:
        if requested_mode is not None and CoTMode(requested_mode) is not CoTMode.NO_COT:
            raise ValueError("human_eval only supports --cot-mode no_cot")
        return CoTMode.NO_COT
    if kind is CodingBenchmarkKind.LIVECODEBENCH:
        if requested_mode is not None and CoTMode(requested_mode) is not CoTMode.COT:
            raise ValueError("livecodebench only supports --cot-mode cot")
        return CoTMode.COT
    if requested_mode is None:
        return CoTMode.NO_COT
    return CoTMode(requested_mode)


def _default_job_name(kind: CodingBenchmarkKind, cot_mode: CoTMode) -> str:
    if kind is CodingBenchmarkKind.HUMAN_EVAL:
        return "code_human_eval"
    if kind is CodingBenchmarkKind.LIVECODEBENCH:
        return "code_livecodebench"
    if cot_mode is CoTMode.NO_COT:
        return "code_mbpp"
    if cot_mode is CoTMode.FAKE_COT:
        return "code_mbpp_fake_cot"
    return "code_mbpp_cot"


def _print_done_message(kind: CodingBenchmarkKind, cot_mode: CoTMode, sample_count: int) -> None:
    if kind is CodingBenchmarkKind.HUMAN_EVAL:
        print(f"✅ HumanEval done: {sample_count} samples")
        return
    if kind is CodingBenchmarkKind.LIVECODEBENCH:
        print(f"✅ LiveCodeBench done: {sample_count} samples")
        return
    if cot_mode is CoTMode.NO_COT:
        print(f"✅ MBPP done: {sample_count} samples")
        return
    if cot_mode is CoTMode.FAKE_COT:
        print(f"✅ fake-CoT MBPP done: {sample_count} samples")
        return
    print(f"✅ CoT MBPP done: {sample_count} samples")


def _sampling_payload(
    kind: CodingBenchmarkKind,
    cot_mode: CoTMode,
    *,
    sampling=None,
    cot_sampling=None,
    final_sampling=None,
) -> dict[str, object]:
    from src.eval.results.schema import sampling_config_to_dict

    if kind is CodingBenchmarkKind.LIVECODEBENCH:
        return {
            "stage1": sampling_config_to_dict(cot_sampling),
            "stage2": sampling_config_to_dict(final_sampling),
        }
    if kind is CodingBenchmarkKind.MBPP and cot_mode is CoTMode.COT:
        return {
            "stage1": sampling_config_to_dict(sampling),
            "stage2": sampling_config_to_dict(sampling),
        }
    return {"stage1": sampling_config_to_dict(sampling)}


def main(
    argv: Sequence[str] | None = None,
    *,
    run_context: "RunContext | None" = None,
    task_spec: "TaskSpec | None" = None,
) -> int:
    del task_spec
    args = parse_args(argv)
    validate_inference_backend_args(args)

    from src.eval.coding.pipeline import CodingPipeline
    from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
    from src.eval.evaluating import TaskRunController, TaskRunState, prepare_task_execution
    from src.eval.execution_plan import build_attempt_keys, build_auto_avg_k_execution_plan, plan_attempt_count
    from src.eval.metrics.code_generation.evaluate import evaluate_human_eval, evaluate_mbpp_dataset
    from src.eval.metrics.code_generation.livecodebench import evaluate_livecodebench_dataset
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
    from src.db.async_writer import CompletionWriteWorker
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    benchmark_kind = _resolve_benchmark_kind(slug, CodingBenchmarkKind(args.benchmark_kind))
    cot_mode = _resolve_cot_mode(benchmark_kind, args.cot_mode)

    dataset_records = JsonlCodeGenerationLoader(str(dataset_path)).load()
    plan = build_auto_avg_k_execution_plan(slug, len(dataset_records))
    attempt_keys = build_attempt_keys(plan, max_pass_k=max(_DEFAULT_PASS_K))
    model_name = resolve_backend_model_name(args)
    batch_size = max(1, args.batch_size)
    sample_limit = batch_size if args.probe_only else args.max_samples

    sampling = None
    cot_sampling = None
    final_sampling = None
    if benchmark_kind is CodingBenchmarkKind.LIVECODEBENCH:
        cot_sampling = _apply_sampling_overrides(
            _require_sampling(slug, model_name, stage="cot", fallback_templates="full_code_cot_default"),
            args,
        )
        final_sampling = _apply_sampling_overrides(
            _require_sampling(slug, model_name, stage="final", fallback_templates="full_code_final_default"),
            args,
        )
    else:
        sampling = _apply_sampling_overrides(
            _require_sampling(slug, model_name, fallback_templates="code_default"),
            args,
        )

    backend = build_inference_backend_from_args(args)
    pipeline = CodingPipeline(backend)

    if args.probe_only:
        if benchmark_kind is CodingBenchmarkKind.HUMAN_EVAL:
            result = pipeline.run_human_eval(
                dataset_path=str(dataset_path),
                sampling=sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                probe_only=True,
                samples_per_task=1,
            )
        elif benchmark_kind is CodingBenchmarkKind.MBPP:
            result = pipeline.run_mbpp(
                dataset_path=str(dataset_path),
                sampling=sampling,
                cot_mode=cot_mode,
                batch_size=batch_size,
                sample_limit=sample_limit,
                probe_only=True,
                samples_per_task=1,
            )
        else:
            result = pipeline.run_livecodebench(
                dataset_path=str(dataset_path),
                cot_sampling=cot_sampling,
                final_sampling=final_sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                probe_only=True,
                samples_per_task=1,
            )
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        return 0

    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    job_name = run_context.job_name if run_context is not None else os.environ.get(
        "RWKV_SKILLS_JOB_NAME",
        _default_job_name(benchmark_kind, cot_mode),
    )
    task_state = prepare_task_execution(
        service=service,
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        job_name=job_name,
        run_mode=(run_context.run_mode if run_context is not None else None),
        sampling_config=build_task_sampling_config(
            cot_mode=cot_mode,
            avg_k=plan.avg_k,
            sampling_config=_sampling_payload(
                benchmark_kind,
                cot_mode,
                sampling=sampling,
                cot_sampling=cot_sampling,
                final_sampling=final_sampling,
            ),
            pass_ks=_DEFAULT_PASS_K,
            effective_sample_count=plan.effective_sample_count,
        ),
    )
    expected_count = plan_attempt_count(plan, max_pass_k=max(_DEFAULT_PASS_K))
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
        if benchmark_kind is CodingBenchmarkKind.HUMAN_EVAL:
            result = pipeline.run_human_eval(
                dataset_path=str(dataset_path),
                sampling=sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                record_indices=plan.sample_indices,
                eval_timeout=args.eval_timeout,
                eval_workers=args.eval_workers,
                pass_k=_DEFAULT_PASS_K,
                samples_per_task=plan.repeat_count,
                probe_only=False,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
            )
        elif benchmark_kind is CodingBenchmarkKind.MBPP:
            result = pipeline.run_mbpp(
                dataset_path=str(dataset_path),
                sampling=sampling,
                cot_mode=cot_mode,
                batch_size=batch_size,
                sample_limit=sample_limit,
                record_indices=plan.sample_indices,
                eval_timeout=args.eval_timeout,
                eval_workers=args.eval_workers,
                pass_k=_DEFAULT_PASS_K,
                samples_per_task=plan.repeat_count,
                probe_only=False,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
            )
        else:
            result = pipeline.run_livecodebench(
                dataset_path=str(dataset_path),
                cot_sampling=cot_sampling,
                final_sampling=final_sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                record_indices=plan.sample_indices,
                eval_timeout=args.eval_timeout,
                eval_workers=args.eval_workers,
                pass_k=_DEFAULT_PASS_K,
                samples_per_task=plan.repeat_count,
                probe_only=False,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
            )
    except BaseException:
        runtime.handle_attempt_stage_failure(writer)
        raise

    completions_payloads = runtime.complete_attempt_stage(writer)
    try:
        if benchmark_kind is CodingBenchmarkKind.HUMAN_EVAL:
            eval_metrics, eval_payloads = evaluate_human_eval(
                completions_payloads,
                dataset_path=str(dataset_path),
                pass_k=_DEFAULT_PASS_K,
                n_workers=args.eval_workers,
                timeout=args.eval_timeout,
            )
        elif benchmark_kind is CodingBenchmarkKind.MBPP:
            eval_metrics, eval_payloads = evaluate_mbpp_dataset(
                completions_payloads,
                dataset_path=str(dataset_path),
                pass_k=_DEFAULT_PASS_K,
                n_workers=args.eval_workers,
                timeout=args.eval_timeout,
            )
        else:
            eval_metrics, eval_payloads = evaluate_livecodebench_dataset(
                completions_payloads,
                dataset_path=str(dataset_path),
                pass_k=_DEFAULT_PASS_K,
                n_workers=args.eval_workers,
                timeout=args.eval_timeout,
            )

        rows = [
            (int(payload["sample_index"]), int(payload["repeat_index"]), bool(payload["is_passed"]))
            for payload in eval_payloads
        ]
        metrics_payload = dict(eval_metrics)
        metrics_payload.update(
            build_avg_k_metrics(
                rows,
                avg_k=plan.avg_k,
                primary_name="pass@1",
                primary_value=float(eval_metrics.get("pass@1", 0.0)),
            )
        )

        runtime.ingest_eval_payloads(eval_payloads)
        runtime.run_checker(model_name=model_name)
        score_payload = make_score_payload(
            slug,
            is_cot=cot_mode.is_cot,
            model_name=model_name,
            metrics=metrics_payload,
            samples=len(completions_payloads),
            problems=result.problem_count,
            task=job_name,
            task_details=build_plan_task_details(plan, cot_mode=cot_mode.value),
            extra={"cot_mode": cot_mode.value},
        )
        runtime.record_score(score_payload)
    except BaseException as exc:
        runtime.fail_task(error=str(exc))
        raise
    _print_done_message(benchmark_kind, cot_mode, result.sample_count)
    return 0


__all__ = ["CodingBenchmarkKind", "main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
