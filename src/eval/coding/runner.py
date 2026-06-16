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
from src.eval.field_common import build_plan_task_details, build_task_sampling_config, resolve_configured_k_plan, set_task_env
from src.eval.k_values import filter_metrics_by_k
from src.eval.long_doc_evidence import LONG_DOC_MODE_CHOICES, LongDocEvidenceConfig
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    require_completion_style_remote_protocol,
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
    SWE_BENCH = "swe_bench"


_HUMAN_EVAL_JOB_NAMES = frozenset({"code_human_eval"})
_MBPP_JOB_NAMES = frozenset({"code_mbpp"})
_LIVECODEBENCH_JOB_NAMES = frozenset({"code_livecodebench"})
_SWE_BENCH_JOB_NAMES = frozenset({"code_swe_bench"})
_DEFAULT_PASS_K: tuple[int, ...] = ()
_DEFAULT_AVG_K: tuple[float, ...] = ()


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
    parser.add_argument("--swebench-run-harness", action="store_true", help="Run the official SWE-bench Docker harness")
    parser.add_argument("--swebench-dataset-name", help="Official SWE-bench dataset_name passed to the harness")
    parser.add_argument("--swebench-run-id", help="Run id passed to the official SWE-bench harness")
    parser.add_argument("--swebench-cache-level", help="SWE-bench harness cache_level")
    parser.add_argument("--swebench-clean", action="store_true", help="Ask the SWE-bench harness to clean resources")
    parser.add_argument("--swebench-predictions-path", help="Where to write official SWE-bench predictions JSONL")
    parser.add_argument("--swebench-max-context-chars", type=int, help="Clamp retrieved context included in SWE-bench prompts")
    parser.add_argument("--swebench-harness-timeout-s", type=float, help="Wall-clock timeout for official SWE-bench harness")
    parser.add_argument(
        "--long-doc-mode",
        choices=LONG_DOC_MODE_CHOICES,
        default="off",
        help="SWE-bench retrieved-context chunk routing mode",
    )
    parser.add_argument("--long-doc-max-chars", type=int, default=1000, help="Long-document chunk max characters")
    parser.add_argument("--long-doc-overlap-lines", type=int, default=3, help="Long-document chunk overlap lines")
    parser.add_argument(
        "--long-doc-min-chars",
        type=int,
        default=6000,
        help="Minimum retrieved-context characters before chunk routing runs",
    )
    parser.add_argument(
        "--long-doc-max-evidence-chunks",
        type=int,
        default=4,
        help="Maximum selected chunks when compacting SWE-bench retrieved context",
    )
    parser.add_argument(
        "--long-doc-max-evidence-chars",
        type=int,
        default=6000,
        help="Maximum selected evidence characters for SWE-bench retrieved context",
    )
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
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to report (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute from generated samples (default: none; can be set in configs/<benchmark>.toml)",
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
    if job_names & _SWE_BENCH_JOB_NAMES:
        return CodingBenchmarkKind.SWE_BENCH
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
    if kind is CodingBenchmarkKind.SWE_BENCH:
        if requested_mode is not None and CoTMode(requested_mode) is not CoTMode.COT:
            raise ValueError("swe_bench only supports --cot-mode cot")
        return CoTMode.COT
    if kind is CodingBenchmarkKind.MBPP:
        if requested_mode is not None and CoTMode(requested_mode) is not CoTMode.NO_COT:
            raise ValueError("mbpp legacy-aligned runner only supports --cot-mode no_cot")
        return CoTMode.NO_COT
    if requested_mode is None:
        return CoTMode.NO_COT
    return CoTMode(requested_mode)


def _default_job_name(kind: CodingBenchmarkKind, cot_mode: CoTMode) -> str:
    if kind is CodingBenchmarkKind.HUMAN_EVAL:
        return "code_human_eval"
    if kind is CodingBenchmarkKind.LIVECODEBENCH:
        return "code_livecodebench"
    if kind is CodingBenchmarkKind.SWE_BENCH:
        return "code_swe_bench"
    return "code_mbpp"


def _print_done_message(kind: CodingBenchmarkKind, cot_mode: CoTMode, sample_count: int) -> None:
    if kind is CodingBenchmarkKind.HUMAN_EVAL:
        print(f"✅ HumanEval done: {sample_count} samples")
        return
    if kind is CodingBenchmarkKind.LIVECODEBENCH:
        print(f"✅ LiveCodeBench done: {sample_count} samples")
        return
    if kind is CodingBenchmarkKind.SWE_BENCH:
        print(f"✅ SWE-bench done: {sample_count} samples")
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
    if kind is CodingBenchmarkKind.SWE_BENCH:
        return {"stage1": sampling_config_to_dict(sampling)}
    if kind is CodingBenchmarkKind.MBPP and cot_mode is CoTMode.COT:
        return {
            "stage1": sampling_config_to_dict(sampling),
            "stage2": sampling_config_to_dict(sampling),
        }
    return {"stage1": sampling_config_to_dict(sampling)}


def _coding_long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
    mode = str(getattr(args, "long_doc_mode", "off") or "off").strip().lower()
    enabled = mode != "off"
    if mode == "off":
        mode = "lexical"
    return LongDocEvidenceConfig(
        enabled=enabled,
        mode=mode,  # type: ignore[arg-type]
        max_chunk_chars=max(1, int(getattr(args, "long_doc_max_chars", 1000) or 1000)),
        overlap_lines=max(0, int(getattr(args, "long_doc_overlap_lines", 3) or 0)),
        min_long_text_chars=max(1, int(getattr(args, "long_doc_min_chars", 6000) or 6000)),
        max_evidence_chunks=max(1, int(getattr(args, "long_doc_max_evidence_chunks", 4) or 4)),
        max_evidence_chars=max(1, int(getattr(args, "long_doc_max_evidence_chars", 6000) or 6000)),
    )


def _long_doc_config_payload(config: LongDocEvidenceConfig) -> dict[str, object]:
    return {
        "enabled": bool(config.enabled),
        "mode": config.mode if config.enabled else "off",
        "max_chunk_chars": int(config.max_chunk_chars),
        "overlap_lines": int(config.overlap_lines),
        "min_long_text_chars": int(config.min_long_text_chars),
        "max_evidence_chunks": int(config.max_evidence_chunks),
        "max_evidence_chars": int(config.max_evidence_chars),
    }


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
    from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
    from src.eval.metrics.code_generation.evaluate import evaluate_human_eval, evaluate_mbpp_dataset
    from src.eval.metrics.at_k import compute_avg_at_k
    from src.eval.metrics.code_generation.livecodebench import evaluate_livecodebench_dataset
    from src.eval.coding.swe_bench import evaluate_swebench_predictions, infer_harness_dataset_name
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
    from src.db.async_writer import CompletionWriteWorker
    from src.db.eval_service import create_eval_service, init_eval_store

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    benchmark_kind = _resolve_benchmark_kind(slug, CodingBenchmarkKind(args.benchmark_kind))
    cot_mode = _resolve_cot_mode(benchmark_kind, args.cot_mode)
    completion_style_remote = False
    if benchmark_kind in {CodingBenchmarkKind.HUMAN_EVAL, CodingBenchmarkKind.MBPP}:
        completion_style_remote = require_completion_style_remote_protocol(
            args,
            benchmark_name=f"{benchmark_kind.value} coding benchmark",
        )

    dataset_records = JsonlCodeGenerationLoader(str(dataset_path)).load()
    model_name = resolve_backend_model_name(args)
    k_plan = resolve_configured_k_plan(
        slug=slug,
        model_name=model_name,
        dataset_len=len(dataset_records),
        args=args,
        default_pass_k=_DEFAULT_PASS_K,
        default_avg_k=_DEFAULT_AVG_K,
    )
    plan = k_plan.plan
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    batch_size = max(1, args.batch_size)
    sample_limit = batch_size if args.probe_only else k_plan.sample_limit

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
    long_doc_config = _coding_long_doc_config(args) if benchmark_kind is CodingBenchmarkKind.SWE_BENCH else None

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
        elif benchmark_kind is CodingBenchmarkKind.LIVECODEBENCH:
            result = pipeline.run_livecodebench(
                dataset_path=str(dataset_path),
                cot_sampling=cot_sampling,
                final_sampling=final_sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                probe_only=True,
                samples_per_task=1,
            )
        else:
            result = pipeline.run_swe_bench(
                dataset_path=str(dataset_path),
                sampling=sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                probe_only=True,
                samples_per_task=1,
                max_context_chars=args.swebench_max_context_chars,
                long_doc_config=long_doc_config,
            )
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        return 0

    init_eval_store(DEFAULT_DB_CONFIG)
    service = create_eval_service()
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
            pass_ks=k_plan.pass_k,
            effective_sample_count=plan.effective_sample_count,
        ),
    )
    expected_count = plan_attempt_count(plan, max_pass_k=1)
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
                pass_k=k_plan.pass_k,
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
                pass_k=k_plan.pass_k,
                samples_per_task=plan.repeat_count,
                probe_only=False,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
            )
        elif benchmark_kind is CodingBenchmarkKind.LIVECODEBENCH:
            result = pipeline.run_livecodebench(
                dataset_path=str(dataset_path),
                cot_sampling=cot_sampling,
                final_sampling=final_sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                record_indices=plan.sample_indices,
                eval_timeout=args.eval_timeout,
                eval_workers=args.eval_workers,
                pass_k=k_plan.pass_k,
                samples_per_task=plan.repeat_count,
                probe_only=False,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
            )
        else:
            result = pipeline.run_swe_bench(
                dataset_path=str(dataset_path),
                sampling=sampling,
                batch_size=batch_size,
                sample_limit=sample_limit,
                record_indices=plan.sample_indices,
                pass_k=k_plan.pass_k,
                samples_per_task=plan.repeat_count,
                probe_only=False,
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
                max_context_chars=args.swebench_max_context_chars,
                long_doc_config=long_doc_config,
            )
    except Exception:
        runtime.handle_attempt_stage_failure(writer)
        raise

    completions_payloads = runtime.complete_attempt_stage(writer)
    try:
        if benchmark_kind is CodingBenchmarkKind.HUMAN_EVAL:
            eval_metrics, eval_payloads = evaluate_human_eval(
                completions_payloads,
                dataset_path=str(dataset_path),
                pass_k=k_plan.pass_k,
                n_workers=args.eval_workers,
                timeout=args.eval_timeout,
            )
        elif benchmark_kind is CodingBenchmarkKind.MBPP:
            eval_metrics, eval_payloads = evaluate_mbpp_dataset(
                completions_payloads,
                dataset_path=str(dataset_path),
                pass_k=k_plan.pass_k,
                n_workers=args.eval_workers,
                timeout=args.eval_timeout,
            )
        elif benchmark_kind is CodingBenchmarkKind.LIVECODEBENCH:
            eval_metrics, eval_payloads = evaluate_livecodebench_dataset(
                completions_payloads,
                dataset_path=str(dataset_path),
                pass_k=k_plan.pass_k,
                n_workers=args.eval_workers,
                timeout=args.eval_timeout,
            )
            predictions_path = None
        else:
            predictions_path = Path(args.swebench_predictions_path) if args.swebench_predictions_path else (
                Path("results") / "swebench_predictions" / f"task_{task_id}" / "predictions.jsonl"
            )
            harness_dataset = args.swebench_dataset_name or infer_harness_dataset_name(dataset_path)
            eval_metrics, eval_payloads, predictions_path = evaluate_swebench_predictions(
                completions_payloads,
                dataset_path=str(dataset_path),
                model_name=model_name,
                predictions_path=predictions_path,
                run_harness=bool(args.swebench_run_harness),
                dataset_name=harness_dataset,
                split="test",
                run_id=args.swebench_run_id or f"rwkv-skills-task-{task_id}",
                max_workers=max(1, int(args.eval_workers or 1)),
                cache_level=args.swebench_cache_level,
                clean=bool(args.swebench_clean),
                timeout_s=args.swebench_harness_timeout_s,
            )

        rows = [
            (int(payload["sample_index"]), int(payload["repeat_index"]), bool(payload["is_passed"]))
            for payload in eval_payloads
        ]
        avg_metrics_all = compute_avg_at_k(rows, k_plan.avg_k)
        metrics_payload: dict[str, float] = {}
        pass_payload: dict[str, float] = {}
        avg_payload: dict[str, float] = {}
        if benchmark_kind is CodingBenchmarkKind.SWE_BENCH:
            metrics_payload.update(eval_metrics or {})
            if avg_metrics_all:
                metrics_payload.update(avg_metrics_all)
        else:
            pass_payload = filter_metrics_by_k(eval_metrics, k_plan.report_pass_k, "pass@")
            if k_plan.report_pass_k and not pass_payload:
                pass_payload = eval_metrics or {}
            if pass_payload:
                metrics_payload.update(pass_payload)
            avg_payload = filter_metrics_by_k(avg_metrics_all, k_plan.report_avg_k, "avg@")
            if k_plan.report_avg_k and not avg_payload:
                avg_payload = avg_metrics_all or {}
            if avg_payload:
                metrics_payload.update(avg_payload)
        task_details: dict[str, object] = build_plan_task_details(plan, cot_mode=cot_mode.value)
        if eval_metrics and pass_payload != eval_metrics:
            task_details["pass_curve"] = eval_metrics
        if avg_metrics_all and avg_payload != avg_metrics_all:
            task_details["avg_curve"] = avg_metrics_all

        runtime.ingest_eval_payloads(eval_payloads)
        if eval_payloads:
            runtime.run_checker(model_name=model_name)
        score_payload = make_score_payload(
            slug,
            is_cot=cot_mode.is_cot,
            model_name=model_name,
            metrics=metrics_payload,
            samples=len(completions_payloads),
            problems=result.problem_count,
            task=job_name,
            task_details=task_details,
            extra={
                "cot_mode": cot_mode.value,
                "infer_protocol": getattr(args, "infer_protocol", "local"),
                "completion_style_remote": completion_style_remote,
                **(
                    {
                        "code_extraction": "think_and_python_fence_v1",
                        "generation_contract": "legacy_completion_style_code",
                    }
                    if benchmark_kind in {CodingBenchmarkKind.HUMAN_EVAL, CodingBenchmarkKind.MBPP}
                    else {}
                ),
                **(
                    {
                        "swebench_predictions_path": str(predictions_path),
                        "swebench_harness_ran": bool(args.swebench_run_harness),
                        "swebench_long_doc": _long_doc_config_payload(long_doc_config),
                    }
                    if benchmark_kind is CodingBenchmarkKind.SWE_BENCH
                    else {}
                ),
            },
        )
        runtime.record_score(score_payload)
    except Exception as exc:
        runtime.fail_task(error=str(exc))
        raise
    _print_done_message(benchmark_kind, cot_mode, result.sample_count)
    return 0


__all__ = ["CodingBenchmarkKind", "main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
