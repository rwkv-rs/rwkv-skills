from __future__ import annotations

"""Field-oriented instruction-following runner aligned with rwkv-rs datasets."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from dataclasses import replace

from src.eval.benchmark_registry import CoTMode
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.datasets.data_loader.instruction_following import JsonlInstructionFollowingLoader
from src.eval.execution_plan import avg_k_metric_key, build_attempt_keys, build_auto_avg_k_execution_plan, plan_attempt_count
from src.eval.field_common import build_plan_task_details, build_task_sampling_config, set_task_env
from src.eval.metrics.instruction_following.metrics import evaluate_instruction_following
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.job_env import ensure_job_id
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.eval.evaluating import prepare_task_execution, run_checker_for_task
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, canonical_slug

DEFAULT_AVG_K: tuple[float, ...] = ()
IFEVAL_AVG_K: tuple[float, ...] = ()


def _max_k(values) -> int:
    return max(values) if values else 0


def _resolve_avg_k(slug: str, args: argparse.Namespace) -> tuple[float, ...]:
    if args.avg_k:
        return tuple(args.avg_k)
    lower_slug = canonical_slug(str(slug))
    if lower_slug.startswith("ifeval"):
        return IFEVAL_AVG_K
    return DEFAULT_AVG_K


def _report_avg_k(slug: str, final_avg_k: tuple[float, ...]) -> tuple[float, ...]:
    lower_slug = canonical_slug(str(slug))
    if lower_slug.startswith("ifeval"):
        return IFEVAL_AVG_K
    return final_avg_k


def _filter_metrics_by_k(metric_map, ks: tuple[float, ...], prefix: str) -> dict[str, float]:
    if not metric_map or not ks:
        return {}
    allowed = {avg_k_metric_key(float(k)) for k in ks if float(k) > 0}
    filtered: dict[str, float] = {}
    for key, value in metric_map.items():
        if key in allowed and key.startswith(prefix):
            filtered[key] = value
    return filtered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV instruction-following evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--enable-think", action="store_true", help="Append <think for think-style prompting")
    parser.add_argument("--stop-token", action="append", type=int, help="Extra stop tokens (can repeat)")
    parser.add_argument("--ban-token", action="append", type=int, help="Tokens to ban (can repeat)")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute from generated samples (IFEval 默认 4)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    from src.eval.instruction_following.pipeline import InstructionFollowingPipeline
    from src.infer.model import ModelLoadConfig

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    dataset_records = JsonlInstructionFollowingLoader(str(dataset_path)).load()
    plan = build_auto_avg_k_execution_plan(slug, len(dataset_records))
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = InstructionFollowingPipeline(config)
    avg_k_final = (plan.avg_k,)
    report_avg_k = (plan.avg_k,)
    samples_per_prompt = max(plan.repeat_count, 1)
    records = dataset_records
    expected_count = plan_attempt_count(plan, max_pass_k=1)

    sampling = resolve_sampling_config(
        slug,
        Path(args.model_path).stem,
        fallback_templates="instruction_following_default",
    )
    if sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({Path(args.model_path).stem})")
    if args.stop_token:
        sampling = replace(sampling, stop_tokens=tuple(args.stop_token))
    ban_tokens = tuple(args.ban_token) if args.ban_token else None

    init_orm(DEFAULT_DB_CONFIG)
    
    service = EvalDbService()
    job_name = os.environ.get("RWKV_SKILLS_JOB_NAME", "instruction_following")
    task_state = prepare_task_execution(
        service=service,
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        job_name=job_name,
        sampling_config=build_task_sampling_config(
            cot_mode=CoTMode.NO_COT,
            avg_k=plan.avg_k,
            sampling_config={"stage1": sampling_config_to_dict(sampling)},
            effective_sample_count=plan.effective_sample_count,
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
    try:
        result = pipeline.run(
            dataset_path=str(dataset_path),
            sampling=sampling,
            batch_size=max(1, args.batch_size),
            record_indices=plan.sample_indices,
            enable_think=bool(args.enable_think),
            stop_tokens=sampling.stop_tokens,
            ban_tokens=ban_tokens,
            samples_per_prompt=samples_per_prompt,
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
    metrics = evaluate_instruction_following(
        completions_payloads,
        dataset_path=str(dataset_path),
        strict=True,
        avg_k=avg_k_final,
    )
    avg_payload = _filter_metrics_by_k(metrics.avg_at_k, report_avg_k, "avg@") or (metrics.avg_at_k or {})
    service.ingest_eval_payloads(payloads=metrics.payloads or [], task_id=task_id)
    run_checker_for_task(service=service, task_id=task_id, model_name=Path(args.model_path).stem)
    score_payload = make_score_payload(
        slug,
        is_cot=False,
        model_name=Path(args.model_path).stem,
        metrics={
            "prompt_accuracy": metrics.prompt_accuracy,
            "instruction_accuracy": metrics.instruction_accuracy,
            **avg_payload,
        },
        samples=metrics.samples,
        task=job_name,
        task_details={
            "tier0_accuracy": metrics.tier0_accuracy,
            "tier1_accuracy": metrics.tier1_accuracy,
            **build_plan_task_details(plan, cot_mode=CoTMode.NO_COT.value),
            **({"avg_curve": metrics.avg_at_k} if metrics.avg_at_k and avg_payload != metrics.avg_at_k else {}),
        },
        extra={"cot_mode": CoTMode.NO_COT.value},
    )
    service.record_score_payload(
        payload=score_payload,
        task_id=task_id,
    )
    print(f"✅ instruction-following done: {result.sample_count} samples")
    return 0


__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
