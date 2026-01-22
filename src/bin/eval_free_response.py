from __future__ import annotations

"""Run chain-of-thought free-form QA evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.eval.metrics.free_response import (
    compute_pass_at_k,
    compute_avg_at_k,
    evaluate_free_response,
)
from src.eval.benchmark_config import resolve_benchmark_model_config, resolve_sampling_config
from src.eval.checkers.llm_checker import run_llm_checker
from src.eval.results.layout import (
    eval_details_path,
    jsonl_path,
    write_scores_json,
)
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.infra.database import DatabaseManager
from src.infra.eval_db_service import EvalDbService
from src.eval.evaluators.free_response import FreeResponsePipeline
from src.infer.model import ModelLoadConfig


DEFAULT_PASS_K = (1,)
DEFAULT_AVG_K: tuple[int, ...] = ()

def _resolve_output_path(dataset: str, model_path: str, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=True, model_name=Path(model_path).stem)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV free-form CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--cot-max-tokens", type=int, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, help="Clamp final answer generation length")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe and skip scoring",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to generate for and compute (default: 1; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--avg-k",
        type=int,
        action="append",
        help="avg@k values to compute from generated samples (default: none; can be set in configs/<benchmark>.toml)",
    )
    return parser.parse_args(argv)


def _max_k(values: Sequence[int] | None) -> int:
    return max(values) if values else 0


def _resolve_pass_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[int, ...]:
    if args.pass_k:
        return tuple(args.pass_k)
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.pass_k is not None:
        return config.pass_k
    return DEFAULT_PASS_K


def _resolve_avg_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[int, ...]:
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


def _report_avg_k(slug: str, model_name: str, avg_k: tuple[int, ...]) -> tuple[int, ...]:
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.report_avg_k is not None:
        return config.report_avg_k
    return avg_k


def _filter_metrics_by_k(metric_map: dict[str, float] | None, ks: tuple[int, ...], prefix: str) -> dict[str, float]:
    if not metric_map or not ks:
        return {}
    allowed = {int(k) for k in ks if int(k) > 0}
    filtered: dict[str, float] = {}
    for key, value in metric_map.items():
        if not key.startswith(prefix):
            continue
        suffix_text = key[len(prefix) :]
        if not suffix_text.isdigit():
            continue
        suffix = int(suffix_text)
        if suffix in allowed:
            filtered[key] = value
    return filtered


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1

    slug = infer_dataset_slug_from_path(str(dataset_path))
    output_path = _resolve_output_path(str(dataset_path), args.model_path, args.output)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = FreeResponsePipeline(config)

    model_name = Path(args.model_path).stem
    pass_k = _resolve_pass_k(slug, model_name, args)
    avg_k = _resolve_avg_k(slug, model_name, args)
    report_pass_k = _report_pass_k(slug, model_name, pass_k)
    report_avg_k = _report_avg_k(slug, model_name, avg_k)

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

    if args.probe_only:
        _ = pipeline.run(
            dataset_path=str(dataset_path),
            output_path=str(output_path),
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=batch_size,
            pad_to_batch=True,
            pass_k=(1,),
            samples_per_task=1,
            probe_only=True,
            write_output=False,
        )
        print(f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}.")
        return 0

    samples_per_task = max(_max_k(pass_k), _max_k(avg_k), 1)
    result = pipeline.run(
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        cot_sampling=cot_sampling,
        final_sampling=final_sampling,
        batch_size=batch_size,
        sample_limit=args.max_samples,
        pad_to_batch=False,
        pass_k=pass_k,
        samples_per_task=samples_per_task,
    )

    eval_path = eval_details_path(slug, is_cot=True, model_name=Path(args.model_path).stem)
    evaluation = evaluate_free_response(
        output_path,
        dataset_path=str(dataset_path),
        eval_output_path=eval_path,
        judge=None,
    )
    pass_metrics_all = compute_pass_at_k(evaluation.rows, pass_k)
    avg_metrics_all = compute_avg_at_k(evaluation.rows, avg_k)
    task_details: dict[str, object] = {"eval_details_path": str(eval_path)}
    metrics_payload = {
        "exact_accuracy": evaluation.exact_accuracy,
        "judge_accuracy": evaluation.judge_accuracy,
    }

    pass_payload = _filter_metrics_by_k(pass_metrics_all, report_pass_k, "pass@")
    if report_pass_k and not pass_payload:
        pass_payload = pass_metrics_all or {}
    if pass_payload:
        metrics_payload.update(pass_payload)
    avg_payload = _filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@")
    if report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    if avg_payload:
        metrics_payload.update(avg_payload)
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all

    score_path = write_scores_json(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics=metrics_payload,
        samples=evaluation.samples,
        problems=result.problem_count,
        log_path=output_path,
        task="free_response",
        task_details=task_details,
    )
    if DEFAULT_DB_CONFIG.enabled:
        db = DatabaseManager.instance()
        db.initialize(DEFAULT_DB_CONFIG)
        service = EvalDbService(db)
        version_id = service.get_or_create_version(
            job_name="eval_free_response",
            job_id=os.environ.get("RWKV_SKILLS_JOB_ID"),
            dataset=str(slug),
            model=Path(args.model_path).stem,
            is_param_search=False,
            allow_resume=True,
        )
        os.environ["RWKV_SKILLS_VERSION_ID"] = version_id
        service.ingest_completions(
            completions_path=output_path,
            version_id=version_id,
            is_param_search=False,
        )
        service.ingest_eval(
            eval_path=eval_path,
            version_id=version_id,
            is_param_search=False,
        )
        service.record_score(
            score_path=score_path,
            version_id=version_id,
            is_param_search=False,
        )
    print(f"✅ CoT free-form done: {result.sample_count} samples -> {result.output_path}")
    print(f"📄 eval details saved: {eval_path}")
    print(f"📊 scores saved: {score_path}")
    run_llm_checker(eval_path, model_name=Path(args.model_path).stem)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
