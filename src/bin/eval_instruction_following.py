from __future__ import annotations

"""Run instruction-following evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from dataclasses import replace

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.metrics.instruction_following.metrics import evaluate_instruction_following
from src.eval.results.payloads import make_score_payload
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.db.database import DatabaseManager
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, canonical_slug
from src.eval.evaluators.instruction_following import InstructionFollowingPipeline
from src.infer.model import ModelLoadConfig

DEFAULT_AVG_K: tuple[int, ...] = ()
IFEVAL_AVG_K = (4,)


def _max_k(values) -> int:
    return max(values) if values else 0


def _resolve_avg_k(slug: str, args: argparse.Namespace) -> tuple[int, ...]:
    if args.avg_k:
        return tuple(args.avg_k)
    lower_slug = canonical_slug(str(slug))
    if lower_slug.startswith("ifeval"):
        return IFEVAL_AVG_K
    return DEFAULT_AVG_K


def _report_avg_k(slug: str, final_avg_k: tuple[int, ...]) -> tuple[int, ...]:
    lower_slug = canonical_slug(str(slug))
    if lower_slug.startswith("ifeval"):
        return IFEVAL_AVG_K
    return final_avg_k


def _filter_metrics_by_k(metric_map, ks: tuple[int, ...], prefix: str) -> dict[str, float]:
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
    parser.add_argument("--db-write-batch", type=int, default=128, help="DB completion write batch size")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
    parser.add_argument(
        "--no-param-search",
        action="store_true",
        help="Compatibility flag (no-op).",
    )
    parser.add_argument(
        "--avg-k",
        type=int,
        action="append",
        help="avg@k values to compute from generated samples (IFEval 默认 4)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    slug = infer_dataset_slug_from_path(str(dataset_path))
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = InstructionFollowingPipeline(config)
    avg_k_final = _resolve_avg_k(slug, args)
    report_avg_k = _report_avg_k(slug, avg_k_final)
    samples_per_prompt = max(_max_k(avg_k_final), 1)

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

    if not DEFAULT_DB_CONFIG.enabled:
        raise RuntimeError("DB 未启用：当前仅支持数据库写入模式。")
    db = DatabaseManager.instance()
    db.initialize(DEFAULT_DB_CONFIG)
    service = EvalDbService(db)
    version_id = service.get_or_create_version(
        job_name="eval_instruction_following",
        job_id=os.environ.get("RWKV_SKILLS_JOB_ID"),
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        allow_resume=True,
    )
    os.environ["RWKV_SKILLS_VERSION_ID"] = version_id
    skip_keys = service.list_completion_keys(
        version_id=version_id,
        is_param_search=False,
    )
    writer = CompletionWriteWorker(
        service=service,
        version_id=version_id,
        is_param_search=False,
        batch_size=args.db_write_batch,
        max_queue=args.db_write_queue,
    )
    result = pipeline.run(
        dataset_path=str(dataset_path),
        sampling=sampling,
        batch_size=max(1, args.batch_size),
        sample_limit=args.max_samples,
        enable_think=bool(args.enable_think),
        stop_tokens=sampling.stop_tokens,
        ban_tokens=ban_tokens,
        samples_per_prompt=samples_per_prompt,
        skip_keys=skip_keys,
        on_record=writer.enqueue,
    )
    writer.close()
    completions_payloads = service.list_completion_payloads(
        version_id=version_id,
        is_param_search=False,
    )
    metrics = evaluate_instruction_following(
        completions_payloads,
        dataset_path=str(dataset_path),
        strict=True,
        avg_k=avg_k_final,
    )
    avg_payload = _filter_metrics_by_k(metrics.avg_at_k, report_avg_k, "avg@") or (metrics.avg_at_k or {})
    service.ingest_eval_payloads(
        payloads=metrics.payloads or [],
        version_id=version_id,
        is_param_search=False,
    )
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
        task="instruction_following",
        task_details={
            "tier0_accuracy": metrics.tier0_accuracy,
            "tier1_accuracy": metrics.tier1_accuracy,
            **({"avg_curve": metrics.avg_at_k} if metrics.avg_at_k and avg_payload != metrics.avg_at_k else {}),
        },
    )
    service.record_score_payload(
        payload=score_payload,
        version_id=version_id,
        is_param_search=False,
    )
    export_version_results(
        service,
        version_id=version_id,
        is_param_search=False,
    )
    print(f"✅ instruction-following done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
