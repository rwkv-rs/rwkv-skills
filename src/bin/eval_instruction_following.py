from __future__ import annotations

"""Run instruction-following evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from dataclasses import replace

from src.eval.metrics.instruction_following.metrics import evaluate_instruction_following
from src.eval.results.layout import eval_details_path, jsonl_path, write_scores_json
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, canonical_slug
from src.eval.evaluators.instruction_following import InstructionFollowingPipeline, DEFAULT_SAMPLING
from src.eval.checkers.llm_checker import run_llm_checker
from src.infer.model import ModelLoadConfig

DEFAULT_AVG_K: tuple[int, ...] = ()
IFEVAL_AVG_K = (4,)


def _resolve_output_path(dataset: str, model_path: str, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=False, model_name=Path(model_path).stem)


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
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
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
    out_path = _resolve_output_path(str(dataset_path), args.model_path, args.output)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = InstructionFollowingPipeline(config)
    avg_k_final = _resolve_avg_k(slug, args)
    report_avg_k = _report_avg_k(slug, avg_k_final)
    samples_per_prompt = max(_max_k(avg_k_final), 1)

    sampling = DEFAULT_SAMPLING
    if args.stop_token:
        sampling = replace(sampling, stop_tokens=tuple(args.stop_token))
    ban_tokens = tuple(args.ban_token) if args.ban_token else None

    result = pipeline.run(
        dataset_path=str(dataset_path),
        output_path=str(out_path),
        sampling=sampling,
        batch_size=max(1, args.batch_size),
        sample_limit=args.max_samples,
        enable_think=bool(args.enable_think),
        stop_tokens=sampling.stop_tokens,
        ban_tokens=ban_tokens,
        samples_per_prompt=samples_per_prompt,
    )
    eval_path = eval_details_path(slug, is_cot=False, model_name=Path(args.model_path).stem)
    metrics = evaluate_instruction_following(
        out_path,
        dataset_path=str(dataset_path),
        eval_output_path=eval_path,
        strict=True,
        avg_k=avg_k_final,
    )
    avg_payload = _filter_metrics_by_k(metrics.avg_at_k, report_avg_k, "avg@") or (metrics.avg_at_k or {})
    score_path = write_scores_json(
        slug,
        is_cot=False,
        model_name=Path(args.model_path).stem,
        metrics={
            "prompt_accuracy": metrics.prompt_accuracy,
            "instruction_accuracy": metrics.instruction_accuracy,
            **avg_payload,
        },
        samples=metrics.samples,
        log_path=out_path,
        task="instruction_following",
        task_details={
            "tier0_accuracy": metrics.tier0_accuracy,
            "tier1_accuracy": metrics.tier1_accuracy,
            "eval_details_path": str(eval_path),
            **({"avg_curve": metrics.avg_at_k} if metrics.avg_at_k and avg_payload != metrics.avg_at_k else {}),
        },
    )
    print(f"✅ instruction-following done: {result.sample_count} samples -> {result.output_path}")
    print(f"📄 eval details saved: {eval_path}")
    print(f"📊 scores saved: {score_path}")
    run_llm_checker(eval_path, model_name=Path(args.model_path).stem)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
