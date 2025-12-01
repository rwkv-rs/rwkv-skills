from __future__ import annotations

"""Run instruction-following evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from dataclasses import replace

from src.eval.metrics.instruction_following.metrics import (
    evaluate_samples,
    load_samples_from_jsonl,
)
from src.eval.results.layout import jsonl_path, write_scores_json
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.instruction_following import InstructionFollowingPipeline, DEFAULT_SAMPLING
from src.infer.model import ModelLoadConfig


def _resolve_output_path(dataset: str, model_path: str, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=False, model_name=Path(model_path).stem)


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
    parser.add_argument("--output", help="Output JSONL path (defaults to results/logs layout)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_path = resolve_or_prepare_dataset(args.dataset)
    except Exception as exc:
        print(f"❌ 数据集准备失败: {exc}")
        return 1
    slug = infer_dataset_slug_from_path(str(dataset_path))
    out_path = _resolve_output_path(str(dataset_path), args.model_path, args.output)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = InstructionFollowingPipeline(config)

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
    )
    samples = load_samples_from_jsonl(out_path)
    metrics = evaluate_samples(samples, strict=True)
    score_path = write_scores_json(
        slug,
        is_cot=False,
        model_name=Path(args.model_path).stem,
        metrics={
            "prompt_accuracy": metrics.prompt_accuracy,
            "instruction_accuracy": metrics.instruction_accuracy,
        },
        samples=len(samples),
        log_path=out_path,
        task="instruction_following",
        task_details={
            "tier0_accuracy": metrics.tier0_accuracy,
            "tier1_accuracy": metrics.tier1_accuracy,
        },
    )
    print(f"✅ instruction-following done: {result.sample_count} samples -> {result.output_path}")
    print(f"📊 scores saved: {score_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
