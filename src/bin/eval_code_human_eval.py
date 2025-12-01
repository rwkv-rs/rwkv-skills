from __future__ import annotations

"""Run HumanEval code generation + evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from dataclasses import replace

from src.eval.results.layout import jsonl_path
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.coding import CodingPipeline, DEFAULT_CODE_SAMPLING
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
    parser = argparse.ArgumentParser(description="RWKV HumanEval evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="HumanEval JSONL path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--samples-per-task", type=int, default=1, help="Number of completions per task")
    parser.add_argument("--max-samples", type=int, help="Limit number of problems for quick runs")
    parser.add_argument("--max-tokens", type=int, help="Clamp generation length")
    parser.add_argument("--temperature", type=float, help="Override sampling temperature")
    parser.add_argument("--top-k", type=int, help="Override sampling top-k")
    parser.add_argument("--top-p", type=float, help="Override sampling top-p")
    parser.add_argument("--eval-timeout", type=float, default=3.0, help="Seconds per test execution")
    parser.add_argument("--eval-workers", type=int, default=4, help="Parallel workers for evaluation")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/logs layout)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = _resolve_output_path(args.dataset, args.model_path, args.output)

    sampling = DEFAULT_CODE_SAMPLING
    if args.max_tokens:
        sampling = sampling.clamp(args.max_tokens)
    if args.temperature is not None:
        sampling = replace(sampling, temperature=args.temperature)
    if args.top_k is not None:
        sampling = replace(sampling, top_k=args.top_k)
    if args.top_p is not None:
        sampling = replace(sampling, top_p=args.top_p)

    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = CodingPipeline(config)
    result = pipeline.run_human_eval(
        dataset_path=args.dataset,
        output_path=str(out_path),
        sampling=sampling,
        batch_size=max(1, args.batch_size),
        samples_per_task=max(1, args.samples_per_task),
        sample_limit=args.max_samples,
        eval_timeout=args.eval_timeout,
        eval_workers=args.eval_workers,
    )

    print(f"✅ HumanEval生成完成：{result.sample_count} completions -> {result.output_path}")
    if result.eval_results:
        print(f"HumanEval 评测: {result.eval_results} (详情: {result.eval_details_path})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
