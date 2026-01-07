from __future__ import annotations

"""Run direct multiple-choice evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.metrics.multi_choice import evaluate_multiple_choice
from src.eval.results.layout import eval_details_path, jsonl_path, write_scores_json
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.eval.checkers.llm_checker import run_llm_checker
from src.infer.model import ModelLoadConfig


def _resolve_output_path(dataset: str, model_path: str, user_path: str | None, is_cot: bool) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=is_cot, model_name=Path(model_path).stem)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV multiple-choice (direct) evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for scoring")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        return 1
    slug = infer_dataset_slug_from_path(str(dataset_path))
    out_path = _resolve_output_path(str(dataset_path), args.model_path, args.output, is_cot=False)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = MultipleChoicePipeline(config, target_token_format=args.target_token_format)

    # Quick validation of dataset readability before heavy model init
    _ = JsonlMultipleChoiceLoader(str(dataset_path)).load()

    result = pipeline.run_direct(
        dataset_path=str(dataset_path),
        output_path=str(out_path),
        sample_limit=args.max_samples,
    )
    eval_path = eval_details_path(slug, is_cot=False, model_name=Path(args.model_path).stem)
    metrics = evaluate_multiple_choice(
        out_path,
        dataset_path=dataset_path,
        eval_output_path=eval_path,
    )
    score_path = write_scores_json(
        slug,
        is_cot=False,
        model_name=Path(args.model_path).stem,
        metrics={"accuracy": metrics.accuracy},
        samples=metrics.samples,
        log_path=out_path,
        task="multiple_choice",
        task_details={
            "accuracy_by_subject": metrics.accuracy_by_subject,
            "eval_details_path": str(eval_path),
        },
    )
    print(f"✅ direct multiple-choice done: {result.sample_count} samples -> {result.output_path}")
    print(f"📄 eval details saved: {eval_path}")
    print(f"📊 scores saved: {score_path}")
    run_llm_checker(eval_path, model_name=Path(args.model_path).stem)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
