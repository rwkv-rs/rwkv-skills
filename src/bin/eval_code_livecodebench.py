from __future__ import annotations

"""Run LiveCodeBench code generation + evaluation for RWKV models."""

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.checkers.llm_checker import run_llm_checker
from src.eval.evaluators.coding import CodingPipeline
from src.eval.metrics.code_generation.livecodebench import evaluate_livecodebench_dataset
from src.eval.results.layout import eval_details_path, jsonl_path, write_scores_json
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.model import ModelLoadConfig


def _resolve_output_path(dataset: str, model_path: str, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=True, model_name=Path(model_path).stem)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV LiveCodeBench evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="LiveCodeBench JSONL path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of problems for quick runs")
    parser.add_argument("--max-tokens", type=int, help="Clamp generation length")
    parser.add_argument("--temperature", type=float, help="Override sampling temperature")
    parser.add_argument("--top-k", type=int, help="Override sampling top-k")
    parser.add_argument("--top-p", type=float, help="Override sampling top-p")
    parser.add_argument("--eval-timeout", type=float, default=3.0, help="Seconds per test execution")
    parser.add_argument("--eval-workers", type=int, default=4, help="Parallel workers for evaluation")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="只跑一批生成用于 batch 探测，不评测、不写盘",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to report (default: 1)",
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
    model_name = Path(args.model_path).stem
    cot_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="cot",
        fallback_templates="full_code_cot_default",
    )
    final_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="final",
        fallback_templates="full_code_final_default",
    )
    if cot_sampling is None or final_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")
    if args.max_tokens:
        cot_sampling = cot_sampling.clamp(args.max_tokens)
        final_sampling = final_sampling.clamp(args.max_tokens)
    if args.temperature is not None:
        cot_sampling = replace(cot_sampling, temperature=args.temperature)
        final_sampling = replace(final_sampling, temperature=args.temperature)
    if args.top_k is not None:
        cot_sampling = replace(cot_sampling, top_k=args.top_k)
        final_sampling = replace(final_sampling, top_k=args.top_k)
    if args.top_p is not None:
        cot_sampling = replace(cot_sampling, top_p=args.top_p)
        final_sampling = replace(final_sampling, top_p=args.top_p)

    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = CodingPipeline(config)
    batch_size = max(1, args.batch_size)
    default_pass_k = (1,)
    pass_k = (1,) if args.probe_only else (tuple(args.pass_k) if args.pass_k else default_pass_k)
    sample_limit = batch_size if args.probe_only else args.max_samples
    result = pipeline.run_livecodebench(
        dataset_path=str(dataset_path),
        output_path=str(out_path),
        cot_sampling=cot_sampling,
        final_sampling=final_sampling,
        batch_size=batch_size,
        sample_limit=sample_limit,
        eval_timeout=args.eval_timeout,
        eval_workers=args.eval_workers,
        pass_k=pass_k,
        probe_only=args.probe_only,
        write_output=not args.probe_only,
    )

    if args.probe_only:
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        return 0

    print(f"✅ LiveCodeBench 生成完成：{result.sample_count} completions -> {result.output_path}")

    eval_path = eval_details_path(slug, is_cot=True, model_name=Path(args.model_path).stem)
    eval_metrics = evaluate_livecodebench_dataset(
        out_path,
        dataset_path=str(dataset_path),
        eval_output_path=eval_path,
        pass_k=pass_k,
        n_workers=args.eval_workers,
        timeout=args.eval_timeout,
    )
    print(f"LiveCodeBench 评测: {eval_metrics} (详情: {eval_path})")
    score_path = write_scores_json(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics=eval_metrics or {},
        samples=result.sample_count,
        problems=result.problem_count,
        log_path=out_path,
        task="code_livecodebench",
        task_details={
            "eval_details_path": str(eval_path),
        },
    )
    print(f"📊 scores saved: {score_path}")
    run_llm_checker(eval_path, model_name=Path(args.model_path).stem)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
