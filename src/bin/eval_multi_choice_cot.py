from __future__ import annotations

"""Run chain-of-thought multiple-choice evaluation for RWKV models."""

import argparse
import os
import time
from pathlib import Path
from typing import Sequence

import torch

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.metrics.multi_choice import evaluate_multiple_choice
from src.eval.results.layout import eval_details_path, jsonl_path, write_scores_json
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, safe_slug
from src.eval.scheduler.profiler import update_batch_cache_locked
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.eval.checkers.llm_checker import run_llm_checker
from src.infer.model import ModelLoadConfig


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text


def _extract_gpu_from_device(device: str) -> str | None:
    if not device:
        return None
    if "cuda" not in device:
        return None
    parts = device.split(":", 1)
    if len(parts) == 2:
        return parts[1]
    return None


def _update_batch_cache(job_name: str, model_slug: str, gpu: str, batch_size: int) -> None:
    log_root = os.environ.get("RUN_LOG_DIR")
    if not log_root:
        return
    cache_path = Path(log_root).expanduser() / "batch_cache.json"
    update_batch_cache_locked(
        cache_path,
        lambda data: _update_batch_cache_record(data, job_name, model_slug, gpu, batch_size),
    )


def _update_batch_cache_record(
    data: dict[str, dict[str, dict[str, dict[str, object]]]],
    job_name: str,
    model_slug: str,
    gpu: str,
    batch_size: int,
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    job_map = data.setdefault(job_name, {})
    model_map = job_map.setdefault(model_slug, {})
    record = model_map.setdefault(gpu, {})
    record["batch"] = batch_size
    record.pop("last_error", None)
    record["last_probe"] = time.time()
    return data


def _resolve_output_path(dataset: str, model_path: str, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=True, model_name=Path(model_path).stem)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV multiple-choice CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation/scoring")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe and skip scoring",
    )
    parser.add_argument(
        "--no-param-search",
        action="store_true",
        help="Compatibility flag (no-op).",
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
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = MultipleChoicePipeline(config, target_token_format=args.target_token_format)

    # Quick validation of dataset readability before heavy model init
    _ = JsonlMultipleChoiceLoader(str(dataset_path)).load()

    cot_sampling = resolve_sampling_config(
        slug,
        model_name,
        fallback_templates="multi_choice_cot_default",
    )
    if cot_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")

    if args.probe_only:
        batch_size = max(1, args.batch_size)
        _ = pipeline.run_chain_of_thought(
            dataset_path=str(dataset_path),
            output_path=str(out_path),
            cot_sampling=cot_sampling,
            batch_size=batch_size,
            sample_limit=batch_size,
            min_prompt_count=batch_size,
            probe_only=True,
            write_output=False,
        )
        print(f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}.")
        return 0

    probe_only = False
    sample_limit: int | None = args.max_samples
    output_path = out_path
    min_prompt_count: int | None = None

    target_batch = max(1, args.batch_size)
    effective_batch = target_batch
    attempt_batch = target_batch
    while True:
        try:
            result = pipeline.run_chain_of_thought(
                dataset_path=str(dataset_path),
                output_path=str(output_path),
                cot_sampling=cot_sampling,
                batch_size=attempt_batch,
                sample_limit=sample_limit,
                min_prompt_count=min_prompt_count,
            )
            effective_batch = attempt_batch
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_oom(exc):
                raise
            if probe_only or attempt_batch <= 1:
                raise
            fallback = max(1, attempt_batch // 2)
            if fallback == attempt_batch:
                raise
            print(
                f"⚠️  CUDA OOM at batch {attempt_batch}; retrying with {fallback}."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            attempt_batch = fallback
            continue

    if effective_batch != target_batch:
        job_name = os.environ.get("RWKV_SKILLS_JOB_NAME")
        model_slug = safe_slug(Path(args.model_path).stem)
        gpu = _extract_gpu_from_device(args.device)
        if job_name and model_slug and gpu:
            _update_batch_cache(job_name, model_slug, gpu, effective_batch)

    eval_path = eval_details_path(slug, is_cot=True, model_name=Path(args.model_path).stem)
    metrics = evaluate_multiple_choice(
        output_path,
        dataset_path=dataset_path,
        eval_output_path=eval_path,
    )
    score_path = write_scores_json(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics={"accuracy": metrics.accuracy},
        samples=metrics.samples,
        log_path=out_path,
        task="multiple_choice_cot",
        task_details={
            "accuracy_by_subject": metrics.accuracy_by_subject,
            "eval_details_path": str(eval_path),
        },
    )
    print(f"✅ CoT multiple-choice done: {result.sample_count} samples -> {result.output_path}")
    print(f"📄 eval details saved: {eval_path}")
    print(f"📊 scores saved: {score_path}")
    run_llm_checker(eval_path, model_name=Path(args.model_path).stem)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
