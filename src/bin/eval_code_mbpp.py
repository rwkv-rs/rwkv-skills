from __future__ import annotations

"""Run MBPP code generation + evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence
from dataclasses import replace

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.results.payloads import make_score_payload
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.db.database import DatabaseManager
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.coding import CodingPipeline
from src.eval.metrics.code_generation.evaluate import evaluate_mbpp_dataset
from src.infer.model import ModelLoadConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV MBPP evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="MBPP JSONL path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of problems for quick runs")
    parser.add_argument("--max-tokens", type=int, help="Clamp generation length")
    parser.add_argument("--temperature", type=float, help="Override sampling temperature")
    parser.add_argument("--top-k", type=int, help="Override sampling top-k")
    parser.add_argument("--top-p", type=float, help="Override sampling top-p")
    parser.add_argument("--eval-timeout", type=float, default=3.0, help="Seconds per test execution")
    parser.add_argument("--eval-workers", type=int, default=4, help="Parallel workers for evaluation")
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
    sampling = resolve_sampling_config(
        slug,
        Path(args.model_path).stem,
        fallback_templates="code_default",
    )
    if sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({Path(args.model_path).stem})")
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
    batch_size = max(1, args.batch_size)
    default_pass_k = (1,)
    pass_k = (1,) if args.probe_only else (tuple(args.pass_k) if args.pass_k else default_pass_k)
    sample_limit = batch_size if args.probe_only else args.max_samples
    if not DEFAULT_DB_CONFIG.enabled:
        raise RuntimeError("DB 未启用：当前仅支持数据库写入模式。")
    db = DatabaseManager.instance()
    db.initialize(DEFAULT_DB_CONFIG)
    service = EvalDbService(db)
    version_id = service.get_or_create_version(
        job_name="eval_code_mbpp",
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
    )
    result = pipeline.run_mbpp(
        dataset_path=str(dataset_path),
        sampling=sampling,
        batch_size=batch_size,
        sample_limit=sample_limit,
        eval_timeout=args.eval_timeout,
        eval_workers=args.eval_workers,
        pass_k=pass_k,
        probe_only=args.probe_only,
        skip_keys=skip_keys,
        on_record=writer.enqueue,
    )

    if args.probe_only:
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        return 0

    print(f"✅ MBPP 生成完成：{result.sample_count} completions")

    writer.close()
    completions_payloads = service.list_completion_payloads(
        version_id=version_id,
        is_param_search=False,
    )
    eval_metrics, eval_payloads = evaluate_mbpp_dataset(
        completions_payloads,
        dataset_path=str(dataset_path),
        pass_k=pass_k,
        n_workers=args.eval_workers,
        timeout=args.eval_timeout,
    )
    print(f"MBPP 评测: {eval_metrics}")
    service.ingest_eval_payloads(
        payloads=eval_payloads,
        version_id=version_id,
        is_param_search=False,
    )
    score_payload = make_score_payload(
        slug,
        is_cot=False,
        model_name=Path(args.model_path).stem,
        metrics=eval_metrics or {},
        samples=result.sample_count,
        problems=result.problem_count,
        task="code_mbpp",
    )
    service.record_score_payload(
        payload=score_payload,
        version_id=version_id,
        is_param_search=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
