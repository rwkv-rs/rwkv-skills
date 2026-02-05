from __future__ import annotations

"""Run LiveCodeBench code generation + evaluation for RWKV models."""

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
from src.eval.evaluators.coding import CodingPipeline
from src.eval.metrics.code_generation.livecodebench import evaluate_livecodebench_dataset
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.job_env import ensure_job_id
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.model import ModelLoadConfig


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
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
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
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
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
    init_orm(DEFAULT_DB_CONFIG)
    
    service = EvalDbService()

    # 三层级联检索：一次查询获取所有续跑信息
    ctx = service.get_resume_context(
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
    )
    sampling_payload = {
        "cot": sampling_config_to_dict(cot_sampling),
        "final": sampling_config_to_dict(final_sampling),
    }
    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name="eval_code_livecodebench",
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config=sampling_payload,
    )
    skip_keys = ctx.completed_keys

    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id
    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
    records = JsonlCodeGenerationLoader(str(dataset_path)).load()
    expected_count = (min(len(records), sample_limit) if sample_limit else len(records)) * max(1, max(pass_k))
    try:
        result = pipeline.run_livecodebench(
            dataset_path=str(dataset_path),
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            eval_timeout=args.eval_timeout,
            eval_workers=args.eval_workers,
            pass_k=pass_k,
            probe_only=args.probe_only,
            skip_keys=skip_keys,
            on_record=writer.enqueue,
        )
    except BaseException:
        try:
            writer.close()
        finally:
            actual = service.count_completions(task_id=task_id)
            status = "completed" if actual == expected_count else "failed"
            service.update_task_status(task_id=task_id, status=status)
        raise
    if args.probe_only:
        writer.close()
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        return 0

    print(f"✅ LiveCodeBench 生成完成：{result.sample_count} completions")

    writer.close()
    try:
        completions_payloads = service.list_completion_payloads(task_id=task_id)
        eval_metrics, eval_payloads = evaluate_livecodebench_dataset(
            completions_payloads,
            dataset_path=str(dataset_path),
            pass_k=pass_k,
            n_workers=args.eval_workers,
            timeout=args.eval_timeout,
        )
        print(f"LiveCodeBench 评测: {eval_metrics}")
        service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
        score_payload = make_score_payload(
            slug,
            is_cot=True,
            model_name=Path(args.model_path).stem,
            metrics=eval_metrics or {},
            samples=result.sample_count,
            problems=result.problem_count,
            task="code_livecodebench",
        )
        service.record_score_payload(
            payload=score_payload,
            task_id=task_id,
        )
        export_version_results(
            service,
            task_id=task_id,
        )
    except BaseException:
        if service.get_score_payload(task_id=task_id) is None:
            service.update_task_status(task_id=task_id, status="failed")
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
