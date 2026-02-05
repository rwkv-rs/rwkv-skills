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
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.job_env import ensure_job_id
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, safe_slug
from src.eval.scheduler.profiler import update_batch_cache_locked
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV multiple-choice CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation/scoring")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
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
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    model_name = Path(args.model_path).stem
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = MultipleChoicePipeline(config, target_token_format=args.target_token_format)

    # Quick validation of dataset readability before heavy model init
    records = JsonlMultipleChoiceLoader(str(dataset_path)).load()

    cot_sampling = resolve_sampling_config(
        slug,
        model_name,
        fallback_templates="multi_choice_cot_default",
    )
    if cot_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")

    init_orm(DEFAULT_DB_CONFIG)
    
    service = EvalDbService()

    # 三层级联检索：一次查询获取所有续跑信息
    ctx = service.get_resume_context(
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
    )
    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name="eval_multi_choice_cot",
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config={"cot": sampling_config_to_dict(cot_sampling)},
    )
    skip_keys = ctx.completed_keys

    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id
    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
    if args.probe_only:
        batch_size = max(1, args.batch_size)
        _ = pipeline.run_chain_of_thought(
            dataset_path=str(dataset_path),
            cot_sampling=cot_sampling,
            batch_size=batch_size,
            sample_limit=batch_size,
            min_prompt_count=batch_size,
            probe_only=True,
        )
        print(
            f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}."
        )
        return 0

    sample_limit: int | None = args.max_samples
    min_prompt_count: int | None = None
    target_batch = max(1, args.batch_size)
    expected_count = min(len(records), sample_limit) if sample_limit else len(records)
    try:
        result = pipeline.run_chain_of_thought(
            dataset_path=str(dataset_path),
            cot_sampling=cot_sampling,
            batch_size=target_batch,
            sample_limit=sample_limit,
            min_prompt_count=min_prompt_count,
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

    writer.close()
    completions_payloads = service.list_completion_payloads(task_id=task_id)
    metrics = evaluate_multiple_choice(
        completions_payloads,
        dataset_path=dataset_path,
    )
    service.ingest_eval_payloads(payloads=metrics.payloads, task_id=task_id)
    score_payload = make_score_payload(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics={"accuracy": metrics.accuracy},
        samples=metrics.samples,
        task="multiple_choice_cot",
        task_details={
            "accuracy_by_subject": metrics.accuracy_by_subject,
        },
    )
    service.record_score_payload(
        payload=score_payload,
        task_id=task_id,
    )
    export_version_results(
        service,
        task_id=task_id,
    )
    print(f"✅ CoT multiple-choice done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
