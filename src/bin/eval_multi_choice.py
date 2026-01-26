from __future__ import annotations

"""Run direct multiple-choice evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.metrics.multi_choice import evaluate_multiple_choice
from src.eval.results.payloads import make_score_payload
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.db.database import DatabaseManager
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.infer.model import ModelLoadConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV multiple-choice (direct) evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for scoring")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument("--db-write-batch", type=int, default=128, help="DB completion write batch size")
    parser.add_argument("--db-write-queue", type=int, default=4096, help="DB completion write queue max size")
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
    pipeline = MultipleChoicePipeline(config, target_token_format=args.target_token_format)

    # Quick validation of dataset readability before heavy model init
    _ = JsonlMultipleChoiceLoader(str(dataset_path)).load()

    if not DEFAULT_DB_CONFIG.enabled:
        raise RuntimeError("DB 未启用：当前仅支持数据库写入模式。")
    db = DatabaseManager.instance()
    db.initialize(DEFAULT_DB_CONFIG)
    service = EvalDbService(db)
    task_id = service.get_or_create_task(
        job_name="eval_multi_choice",
        job_id=os.environ.get("RWKV_SKILLS_JOB_ID"),
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config={"mode": "logits_only"},
        allow_resume=True,
    )
    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id
    skip_keys = service.list_completion_keys(
        task_id=task_id,
    )
    try:
        writer = CompletionWriteWorker(
            service=service,
            task_id=task_id,
            batch_size=args.db_write_batch,
            max_queue=args.db_write_queue,
        )
        result = pipeline.run_direct(
            dataset_path=str(dataset_path),
            sample_limit=args.max_samples,
            skip_keys=skip_keys,
            on_record=writer.enqueue,
        )
        writer.close()
        completions_payloads = service.list_completion_payloads(task_id=task_id)
        metrics = evaluate_multiple_choice(
            completions_payloads,
            dataset_path=dataset_path,
        )
        service.ingest_eval_payloads(payloads=metrics.payloads, task_id=task_id)
        score_payload = make_score_payload(
            slug,
            is_cot=False,
            model_name=Path(args.model_path).stem,
            metrics={"accuracy": metrics.accuracy},
            samples=metrics.samples,
            task="multiple_choice",
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
        print(f"✅ direct multiple-choice done: {result.sample_count} samples")
    except Exception:
        service.update_task_status(task_id=task_id, status="failed")
        raise
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
