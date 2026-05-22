from __future__ import annotations

"""Run multi-turn function-calling agent evaluation for RWKV models."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.db.async_writer import CompletionWriteWorker
from src.db.eval_db_service import EvalDbService
from src.db.export_results import export_version_results
from src.db.orm import init_orm
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.function_calling.agent.pipeline import FunctionCallAgentPipeline, load_agent_records
from src.eval.function_calling.agent.scorer import evaluate_function_call_agent
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.infer.model import ModelLoadConfig


AGENT_JOB_BY_DATASET: dict[str, str] = {
    "apibank_l2_test": "function_agent_apibank_l2",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV function-calling agent evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="Agent function-calling dataset path or registered slug")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for scheduler compatibility")
    parser.add_argument("--max-samples", type=int, help="Limit number of tasks for quick runs")
    parser.add_argument("--db-write-queue", type=int, default=1024, help="DB completion write queue max size")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    model_name = Path(args.model_path).stem
    records, _resolved = load_agent_records(str(dataset_path), args.max_samples)
    if not records:
        raise ValueError(f"{slug} 没有可运行的 function-calling agent 样本")
    job_name = AGENT_JOB_BY_DATASET.get(str(slug))
    if job_name is None:
        raise ValueError(f"function_call agent 暂不支持数据集: {slug}")
    sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    if sampling is not None:
        sampling = sampling.clamp(768)
    if sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")

    pipeline = FunctionCallAgentPipeline(ModelLoadConfig(weights_path=args.model_path, device=args.device))

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    force_new_task = os.environ.get("RWKV_SCHEDULER_OVERWRITE") == "1"
    ctx = service.get_resume_context(
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        force_new_task=force_new_task,
    )
    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name=job_name,
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        sampling_config=sampling_config_to_dict(sampling),
    )
    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id
    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
    expected_count = service.expected_completion_count(
        dataset=str(slug),
        sample_limit=args.max_samples,
        repeats_per_problem=1,
    )
    if expected_count is None:
        expected_count = len(records)
    try:
        result = pipeline.run(
            dataset_path=str(dataset_path),
            sampling=sampling,
            batch_size=max(1, args.batch_size),
            sample_limit=args.max_samples,
            samples_per_task=1,
            skip_keys=ctx.completed_keys,
            on_record=writer.enqueue,
        )
    except BaseException:
        try:
            writer.close()
        finally:
            actual = service.count_completions(task_id=task_id, status="answer")
            status = "completed" if actual == expected_count else "failed"
            service.update_task_status(task_id=task_id, status=status)
            session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
            if session_task_id:
                try:
                    service.update_task_session_status(task_id=session_task_id, session_status="failed")
                except Exception:
                    pass
        raise
    writer.close()

    completions_payloads = service.list_completion_payloads(task_id=task_id, status="answer")
    metrics = evaluate_function_call_agent(completions_payloads)
    service.ingest_eval_payloads(payloads=metrics.payloads or [], task_id=task_id)
    score_payload = make_score_payload(
        slug,
        is_cot=False,
        model_name=model_name,
        metrics={
            "success_rate": metrics.success_rate,
            "official_score": metrics.official_score,
            "avg_steps": metrics.avg_steps,
            "invalid_action_rate": metrics.invalid_action_rate,
            "timeout_rate": metrics.timeout_rate,
            "parse_error_rate": metrics.parse_error_rate,
        },
        samples=metrics.samples,
        task=job_name,
        task_details={"subtype": "agent", "benchmark": "apibank"},
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
    if session_task_id:
        try:
            service.update_task_session_status(task_id=session_task_id, session_status="completed")
        except Exception:
            pass
    export_version_results(service, task_id=task_id)
    print(f"✅ function_call agent done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
