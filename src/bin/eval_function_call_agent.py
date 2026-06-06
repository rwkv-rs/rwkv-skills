from __future__ import annotations

"""Run multi-turn function-calling agent evaluation for RWKV models."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rwkv_agent_eval_plugin import agent_plugin_config_from_sources

from src.db.async_writer import CompletionWriteWorker
from src.db.eval_db_service import EvalDbService
from src.db.export_results import export_version_results
from src.db.orm import init_orm
from src.eval.agent_bench.tau_specs import TAU_AGENT_JOB_BY_DATASET
from src.eval.benchmark_config import resolve_benchmark_model_config, resolve_sampling_config
from src.eval.function_calling.agent.adapters.browsecomp_plus_judge import (
    BrowseCompPlusJudgeConfig,
    default_browsecomp_plus_eval_dir,
    evaluate_browsecomp_plus_completions,
)
from src.eval.function_calling.agent.adapters.complexfuncbench import (
    summarize_complexfuncbench_official_payloads,
)
from src.eval.function_calling.agent.pipeline import FunctionCallAgentPipeline, load_agent_records
from src.eval.function_calling.agent.scorer import evaluate_function_call_agent
from src.eval.function_calling.agent.tau_official_runner import (
    TauOfficialAgentPipeline,
    TauOfficialRunnerOptions,
)
from src.eval.function_calling.common.benchmarks import function_calling_benchmark_spec
from src.eval.function_calling.long_context_router import (
    long_context_routing_config_from_args,
    long_context_routing_config_from_benchmark_config,
)
from src.eval.function_calling.tool_router import (
    tool_routing_config_from_args,
    tool_routing_config_from_benchmark_config,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import canonical_slug, infer_dataset_slug_from_path


AGENT_JOB_BY_DATASET: dict[str, str] = {
    "apibank_level2_test": "function_agent_apibank_l2",
    "complexfuncbench_subset_test": "function_agent_complexfuncbench",
    "browsecomp_plus_test": "function_agent_browsecomp_plus",
    **TAU_AGENT_JOB_BY_DATASET,
}
TAU_AGENT_JOBS = frozenset(TAU_AGENT_JOB_BY_DATASET.values())


@dataclass(slots=True)
class LocalModelLoadConfig:
    weights_path: str
    device: str = "cuda"
    tokenizer_path: str | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV function-calling agent evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="Agent function-calling dataset path or registered slug")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for scheduler compatibility")
    parser.add_argument("--max-samples", type=int, help="Limit number of tasks for quick runs")
    parser.add_argument("--db-write-queue", type=int, default=1024, help="DB completion write queue max size")
    parser.add_argument(
        "--agent-plugin-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable the RWKV multi-turn agent benchmark plugin defaults",
    )
    parser.add_argument("--tool-router-mode", choices=("off", "lexical"), help="Tool window router mode")
    parser.add_argument("--tool-router-max-tools", type=int, help="Maximum tools exposed after routing")
    parser.add_argument("--tool-router-trigger-tool-count", type=int, help="Route when tool count reaches this value")
    parser.add_argument("--tool-router-trigger-catalog-chars", type=int, help="Route when catalog JSON reaches this size")
    parser.add_argument("--tool-router-context-chars", type=int, help="Recent context characters used by the router")
    parser.add_argument("--tool-router-description-chars", type=int, help="Per-tool description chars used by the router")
    parser.add_argument(
        "--long-context-router-mode",
        "--long-doc-mode",
        dest="long_context_router_mode",
        choices=("off", "lexical"),
        help="Long context router mode",
    )
    parser.add_argument(
        "--long-context-min-chars",
        "--long-doc-min-chars",
        dest="long_context_min_chars",
        type=int,
        help="Compact text at or above this character length",
    )
    parser.add_argument(
        "--long-context-chunk-chars",
        "--long-doc-max-chars",
        dest="long_context_chunk_chars",
        type=int,
        help="Chunk size for lexical long-context compaction",
    )
    parser.add_argument(
        "--long-context-overlap-lines",
        "--long-doc-overlap-lines",
        dest="long_context_overlap_lines",
        type=int,
        help="Line overlap between lexical chunks",
    )
    parser.add_argument(
        "--long-context-max-evidence-chunks",
        "--long-doc-max-evidence-chunks",
        dest="long_context_max_evidence_chunks",
        type=int,
        help="Maximum selected evidence chunks",
    )
    parser.add_argument(
        "--long-context-max-evidence-chars",
        "--long-doc-max-evidence-chars",
        dest="long_context_max_evidence_chars",
        type=int,
        help="Maximum selected evidence characters",
    )
    parser.add_argument(
        "--long-context-query-chars",
        "--long-doc-query-chars",
        dest="long_context_query_chars",
        type=int,
        help="Recent query characters used by long-context router",
    )
    parser.add_argument("--history-max-chars", type=int, help="TAU official trajectory history character budget")
    parser.add_argument("--prompt-max-chars", type=int, help="TAU official rendered prompt character budget")
    parser.add_argument("--max-steps", type=int, help="TAU official max simulation steps")
    parser.add_argument("--max-tool-errors", type=int, help="TAU official max tool/runtime errors before stopping")
    parser.add_argument("--decision-max-tokens", type=int, help="TAU official per-step generation token budget")
    parser.add_argument("--max-repeated-tool-calls", type=int, help="TAU official repeated tool-call guard threshold")
    parser.add_argument(
        "--tau-sample-workers",
        type=int,
        help="TAU attempt-level sample workers for one model task; not model generation batch size",
    )
    parser.add_argument("--tau-attempt-retries", type=int, help="TAU attempt retries before recording a failed attempt")
    parser.add_argument("--tau-judge-concurrency", type=int, help="TAU external judge concurrency limit")
    parser.add_argument("--user-model", help="TAU official user simulator model name")
    parser.add_argument("--user-api-key", help="TAU official user simulator API key")
    parser.add_argument("--user-base-url", help="TAU official user simulator OpenAI-compatible base URL")
    parser.add_argument("--judge-model", help="TAU official NL assertion judge model name")
    parser.add_argument("--judge-api-key", help="TAU official NL assertion judge API key")
    parser.add_argument("--judge-base-url", help="TAU official NL assertion judge OpenAI-compatible base URL")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    scheduler_slug = os.environ.get("RWKV_SKILLS_DATASET_SLUG")
    slug = canonical_slug(scheduler_slug or infer_dataset_slug_from_path(str(dataset_path)))
    model_name = Path(args.model_path).stem
    records, _resolved = load_agent_records(str(dataset_path), args.max_samples)
    if not records:
        raise ValueError(f"{slug} 没有可运行的 function-calling agent 样本")
    job_name = AGENT_JOB_BY_DATASET.get(str(slug))
    if job_name is None:
        raise ValueError(f"function_call agent 暂不支持数据集: {slug}")
    benchmark_config = resolve_benchmark_model_config(slug, model_name, stage="tool")
    agent_plugin_config = agent_plugin_config_from_sources(args, benchmark_config)
    tool_router_fallback_mode = agent_plugin_config.tool_router_mode if agent_plugin_config.enabled else "off"
    tool_routing_config = tool_routing_config_from_args(
        args,
        base=tool_routing_config_from_benchmark_config(
            benchmark_config,
            fallback_mode=tool_router_fallback_mode,
        ),
    )
    sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    is_tau_job = job_name in TAU_AGENT_JOBS
    if sampling is not None and job_name != "function_agent_complexfuncbench" and not is_tau_job:
        sampling = sampling.clamp(768)
    if sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")

    tau_options = TauOfficialRunnerOptions.from_sources(args, benchmark_config) if is_tau_job else None
    if agent_plugin_config.enabled:
        long_context_mode = agent_plugin_config.long_context_router_mode
    else:
        long_context_mode = tool_routing_config.mode if is_tau_job and tool_routing_config.enabled else "off"
    long_context_routing_config = long_context_routing_config_from_args(
        args,
        base=long_context_routing_config_from_benchmark_config(
            benchmark_config,
            fallback_mode=long_context_mode,
        ),
    )
    if is_tau_job:
        pipeline = TauOfficialAgentPipeline(_model_load_config(weights_path=args.model_path, device=args.device))
    else:
        pipeline = FunctionCallAgentPipeline(_model_load_config(weights_path=args.model_path, device=args.device))

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    force_new_task = os.environ.get("RWKV_SCHEDULER_OVERWRITE") == "1" or (
        is_tau_job and os.environ.get("RWKV_TAU_FORCE_NEW_TASK") == "1"
    )
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
        if is_tau_job:
            result = pipeline.run(
                dataset_path=str(dataset_path),
                sampling=sampling,
                options=tau_options,
                dataset_name=str(slug),
                sample_limit=args.max_samples,
                samples_per_task=1,
                tau_sample_workers=args.tau_sample_workers,
                skip_keys=ctx.completed_keys,
                tool_routing_config=tool_routing_config,
                long_context_routing_config=long_context_routing_config,
                on_record=writer.enqueue,
            )
        else:
            result = pipeline.run(
                dataset_path=str(dataset_path),
                sampling=sampling,
                batch_size=max(1, args.batch_size),
                sample_limit=args.max_samples,
                samples_per_task=1,
                skip_keys=ctx.completed_keys,
                config=benchmark_config,
                tool_routing_config=tool_routing_config,
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

    actual_count = service.count_completions(task_id=task_id, status="answer")
    if actual_count != expected_count:
        service.update_task_status(task_id=task_id, status="failed")
        session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
        if session_task_id:
            try:
                service.update_task_session_status(task_id=session_task_id, session_status="failed")
            except Exception:
                pass
        raise RuntimeError(
            f"{job_name} produced {actual_count}/{expected_count} answer completions; refusing to record score"
        )

    completions_payloads = service.list_completion_payloads(task_id=task_id, status="answer")
    try:
        metrics = evaluate_function_call_agent(completions_payloads)
        eval_payloads = metrics.payloads or []
        score_metrics: dict[str, Any] = {
            "avg_steps": metrics.avg_steps,
            "invalid_action_rate": metrics.invalid_action_rate,
            "timeout_rate": metrics.timeout_rate,
            "parse_error_rate": metrics.parse_error_rate,
        }
        if metrics.official_score is not None:
            score_metrics["avg@1"] = metrics.official_score
            score_metrics["official_score"] = metrics.official_score
            score_metrics["success_rate"] = metrics.success_rate
        benchmark_spec = function_calling_benchmark_spec(job_name)
        task_details: dict[str, Any] = {
            "subtype": benchmark_spec.subtype if benchmark_spec else "agent",
            "benchmark": benchmark_spec.benchmark if benchmark_spec else "",
        }
        if job_name == "function_agent_browsecomp_plus":
            judge_config = BrowseCompPlusJudgeConfig.from_benchmark_config(benchmark_config)
            judge_eval_dir = default_browsecomp_plus_eval_dir(task_id)
            judge_metrics = evaluate_browsecomp_plus_completions(
                completions_payloads,
                config=judge_config,
                eval_dir=judge_eval_dir,
            )
            eval_payloads = judge_metrics.payloads
            score_metrics["avg@1"] = judge_metrics.accuracy
            score_metrics["success_rate"] = judge_metrics.accuracy
            score_metrics["official_score"] = judge_metrics.accuracy
            if judge_metrics.retrieval_recall is not None:
                score_metrics["browsecomp_plus_retrieval_recall"] = judge_metrics.retrieval_recall
            if judge_metrics.calibration_error is not None:
                score_metrics["browsecomp_plus_calibration_error"] = judge_metrics.calibration_error
            score_metrics["browsecomp_plus_accuracy_percent"] = judge_metrics.summary.get("Accuracy (%)", 0.0)
            score_metrics["browsecomp_plus_recall_percent"] = judge_metrics.summary.get("Recall (%)")
            task_details["browsecomp_plus_judge"] = {
                key: value
                for key, value in judge_metrics.summary.items()
                if key != "per_query_metrics"
            }
            task_details["browsecomp_plus_eval_dir"] = str(judge_eval_dir)

        if job_name == "function_agent_complexfuncbench":
            complex_metrics = summarize_complexfuncbench_official_payloads(completions_payloads)
            score_metrics["avg@1"] = complex_metrics.success_rate
            score_metrics["success_rate"] = complex_metrics.success_rate
            score_metrics["official_score"] = complex_metrics.success_rate
            score_metrics["call_accuracy"] = complex_metrics.call_accuracy
            score_metrics["completeness"] = complex_metrics.completeness
            score_metrics["correctness"] = complex_metrics.correctness
            score_metrics["response_eval_samples"] = float(complex_metrics.response_eval_samples)
            task_details["complexfuncbench_official"] = {
                "call_accuracy": complex_metrics.call_accuracy,
                "response_eval_samples": complex_metrics.response_eval_samples,
            }

        service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
        if score_metrics.get("official_score") is None:
            raise RuntimeError(f"{job_name} produced no official score; refusing to record fallback score")
        score_payload = make_score_payload(
            slug,
            is_cot=False,
            model_name=model_name,
            metrics=score_metrics,
            samples=metrics.samples,
            task=job_name,
            task_details=task_details,
        )
        service.record_score_payload(payload=score_payload, task_id=task_id)
    except BaseException:
        service.update_task_status(task_id=task_id, status="failed")
        session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
        if session_task_id:
            try:
                service.update_task_session_status(task_id=session_task_id, session_status="failed")
            except Exception:
                pass
        raise
    session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
    if session_task_id:
        try:
            service.update_task_session_status(task_id=session_task_id, session_status="completed")
        except Exception:
            pass
    export_version_results(service, task_id=task_id)
    print(f"✅ function_call agent done: {result.sample_count} samples")
    return 0


def _model_load_config(*, weights_path: str, device: str) -> Any:
    return LocalModelLoadConfig(weights_path=weights_path, device=device)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
