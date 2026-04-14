from __future__ import annotations

"""Unified function-calling runner for BrowseComp, MCP-Bench, tau_bench, and tau2_bench."""

import argparse
import json
import os
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any, Mapping, Sequence

from src.eval.agent_bench.envs.tau_v2 import TauV2Env
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import BenchmarkField, CoTMode, resolve_benchmark_metadata
from src.eval.env_config import load_env_file, resolve_judge_model_config
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import (
    AttemptKey,
    build_attempt_keys,
    build_auto_avg_k_execution_plan,
    build_avg_k_execution_plan,
    plan_attempt_count,
)
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling import (
    BFCL_COT_STOP_SUFFIX,
    BFCL_DECISION_STOP_SUFFIXES,
    BFCL_ADDITIONAL_FUNCTION_PROMPT,
    DEFAULT_HISTORY_MAX_CHARS,
    BFCL_ROUTER_LABELS,
    BrowseCompJudgeConfig,
    BfclTaskRecord,
    BFCL_V3_MAX_COT_CHARS,
    McpBenchExecutionResult,
    McpBenchItem,
    McpBenchWorkerClient,
    TauManifestRecord,
    TauToolCall,
    apply_bfcl_tool_call,
    append_round_summary,
    build_bfcl_ask_prompt,
    build_bfcl_cot_prompt,
    build_bfcl_handoff_prompt,
    build_bfcl_ref_answer,
    build_bfcl_router_prompt,
    build_bfcl_system_prompt,
    build_bfcl_tool_prompt,
    build_bfcl_tool_result_payload,
    build_bfcl_tool_result_message,
    build_bfcl_user_block,
    collect_bfcl_dataset_issues,
    decode_bfcl_exec_response,
    build_browsecomp_answer_prompt,
    build_browsecomp_expected_context,
    build_browsecomp_user_prompt,
    build_expected_context,
    build_final_answer_prompt,
    build_mcp_bench_ref_answer,
    build_planning_context,
    build_planning_decision_prompt,
    build_tau_system_prompt,
    build_turn_completion_prompt,
    collapse_mcp_bench_pass,
    extract_bfcl_cot_hidden_summary,
    execute_bfcl_official_tool_call,
    has_bfcl_official_turns,
    judge_browsecomp_answers,
    evaluate_bfcl_v3_episode,
    normalize_bfcl_decision_output,
    normalize_bfcl_tool_output_safe,
    parse_bfcl_assistant_output,
    parse_bfcl_router_output,
    reconstruct_bfcl_tool_output,
    render_bfcl_assistant_tool_message,
    render_bfcl_official_call,
    render_bfcl_turn_request,
    load_bfcl_v3_manifest_records,
    load_browsecomp_manifest_records,
    load_mcp_bench_manifest_records,
    load_tau_manifest_records,
    normalize_planned_tool_call,
    parse_planning_decision,
    render_assistant_tool_message,
    start_bfcl_runtime,
    render_tau_user_prompt,
    render_tool_result,
    summarize_mcp_bench_evaluation,
    trim_message_history,
)
from src.eval.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    finalize_function_calling_run,
    load_function_calling_backend,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.config import REPO_ROOT
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split
from src.infer.backend import InferenceBackend, add_inference_backend_arguments, validate_inference_backend_args
from src.infer.constraints import LiteralChoiceConstraint, build_bfcl_tool_call_constraint

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext, TaskSpec

DEFAULT_MAX_STEPS = 16
DEFAULT_MAX_TOOL_ERRORS = 4
_TEMPLATE_LEAK_MARKERS = (
    "<system message>",
    "</system message>",
    "<assistant>",
    "</assistant>",
    "<user_input>",
    "</user_input>",
)


class FunctionCallingBenchmarkKind(str, Enum):
    AUTO = "auto"
    BROWSECOMP = "browsecomp"
    MCP_BENCH = "mcp_bench"
    BFCL_V3 = "bfcl_v3"
    TAU_BENCH = "tau_bench"
    TAU2_BENCH = "tau2_bench"


@dataclass(slots=True)
class ResolvedFunctionCallingRun:
    benchmark_kind: FunctionCallingBenchmarkKind
    dataset_path: Path
    dataset_slug: str
    benchmark_name: str
    dataset_split: str
    model_name: str
    engine: InferenceBackend


@dataclass(slots=True)
class _ActiveEpisode:
    sample_index: int
    repeat_index: int
    pass_index: int
    record: TauManifestRecord
    runtime_env: TauV2Env
    task: Any
    environment: Any
    system_prompt: str
    prompt_messages: list[dict[str, str]]
    trajectory: list[Any]
    stages: list[StageRecord] = field(default_factory=list)
    tool_calls: list[TauToolCall] = field(default_factory=list)
    turn_count: int = 0
    tool_errors: int = 0
    final_answer: str = ""
    termination_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class _ActiveBfclEpisode:
    sample_index: int
    repeat_index: int
    pass_index: int
    record: BfclTaskRecord
    system_prompt: str
    prompt_messages: list[dict[str, str]]
    active_tools: list[dict[str, Any]]
    runtime_state: Any
    stages: list[StageRecord] = field(default_factory=list)
    tool_calls: list[TauToolCall] = field(default_factory=list)
    step_count: int = 0
    turn_count: int = 0
    tool_errors: int = 0
    final_answer: str = ""
    termination_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class _BfclGenerationStepOutcome:
    ok: bool
    trace_entry: dict[str, object]
    action_type: str | None = None
    tool_call: TauToolCall | None = None
    final_answer: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV unified function-calling benchmark runner")
    parser.add_argument("--dataset", required=True, help="Prepared function-calling JSONL dataset path")
    parser.add_argument(
        "--benchmark-kind",
        choices=[kind.value for kind in FunctionCallingBenchmarkKind],
        default=FunctionCallingBenchmarkKind.AUTO.value,
        help="Explicit function-calling benchmark family (defaults to auto-detect from dataset slug)",
    )
    add_inference_backend_arguments(parser)
    parser.add_argument("--batch-size", type=int, help="Generation batch size for batched runners")
    parser.add_argument("--max-samples", type=int, help="Limit source task count before avg@k planning")
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        dest="avg_k",
        help="Override auto avg@k planning; function-calling runners accept exactly one explicit avg_k",
    )
    parser.add_argument("--db-write-queue", type=int, help="DB completion write queue max size")
    parser.add_argument("--db-close-timeout-s", type=float, default=30.0, help="DB close timeout")
    parser.add_argument("--probe-only", action="store_true", help="Run a minimal probe and skip scoring")
    parser.add_argument(
        "--history-max-chars",
        type=int,
        default=DEFAULT_HISTORY_MAX_CHARS,
        help="Clamp accumulated conversation/tool history length",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=2048, help="Clamp CoT generation length")
    parser.add_argument("--answer-max-tokens", type=int, default=1024, help="Clamp final answer generation length")
    parser.add_argument("--planning-max-tokens", type=int, default=2048, help="Clamp MCP planning generation length")
    parser.add_argument("--decision-max-tokens", type=int, help="Clamp tool/final-decision generation length")
    parser.add_argument("--final-max-tokens", type=int, default=3072, help="Clamp MCP final synthesis generation length")
    parser.add_argument("--max-rounds", type=int, default=20, help="Maximum MCP planning rounds per task")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum tau turns per task")
    parser.add_argument(
        "--max-tool-errors",
        type=int,
        default=DEFAULT_MAX_TOOL_ERRORS,
        help="Abort one tau task after this many tool-call errors",
    )
    return parser.parse_args(argv)


def _infer_benchmark_kind(dataset_arg: str) -> FunctionCallingBenchmarkKind:
    dataset_slug = infer_dataset_slug_from_path(dataset_arg)
    metadata = resolve_benchmark_metadata(dataset_slug)
    if metadata.field is not BenchmarkField.FUNCTION_CALLING:
        raise ValueError(f"dataset {dataset_slug!r} 不是 function-calling benchmark，无法用 function_calling runner 运行。")

    job_names = frozenset(metadata.scheduler_jobs)
    if "function_browsecomp" in job_names:
        return FunctionCallingBenchmarkKind.BROWSECOMP
    if "function_mcp_bench" in job_names:
        return FunctionCallingBenchmarkKind.MCP_BENCH
    if "function_bfcl_v3" in job_names:
        return FunctionCallingBenchmarkKind.BFCL_V3
    if "function_tau2_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU2_BENCH
    if "function_tau_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU_BENCH
    raise ValueError(f"dataset {dataset_slug!r} 没有已知的 function-calling scheduler job。")


def _resolve_run(args: argparse.Namespace) -> ResolvedFunctionCallingRun:
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    dataset_slug = infer_dataset_slug_from_path(str(dataset_path))
    detected_kind = _infer_benchmark_kind(str(dataset_path))
    requested_kind = FunctionCallingBenchmarkKind(args.benchmark_kind)
    if requested_kind is FunctionCallingBenchmarkKind.AUTO:
        benchmark_kind = detected_kind
    else:
        if requested_kind is not detected_kind:
            raise ValueError(
                f"dataset {dataset_slug!r} 对应 {detected_kind.value}，但收到了不匹配的 --benchmark-kind={requested_kind.value}"
            )
        benchmark_kind = requested_kind
    benchmark_name, dataset_split = split_benchmark_and_split(dataset_slug)
    model_name, engine = load_function_calling_backend(args)
    return ResolvedFunctionCallingRun(
        benchmark_kind=benchmark_kind,
        dataset_path=dataset_path,
        dataset_slug=dataset_slug,
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        model_name=model_name,
        engine=engine,
    )


def _normalize_final_answer(text: str, *, locale: str) -> str:
    body = text.strip()
    if not body:
        return ""
    prefix = "解释:" if locale == "zh" else "Explanation:"
    return body if body.startswith(prefix) else f"{prefix} {body}"


def _browsecomp_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("judge_reason") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=str(agent_info.get("response") or ""),
        ref_answer=str(agent_info.get("reference_answer") or ""),
    )


def _resolve_job_name(default_job_name: str, *, run_context: "RunContext | None" = None) -> str:
    if run_context is not None:
        return run_context.job_name
    return os.environ.get("RWKV_SKILLS_JOB_NAME", default_job_name)


def _looks_like_template_leak(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if "<system message>" in lowered and "you are a helpful assistant" in lowered:
        return True
    marker_hits = sum(lowered.count(marker) for marker in _TEMPLATE_LEAK_MARKERS)
    return marker_hits >= 3


def _bounded_prompt_messages(
    messages: Sequence[dict[str, str]],
    *,
    history_max_chars: int,
) -> list[dict[str, str]]:
    return trim_message_history(messages, max_chars=max(0, int(history_max_chars)))


def _resolve_function_calling_plan(
    dataset_slug: str,
    dataset_len: int,
    *,
    avg_ks: Sequence[float] | None,
):
    explicit = tuple(float(item) for item in (avg_ks or ()))
    if not explicit:
        return build_auto_avg_k_execution_plan(dataset_slug, dataset_len)
    if len(explicit) != 1:
        rendered = ", ".join(str(item) for item in explicit)
        raise ValueError(f"function-calling runner accepts exactly one avg_k override, got: {rendered}")
    return build_avg_k_execution_plan(dataset_slug, dataset_len, avg_k=explicit[0])


def _run_browsecomp(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_browsecomp_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("BrowseComp manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(records), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    cot_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    answer_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="final",
        fallback_templates="free_response_cot_default",
    )
    if cot_sampling is None or answer_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    answer_sampling = answer_sampling.clamp(args.answer_max_tokens)

    batch_size = max(1, int(args.batch_size or 32))
    selected_entries = [(int(sample_index), records[int(sample_index)]) for sample_index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        prompts = [
            build_browsecomp_expected_context(
                build_browsecomp_user_prompt(record.question, locale=record.locale)
            )
            for _, record in repeated
        ]
        run.engine.generate(
            prompts,
            sampling=cot_sampling,
            batch_size=len(prompts),
            progress_desc="BrowseComp-Probe",
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("BrowseComp requires JUDGE_MODEL / judge_model_name and judge API key")
    judge = BrowseCompJudgeConfig(
        api_key=judge_cfg.api_key,
        model=judge_cfg.model_name,
        base_url=judge_cfg.base_url,
    )

    job_name = _resolve_job_name("function_browsecomp", run_context=run_context)
    sampling_payload = normalize_sampling_config_by_stage([(1, cot_sampling), (2, answer_sampling)])
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 32),
        run_context=run_context,
        judger_model_name=judge.model,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_browsecomp_completion_to_eval_payload,
        runner_name="browsecomp",
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=_flush_partial_eval,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)

                for start in range(0, len(pending), batch_size):
                    chunk = pending[start : start + batch_size]
                    cot_prompts = [
                        build_browsecomp_expected_context(
                            build_browsecomp_user_prompt(record.question, locale=record.locale)
                        )
                        for _key, record in chunk
                    ]
                    cot_outputs = run.engine.generate(
                        cot_prompts,
                        sampling=cot_sampling,
                        batch_size=len(cot_prompts),
                        progress_desc="BrowseComp-CoT",
                        prompt_seeds=[
                            sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=1,
                            )
                            for key, _record in chunk
                        ],
                    )
                    cot_by_index = {int(output.prompt_index): output for output in cot_outputs}
                    answer_prompts: list[str] = []
                    answer_stage_prompts: list[str] = []
                    for index, (_key, record) in enumerate(chunk):
                        cot_output = cot_by_index[index]
                        answer_prompt = build_browsecomp_answer_prompt(
                            cot_prompts[index],
                            cot_output.text,
                            locale=record.locale,
                        )
                        answer_prompts.append(answer_prompt)
                        answer_stage_prompts.append(prompt_delta(answer_prompt, f"{cot_output.prompt}{cot_output.text}"))
                    answer_outputs = run.engine.generate(
                        answer_prompts,
                        sampling=answer_sampling,
                        batch_size=len(answer_prompts),
                        progress_desc="BrowseComp-Answer",
                        prompt_seeds=[
                            sample_repeat_seed(
                                key.sample_index,
                                key.repeat_index,
                                pass_index=key.pass_index,
                                stage=2,
                            )
                            for key, _record in chunk
                        ],
                    )
                    answer_by_index = {int(output.prompt_index): output for output in answer_outputs}
                    judged = judge_browsecomp_answers(
                        [
                            (
                                record,
                                _normalize_final_answer(answer_by_index[index].text, locale=record.locale),
                            )
                            for index, (_key, record) in enumerate(chunk)
                        ],
                        config=judge,
                    )
                    for index, ((key, record), outcome) in enumerate(zip(chunk, judged)):
                        cot_output = cot_by_index[index]
                        answer_output = answer_by_index[index]
                        final_answer = _normalize_final_answer(answer_output.text, locale=record.locale)
                        stages = [
                            StageRecord(
                                prompt=cot_prompts[index],
                                completion=cot_output.text,
                                stop_reason=cot_output.finish_reason,
                            ),
                            StageRecord(
                                prompt=answer_stage_prompts[index],
                                completion=answer_output.text,
                                stop_reason=answer_output.finish_reason,
                            ),
                        ]
                        payload = SampleRecord(
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            stages=stages,
                            sampling_config=sampling_payload,
                        ).as_payload()
                        payload["agent_result"] = {
                            "reward": 1.0 if outcome.is_passed else 0.0,
                            "num_turns": 2,
                            "cost": 0.0,
                            "is_passed": bool(outcome.is_passed),
                        }
                        payload["agent_info"] = {
                            "question": record.question,
                            "reference_answer": record.answer,
                            "response": final_answer,
                            "judge_reason": outcome.reason,
                            "locale": record.locale,
                            "cot_mode": CoTMode.COT.value,
                            "topic": record.topic or "",
                        }
                        payload["agent_trace"] = [
                            {"stage": "cot", "text": cot_output.text},
                            {"stage": "answer", "text": final_answer},
                        ]
                        payload["task_id"] = record.task_id
                        payload["domain"] = "function_call"
                        payload["instruction"] = record.question
                        writer.enqueue(payload)
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _flush_partial_eval("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_browsecomp_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=len(records),
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={
                    "sampling_config": sampling_payload,
                    "judger_model_name": judge.model,
                    "cot_mode": CoTMode.COT.value,
                },
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"browsecomp done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


def _mcp_bench_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("fail_reason") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=str(agent_info.get("final_answer") or ""),
        ref_answer=str(agent_info.get("ref_answer") or ""),
    )


def _presented_task(item: McpBenchItem) -> str:
    return item.task.fuzzy_description.strip() or item.task.task_description.strip()


def _execution_to_dict(entry: McpBenchExecutionResult) -> dict[str, object]:
    return {
        "tool": entry.tool,
        "server": entry.server,
        "parameters": dict(entry.parameters),
        "round_num": int(entry.round_num),
        "planned_layer": entry.planned_layer,
        "success": bool(entry.success),
        "result": entry.result,
        "error": entry.error,
    }


def _run_mcp_bench(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    items = load_mcp_bench_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        items = items[: int(args.max_samples)]
    if not items:
        raise ValueError("MCP-Bench manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(items), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    base_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="cot",
        fallback_templates="instruction_following_default",
    )
    if base_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    planning_sampling = base_sampling.clamp(args.planning_max_tokens)
    decision_sampling = base_sampling.clamp(args.decision_max_tokens or 2048)
    final_sampling = base_sampling.clamp(args.final_max_tokens)

    runtime_root = Path(items[0].runtime_root or "").expanduser().resolve()
    worker_script = REPO_ROOT / "src" / "eval" / "function_calling" / "mcp_bench_worker.py"
    if args.probe_only:
        worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
        try:
            available_tools = worker.open_task(items[0])
            prompt = build_planning_context(items[0], available_tools, "")
            run.engine.generate(
                [prompt],
                sampling=planning_sampling,
                batch_size=1,
                progress_desc="MCPBench-Probe",
            )
            worker.close_task()
        finally:
            worker.close()
        print("probe-only run completed: 1 task")
        return 0

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("MCP-Bench requires JUDGE_MODEL / judge_model_name and judge API key")

    job_name = _resolve_job_name("function_mcp_bench", run_context=run_context)
    sampling_payload = normalize_sampling_config_by_stage(
        [(1, planning_sampling), (2, decision_sampling), (3, final_sampling)]
    )
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 8),
        run_context=run_context,
        judger_model_name=judge_cfg.model_name,
    )
    worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
    runtime = ctx.runtime
    writer = ctx.writer
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_mcp_bench_completion_to_eval_payload,
        runner_name="mcp_bench",
    )

    def _handle_runtime_interrupt(signame: str) -> None:
        try:
            worker.close()
        finally:
            _flush_partial_eval(signame)

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=_handle_runtime_interrupt,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, items, skip_keys=ctx.skip_keys)
                max_rounds = max(1, int(args.max_rounds))
                for key, item in pending:
                    sample_index = key.sample_index
                    repeat_index = key.repeat_index
                    stages: list[StageRecord] = []
                    steps: list[dict[str, object]] = []
                    accumulated_information = ""
                    execution_results: list[McpBenchExecutionResult] = []
                    final_answer = ""
                    fail_reason = ""
                    evaluation_summary = ""
                    is_passed = False
                    try:
                        available_tools = worker.open_task(item)
                        total_planned_tools = 0
                        valid_planned_tools = 0
                        executed_rounds = 0
                        for round_num in range(1, max_rounds + 1):
                            cot_context = build_planning_context(item, available_tools, accumulated_information)
                            cot_output = run.engine.generate(
                                [cot_context],
                                sampling=planning_sampling,
                                batch_size=1,
                                progress_desc="MCPBench-Plan",
                                prompt_seeds=[
                                    sample_repeat_seed(
                                        sample_index,
                                        repeat_index,
                                        pass_index=key.pass_index,
                                        stage=(round_num - 1) * 3 + 1,
                                    )
                                ],
                            )[0]
                            decision_prompt = build_planning_decision_prompt(cot_context, cot_output.text)
                            decision_output = run.engine.generate(
                                [decision_prompt],
                                sampling=decision_sampling,
                                batch_size=1,
                                progress_desc="MCPBench-Decision",
                                prompt_seeds=[
                                    sample_repeat_seed(
                                        sample_index,
                                        repeat_index,
                                        pass_index=key.pass_index,
                                        stage=(round_num - 1) * 3 + 2,
                                    )
                                ],
                            )[0]
                            stages.append(
                                StageRecord(
                                    prompt=cot_context,
                                    completion=cot_output.text,
                                    stop_reason=cot_output.finish_reason,
                                )
                            )
                            stages.append(
                                StageRecord(
                                    prompt=prompt_delta(decision_prompt, f"{cot_context}{cot_output.text}"),
                                    completion=decision_output.text,
                                    stop_reason=decision_output.finish_reason,
                                )
                            )
                            try:
                                decision = parse_planning_decision(decision_output.text)
                            except Exception as exc:
                                fail_reason = str(exc)
                                steps.append(
                                    {
                                        "round_num": round_num,
                                        "cot": cot_output.text,
                                        "decision": {"raw": decision_output.text, "parse_error": fail_reason},
                                        "executions": [],
                                    }
                                )
                                break

                            round_executions: list[McpBenchExecutionResult] = []
                            if decision.should_continue:
                                for planned_layer, raw_call in enumerate(decision.tool_calls):
                                    total_planned_tools += 1
                                    try:
                                        normalized = normalize_planned_tool_call(raw_call, available_tools)
                                        valid_planned_tools += 1
                                        tool_response = worker.call_tool(normalized.full_name, normalized.arguments)
                                        success = bool(tool_response.get("success", False))
                                        round_executions.append(
                                            McpBenchExecutionResult(
                                                tool=normalized.full_name,
                                                server=normalized.server,
                                                parameters=dict(normalized.arguments),
                                                round_num=round_num,
                                                planned_layer=planned_layer,
                                                success=success,
                                                result=str(tool_response.get("result") or "") or None,
                                                error=str(tool_response.get("error") or "") or None,
                                            )
                                        )
                                    except Exception as exc:
                                        tool_name = raw_call.full_name if raw_call.server.strip() else raw_call.tool
                                        round_executions.append(
                                            McpBenchExecutionResult(
                                                tool=tool_name,
                                                server=raw_call.server.strip() or "unknown",
                                                parameters=dict(raw_call.arguments),
                                                round_num=round_num,
                                                planned_layer=planned_layer,
                                                success=False,
                                                error=str(exc),
                                            )
                                        )
                            steps.append(
                                {
                                    "round_num": round_num,
                                    "cot": cot_output.text,
                                    "decision": {
                                        "reasoning": decision.reasoning,
                                        "should_continue": decision.should_continue,
                                        "tool_calls": [
                                            {
                                                "server": call.server,
                                                "tool": call.tool,
                                                "arguments": dict(call.arguments),
                                            }
                                            for call in decision.tool_calls
                                        ],
                                    },
                                    "executions": [_execution_to_dict(entry) for entry in round_executions],
                                }
                            )
                            if round_executions:
                                accumulated_information = append_round_summary(
                                    accumulated_information,
                                    round_num,
                                    decision.reasoning,
                                    round_executions,
                                )
                                execution_results.extend(round_executions)
                                executed_rounds = round_num
                            if not decision.should_continue or not round_executions:
                                break

                        if not fail_reason:
                            final_prompt = build_final_answer_prompt(item, accumulated_information)
                            final_output = run.engine.generate(
                                [final_prompt],
                                sampling=final_sampling,
                                batch_size=1,
                                progress_desc="MCPBench-Final",
                                prompt_seeds=[
                                    sample_repeat_seed(
                                        sample_index,
                                        repeat_index,
                                        pass_index=key.pass_index,
                                        stage=max_rounds * 3,
                                    )
                                ],
                            )[0]
                            stages.append(
                                StageRecord(
                                    prompt=final_prompt,
                                    completion=final_output.text,
                                    stop_reason=final_output.finish_reason,
                                )
                            )
                            final_answer = final_output.text.strip()
                            planning_json_compliance = (
                                valid_planned_tools / total_planned_tools if total_planned_tools > 0 else 1.0
                            )
                            evaluation = worker.evaluate(
                                {
                                    "judge_config": {
                                        "api_key": judge_cfg.api_key,
                                        "base_url": judge_cfg.base_url or "",
                                        "model": judge_cfg.model_name,
                                    },
                                    "task": _presented_task(item),
                                    "final_solution": final_answer,
                                    "total_rounds": executed_rounds,
                                    "available_tools": available_tools,
                                    "planning_json_compliance": planning_json_compliance,
                                    "accumulated_information": accumulated_information,
                                    "concrete_task_description": (
                                        item.task.task_description if item.task.fuzzy_description.strip() else ""
                                    ),
                                    "dependency_analysis": item.task.dependency_analysis,
                                    "execution_results": [_execution_to_dict(entry) for entry in execution_results],
                                }
                            )
                            is_passed = collapse_mcp_bench_pass(evaluation)
                            evaluation_summary = summarize_mcp_bench_evaluation(evaluation)
                            if not is_passed:
                                fail_reason = evaluation_summary
                        if not final_answer and not fail_reason:
                            fail_reason = "mcp_bench produced no final answer"
                        ref_answer = build_mcp_bench_ref_answer(item)
                        payload = SampleRecord(
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sample_index=sample_index,
                            repeat_index=repeat_index,
                            pass_index=key.pass_index,
                            stages=stages,
                            sampling_config=sampling_payload,
                        ).as_payload()
                        payload["agent_result"] = {
                            "reward": 1.0 if is_passed else 0.0,
                            "num_turns": len(steps),
                            "cost": 0.0,
                            "is_passed": is_passed,
                            "error": fail_reason or None,
                        }
                        payload["agent_info"] = {
                            "final_answer": final_answer,
                            "ref_answer": ref_answer,
                            "fail_reason": fail_reason,
                            "evaluation_summary": evaluation_summary,
                            "cot_mode": CoTMode.COT.value,
                            "execution_count": len(execution_results),
                        }
                        payload["agent_trace"] = steps
                        payload["task_id"] = item.task.task_id
                        payload["domain"] = "function_call"
                        payload["instruction"] = _presented_task(item)
                        writer.enqueue(payload)
                    except Exception as exc:
                        ref_answer = build_mcp_bench_ref_answer(item)
                        payload = SampleRecord(
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sample_index=sample_index,
                            repeat_index=repeat_index,
                            pass_index=key.pass_index,
                            stages=stages,
                            sampling_config=sampling_payload,
                        ).as_payload()
                        payload["agent_result"] = {
                            "reward": 0.0,
                            "num_turns": len(steps),
                            "cost": 0.0,
                            "is_passed": False,
                            "error": str(exc),
                        }
                        payload["agent_info"] = {
                            "final_answer": final_answer,
                            "ref_answer": ref_answer,
                            "fail_reason": str(exc),
                            "evaluation_summary": "",
                            "cot_mode": CoTMode.COT.value,
                            "execution_count": len(execution_results),
                        }
                        payload["agent_trace"] = steps
                        payload["task_id"] = item.task.task_id
                        payload["domain"] = "function_call"
                        payload["instruction"] = _presented_task(item)
                        writer.enqueue(payload)
                    finally:
                        try:
                            worker.close_task()
                        except Exception:
                            pass
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _handle_runtime_interrupt("exception"),
                )
                raise
    finally:
        worker.close()

    try:
        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_mcp_bench_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=len(items),
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={
                    "sampling_config": sampling_payload,
                    "judger_model_name": judge_cfg.model_name,
                    "cot_mode": CoTMode.COT.value,
                },
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"mcp_bench done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


def _bfcl_v3_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("fail_reason") or agent_result.get("error") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=str(agent_info.get("final_answer") or ""),
        ref_answer=str(agent_info.get("ref_answer") or ""),
    )


def _start_bfcl_episode(
    *,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    record: BfclTaskRecord,
) -> _ActiveBfclEpisode:
    active_tools = [dict(tool) for tool in record.tools]
    system_prompt = build_bfcl_system_prompt(active_tools)
    prompt_messages = (
        []
        if has_bfcl_official_turns(record)
        else [
            {
                "role": "user",
                "content": build_bfcl_user_block(record.instruction.strip()),
            }
        ]
    )
    runtime_state = start_bfcl_runtime(record)
    runtime_state.official_model_name = (
        f"rwkv_bfcl_{sample_index}_{repeat_index}_{pass_index}_{uuid.uuid4().hex}"
    )
    return _ActiveBfclEpisode(
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        record=record,
        system_prompt=system_prompt,
        prompt_messages=prompt_messages,
        active_tools=active_tools,
        runtime_state=runtime_state,
    )


def _merge_bfcl_tools(
    current: Sequence[dict[str, Any]],
    additions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {str(tool.get("name") or ""): dict(tool) for tool in current}
    for tool in additions:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        merged[name] = dict(tool)
    return [tool for name, tool in merged.items() if name]


def _next_bfcl_stage_seed(state: _ActiveBfclEpisode) -> int:
    return sample_repeat_seed(
        state.sample_index,
        state.repeat_index,
        pass_index=state.pass_index,
        stage=max(1, len(state.stages) + 1),
    )


def _generate_bfcl_stage(
    *,
    state: _ActiveBfclEpisode,
    run: ResolvedFunctionCallingRun,
    prompt: str,
    sampling: Any,
    progress_desc: str,
    stop_suffixes: Sequence[str] | None = None,
    constraint: Any | None = None,
    constraint_mode: str = "off",
    completion_for_record: str | None = None,
) -> Any:
    output = run.engine.generate(
        [prompt],
        sampling=sampling,
        batch_size=1,
        progress_desc=progress_desc,
        prompt_stop_suffixes=None if stop_suffixes is None else [list(stop_suffixes)],
        constraints=None if constraint is None else [constraint],
        constraint_mode=constraint_mode,
        prompt_seeds=[_next_bfcl_stage_seed(state)],
    )[0]
    state.stages.append(
        StageRecord(
            prompt=prompt,
            completion=output.text if completion_for_record is None else completion_for_record,
            stop_reason=output.finish_reason,
        )
    )
    return output


def _failed_bfcl_step(
    state: _ActiveBfclEpisode,
    trace_entry: dict[str, object],
    *,
    termination_reason: str,
    error: str,
) -> _BfclGenerationStepOutcome:
    state.termination_reason = termination_reason
    state.error = error
    trace_entry["error"] = error
    return _BfclGenerationStepOutcome(ok=False, trace_entry=trace_entry)


def _run_bfcl_generation_step(
    *,
    state: _ActiveBfclEpisode,
    run: ResolvedFunctionCallingRun,
    user_request: str,
    cot_sampling: Any,
    router_sampling: Any,
    tool_sampling: Any,
    ask_sampling: Any,
    handoff_sampling: Any,
    progress_suffix: str,
    recent_tool_result: dict[str, Any] | None,
    previous_state_snapshot: Mapping[str, Any] | None,
) -> _BfclGenerationStepOutcome:
    cot_prompt = build_bfcl_cot_prompt(
        state.system_prompt,
        user_request=user_request,
        current_state_snapshot=state.runtime_state.current_state,
        previous_tool_result=recent_tool_result,
    )
    cot_output = _generate_bfcl_stage(
        state=state,
        run=run,
        prompt=cot_prompt,
        sampling=cot_sampling,
        progress_desc=f"BFCLV3-CoT {progress_suffix}",
        stop_suffixes=[BFCL_COT_STOP_SUFFIX],
    )
    trace_entry: dict[str, object] = {
        "cot": cot_output.text,
        "cot_stop_reason": cot_output.finish_reason,
    }
    if _looks_like_template_leak(cot_output.text):
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="template_leak",
            error="cot stage leaked internal template/control tokens",
        )
    if cot_output.finish_reason == "max_length":
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="cot_max_length",
            error="cot stage reached max_length before a bounded decision handoff",
        )
    if len(cot_output.text) > BFCL_V3_MAX_COT_CHARS:
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="cot_too_long",
            error="cot stage exceeded bounded reasoning budget",
        )

    cot_hidden_summary = extract_bfcl_cot_hidden_summary(cot_output.text)
    prompt_kwargs = {
        "user_request": user_request,
        "cot_hidden_summary": cot_hidden_summary,
        "recent_tool_window": state.runtime_state.executed_tool_calls,
        "current_state_snapshot": state.runtime_state.current_state,
        "previous_state_snapshot": previous_state_snapshot,
        "previous_tool_result": recent_tool_result,
    }
    router_prompt = build_bfcl_router_prompt(state.system_prompt, **prompt_kwargs)
    router_output = _generate_bfcl_stage(
        state=state,
        run=run,
        prompt=router_prompt,
        sampling=router_sampling,
        progress_desc=f"BFCLV3-Router {progress_suffix}",
        stop_suffixes=["\n"],
        constraint=LiteralChoiceConstraint(tuple(BFCL_ROUTER_LABELS)),
        constraint_mode="strict",
    )
    trace_entry["router_text"] = router_output.text
    trace_entry["router_stop_reason"] = router_output.finish_reason
    if _looks_like_template_leak(router_output.text):
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="template_leak",
            error="router stage leaked internal template/control tokens",
        )
    if router_output.finish_reason == "max_length":
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="router_max_length",
            error="router stage reached max_length before producing TOOL/ASK/HANDOFF",
        )
    try:
        action_type = parse_bfcl_router_output(router_output.text)
    except Exception as exc:
        trace_entry["parse_error"] = str(exc)
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="invalid_router_output",
            error=str(exc),
        )

    trace_entry["action_type"] = action_type
    state.step_count += 1
    if action_type == "TOOL":
        tool_prompt = build_bfcl_tool_prompt(state.system_prompt, **prompt_kwargs)
        tool_output = _generate_bfcl_stage(
            state=state,
            run=run,
            prompt=tool_prompt,
            sampling=tool_sampling,
            progress_desc=f"BFCLV3-Tool {progress_suffix}",
            stop_suffixes=BFCL_DECISION_STOP_SUFFIXES,
            constraint=build_bfcl_tool_call_constraint(state.active_tools),
            constraint_mode="strict",
        )
        tool_text = normalize_bfcl_tool_output_safe(reconstruct_bfcl_tool_output(tool_output.text))
        trace_entry["tool_completion"] = tool_output.text
        trace_entry["tool_text"] = tool_text
        if _looks_like_template_leak(tool_text):
            return _failed_bfcl_step(
                state,
                trace_entry,
                termination_reason="template_leak",
                error="tool stage leaked internal template/control tokens",
            )
        if tool_output.finish_reason == "max_length":
            return _failed_bfcl_step(
                state,
                trace_entry,
                termination_reason="decision_max_length",
                error="tool stage reached max_length before producing a bounded tool call",
            )
        try:
            decoded_calls, turn_handoff = decode_bfcl_exec_response(
                tool_text,
                tools=state.active_tools,
            )
        except Exception as exc:
            trace_entry["parse_error"] = str(exc)
            return _failed_bfcl_step(
                state,
                trace_entry,
                termination_reason="invalid_tool_call",
                error=str(exc),
            )
        if not decoded_calls:
            trace_entry["tool_handoff"] = turn_handoff
            return _failed_bfcl_step(
                state,
                trace_entry,
                termination_reason="invalid_tool_call",
                error="router selected TOOL but model returned plain text",
            )
        return _BfclGenerationStepOutcome(
            ok=True,
            trace_entry=trace_entry,
            action_type=action_type,
            tool_call=decoded_calls[0],
        )

    branch_prompt_builder = build_bfcl_ask_prompt if action_type == "ASK" else build_bfcl_handoff_prompt
    branch_sampling = ask_sampling if action_type == "ASK" else handoff_sampling
    branch_progress = "BFCLV3-Ask" if action_type == "ASK" else "BFCLV3-Handoff"
    branch_prompt = branch_prompt_builder(state.system_prompt, **prompt_kwargs)
    branch_output = _generate_bfcl_stage(
        state=state,
        run=run,
        prompt=branch_prompt,
        sampling=branch_sampling,
        progress_desc=f"{branch_progress} {progress_suffix}",
        stop_suffixes=["\nUser:", "\nSystem:", "\nAssistant:"],
    )
    branch_text = normalize_bfcl_decision_output(branch_output.text)
    trace_entry["branch_text"] = branch_text
    if _looks_like_template_leak(branch_text):
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="template_leak",
            error=f"{action_type.lower()} stage leaked internal template/control tokens",
        )
    if branch_output.finish_reason == "max_length":
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="decision_max_length",
            error=f"{action_type.lower()} stage reached max_length before producing a bounded reply",
        )
    try:
        decision = parse_bfcl_assistant_output(branch_text)
    except Exception as exc:
        trace_entry["parse_error"] = str(exc)
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="invalid_plain_response",
            error=str(exc),
        )
    if decision.is_tool_call:
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="invalid_plain_response",
            error=f"router selected {action_type} but model returned a tool call",
        )
    final_answer = decision.final_answer.strip()
    trace_entry["final_answer"] = final_answer
    return _BfclGenerationStepOutcome(
        ok=True,
        trace_entry=trace_entry,
        action_type=action_type,
        final_answer=final_answer,
    )


def _run_bfcl_v3_official_episode(
    *,
    state: _ActiveBfclEpisode,
    run: ResolvedFunctionCallingRun,
    cot_sampling: Any,
    router_sampling: Any,
    tool_sampling: Any,
    ask_sampling: Any,
    handoff_sampling: Any,
    max_steps: int,
    max_tool_errors: int,
    history_max_chars: int,
) -> list[dict[str, object]]:
    trace: list[dict[str, object]] = []

    for turn_index, turn in enumerate(state.record.turns):
        state.runtime_state.current_turn_index = turn_index
        if turn.tool_additions:
            state.active_tools = _merge_bfcl_tools(state.active_tools, turn.tool_additions)
            state.system_prompt = build_bfcl_system_prompt(state.active_tools)

        turn_request = render_bfcl_turn_request(turn.messages)
        if not turn_request and turn.tool_additions:
            turn_request = BFCL_ADDITIONAL_FUNCTION_PROMPT
        state.prompt_messages.append(
            {
                "role": "user",
                "content": build_bfcl_user_block(turn_request),
            }
        )

        current_turn_outputs: list[list[str]] = []
        step_in_turn = 0
        turn_finished = False
        recent_tool_result: dict[str, Any] | None = None
        previous_state_snapshot: dict[str, Any] | None = None

        while step_in_turn < max_steps:
            progress_suffix = f"sample {state.sample_index} turn {turn_index + 1} step {step_in_turn + 1}"
            outcome = _run_bfcl_generation_step(
                state=state,
                run=run,
                user_request=turn_request,
                cot_sampling=cot_sampling,
                router_sampling=router_sampling,
                tool_sampling=tool_sampling,
                ask_sampling=ask_sampling,
                handoff_sampling=handoff_sampling,
                progress_suffix=progress_suffix,
                recent_tool_result=recent_tool_result,
                previous_state_snapshot=previous_state_snapshot,
            )
            trace_entry = {
                "turn_index": turn_index,
                "step_index": step_in_turn,
                **outcome.trace_entry,
            }
            if not outcome.ok:
                state.final_answer = outcome.final_answer.strip()
                trace.append(trace_entry)
                break

            if outcome.tool_call is not None:
                tool_call = outcome.tool_call
                previous_state_snapshot = dict(state.runtime_state.current_state)
                state.tool_calls.append(tool_call)
                state.prompt_messages.append(
                    {"role": "assistant", "content": render_bfcl_assistant_tool_message(tool_call)}
                )
                try:
                    execution = execute_bfcl_official_tool_call(state.record, state.runtime_state, tool_call)
                except Exception as exc:
                    state.tool_errors += 1
                    recent_tool_result = build_bfcl_tool_result_payload(
                        tool_call,
                        ok=False,
                        error=str(exc),
                    )
                    state.prompt_messages.append(
                        {
                            "role": "user",
                            "content": build_bfcl_tool_result_message(
                                recent_tool_result,
                                current_state_snapshot=state.runtime_state.current_state,
                                previous_state_snapshot=previous_state_snapshot,
                            ),
                        }
                    )
                    trace_entry["tool_call"] = {
                        "name": tool_call.name,
                        "arguments": dict(tool_call.arguments),
                    }
                    trace_entry["tool_error"] = str(exc)
                    trace.append(trace_entry)
                    if state.tool_errors >= max_tool_errors:
                        state.termination_reason = "too_many_errors"
                        state.error = str(exc)
                        break
                    step_in_turn += 1
                    continue
                recent_tool_result = build_bfcl_tool_result_payload(
                    tool_call,
                    ok=execution.success,
                    output=execution.result,
                    error=execution.error,
                )
                state.prompt_messages.append(
                    {
                        "role": "user",
                        "content": build_bfcl_tool_result_message(
                            recent_tool_result,
                            current_state_snapshot=state.runtime_state.current_state,
                            previous_state_snapshot=previous_state_snapshot,
                        ),
                    }
                )
                current_turn_outputs.append([render_bfcl_official_call(tool_call)])
                if not execution.matched_expectation:
                    state.tool_errors += 1
                trace_entry["tool_calls"] = [
                    {
                        "name": tool_call.name,
                        "arguments": dict(tool_call.arguments),
                    }
                ]
                trace_entry["tool_success"] = execution.success
                trace_entry["tool_result"] = execution.result
                trace_entry["tool_error"] = execution.error
                trace_entry["state_snapshot"] = dict(execution.state_snapshot)
                trace.append(trace_entry)
                if state.tool_errors >= max_tool_errors:
                    state.termination_reason = "too_many_errors"
                    state.error = "too many BFCL tool execution errors"
                    break
                step_in_turn += 1
                continue

            state.final_answer = outcome.final_answer.strip()
            if state.final_answer:
                state.prompt_messages.append({"role": "assistant", "content": state.final_answer})
            trace_entry["turn_handoff"] = state.final_answer
            trace.append(trace_entry)
            turn_finished = True
            break

        state.runtime_state.decoded_turn_outputs.append(current_turn_outputs)
        state.turn_count += 1

        if state.termination_reason is not None:
            break
        if not turn_finished and step_in_turn >= max_steps:
            state.termination_reason = "max_steps"
            state.error = f"BFCL turn {turn_index + 1} exceeded max_steps={max_steps}"
            break

    if state.termination_reason is None:
        state.termination_reason = "agent_stop"
    return trace


def _run_bfcl_v3(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_bfcl_v3_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("BFCL V3 manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(records), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    cot_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    router_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="router",
        fallback_templates="instruction_following_default",
    )
    tool_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    ask_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="ask",
        fallback_templates="instruction_following_default",
    )
    handoff_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="handoff",
        fallback_templates="instruction_following_default",
    )
    if (
        cot_sampling is None
        or router_sampling is None
        or tool_sampling is None
        or ask_sampling is None
        or handoff_sampling is None
    ):
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    router_sampling = router_sampling.clamp(min(int(args.decision_max_tokens or 8), 8))
    tool_sampling = tool_sampling.clamp(min(int(args.decision_max_tokens or 192), 192))
    ask_sampling = ask_sampling.clamp(min(int(args.decision_max_tokens or 96), 96))
    handoff_sampling = handoff_sampling.clamp(min(int(args.decision_max_tokens or 96), 96))
    sampling_payload = normalize_sampling_config_by_stage(
        [
            (1, cot_sampling),
            (2, router_sampling),
            (3, tool_sampling),
            (4, ask_sampling),
            (5, handoff_sampling),
        ]
    )

    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]
    dataset_issues = collect_bfcl_dataset_issues([record for _index, record in selected_entries])
    if dataset_issues:
        preview = "\n".join(f"- {issue}" for issue in dataset_issues[:10])
        remainder = len(dataset_issues) - min(len(dataset_issues), 10)
        if remainder > 0:
            preview += f"\n- ... and {remainder} more"
        raise ValueError(
            "BFCL V3 dataset/support assets are incomplete; fix the official possible_answer/function-doc setup before scoring:\n"
            + preview
        )
    batch_size = max(1, int(args.batch_size or 16))
    max_steps = max(1, int(args.max_steps))
    max_tool_errors = max(1, int(args.max_tool_errors))
    history_max_chars = max(0, int(args.history_max_chars))

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        probe_states = [
            _start_bfcl_episode(
                sample_index=sample_index,
                repeat_index=0,
                pass_index=0,
                record=record,
            )
            for sample_index, record in repeated
        ]
        cot_prompts = [
            build_bfcl_cot_prompt(
                build_bfcl_system_prompt(
                    state.active_tools
                    if not has_bfcl_official_turns(state.record)
                    else _merge_bfcl_tools(
                        state.active_tools,
                        state.record.turns[0].tool_additions if state.record.turns else (),
                    )
                ),
                user_request=(
                    render_bfcl_turn_request(state.record.turns[0].messages)
                    if state.record.turns
                    else state.record.instruction.strip()
                ),
                current_state_snapshot=state.runtime_state.current_state,
            )
            for state in probe_states
        ]
        cot_outputs = run.engine.generate(
            cot_prompts,
            sampling=cot_sampling,
            batch_size=len(cot_prompts),
            progress_desc="BFCLV3-Probe-CoT",
            prompt_stop_suffixes=[[BFCL_COT_STOP_SUFFIX] for _ in cot_prompts],
            prompt_seeds=[
                sample_repeat_seed(state.sample_index, state.repeat_index, stage=1)
                for state in probe_states
            ],
        )
        cot_summaries = [extract_bfcl_cot_hidden_summary(output.text) for output in cot_outputs]
        probe_tool_sets = [
            (
                state.active_tools
                if not has_bfcl_official_turns(state.record)
                else _merge_bfcl_tools(
                    state.active_tools,
                    state.record.turns[0].tool_additions if state.record.turns else (),
                )
            )
            for state in probe_states
        ]
        router_prompts = [
            build_bfcl_router_prompt(
                build_bfcl_system_prompt(
                    probe_tool_sets[index]
                ),
                user_request=(
                    render_bfcl_turn_request(state.record.turns[0].messages)
                    if state.record.turns
                    else state.record.instruction.strip()
                ),
                cot_hidden_summary=cot_summaries[index],
                current_state_snapshot=state.runtime_state.current_state,
            )
            for index, state in enumerate(probe_states)
        ]
        run.engine.generate(
            router_prompts,
            sampling=router_sampling,
            batch_size=len(router_prompts),
            progress_desc="BFCLV3-Probe-Router",
            prompt_stop_suffixes=[["\n"] for _ in router_prompts],
            constraints=[LiteralChoiceConstraint(tuple(BFCL_ROUTER_LABELS)) for _ in router_prompts],
            constraint_mode="strict",
            prompt_seeds=[
                sample_repeat_seed(state.sample_index, state.repeat_index, stage=2)
                for state in probe_states
            ],
        )
        tool_prompts = [
            build_bfcl_tool_prompt(
                build_bfcl_system_prompt(
                    probe_tool_sets[index]
                ),
                user_request=(
                    render_bfcl_turn_request(state.record.turns[0].messages)
                    if state.record.turns
                    else state.record.instruction.strip()
                ),
                cot_hidden_summary=cot_summaries[index],
                current_state_snapshot=state.runtime_state.current_state,
            )
            for index, state in enumerate(probe_states)
        ]
        run.engine.generate(
            tool_prompts,
            sampling=tool_sampling,
            batch_size=len(tool_prompts),
            progress_desc="BFCLV3-Probe-Tool",
            prompt_stop_suffixes=[list(BFCL_DECISION_STOP_SUFFIXES) for _ in tool_prompts],
            constraints=[build_bfcl_tool_call_constraint(tool_set) for tool_set in probe_tool_sets],
            constraint_mode="strict",
            prompt_seeds=[
                sample_repeat_seed(state.sample_index, state.repeat_index, stage=3)
                for state in probe_states
            ],
        )
        print(f"probe-only run completed: {len(probe_states)} prompt(s)")
        return 0

    job_name = _resolve_job_name("function_bfcl_v3", run_context=run_context)
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 32),
        run_context=run_context,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_bfcl_v3_completion_to_eval_payload,
        runner_name="bfcl_v3",
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=_flush_partial_eval,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                for key, record in pending:
                    state = _start_bfcl_episode(
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        record=record,
                    )
                    trace: list[dict[str, object]] = []
                    if has_bfcl_official_turns(record):
                        trace = _run_bfcl_v3_official_episode(
                            state=state,
                            run=run,
                            cot_sampling=cot_sampling,
                            router_sampling=router_sampling,
                            tool_sampling=tool_sampling,
                            ask_sampling=ask_sampling,
                            handoff_sampling=handoff_sampling,
                            max_steps=max_steps,
                            max_tool_errors=max_tool_errors,
                            history_max_chars=history_max_chars,
                        )
                    else:
                        current_request = record.instruction.strip()
                        recent_tool_result: dict[str, Any] | None = None
                        previous_state_snapshot: dict[str, Any] | None = None
                        for _ in range(max_steps):
                            progress_suffix = f"sample {state.sample_index} step {state.turn_count + 1}"
                            outcome = _run_bfcl_generation_step(
                                state=state,
                                run=run,
                                user_request=current_request,
                                cot_sampling=cot_sampling,
                                router_sampling=router_sampling,
                                tool_sampling=tool_sampling,
                                ask_sampling=ask_sampling,
                                handoff_sampling=handoff_sampling,
                                progress_suffix=progress_suffix,
                                recent_tool_result=recent_tool_result,
                                previous_state_snapshot=previous_state_snapshot,
                            )
                            state.turn_count += 1
                            trace_entry = {
                                "round_num": state.turn_count,
                                **outcome.trace_entry,
                            }
                            if not outcome.ok:
                                state.final_answer = outcome.final_answer.strip()
                                trace.append(trace_entry)
                                break

                            if outcome.tool_call is not None:
                                tool_call = outcome.tool_call
                                previous_state_snapshot = dict(state.runtime_state.current_state)
                                state.tool_calls.append(tool_call)
                                state.prompt_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": render_bfcl_assistant_tool_message(tool_call),
                                    }
                                )
                                execution = apply_bfcl_tool_call(record, state.runtime_state, tool_call)
                                recent_tool_result = build_bfcl_tool_result_payload(
                                    tool_call,
                                    ok=execution.success,
                                    output=execution.result,
                                    error=execution.error,
                                )
                                state.prompt_messages.append(
                                    {
                                        "role": "user",
                                        "content": build_bfcl_tool_result_message(
                                            recent_tool_result,
                                            current_state_snapshot=state.runtime_state.current_state,
                                            previous_state_snapshot=previous_state_snapshot,
                                        ),
                                    }
                                )
                                if not execution.matched_expectation:
                                    state.tool_errors += 1
                                trace_entry["tool_calls"] = [
                                    {
                                        "name": tool_call.name,
                                        "arguments": dict(tool_call.arguments),
                                    }
                                ]
                                trace_entry["matched_expectation"] = execution.matched_expectation
                                trace_entry["tool_success"] = execution.success
                                trace_entry["tool_result"] = execution.result
                                trace_entry["tool_error"] = execution.error
                                trace_entry["state_snapshot"] = dict(execution.state_snapshot)
                                trace.append(trace_entry)
                                if state.tool_errors >= max_tool_errors:
                                    state.termination_reason = "too_many_errors"
                                    state.error = "too many BFCL tool execution errors"
                                    break
                                continue

                            state.final_answer = outcome.final_answer.strip()
                            state.prompt_messages.append({"role": "assistant", "content": state.final_answer})
                            state.termination_reason = "agent_stop"
                            trace.append(trace_entry)
                            break

                        if state.termination_reason is None:
                            state.termination_reason = "max_steps"

                    evaluation = evaluate_bfcl_v3_episode(
                        record,
                        state.runtime_state,
                        state.final_answer,
                        termination_reason=state.termination_reason,
                        error=state.error,
                    )
                    payload = SampleRecord(
                        benchmark_name=run.benchmark_name,
                        dataset_split=run.dataset_split,
                        sample_index=state.sample_index,
                        repeat_index=state.repeat_index,
                        pass_index=state.pass_index,
                        stages=list(state.stages),
                        sampling_config=sampling_payload,
                    ).as_payload()
                    payload["agent_result"] = {
                        "reward": float(evaluation.reward),
                        "num_turns": int(state.turn_count),
                        "cost": 0.0,
                        "is_passed": bool(evaluation.is_passed),
                        "error": state.error or evaluation.fail_reason or None,
                    }
                    payload["agent_info"] = {
                        **dict(evaluation.details),
                        "termination_reason": state.termination_reason,
                        "tool_errors": state.tool_errors,
                        "num_steps": state.step_count or state.turn_count,
                        "final_answer": state.final_answer,
                        "ref_answer": build_bfcl_ref_answer(record),
                        "fail_reason": evaluation.fail_reason,
                        "cot_mode": CoTMode.COT.value,
                    }
                    payload["agent_trace"] = trace
                    payload["task_id"] = record.task_id
                    payload["domain"] = "function_call"
                    payload["instruction"] = record.instruction
                    writer.enqueue(payload)
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _flush_partial_eval("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_bfcl_v3_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={
                    "cot_mode": CoTMode.COT.value,
                    "history_max_chars": history_max_chars,
                },
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"bfcl_v3 function-calling done: {len(completions_payloads)} samples")
    return 0


def _default_job_name(dataset_slug: str) -> str:
    lower = dataset_slug.lower()
    if lower.startswith("tau2_bench"):
        return "function_tau2_bench"
    return "function_tau_bench"


def _assistant_tools_schema(runtime_env: TauV2Env, environment: Any) -> list[dict[str, Any]]:
    return runtime_env.tools_schema(environment)


def _user_tools_schema(runtime_env: TauV2Env, environment: Any) -> list[dict[str, Any]]:
    return runtime_env.user_tools_schema(environment)


def _trajectory_to_prompt_messages(trajectory: Sequence[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in trajectory:
        role = str(getattr(item, "role", "") or "").strip().lower()
        if role == "assistant":
            tool_calls = getattr(item, "tool_calls", None)
            content = str(getattr(item, "content", "") or "").strip()
            if tool_calls:
                blocks: list[str] = []
                if content:
                    blocks.append(content)
                for tool_call in tool_calls:
                    payload = {
                        "requestor": str(getattr(tool_call, "requestor", "assistant") or "assistant"),
                        "name": str(getattr(tool_call, "name", "") or ""),
                        "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
                    }
                    blocks.append(f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```")
                rendered = "\n".join(blocks).strip()
                if rendered:
                    messages.append({"role": "assistant", "content": rendered})
            elif content:
                messages.append({"role": "assistant", "content": content})
        elif role == "user":
            content = str(getattr(item, "content", "") or "").strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif role == "tool":
            content = str(getattr(item, "content", "") or "").strip()
            if content:
                messages.append({"role": "user", "content": content})
    return messages


def _start_episode(
    *,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    record: TauManifestRecord,
    runtime_env: TauV2Env,
) -> _ActiveEpisode:
    task = runtime_env.load_task(record.task)
    environment = runtime_env.create_environment(solo_mode=False)
    trajectory = runtime_env.apply_initial_state(environment=environment, task=task)
    system_prompt = build_tau_system_prompt(
        runtime_env.system_prompt(environment),
        assistant_tools=_assistant_tools_schema(runtime_env, environment),
        user_tools=_user_tools_schema(runtime_env, environment),
    )
    prompt_messages = _trajectory_to_prompt_messages(trajectory)
    user_prompt = render_tau_user_prompt(record.task).strip() or record.instruction.strip()
    if user_prompt and not any(item.get("role") == "user" for item in prompt_messages):
        prompt_messages.append({"role": "user", "content": user_prompt})
    return _ActiveEpisode(
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        record=record,
        runtime_env=runtime_env,
        task=task,
        environment=environment,
        system_prompt=system_prompt,
        prompt_messages=prompt_messages,
        trajectory=list(trajectory),
    )


def _tool_output_payload(message: Any) -> tuple[bool, Any, str | None]:
    error_flag = bool(getattr(message, "error", False))
    raw = getattr(message, "content", "")
    text = str(raw or "")
    if error_flag:
        return False, None, text
    try:
        return True, json.loads(text), None
    except json.JSONDecodeError:
        return True, text, None


def _trajectory_dump(trajectory: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in trajectory:
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                rows.append(dumped)
                continue
        rows.append(
            {
                "role": str(getattr(message, "role", "unknown")),
                "content": str(getattr(message, "content", "")),
            }
        )
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sum_message_costs(trajectory: Sequence[Any]) -> float:
    total = 0.0
    for item in trajectory:
        total += _safe_float(getattr(item, "cost", None), 0.0)
    return total


def _ref_answer(state: _ActiveEpisode) -> str:
    task_id = str(getattr(state.task, "id", "") or state.record.task_id)
    return f"domain={state.record.domain}\ntask_id={task_id}\nbenchmark_version={state.record.benchmark_version}"


def _tau_completion_payload(
    state: _ActiveEpisode,
    *,
    benchmark_name: str,
    dataset_split: str,
    sampling_payload: dict[str, Any],
) -> dict[str, Any]:
    evaluation = state.runtime_env.evaluate(
        task=state.task,
        trajectory=state.trajectory,
        termination_reason=state.termination_reason or "max_steps",
        solo_mode=False,
    )
    info = dict(evaluation.details)
    info["termination_reason"] = evaluation.termination_reason
    info["tool_errors"] = state.tool_errors
    info["domain"] = state.record.domain
    info["task_id"] = str(getattr(state.task, "id", "") or state.record.task_id)
    info["benchmark_version"] = state.record.benchmark_version
    info["tool_calls"] = [
        {
            "requestor": call.requestor,
            "name": call.name,
            "arguments": dict(call.arguments),
        }
        for call in state.tool_calls
    ]
    info["final_answer"] = state.final_answer
    info["ref_answer"] = _ref_answer(state)

    payload = SampleRecord(
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        sample_index=state.sample_index,
        repeat_index=state.repeat_index,
        pass_index=state.pass_index,
        stages=list(state.stages),
        sampling_config=sampling_payload,
    ).as_payload()
    payload["_stage"] = "answer"
    payload["agent_result"] = {
        "task_id": info["task_id"],
        "domain": state.record.domain,
        "reward": float(evaluation.reward),
        "num_turns": int(state.turn_count),
        "cost": _sum_message_costs(state.trajectory),
        "is_passed": bool(evaluation.is_passed),
        "error": state.error,
    }
    payload["agent_info"] = info
    payload["agent_trace"] = _trajectory_dump(state.trajectory)
    return payload


def _tau_completion_to_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("agent_result")
    if not isinstance(result, dict):
        result = {}
    info = payload.get("agent_info")
    if not isinstance(info, dict):
        info = {}
    passed = bool(result.get("is_passed", False))
    fail_reason = ""
    if not passed:
        fail_reason = str(result.get("error") or info.get("termination_reason") or "tau_bench evaluation failed")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=fail_reason,
        answer=str(info.get("final_answer") or ""),
        ref_answer=str(info.get("ref_answer") or ""),
    )


def _run_tau(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_tau_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("tau_bench/tau2_bench manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(records), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    cot_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    decision_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="final",
        fallback_templates="instruction_following_default",
    )
    if cot_sampling is None or decision_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    decision_sampling = decision_sampling.clamp(args.decision_max_tokens or 1024)
    sampling_payload = normalize_sampling_config_by_stage([(1, cot_sampling), (2, decision_sampling)])

    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]
    batch_size = max(1, int(args.batch_size or 16))
    max_steps = max(1, int(args.max_steps))
    max_tool_errors = max(1, int(args.max_tool_errors))
    history_max_chars = max(0, int(args.history_max_chars))

    runtime_cache: dict[str, TauV2Env] = {}

    def _runtime_for_domain(domain: str) -> TauV2Env:
        cached = runtime_cache.get(domain)
        if cached is None:
            cached = TauV2Env(domain=domain, judge=None)
            runtime_cache[domain] = cached
        return cached

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        probe_states = [
            _start_episode(
                sample_index=sample_index,
                repeat_index=0,
                pass_index=0,
                record=record,
                runtime_env=_runtime_for_domain(record.domain),
            )
            for sample_index, record in repeated
        ]
        cot_prompts = [
            build_expected_context(
                state.system_prompt,
                _bounded_prompt_messages(state.prompt_messages, history_max_chars=history_max_chars),
            )
            for state in probe_states
        ]
        cot_outputs = run.engine.generate(
            cot_prompts,
            sampling=cot_sampling,
            batch_size=len(cot_prompts),
            progress_desc="TauBench-Probe-CoT",
            prompt_seeds=[
                sample_repeat_seed(state.sample_index, state.repeat_index, stage=1)
                for state in probe_states
            ],
        )
        by_index = {int(item.prompt_index): item for item in cot_outputs}
        decision_prompts = [
            build_turn_completion_prompt(cot_prompts[idx], by_index[idx].text if idx in by_index else "")
            for idx in range(len(cot_prompts))
        ]
        run.engine.generate(
            decision_prompts,
            sampling=decision_sampling,
            batch_size=len(decision_prompts),
            progress_desc="TauBench-Probe-Decision",
            prompt_seeds=[
                sample_repeat_seed(state.sample_index, state.repeat_index, stage=2)
                for state in probe_states
            ],
        )
        print(f"probe-only run completed: {len(probe_states)} prompt(s)")
        return 0

    job_name = _resolve_job_name(_default_job_name(run.dataset_slug), run_context=run_context)
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 32),
        run_context=run_context,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_tau_completion_to_eval_payload,
        runner_name="tau_bench",
    )

    pending: deque[tuple[AttemptKey, TauManifestRecord]] = deque(
        build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=_flush_partial_eval,
        ):
            try:
                active: list[_ActiveEpisode] = []
                while pending or active:
                    while pending and len(active) < batch_size:
                        key, record = pending.popleft()
                        active.append(
                            _start_episode(
                                sample_index=key.sample_index,
                                repeat_index=key.repeat_index,
                                pass_index=key.pass_index,
                                record=record,
                                runtime_env=_runtime_for_domain(record.domain),
                            )
                        )

                    if not active:
                        break

                    cot_prompts = [
                        build_expected_context(
                            state.system_prompt,
                            _bounded_prompt_messages(state.prompt_messages, history_max_chars=history_max_chars),
                        )
                        for state in active
                    ]
                    cot_outputs = run.engine.generate(
                        cot_prompts,
                        sampling=cot_sampling,
                        batch_size=len(cot_prompts),
                        progress_desc="TauBench-CoT",
                        prompt_seeds=[
                            sample_repeat_seed(
                                state.sample_index,
                                state.repeat_index,
                                pass_index=state.pass_index,
                                stage=state.turn_count * 2 + 1,
                            )
                            for state in active
                        ],
                    )
                    cot_by_index = {int(item.prompt_index): item for item in cot_outputs}

                    decision_prompts: list[str] = []
                    for idx, cot_prompt in enumerate(cot_prompts):
                        cot_output = cot_by_index.get(idx)
                        decision_prompts.append(
                            build_turn_completion_prompt(cot_prompt, cot_output.text if cot_output is not None else "")
                        )
                    decision_outputs = run.engine.generate(
                        decision_prompts,
                        sampling=decision_sampling,
                        batch_size=len(decision_prompts),
                        progress_desc="TauBench-Decision",
                        prompt_seeds=[
                            sample_repeat_seed(
                                state.sample_index,
                                state.repeat_index,
                                pass_index=state.pass_index,
                                stage=state.turn_count * 2 + 2,
                            )
                            for state in active
                        ],
                    )
                    decision_by_index = {int(item.prompt_index): item for item in decision_outputs}

                    finished_slots: list[int] = []
                    for slot_index, state in enumerate(active):
                        cot_output = cot_by_index.get(slot_index)
                        decision_output = decision_by_index.get(slot_index)
                        cot_text = cot_output.text if cot_output is not None else ""
                        decision_text = decision_output.text if decision_output is not None else ""
                        prior_context = cot_prompts[slot_index].replace("<|completions_of_cot|>", cot_text)
                        state.stages.append(
                            StageRecord(
                                prompt=cot_prompts[slot_index],
                                completion=cot_text,
                                stop_reason=cot_output.finish_reason if cot_output is not None else "missing_output",
                            )
                        )
                        state.stages.append(
                            StageRecord(
                                prompt=prompt_delta(decision_prompts[slot_index], prior_context),
                                completion=decision_text,
                                stop_reason=(
                                    decision_output.finish_reason if decision_output is not None else "missing_output"
                                ),
                            )
                        )
                        state.turn_count += 1

                        try:
                            decision = parse_tool_call_or_final_answer(decision_text)
                        except Exception as exc:
                            state.termination_reason = "parse_error"
                            state.error = str(exc)
                            finished_slots.append(slot_index)
                            continue

                        if decision.is_tool_call and decision.tool_call is not None:
                            tool_call = decision.tool_call
                            state.tool_calls.append(tool_call)
                            state.prompt_messages.append(
                                {"role": "assistant", "content": render_assistant_tool_message(cot_text, tool_call)}
                            )
                            try:
                                tool_call_model = state.runtime_env.build_tool_call(
                                    tool_call_id=(
                                        f"{state.sample_index}-{state.repeat_index}-{state.pass_index}-"
                                        f"{uuid.uuid4().hex[:8]}"
                                    ),
                                    name=tool_call.name,
                                    arguments=tool_call.arguments,
                                    requestor=tool_call.requestor,
                                )
                                assistant_message = state.runtime_env.build_assistant_message(
                                    content=None,
                                    tool_calls=[tool_call_model],
                                )
                                state.trajectory.append(assistant_message)
                                tool_message = state.runtime_env.call_tool(
                                    environment=state.environment,
                                    tool_call=tool_call_model,
                                )
                                state.trajectory.append(tool_message)
                                ok, output, error_text = _tool_output_payload(tool_message)
                                state.prompt_messages.append(
                                    {
                                        "role": "user",
                                        "content": render_tool_result(
                                            tool_call,
                                            ok=ok,
                                            output=output,
                                            error=error_text,
                                        ),
                                    }
                                )
                                if not ok:
                                    state.tool_errors += 1
                            except Exception as exc:
                                state.tool_errors += 1
                                state.prompt_messages.append(
                                    {
                                        "role": "user",
                                        "content": render_tool_result(tool_call, ok=False, error=str(exc)),
                                    }
                                )

                            if state.tool_errors >= max_tool_errors:
                                state.termination_reason = "too_many_errors"
                                finished_slots.append(slot_index)
                                continue
                            if state.turn_count >= max_steps:
                                state.termination_reason = "max_steps"
                                finished_slots.append(slot_index)
                            continue

                        state.final_answer = decision.final_answer.strip()
                        state.prompt_messages.append({"role": "assistant", "content": state.final_answer})
                        state.trajectory.append(
                            state.runtime_env.build_assistant_message(content=state.final_answer)
                        )
                        state.termination_reason = "agent_stop"
                        finished_slots.append(slot_index)

                    completed_payloads = [
                        _tau_completion_payload(
                            active[slot_index],
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sampling_payload=sampling_payload,
                        )
                        for slot_index in finished_slots
                    ]
                    for payload in completed_payloads:
                        writer.enqueue(payload)

                    for slot_index in reversed(finished_slots):
                        active.pop(slot_index)
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _flush_partial_eval("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_tau_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={"cot_mode": CoTMode.COT.value},
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"tau function-calling done: {len(completions_payloads)} samples")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    run_context: "RunContext | None" = None,
    task_spec: "TaskSpec | None" = None,
) -> int:
    del task_spec
    load_env_file(Path(".env"))
    args = parse_args(argv)
    validate_inference_backend_args(args)
    run = _resolve_run(args)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BROWSECOMP:
        return _run_browsecomp(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.MCP_BENCH:
        return _run_mcp_bench(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BFCL_V3:
        return _run_bfcl_v3(args, run, run_context=run_context)
    return _run_tau(args, run, run_context=run_context)


__all__ = ["FunctionCallingBenchmarkKind", "main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
