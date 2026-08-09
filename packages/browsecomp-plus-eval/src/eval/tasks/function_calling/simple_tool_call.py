from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.tasks.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.tasks.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS, normalize_rwkv_text, truncate_text
from src.eval.tasks.function_calling.native_tool_calls import (
    run_native_tool_call_decision,
)
from src.eval.tasks.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.tasks.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    assistant_json_prefix,
    build_rwkv_json_call_prompt,
)
from src.eval.tasks.function_calling.api_bank_prompt import normalize_api_bank_instruction_for_prompt
from src.eval.tasks.function_calling.tool_call_contract import parse_tool_calls_text
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext
    from src.eval.experiments.parallel_candidate_router.router import ParallelCandidateRouterConfig

_MAX_TOOL_DESCRIPTION_CHARS = 700
_MAX_TOOL_SCHEMA_CHARS = 1200
_GENERIC_CANDIDATE_ROUTER_POLICY = (
    "Select exactly one JSON function call for the user request. "
    "Use only the provided tools and include final argument values only."
)
_AUTO_CANDIDATE_ROUTER_MIN_TOOLS = 16
DEFAULT_TOOL_CALL_IO = "rwkv-json"
_TOOL_CALL_IO_CHOICES = frozenset({"native", "rwkv-json"})


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    name: str
    arguments: dict[str, Any]
    argument_options: dict[str, tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class SimpleToolCallRecord:
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ToolCallExpectation, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SimpleToolCallEvaluation:
    reward: float
    is_passed: bool
    fail_reason: str
    details: dict[str, Any]


def load_simple_tool_call_manifest_records(path: str | Path) -> list[SimpleToolCallRecord]:
    target = Path(path)
    records: list[SimpleToolCallRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            records.append(normalize_simple_tool_call_manifest_row(payload, index=index, source_path=target))
    return records


def normalize_simple_tool_call_manifest_row(
    payload: Mapping[str, Any],
    *,
    index: int,
    source_path: str | Path | None = None,
) -> SimpleToolCallRecord:
    task_id = str(payload.get("task_id") or payload.get("id") or f"tool_call_{index:04d}")
    instruction = str(payload.get("instruction") or payload.get("question") or "").strip()
    if not instruction:
        raise ValueError(f"simple tool-call row {task_id!r} is missing instruction")
    metadata = dict(payload.get("metadata") or {})
    if source_path is not None:
        metadata.setdefault("manifest_path", str(Path(source_path)))
    tool_payloads = _coerce_list(payload.get("tools"))
    expected_tool_calls = _coerce_list(payload.get("expected_tool_calls"))
    return SimpleToolCallRecord(
        task_id=task_id,
        instruction=instruction,
        tools=tuple(_normalize_tool_schema(tool) for tool in tool_payloads),
        expected_tool_calls=tuple(_normalize_tool_expectation(item) for item in expected_tool_calls),
        metadata=metadata,
    )


def load_bfcl_ast_rows_from_sources(
    question_path: str | Path,
    possible_answer_path: str | Path,
    *,
    category: str,
) -> list[dict[str, Any]]:
    questions = _read_json_or_jsonl_items(Path(question_path))
    answer_lookup = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_json_or_jsonl_items(Path(possible_answer_path))
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{category}_{index}")
        answer = answer_lookup.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        if not instruction:
            raise ValueError(f"BFCL row {task_id!r} is missing question content")
        rows.append(
            {
                "task_id": task_id,
                "instruction": instruction,
                "tools": [_normalize_tool_schema(tool) for tool in _coerce_list(item.get("function"))],
                "expected_tool_calls": _normalize_bfcl_ground_truth_calls(answer.get("ground_truth")),
                "metadata": {
                    "source_format": "official_bfcl_v4_ast",
                    "category": category,
                    "source_path": str(Path(question_path)),
                    "possible_answer_path": str(Path(possible_answer_path)),
                    "execution_result_type": _coerce_list(answer.get("execution_result_type")),
                    "bfcl_official_function": _coerce_list(item.get("function")),
                    "bfcl_official_ground_truth": _coerce_list(answer.get("ground_truth")),
                    "bfcl_official_language": _infer_bfcl_language(category, item),
                },
            }
        )
    return rows


def build_simple_tool_call_prompt(
    record: SimpleToolCallRecord,
    *,
    history_max_chars: int,
    prefill_object: bool = False,
) -> str:
    instruction = _simple_tool_call_instruction(record)
    system_prompt = _simple_tool_call_system_prompt(record)
    return build_rwkv_json_call_prompt(
        system_prompt,
        [{"role": "user", "content": instruction}],
        history_max_chars=history_max_chars,
        assistant_prefix=assistant_json_prefix(enable_think=False, prefill_object=prefill_object),
    )


def build_simple_tool_call_messages(record: SimpleToolCallRecord) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _simple_tool_call_native_system_prompt(record)},
        {"role": "user", "content": _simple_tool_call_instruction(record)},
    ]


def _simple_tool_call_instruction(record: SimpleToolCallRecord) -> str:
    instruction = normalize_rwkv_text(record.instruction)
    if _is_api_bank_record(record):
        return normalize_api_bank_instruction_for_prompt(instruction)
    return instruction


def _simple_tool_call_system_prompt(record: SimpleToolCallRecord) -> str:
    return normalize_rwkv_text(
        "\n".join(
            [
                "Tools:",
                _render_tool_catalog(record.tools),
                "Return only a JSON function call.",
                "For one tool call, return one JSON object.",
                "For multiple required tool calls, return a JSON array containing every required call in execution order; do not stop after the first call.",
                "Each arguments object must contain only final argument values for that tool.",
                "The arguments field must be a JSON object, not a quoted JSON string.",
                *_simple_tool_call_date_instructions(record),
                "Do not copy tool schemas, descriptions, type/items/properties/required/default fields, or wrapper objects like {\"type\":...,\"value\":...} into arguments.",
                "Do not include id, call_id, type, response, output, exception, api_name, input, schema, analysis, or markdown fields.",
                "Previous API or function outputs in the conversation are history only; never copy response/output objects into the next call.",
                "Use only listed tool names.",
                "Return no prose, no markdown, and no extra text outside the JSON value.",
            ]
        )
    )


def _simple_tool_call_native_system_prompt(record: SimpleToolCallRecord) -> str:
    return normalize_rwkv_text(
        "\n".join(
            [
                "Use the provided tools when a function call is needed.",
                "Call only provided tool names and supply valid JSON arguments that match the tool schema.",
                "For multiple required tool calls, return every required call in execution order.",
                "If no tool is needed, answer directly.",
                *_simple_tool_call_date_instructions(record),
            ]
        )
    )


def _simple_tool_call_date_instructions(record: SimpleToolCallRecord) -> list[str]:
    instructions = [
        "For dates and times, use only dates/times stated or implied by the conversation or function outputs; do not use the real current date.",
    ]
    if _is_api_bank_record(record):
        instructions.append(
            "API-Bank date convention: if a month/day or relative date has no explicit year and the conversation does not state today's date, use year 2023."
        )
    return instructions


def _is_api_bank_record(record: SimpleToolCallRecord) -> bool:
    return str(record.metadata.get("source_format") or "").strip() in {
        "official_api_bank",
        "official_apibank",
    }


def decode_simple_tool_call_response(response: str) -> list[dict[str, Any]]:
    calls = parse_tool_calls_text(response, context_label="tool-call selection", recover_partial=True)
    return [{"name": call.name, "arguments": dict(call.arguments)} for call in calls]


def evaluate_simple_tool_calls(
    record: SimpleToolCallRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    expected = list(record.expected_tool_calls)
    actual = [
        {"name": str(item.get("name") or ""), "arguments": dict(item.get("arguments") or {})}
        for item in decoded_calls
    ]
    details: dict[str, Any] = {
        "expected_tool_calls": [_expectation_payload(item) for item in expected],
        "decoded_tool_calls": actual,
        "tool_count_ok": len(actual) == len(expected),
        "call_matches": [],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return SimpleToolCallEvaluation(
            reward=0.0,
            is_passed=False,
            fail_reason=parse_error,
            details=details,
        )

    max_len = max(len(expected), len(actual))
    passed_count = 0
    failure_bits: list[str] = []
    for index in range(max_len):
        if index >= len(expected):
            details["call_matches"].append({"index": index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(actual):
            details["call_matches"].append({"index": index, "ok": False, "reason": "missing_call"})
            failure_bits.append(f"call_{index}:missing_call")
            continue
        ok, reason = _call_matches_expectation(actual[index], expected[index])
        details["call_matches"].append({"index": index, "ok": ok, "reason": reason})
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")

    denominator = max(1, len(expected))
    reward = passed_count / denominator
    is_passed = len(actual) == len(expected) and passed_count == len(expected)
    if not expected:
        is_passed = len(actual) == 0
        reward = 1.0 if is_passed else 0.0
    return SimpleToolCallEvaluation(
        reward=float(reward),
        is_passed=bool(is_passed),
        fail_reason="; ".join(failure_bits),
        details=details,
    )


def _simple_tool_call_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
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
        answer=json.dumps(agent_info.get("decoded_tool_calls") or [], ensure_ascii=False, sort_keys=True),
        ref_answer=json.dumps(agent_info.get("expected_tool_calls") or [], ensure_ascii=False, sort_keys=True),
    )


def _simple_tool_call_payload(
    *,
    run: ResolvedFunctionCallingRun,
    record: SimpleToolCallRecord,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    prompt: str,
    completion: str,
    finish_reason: str,
    evaluation: SimpleToolCallEvaluation,
    trace_entry: dict[str, Any],
    sampling_payload: Any,
) -> dict[str, Any]:
    stage = StageRecord(prompt=prompt, completion=completion, stop_reason=finish_reason)
    payload = SampleRecord(
        benchmark_name=run.benchmark_name,
        dataset_split=run.dataset_split,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        stages=[stage],
        sampling_config=sampling_payload,
    ).as_payload()
    payload["agent_result"] = {
        "reward": float(evaluation.reward),
        "num_turns": 1,
        "cost": 0.0,
        "is_passed": bool(evaluation.is_passed),
        "error": evaluation.fail_reason or None,
    }
    payload["agent_info"] = {
        **dict(evaluation.details),
        "fail_reason": evaluation.fail_reason,
        "cot_mode": CoTMode.NO_COT.value,
    }
    payload["agent_trace"] = [trace_entry]
    payload["task_id"] = record.task_id
    payload["domain"] = "function_call"
    payload["instruction"] = record.instruction
    payload["metadata"] = dict(record.metadata)
    return payload


def _simple_tool_call_failure_payload(
    *,
    run: ResolvedFunctionCallingRun,
    record: SimpleToolCallRecord,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    reason: str,
    sampling_payload: Any,
) -> dict[str, Any]:
    expected_calls = [
        {"name": item.name, "arguments": item.arguments}
        for item in record.expected_tool_calls
    ]
    evaluation = SimpleToolCallEvaluation(
        reward=0.0,
        is_passed=False,
        fail_reason=reason,
        details={
            "expected_tool_calls": expected_calls,
            "decoded_tool_calls": [],
            "parse_error": reason,
            "sample_exception": reason,
        },
    )
    trace_entry = {
        "tool_call_io": "rwkv-json",
        "request": {
            "instruction": record.instruction,
            "tool_names": [str(tool.get("name") or "") for tool in record.tools],
        },
        "completion": "",
        "response_source": "sample_exception",
        "decoded_calls": [],
        "parse_error": reason,
    }
    return _simple_tool_call_payload(
        run=run,
        record=record,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        prompt=record.instruction,
        completion="",
        finish_reason="sample_exception",
        evaluation=evaluation,
        trace_entry=trace_entry,
        sampling_payload=sampling_payload,
    )


def _run_simple_tool_call(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    default_job_name: str,
    evaluator: Callable[..., SimpleToolCallEvaluation] = evaluate_simple_tool_calls,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_simple_tool_call_manifest_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        run.dataset_slug,
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]
    if not records:
        raise ValueError("simple tool-call manifest is empty")

    plan = _resolve_function_calling_plan(
        run.dataset_slug,
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    tool_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="function_call_default",
    )
    if tool_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    tool_sampling = clamp_function_calling_sampling(tool_sampling, max(1, int(args.decision_max_tokens or 768)))
    sampling_payload = normalize_sampling_config_by_stage([(1, tool_sampling)])
    history_max_chars = max(0, int(args.history_max_chars or DEFAULT_HISTORY_MAX_CHARS))
    batch_size = max(1, int(args.batch_size or 16))
    candidate_router_mode = _simple_candidate_router_mode(args)
    tool_call_io = _simple_tool_call_io(args)
    if tool_call_io == "native" and not callable(getattr(run.engine, "generate_tool_calls", None)):
        raise NotImplementedError("simple tool-call evaluation requires an inference backend with generate_tool_calls")
    candidate_router_config = (
        _simple_candidate_router_config_from_args(args) if candidate_router_mode == "parallel" else None
    )
    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        probe_count = 0
        for index, record in repeated:
            prompt_seed = sample_repeat_seed(index, 0, stage=1)
            if tool_call_io == "rwkv-json":
                _run_json_prompt_simple_tool_call(
                    run,
                    record,
                    tool_sampling=tool_sampling,
                    history_max_chars=history_max_chars,
                    prompt_seed=prompt_seed,
                    progress_desc="ToolCall-JSON-Probe",
                )
            else:
                run_native_tool_call_decision(
                    engine=run.engine,
                    messages=build_simple_tool_call_messages(record),
                    tools=record.tools,
                    sampling=tool_sampling,
                    progress_desc="ToolCall-Native-Probe",
                    prompt_seed=prompt_seed,
                    parallel_tool_calls=True,
                    context_label="tool-call selection",
                )
            probe_count += 1
        print(f"probe-only run completed: {probe_count} prompt(s)")
        return 0

    job_name = _resolve_job_name(default_job_name, run_context=run_context)
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
    flush_partial = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_simple_tool_call_completion_to_eval_payload,
        runner_name=default_job_name,
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=flush_partial,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                for key, record in pending:
                    try:
                        prompt_seed = sample_repeat_seed(key.sample_index, key.repeat_index, stage=1)
                        trace_entry: dict[str, Any]
                        active_candidate_router_config = candidate_router_config
                        if active_candidate_router_config is None and candidate_router_mode == "auto":
                            active_candidate_router_config = _auto_candidate_router_config(args, record)
                        if active_candidate_router_config is not None and record.tools:
                            (
                                prompt,
                                completion,
                                finish_reason,
                                decoded_calls,
                                parse_error,
                                trace_entry,
                            ) = _run_candidate_routed_simple_tool_call(
                                args,
                                run,
                                record,
                                tool_sampling=tool_sampling,
                                config=active_candidate_router_config,
                                prompt_seed=prompt_seed,
                                progress_desc=f"ToolCall-CandidateRouter sample {key.sample_index}",
                            )
                        elif record.tools:
                            if tool_call_io == "rwkv-json":
                                (
                                    prompt,
                                    completion,
                                    finish_reason,
                                    decoded_calls,
                                    parse_error,
                                    trace_entry,
                                ) = _run_json_prompt_simple_tool_call(
                                    run,
                                    record,
                                    tool_sampling=tool_sampling,
                                    history_max_chars=history_max_chars,
                                    prompt_seed=prompt_seed,
                                    progress_desc=f"ToolCall-JSON sample {key.sample_index}",
                                )
                            else:
                                decision = run_native_tool_call_decision(
                                    engine=run.engine,
                                    messages=build_simple_tool_call_messages(record),
                                    tools=record.tools,
                                    sampling=tool_sampling,
                                    progress_desc=f"ToolCall-Native sample {key.sample_index}",
                                    prompt_seed=prompt_seed,
                                    parallel_tool_calls=True,
                                    context_label="tool-call selection",
                                )
                                prompt = decision.prompt
                                completion = decision.completion
                                finish_reason = decision.finish_reason
                                decoded_calls = decision.decoded_calls
                                parse_error = decision.parse_error
                                trace_entry = decision.trace
                        else:
                            messages = build_simple_tool_call_messages(record)
                            prompt = json.dumps(
                                {"messages": messages, "tools": [], "tool_choice": "auto"},
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            completion = json.dumps(
                                {"role": "assistant", "content": "", "tool_calls": []},
                                ensure_ascii=False,
                                separators=(",", ":"),
                                sort_keys=True,
                            )
                            finish_reason = "stop"
                            parse_error = None
                            decoded_calls = []
                            trace_entry = {
                                "tool_call_io": "openai-tools",
                                "request": {"messages": messages, "tools": [], "tool_choice": "auto"},
                                "assistant_message": {"role": "assistant", "content": "", "tool_calls": []},
                                "response_source": "no_tools",
                                "decoded_calls": [],
                                "parse_error": "",
                            }
                        evaluation = evaluator(record, decoded_calls, parse_error=parse_error)
                        payload = _simple_tool_call_payload(
                            run=run,
                            record=record,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            prompt=prompt,
                            completion=completion,
                            finish_reason=finish_reason,
                            evaluation=evaluation,
                            trace_entry=trace_entry,
                            sampling_payload=sampling_payload,
                        )
                    except Exception as exc:  # noqa: BLE001 - sample-level infra failures become failed rows.
                        reason = f"sample_exception: {type(exc).__name__}: {exc}"
                        print(
                            f"{default_job_name} sample {key.sample_index} failed as scoreable zero: {reason}",
                            flush=True,
                        )
                        payload = _simple_tool_call_failure_payload(
                            run=run,
                            record=record,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            reason=reason,
                            sampling_payload=sampling_payload,
                        )
                    writer.enqueue(payload)
            except Exception:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_simple_tool_call_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=False,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.NO_COT.value),
                extra={"cot_mode": CoTMode.NO_COT.value, "history_max_chars": history_max_chars},
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"{default_job_name} done: {len(completions_payloads)} samples")
    return 0


def _simple_candidate_router_config_from_args(args: argparse.Namespace) -> ParallelCandidateRouterConfig | None:
    mode = _simple_candidate_router_mode(args)
    if mode in {"off", "auto"}:
        return None
    if mode != "parallel":
        raise ValueError(f"unsupported candidate_router_mode={mode!r}; expected off, auto, or parallel")
    return _candidate_router_config_from_args(args)


def _simple_candidate_router_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "candidate_router_mode", "off") or "off").strip().lower()


def _simple_tool_call_io(args: argparse.Namespace) -> str:
    value = str(getattr(args, "tool_call_io", DEFAULT_TOOL_CALL_IO) or DEFAULT_TOOL_CALL_IO).strip().lower()
    if value not in _TOOL_CALL_IO_CHOICES:
        raise ValueError(f"unsupported tool_call_io={value!r}; expected native or rwkv-json")
    return value


def _run_json_prompt_simple_tool_call(
    run: ResolvedFunctionCallingRun,
    record: SimpleToolCallRecord,
    *,
    tool_sampling: Any,
    history_max_chars: int,
    prompt_seed: int,
    progress_desc: str,
) -> tuple[str, str, str, list[dict[str, Any]], str | None, dict[str, Any]]:
    prompt = build_simple_tool_call_prompt(record, history_max_chars=history_max_chars, prefill_object=True)
    output = run.engine.generate(
        [prompt],
        sampling=tool_sampling,
        batch_size=1,
        progress_desc=progress_desc,
        prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
        prompt_seeds=[prompt_seed],
    )[0]
    completion = output.text
    decoded_calls: list[dict[str, Any]] = []
    parse_error: str | None = None
    try:
        decoded_calls = decode_simple_tool_call_response(completion)
    except Exception as exc:  # noqa: BLE001 - parse errors are sample-level failures
        parse_error = str(exc)
    trace_entry = {
        "tool_call_io": "rwkv-json",
        "request": {
            "prompt": prompt,
            "tool_names": [str(tool.get("name") or "") for tool in record.tools],
        },
        "completion": completion,
        "response_source": "content",
        "decoded_calls": decoded_calls,
        "parse_error": parse_error or "",
    }
    return prompt, completion, str(output.finish_reason or "stop"), decoded_calls, parse_error, trace_entry


def _candidate_router_config_from_args(args: argparse.Namespace) -> ParallelCandidateRouterConfig:
    router_config_type, _route = _parallel_candidate_router_api()
    defaults = router_config_type()
    schema_mode = str(
        getattr(args, "candidate_router_tool_schema_mode", defaults.tool_schema_mode) or defaults.tool_schema_mode
    )
    return router_config_type(
        chunk_tools=_positive_int(getattr(args, "candidate_router_chunk_tools", None), defaults.chunk_tools),
        batch_size=_positive_int(getattr(args, "candidate_router_batch_size", None), defaults.batch_size),
        context_chars=_positive_int(getattr(args, "candidate_router_context_chars", None), defaults.context_chars),
        prompt_max_chars=_positive_int(
            getattr(args, "candidate_router_prompt_max_chars", None),
            defaults.prompt_max_chars,
        ),
        candidate_max_tokens=_positive_int(
            getattr(args, "candidate_router_candidate_max_tokens", None),
            defaults.candidate_max_tokens,
        ),
        aggregate_max_tokens=_positive_int(
            getattr(args, "candidate_router_aggregate_max_tokens", None),
            defaults.aggregate_max_tokens,
        ),
        max_candidates=_positive_int(getattr(args, "candidate_router_max_candidates", None), defaults.max_candidates),
        tool_schema_mode=schema_mode,
        include_respond=False,
        fallback_to_highest_confidence=defaults.fallback_to_highest_confidence,
        evidence_chars=_positive_int(getattr(args, "candidate_router_evidence_chars", None), defaults.evidence_chars),
        policy_chars=_positive_int(getattr(args, "candidate_router_policy_chars", None), defaults.policy_chars),
        ground_identifier_arguments=not bool(getattr(args, "disable_candidate_router_grounding", False)),
    )


def _auto_candidate_router_config(
    args: argparse.Namespace,
    record: SimpleToolCallRecord,
) -> ParallelCandidateRouterConfig | None:
    if not record.tools:
        return None
    config = _candidate_router_config_from_args(args)
    tool_count = len(record.tools)
    instruction_chars = len(record.instruction)
    facts_chars = len(_candidate_router_facts_text(record) or "")
    tool_schema_chars = len(json.dumps(record.tools, ensure_ascii=False, sort_keys=True))
    if tool_count >= max(_AUTO_CANDIDATE_ROUTER_MIN_TOOLS, int(config.chunk_tools) * 2):
        return config
    if instruction_chars + facts_chars > int(config.context_chars):
        return config
    if tool_schema_chars > max(1, int(config.prompt_max_chars) // 2):
        return config
    return None


def _positive_int(raw: object, default: int) -> int:
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return max(1, value)


def _run_candidate_routed_simple_tool_call(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    record: SimpleToolCallRecord,
    *,
    tool_sampling: Any,
    config: ParallelCandidateRouterConfig,
    prompt_seed: int,
    progress_desc: str,
) -> tuple[str, str, str, list[dict[str, Any]], str | None, dict[str, Any]]:
    del args
    _router_config_type, route_parallel_candidate_tool_call = _parallel_candidate_router_api()
    messages = [{"role": "user", "content": normalize_rwkv_text(record.instruction)}]
    route = route_parallel_candidate_tool_call(
        tools=record.tools,
        messages=messages,
        domain_policy=_GENERIC_CANDIDATE_ROUTER_POLICY,
        domain=run.benchmark_name,
        facts_text=_candidate_router_facts_text(record),
        engine=run.engine,
        sampling=tool_sampling,
        config=config,
        progress_desc=progress_desc,
        prompt_seed=prompt_seed,
    )
    selected = route.selected
    decoded_calls: list[dict[str, Any]] = []
    parse_error: str | None = None
    if selected is None:
        parse_error = str(route.aggregate_error or "candidate router did not select a tool call")
        decision_text = ""
    else:
        decoded_calls = [{"name": selected.name, "arguments": dict(selected.arguments)}]
        decision_text = json.dumps(decoded_calls[0], ensure_ascii=False, sort_keys=True)
    completion = route.aggregate_completion or decision_text
    finish_reason = route.aggregate_finish_reason or ("candidate_router_empty" if selected is None else "stop")
    trace_entry = {
        "decision_completion": completion,
        "decision_text": decision_text,
        "decision_stop_reason": finish_reason,
        "decoded_calls": decoded_calls,
        "parse_error": parse_error or "",
        "candidate_router": route.trace_payload(include_prompts=True),
    }
    return route.aggregate_prompt, completion, finish_reason, decoded_calls, parse_error, trace_entry


def _parallel_candidate_router_api():
    from src.eval.experiments.parallel_candidate_router.router import (
        ParallelCandidateRouterConfig,
        route_parallel_candidate_tool_call,
    )

    return ParallelCandidateRouterConfig, route_parallel_candidate_tool_call


def _candidate_router_facts_text(record: SimpleToolCallRecord) -> str | None:
    for key in ("facts_text", "facts", "context", "source_context", "document", "documents", "policy"):
        raw = record.metadata.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, str):
            return truncate_text(normalize_rwkv_text(raw), 4000)
        return truncate_text(json.dumps(raw, ensure_ascii=False, sort_keys=True), 4000)
    return None


def _normalize_tool_expectation(raw: Any) -> ToolCallExpectation:
    if not isinstance(raw, Mapping):
        return ToolCallExpectation(name="unknown_tool", arguments={}, argument_options={})
    name = str(raw.get("name") or raw.get("tool_name") or raw.get("function_name") or "").strip()
    arguments = raw.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            arguments = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    raw_options = raw.get("argument_options")
    argument_options: dict[str, tuple[Any, ...]] = {}
    if isinstance(raw_options, Mapping):
        for key, value in raw_options.items():
            options = tuple(_coerce_list(value) or [value])
            argument_options[str(key)] = options
    for key, value in arguments.items():
        argument_options.setdefault(str(key), (value,))
    return ToolCallExpectation(
        name=name or "unknown_tool",
        arguments=dict(arguments),
        argument_options=argument_options,
    )


def _call_matches_expectation(actual: Mapping[str, Any], expected: ToolCallExpectation) -> tuple[bool, str]:
    actual_name = str(actual.get("name") or "").strip()
    if actual_name != expected.name:
        return False, f"name_mismatch(expected={expected.name}, actual={actual_name})"
    arguments = actual.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments_not_object"
    actual_arguments = dict(arguments)
    for key, options in expected.argument_options.items():
        if key not in actual_arguments:
            if any(_is_absent_option(option) for option in options):
                continue
            return False, f"missing_argument({key})"
        actual_value = actual_arguments[key]
        if not any(_value_matches(actual_value, option) for option in options):
            return False, f"argument_mismatch({key})"
    for key, value in actual_arguments.items():
        if key not in expected.argument_options and not _is_absent_option(value):
            return False, f"unexpected_argument({key})"
    return True, "ok"


def _value_matches(actual: Any, expected: Any) -> bool:
    if _is_absent_option(expected):
        return _is_absent_option(actual)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(actual, str) and not isinstance(expected, str):
        parsed = _try_parse_json_scalar(actual)
        if parsed is not actual:
            return _value_matches(parsed, expected)
    if isinstance(expected, str) and not isinstance(actual, str):
        parsed = _try_parse_json_scalar(expected)
        if parsed is not expected:
            return _value_matches(actual, parsed)
    if isinstance(actual, str) and isinstance(expected, str):
        return normalize_rwkv_text(actual).strip() == normalize_rwkv_text(expected).strip()
    return actual == expected


def _is_absent_option(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _try_parse_json_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{\"-0123456789tfn":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _expectation_payload(expectation: ToolCallExpectation) -> dict[str, Any]:
    return {
        "name": expectation.name,
        "arguments": dict(expectation.arguments),
        "argument_options": {key: list(value) for key, value in expectation.argument_options.items()},
    }


def _normalize_bfcl_ground_truth_calls(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            name, arguments = _parse_python_call(item)
            calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "argument_options": {key: [value] for key, value in arguments.items()},
                }
            )
            continue
        if isinstance(item, Mapping):
            if "name" in item:
                call = _normalize_tool_expectation(item)
                calls.append(_expectation_payload(call))
                continue
            if len(item) != 1:
                continue
            name, argument_options = next(iter(item.items()))
            if not isinstance(argument_options, Mapping):
                argument_options = {}
            canonical_arguments = {
                str(key): _canonical_option_value(_coerce_list(value) or [value])
                for key, value in argument_options.items()
            }
            calls.append(
                {
                    "name": str(name),
                    "arguments": canonical_arguments,
                    "argument_options": {
                        str(key): list(_coerce_list(value) or [value]) for key, value in argument_options.items()
                    },
                }
            )
    return calls


def _infer_bfcl_language(category: str, item: Mapping[str, Any]) -> str:
    raw = str(item.get("language") or item.get("programming_language") or "").strip().lower()
    if raw:
        return raw
    category_text = str(category or "").lower()
    if "javascript" in category_text or category_text.endswith("_js"):
        return "javascript"
    if "java" in category_text:
        return "java"
    return "python"


def _canonical_option_value(options: Sequence[Any]) -> Any:
    for option in options:
        if not _is_absent_option(option):
            return option
    return options[0] if options else None


def _parse_python_call(text: str) -> tuple[str, dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"BFCL ground-truth expression is not a function call: {text}")
    name = _render_ast_call_name(parsed.body.func)
    arguments: dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is None:
            continue
        arguments[keyword.arg] = _literal_from_ast(keyword.value)
    return name, arguments


def _render_ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _render_ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in {"true", "True"}:
            return True
        if node.id in {"false", "False"}:
            return False
        if node.id in {"null", "None"}:
            return None
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.BinOp):
        left = _literal_from_ast(node.left)
        right = _literal_from_ast(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
    return ast.literal_eval(node)


def _render_bfcl_question(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    turns = _coerce_list(raw)
    parts: list[str] = []
    for turn in turns:
        messages = _coerce_list(turn)
        for message in messages:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or "").strip()
                if content:
                    parts.append(f"{role.title()}: {content}")
            elif str(message or "").strip():
                parts.append(str(message).strip())
    return "\n".join(parts).strip()


def _normalize_tool_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "name": "unknown_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    parameters = source.get("parameters") or {"type": "object", "properties": {}, "required": []}
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = dict(parameters)
    if str(parameters.get("type") or "").lower() == "dict":
        parameters["type"] = "object"
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    normalized = {
        "name": str(source.get("name") or raw.get("name") or "unknown_tool"),
        "description": str(source.get("description") or raw.get("description") or ""),
        "parameters": parameters,
    }
    metadata = raw.get("metadata") or source.get("metadata")
    if isinstance(metadata, Mapping):
        normalized["metadata"] = dict(metadata)
    return normalized


def _render_tool_catalog(tools: Sequence[Mapping[str, Any]]) -> str:
    rendered_tools: list[dict[str, Any]] = []
    for tool in tools:
        parameters = tool.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}, "required": []}
        raw_properties = parameters.get("properties")
        rendered_arguments: Any = dict(raw_properties) if isinstance(raw_properties, Mapping) else dict(parameters)
        rendered_schema = json.dumps(rendered_arguments, ensure_ascii=False, sort_keys=True)
        if len(rendered_schema) > _MAX_TOOL_SCHEMA_CHARS:
            rendered_arguments = {
                "_truncated": True,
                "preview": truncate_text(rendered_schema, _MAX_TOOL_SCHEMA_CHARS),
            }
        rendered_tools.append(
            {
                "name": str(tool.get("name") or ""),
                "description": truncate_text(
                    normalize_rwkv_text(str(tool.get("description") or "")),
                    _MAX_TOOL_DESCRIPTION_CHARS,
                ),
                "arguments": rendered_arguments,
                **(
                    {"required": list(parameters.get("required") or [])}
                    if isinstance(parameters.get("required"), list) and parameters.get("required")
                    else {}
                ),
            }
        )
    return json.dumps(rendered_tools, ensure_ascii=False, indent=2, sort_keys=False)


def _read_json_or_jsonl_items(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError(f"unsupported JSON payload: {path}")


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _slug(value: str) -> str:
    rendered = []
    for char in str(value).lower():
        rendered.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(rendered).split("_") if part) or "item"


__all__ = [
    "SimpleToolCallEvaluation",
    "SimpleToolCallRecord",
    "ToolCallExpectation",
    "build_simple_tool_call_prompt",
    "decode_simple_tool_call_response",
    "evaluate_simple_tool_calls",
    "load_bfcl_ast_rows_from_sources",
    "load_simple_tool_call_manifest_records",
    "normalize_simple_tool_call_manifest_row",
    "_run_simple_tool_call",
]
