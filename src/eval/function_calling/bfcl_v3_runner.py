from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.bfcl_v3 import (
    BFCL_ADDITIONAL_FUNCTION_PROMPT,
    BFCL_COT_STOP_SUFFIX,
    BFCL_DECISION_STOP_SUFFIXES,
    BFCL_ROUTER_LABELS,
    BFCL_V3_MAX_COT_CHARS,
    BfclTaskRecord,
    apply_bfcl_tool_call,
    build_bfcl_ask_prompt,
    build_bfcl_cot_prompt,
    build_bfcl_handoff_prompt,
    build_bfcl_ref_answer,
    build_bfcl_router_prompt,
    build_bfcl_system_prompt,
    build_bfcl_tool_prompt,
    build_bfcl_tool_result_message,
    build_bfcl_tool_result_payload,
    build_bfcl_user_block,
    collect_bfcl_dataset_issues,
    decode_bfcl_exec_response,
    evaluate_bfcl_v3_episode,
    execute_bfcl_official_tool_call,
    extract_bfcl_cot_hidden_summary,
    has_bfcl_official_turns,
    load_bfcl_v3_manifest_records,
    normalize_bfcl_decision_output,
    parse_bfcl_assistant_output,
    parse_bfcl_router_output,
    render_bfcl_assistant_tool_message,
    render_bfcl_official_call,
    render_bfcl_turn_request,
    start_bfcl_runtime,
)
from src.eval.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.function_calling.tau_bench import TauToolCall
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage
from src.infer.constraints import LiteralChoiceConstraint, build_bfcl_tool_call_constraint
from src.infer.backend import RemoteInferenceBackend

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

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
    effective_constraint = None if isinstance(run.engine, RemoteInferenceBackend) else constraint
    effective_constraint_mode = "off" if effective_constraint is None else constraint_mode
    output = run.engine.generate(
        [prompt],
        sampling=sampling,
        batch_size=1,
        progress_desc=progress_desc,
        prompt_stop_suffixes=None if stop_suffixes is None else [list(stop_suffixes)],
        constraints=None if effective_constraint is None else [effective_constraint],
        constraint_mode=effective_constraint_mode,
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
        tool_text = normalize_bfcl_decision_output(tool_output.text)
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
        stop_suffixes=BFCL_DECISION_STOP_SUFFIXES,
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
                state.prompt_messages.append(
                    {
                        "role": "assistant",
                        "content": str(outcome.trace_entry.get("branch_text") or state.final_answer).strip(),
                    }
                )
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
                            state.prompt_messages.append(
                                {
                                    "role": "assistant",
                                    "content": str(outcome.trace_entry.get("branch_text") or state.final_answer).strip(),
                                }
                            )
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
