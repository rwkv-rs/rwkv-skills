from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.concurrent_runner import run_episodes
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import AttemptKey, build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.tasks.function_calling.bfcl_v3 import (
    BFCL_ADDITIONAL_FUNCTION_PROMPT,
    BFCL_DECISION_STOP_SUFFIXES,
    BfclEvaluation,
    BfclTaskRecord,
    apply_bfcl_tool_call,
    build_bfcl_ref_answer,
    build_bfcl_rwkv_prompt,
    build_bfcl_system_prompt,
    build_bfcl_tool_result_message,
    build_bfcl_tool_result_payload,
    build_bfcl_user_block,
    collect_bfcl_dataset_issues,
    decode_bfcl_exec_response,
    evaluate_bfcl_v3_episode,
    execute_bfcl_official_tool_call,
    has_bfcl_official_turns,
    load_bfcl_v3_manifest_records,
    normalize_bfcl_decision_output,
    render_bfcl_assistant_tool_message,
    render_bfcl_official_call,
    render_bfcl_turn_request,
    start_bfcl_runtime,
    _bfcl_tools_with_control_functions,
)
from src.eval.tasks.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.tasks.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.tasks.function_calling.rwkv_prompt import (
    RWKV_OFFICIAL_JSON_PROMPT_STYLE,
    normalize_function_prompt_style,
)
from src.eval.tasks.function_calling.tool_call_contract import parse_tool_calls_text
from src.eval.tasks.function_calling.parallel_candidate_router import (
    CandidateToolCall,
    ParallelCandidateRouterConfig,
    route_parallel_candidate_tool_call,
)
from src.eval.tasks.function_calling.tau_bench import TauToolCall
from src.eval.tasks.function_calling.tool_router import (
    ToolRoutingConfig,
    route_tools_for_prompt,
    tool_routing_config_from_args,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage
from src.infer.constraints import build_bfcl_tool_call_constraint
from src.infer.backend import RemoteInferenceBackend

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext


_LOG = logging.getLogger(__name__)

BFCL_V3_DEFAULT_PROMPT_MAX_CHARS = 28000
DEFAULT_BFCL_V3_MAX_STEPS = 20
DEFAULT_BFCL_V3_MAX_TOOL_ERRORS = 20
# The unified function-calling CLI inherits Tau defaults. Treat those inherited
# values as "not explicitly set" so BFCL V3 keeps its benchmark-specific budget.
_INHERITED_TAU_DEFAULT_MAX_STEPS = 200
_INHERITED_TAU_DEFAULT_MAX_TOOL_ERRORS = 10
_BFCL_CANDIDATE_ROUTER_POLICY = (
    "BFCL v3 tool-call policy: return exactly one JSON function call for the next official sandbox action. "
    "Use only listed tool names. Use ask_user only when required information is missing. "
    "Use final_answer only when no environment tool should be called for this turn. "
    "Do not invent tool names, arguments, tool results, state transitions, IDs, dates, or files."
)

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
    tool_calls: list[TauToolCall] = field(default_factory=list)
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


def _bfcl_decision_error_type(exc: BaseException) -> str:
    message = str(exc).lower()
    if "unknown bfcl tool name" in message:
        return "unknown_tool"
    if "invalid arguments for bfcl tool" in message:
        return "schema_mismatch"
    if "arguments must be a json object" in message:
        return "invalid_arguments"
    if "empty response" in message:
        return "empty_response"
    if "max length" in message or "max_length" in message:
        return "max_length"
    return "invalid_json_tool_call"


def _outcome_tool_calls(outcome: _BfclGenerationStepOutcome) -> list[TauToolCall]:
    if outcome.tool_calls:
        return list(outcome.tool_calls)
    if outcome.tool_call is not None:
        return [outcome.tool_call]
    return []


def _trace_tool_calls(tool_calls: Sequence[TauToolCall]) -> list[dict[str, object]]:
    return [{"name": tool_call.name, "arguments": dict(tool_call.arguments)} for tool_call in tool_calls]


def _candidate_router_config_from_args(args: argparse.Namespace) -> ParallelCandidateRouterConfig | None:
    mode = str(getattr(args, "candidate_router_mode", "off") or "off").strip().lower()
    if mode in {"off", "auto"}:
        return None
    if mode != "parallel":
        raise ValueError(f"unsupported candidate_router_mode={mode!r}; expected off, auto, or parallel")
    defaults = ParallelCandidateRouterConfig()
    tool_schema_mode = str(
        getattr(args, "candidate_router_tool_schema_mode", defaults.tool_schema_mode) or defaults.tool_schema_mode
    )
    if tool_schema_mode not in {"minimal", "compact", "full"}:
        tool_schema_mode = defaults.tool_schema_mode
    return ParallelCandidateRouterConfig(
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
        tool_schema_mode=tool_schema_mode,
        include_respond=False,
        fallback_to_highest_confidence=True,
        evidence_chars=_positive_int(getattr(args, "candidate_router_evidence_chars", None), defaults.evidence_chars),
        policy_chars=_positive_int(getattr(args, "candidate_router_policy_chars", None), defaults.policy_chars),
        ground_identifier_arguments=not bool(getattr(args, "disable_candidate_router_grounding", False)),
    )


def _positive_int(raw: object, default: int) -> int:
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)


def _resolve_bfcl_v3_max_steps(raw: object) -> int:
    if raw is None:
        return DEFAULT_BFCL_V3_MAX_STEPS
    value = int(raw)
    if value == _INHERITED_TAU_DEFAULT_MAX_STEPS:
        return DEFAULT_BFCL_V3_MAX_STEPS
    return max(1, value)


def _resolve_bfcl_v3_max_tool_errors(raw: object) -> int:
    if raw is None:
        return DEFAULT_BFCL_V3_MAX_TOOL_ERRORS
    value = int(raw)
    if value == _INHERITED_TAU_DEFAULT_MAX_TOOL_ERRORS:
        return DEFAULT_BFCL_V3_MAX_TOOL_ERRORS
    return max(1, value)


def _build_bfcl_prompt_with_budget(
    *,
    system_prompt: str,
    messages: Sequence[Mapping[str, object]],
    history_max_chars: int,
    prompt_max_chars: int | None,
) -> tuple[str, int, bool]:
    requested_history = max(0, int(history_max_chars))
    candidates = [requested_history]
    if prompt_max_chars is not None and int(prompt_max_chars) > 0:
        budget = int(prompt_max_chars)
        candidates.extend(
            [
                min(requested_history, max(0, budget - len(system_prompt) - 256)),
                min(requested_history, max(0, budget // 2)),
                0,
            ]
        )
    seen: set[int] = set()
    last_prompt = ""
    last_history = requested_history
    for candidate_history in candidates:
        candidate_history = max(0, int(candidate_history))
        if candidate_history in seen:
            continue
        seen.add(candidate_history)
        prompt = build_bfcl_rwkv_prompt(
            system_prompt,
            messages,
            history_max_chars=candidate_history,
        )
        last_prompt = prompt
        last_history = candidate_history
        if prompt_max_chars is None or int(prompt_max_chars) <= 0 or len(prompt) <= int(prompt_max_chars):
            return prompt, candidate_history, candidate_history != requested_history
    return last_prompt, last_history, last_history != requested_history


def _prompt_over_budget_error(prompt: str, *, prompt_max_chars: int | None, label: str) -> str | None:
    if prompt_max_chars is None or int(prompt_max_chars) <= 0:
        return None
    if len(prompt) <= int(prompt_max_chars):
        return None
    return f"{label} prompt_chars={len(prompt)} exceeds prompt_max_chars={int(prompt_max_chars)}"


def _candidate_decision_text(candidate: CandidateToolCall) -> str:
    return json.dumps(
        {"name": candidate.name, "arguments": dict(candidate.arguments)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _bfcl_candidate_facts_text(state: _ActiveBfclEpisode) -> str | None:
    current_state = getattr(state.runtime_state, "current_state", None)
    if not isinstance(current_state, Mapping) or not current_state:
        return None
    return json.dumps({"current_state": dict(current_state)}, ensure_ascii=False, sort_keys=True)


def _bfcl_prompt_context_trace(
    *,
    state: _ActiveBfclEpisode,
    system_prompt: str,
    prompt: str,
    history_max_chars: int,
    prompt_max_chars: int | None,
    requested_history_max_chars: int,
    active_tools: Sequence[Mapping[str, Any]],
    routed_tools: Sequence[Mapping[str, Any]],
    candidate_router_tools: Sequence[Mapping[str, Any]] | None = None,
    history_reduced: bool = False,
) -> dict[str, object]:
    return {
        "system_prompt": system_prompt,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "prompt_max_chars": int(prompt_max_chars) if prompt_max_chars is not None else None,
        "history_max_chars": int(history_max_chars),
        "requested_history_max_chars": int(requested_history_max_chars),
        "history_reduced_for_budget": bool(history_reduced),
        "prompt_messages": [dict(message) for message in state.prompt_messages],
        "official_prompt_messages": [dict(message) for message in _bfcl_official_prompt_messages(state.prompt_messages)],
        "active_tools": [dict(tool) for tool in active_tools],
        "routed_tools": [dict(tool) for tool in routed_tools],
        "candidate_router_tools": (
            [dict(tool) for tool in candidate_router_tools] if candidate_router_tools is not None else None
        ),
        "runtime_state_snapshot": dict(getattr(state.runtime_state, "current_state", {}) or {}),
        "turn_count": int(state.turn_count),
        "step_count": int(state.step_count),
        "tool_errors": int(state.tool_errors),
    }


def _bfcl_action_type_from_decision_text(text: str) -> str:
    try:
        calls = parse_tool_calls_text(text, context_label="BFCL tool call", recover_partial=True)
    except Exception:  # noqa: BLE001
        return "TOOL"
    name = calls[0].name if calls else ""
    if name == "ask_user":
        return "ASK"
    if name == "final_answer":
        return "HANDOFF"
    return "TOOL"


def _bfcl_official_prompt_messages(
    messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    official_messages: list[dict[str, object]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "")
        if role == "user":
            if content.startswith("User:Request:\n"):
                content = content[len("User:Request:\n") :]
            elif content.startswith("User: Request:\n"):
                content = content[len("User: Request:\n") :]
            elif content.startswith("Request:\n"):
                content = content[len("Request:\n") :]
        official_messages.append({"role": role, "content": content})
    return official_messages


def _route_bfcl_tools(
    *,
    state: _ActiveBfclEpisode,
    run: ResolvedFunctionCallingRun,
    tool_sampling: Any,
    tool_routing_config: ToolRoutingConfig,
    progress_desc: str,
    prompt_seed: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    route = route_tools_for_prompt(
        state.active_tools,
        _bfcl_official_prompt_messages(state.prompt_messages),
        config=tool_routing_config,
        engine=run.engine,
        sampling=tool_sampling,
        control_tool_names=("ask_user", "final_answer"),
        progress_desc=progress_desc,
        prompt_seed=prompt_seed,
    )
    routed_tools = [dict(tool) for tool in route.selected_tools if isinstance(tool, Mapping)]
    return route, routed_tools


def _run_bfcl_official_json_generation_step(
    *,
    state: _ActiveBfclEpisode,
    run: ResolvedFunctionCallingRun,
    tool_sampling: Any,
    progress_suffix: str,
    history_max_chars: int,
    prompt_max_chars: int | None,
    tool_routing_config: ToolRoutingConfig,
    candidate_router_config: ParallelCandidateRouterConfig | None = None,
) -> _BfclGenerationStepOutcome:
    route, routed_tools = _route_bfcl_tools(
        state=state,
        run=run,
        tool_sampling=tool_sampling,
        tool_routing_config=tool_routing_config,
        progress_desc=f"BFCLV3-ToolRouter {progress_suffix}",
        prompt_seed=_next_bfcl_stage_seed(state) + 10_000,
    )
    system_prompt = build_bfcl_system_prompt(routed_tools)
    official_messages = _bfcl_official_prompt_messages(state.prompt_messages)

    if candidate_router_config is not None:
        pre_router_prompt = build_bfcl_rwkv_prompt(
            system_prompt,
            official_messages,
            history_max_chars=max(0, int(history_max_chars)),
        )
        candidate_router_tools = _bfcl_tools_with_control_functions(routed_tools)
        candidate_route = route_parallel_candidate_tool_call(
            tools=candidate_router_tools,
            messages=official_messages,
            domain_policy=_BFCL_CANDIDATE_ROUTER_POLICY,
            domain="bfcl_v3",
            facts_text=_bfcl_candidate_facts_text(state),
            engine=run.engine,
            sampling=tool_sampling,
            config=candidate_router_config,
            progress_desc=f"BFCLV3-CandidateRouter {progress_suffix}",
            prompt_seed=_next_bfcl_stage_seed(state) + 20_000,
        )
        selected_candidate = candidate_route.selected
        decision_text = "" if selected_candidate is None else _candidate_decision_text(selected_candidate)
        state.stages.append(
            StageRecord(
                prompt=candidate_route.aggregate_prompt,
                completion=candidate_route.aggregate_completion or decision_text,
                stop_reason=candidate_route.aggregate_finish_reason,
            )
        )
        trace_entry: dict[str, object] = {
            "prompt_style": RWKV_OFFICIAL_JSON_PROMPT_STYLE,
            "decision_completion": candidate_route.aggregate_completion,
            "decision_text": decision_text,
            "decision_stop_reason": candidate_route.aggregate_finish_reason,
            "tool_route": route.trace_payload(include_prompt=True),
            "candidate_router": candidate_route.trace_payload(include_prompts=True),
            "full_context": _bfcl_prompt_context_trace(
                state=state,
                system_prompt=system_prompt,
                prompt=pre_router_prompt,
                history_max_chars=max(0, int(history_max_chars)),
                prompt_max_chars=prompt_max_chars,
                requested_history_max_chars=max(0, int(history_max_chars)),
                active_tools=state.active_tools,
                routed_tools=routed_tools,
                candidate_router_tools=candidate_router_tools,
                history_reduced=False,
            ),
        }
        if selected_candidate is None:
            return _failed_bfcl_step(
                state,
                trace_entry,
                termination_reason="candidate_router_empty",
                error=str(candidate_route.aggregate_error or "candidate router did not select a BFCL action"),
            )
    else:
        prompt, rendered_history_max_chars, history_reduced = _build_bfcl_prompt_with_budget(
            system_prompt=system_prompt,
            messages=official_messages,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
        )
        trace_entry = {
            "prompt_style": RWKV_OFFICIAL_JSON_PROMPT_STYLE,
            "tool_route": route.trace_payload(include_prompt=True),
            "full_context": _bfcl_prompt_context_trace(
                state=state,
                system_prompt=system_prompt,
                prompt=prompt,
                history_max_chars=rendered_history_max_chars,
                prompt_max_chars=prompt_max_chars,
                requested_history_max_chars=max(0, int(history_max_chars)),
                active_tools=state.active_tools,
                routed_tools=routed_tools,
                history_reduced=history_reduced,
            ),
        }
        budget_error = _prompt_over_budget_error(
            prompt,
            prompt_max_chars=prompt_max_chars,
            label="BFCL v3 decision",
        )
        if budget_error is not None:
            return _failed_bfcl_step(
                state,
                trace_entry,
                termination_reason="prompt_over_budget",
                error=budget_error,
            )
        output = _generate_bfcl_stage(
            state=state,
            run=run,
            prompt=prompt,
            sampling=tool_sampling,
            progress_desc=f"BFCLV3-Decision {progress_suffix}",
            stop_suffixes=BFCL_DECISION_STOP_SUFFIXES,
            constraint=build_bfcl_tool_call_constraint(
                _bfcl_tools_with_control_functions(routed_tools),
                prefilled_object=False,
            ),
            constraint_mode="strict",
        )
        decision_text = normalize_bfcl_decision_output(output.text)
        trace_entry.update(
            {
                "decision_completion": output.text,
                "decision_text": decision_text,
                "decision_stop_reason": output.finish_reason,
            }
        )
    if _looks_like_template_leak(decision_text):
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="template_leak",
            error="decision stage leaked internal template/control tokens",
        )
    if str(trace_entry.get("decision_stop_reason") or "") == "max_length":
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="decision_max_length",
            error="decision stage reached max_length before producing a bounded JSON function call",
        )

    try:
        decoded_calls, final_answer = decode_bfcl_exec_response(
            decision_text,
            tools=routed_tools,
        )
    except Exception as exc:
        trace_entry["parse_error_type"] = _bfcl_decision_error_type(exc)
        trace_entry["parse_error"] = str(exc)
        return _failed_bfcl_step(
            state,
            trace_entry,
            termination_reason="invalid_decision_output",
            error=str(exc),
        )

    action_type = _bfcl_action_type_from_decision_text(decision_text)
    trace_entry["action_type"] = action_type
    state.step_count += 1
    if decoded_calls:
        tool_call = decoded_calls[0]
        return _BfclGenerationStepOutcome(
            ok=True,
            trace_entry=trace_entry,
            action_type="TOOL",
            tool_call=tool_call,
            tool_calls=list(decoded_calls),
        )

    final_answer = final_answer.strip()
    trace_entry["final_answer"] = final_answer
    return _BfclGenerationStepOutcome(
        ok=True,
        trace_entry=trace_entry,
        action_type=action_type,
        final_answer=final_answer,
    )


def _run_bfcl_generation_step(
    *,
    state: _ActiveBfclEpisode,
    run: ResolvedFunctionCallingRun,
    tool_sampling: Any,
    progress_suffix: str,
    prompt_style: str = RWKV_OFFICIAL_JSON_PROMPT_STYLE,
    history_max_chars: int = 0,
    prompt_max_chars: int | None = None,
    tool_routing_config: ToolRoutingConfig | None = None,
    candidate_router_config: ParallelCandidateRouterConfig | None = None,
) -> _BfclGenerationStepOutcome:
    normalize_function_prompt_style(prompt_style)
    return _run_bfcl_official_json_generation_step(
        state=state,
        run=run,
        tool_sampling=tool_sampling,
        progress_suffix=progress_suffix,
        history_max_chars=history_max_chars,
        prompt_max_chars=prompt_max_chars,
        tool_routing_config=tool_routing_config or ToolRoutingConfig(),
        candidate_router_config=candidate_router_config,
    )


def _run_bfcl_v3_official_episode(
    *,
    state: _ActiveBfclEpisode,
    run: ResolvedFunctionCallingRun,
    tool_sampling: Any,
    max_steps: int,
    max_tool_errors: int,
    history_max_chars: int,
    prompt_max_chars: int | None = None,
    tool_routing_config: ToolRoutingConfig | None = None,
    candidate_router_config: ParallelCandidateRouterConfig | None = None,
    prompt_style: str = RWKV_OFFICIAL_JSON_PROMPT_STYLE,
) -> list[dict[str, object]]:
    prompt_style = normalize_function_prompt_style(prompt_style)
    tool_routing_config = tool_routing_config or ToolRoutingConfig()
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

        while step_in_turn < max_steps:
            progress_suffix = f"sample {state.sample_index} turn {turn_index + 1} step {step_in_turn + 1}"
            try:
                outcome = _run_bfcl_generation_step(
                    state=state,
                    run=run,
                    tool_sampling=tool_sampling,
                    progress_suffix=progress_suffix,
                    prompt_style=prompt_style,
                    history_max_chars=history_max_chars,
                    prompt_max_chars=prompt_max_chars,
                    tool_routing_config=tool_routing_config,
                    candidate_router_config=candidate_router_config,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                _LOG.warning(
                    "bfcl_v3 sample %s turn %s step %s parse failed: %s",
                    state.sample_index,
                    turn_index + 1,
                    step_in_turn + 1,
                    exc,
                )
                state.termination_reason = "parse_error"
                state.error = f"decode_failed:{exc}"
                trace.append(
                    {
                        "turn_index": turn_index,
                        "step_index": step_in_turn,
                        "error": state.error,
                        "termination_reason": "parse_error",
                    }
                )
                return trace
            trace_entry = {
                "turn_index": turn_index,
                "step_index": step_in_turn,
                **outcome.trace_entry,
            }
            if not outcome.ok:
                state.final_answer = outcome.final_answer.strip()
                trace.append(trace_entry)
                break

            outcome_tool_calls = _outcome_tool_calls(outcome)
            if outcome_tool_calls:
                current_turn_outputs.append([render_bfcl_official_call(tool_call) for tool_call in outcome_tool_calls])
                trace_entry["tool_calls"] = _trace_tool_calls(outcome_tool_calls)
                tool_results: list[dict[str, object]] = []

                for tool_call in outcome_tool_calls:
                    state_before_tool = dict(state.runtime_state.current_state)
                    state.tool_calls.append(tool_call)
                    state.prompt_messages.append(
                        {"role": "assistant", "content": render_bfcl_assistant_tool_message(tool_call)}
                    )
                    try:
                        execution = execute_bfcl_official_tool_call(state.record, state.runtime_state, tool_call)
                    except Exception as exc:
                        state.tool_errors += 1
                        if isinstance(exc, TimeoutError):
                            state.termination_reason = "tool_timeout"
                            state.error = str(exc)
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
                                    previous_state_snapshot=state_before_tool,
                                ),
                            }
                        )
                        tool_results.append(
                            {
                                "name": tool_call.name,
                                "arguments": dict(tool_call.arguments),
                                "success": False,
                                "matched_expectation": False,
                                "result": None,
                                "error": str(exc),
                                "state_snapshot": dict(state.runtime_state.current_state),
                            }
                        )
                        if state.tool_errors >= max_tool_errors:
                            state.termination_reason = "too_many_errors"
                            state.error = str(exc)
                            break
                        if state.termination_reason == "tool_timeout":
                            break
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
                                previous_state_snapshot=state_before_tool,
                            ),
                        }
                    )
                    if not execution.matched_expectation:
                        state.tool_errors += 1
                    tool_results.append(
                        {
                            "name": tool_call.name,
                            "arguments": dict(tool_call.arguments),
                            "success": execution.success,
                            "matched_expectation": execution.matched_expectation,
                            "result": execution.result,
                            "error": execution.error,
                            "state_snapshot": dict(execution.state_snapshot),
                        }
                    )
                    if state.tool_errors >= max_tool_errors:
                        state.termination_reason = "too_many_errors"
                        state.error = "too many BFCL tool execution errors"
                        break

                trace_entry["tool_results"] = tool_results
                if tool_results:
                    first_result = tool_results[0]
                    trace_entry["tool_success"] = first_result.get("success")
                    trace_entry["tool_result"] = first_result.get("result")
                    trace_entry["tool_error"] = first_result.get("error")
                    trace_entry["state_snapshot"] = first_result.get("state_snapshot")
                trace.append(trace_entry)
                step_in_turn += 1
                if state.termination_reason is not None:
                    break
                continue

            state.final_answer = outcome.final_answer.strip()
            if state.final_answer:
                state.prompt_messages.append(
                    {
                        "role": "assistant",
                        "content": str(
                            outcome.trace_entry.get("branch_text")
                            or outcome.trace_entry.get("decision_text")
                            or state.final_answer
                        ).strip(),
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


def _checker_failure_evaluation(
    *,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> BfclEvaluation:
    """Build a standard failed evaluation after checker-local failures."""

    return BfclEvaluation(
        reward=0.0,
        is_passed=False,
        fail_reason=reason,
        details=dict(details or {}),
    )


def _build_bfcl_v3_attempt_payload(
    *,
    state: _ActiveBfclEpisode,
    record: BfclTaskRecord,
    run: ResolvedFunctionCallingRun,
    sampling_payload: Mapping[str, object],
    evaluation: BfclEvaluation,
    trace: Sequence[Mapping[str, object]],
    history_max_chars: int,
    prompt_max_chars: int,
    candidate_router_config: ParallelCandidateRouterConfig | None,
) -> dict[str, object]:
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
        "cot_mode": CoTMode.NO_COT.value,
        "history_max_chars": history_max_chars,
        "prompt_max_chars": prompt_max_chars,
        "candidate_router_mode": "parallel" if candidate_router_config is not None else "off",
        "final_prompt_messages": [dict(message) for message in state.prompt_messages],
        "final_runtime_state_snapshot": dict(getattr(state.runtime_state, "current_state", {}) or {}),
        "active_tools": [dict(tool) for tool in state.active_tools],
    }
    payload["agent_trace"] = [dict(item) for item in trace]
    payload["task_id"] = record.task_id
    payload["domain"] = "function_call"
    payload["instruction"] = record.instruction
    return payload


def _run_one_bfcl_v3_attempt(
    *,
    key: AttemptKey,
    record: BfclTaskRecord,
    run: ResolvedFunctionCallingRun,
    tool_sampling: Any,
    sampling_payload: Mapping[str, object],
    max_steps: int,
    max_tool_errors: int,
    history_max_chars: int,
    prompt_max_chars: int,
    prompt_style: str,
    tool_routing_config: ToolRoutingConfig,
    candidate_router_config: ParallelCandidateRouterConfig | None,
) -> dict[str, object]:
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
            tool_sampling=tool_sampling,
            max_steps=max_steps,
            max_tool_errors=max_tool_errors,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            prompt_style=prompt_style,
            tool_routing_config=tool_routing_config,
            candidate_router_config=candidate_router_config,
        )
    else:
        for _ in range(max_steps):
            progress_suffix = f"sample {state.sample_index} step {state.turn_count + 1}"
            try:
                outcome = _run_bfcl_generation_step(
                    state=state,
                    run=run,
                    tool_sampling=tool_sampling,
                    progress_suffix=progress_suffix,
                    prompt_style=prompt_style,
                    history_max_chars=history_max_chars,
                    prompt_max_chars=prompt_max_chars,
                    tool_routing_config=tool_routing_config,
                    candidate_router_config=candidate_router_config,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                _LOG.warning(
                    "bfcl_v3 sample %s step %s parse failed: %s",
                    state.sample_index,
                    state.turn_count + 1,
                    exc,
                )
                state.termination_reason = "parse_error"
                state.error = f"decode_failed:{exc}"
                trace.append(
                    {
                        "round_num": state.turn_count + 1,
                        "error": state.error,
                        "termination_reason": "parse_error",
                    }
                )
                break
            state.turn_count += 1
            trace_entry = {
                "round_num": state.turn_count,
                **outcome.trace_entry,
            }
            if not outcome.ok:
                state.final_answer = outcome.final_answer.strip()
                trace.append(trace_entry)
                break

            outcome_tool_calls = _outcome_tool_calls(outcome)
            if outcome_tool_calls:
                trace_entry["tool_calls"] = _trace_tool_calls(outcome_tool_calls)
                tool_results: list[dict[str, object]] = []

                for tool_call in outcome_tool_calls:
                    state_before_tool = dict(state.runtime_state.current_state)
                    state.tool_calls.append(tool_call)
                    state.prompt_messages.append(
                        {
                            "role": "assistant",
                            "content": render_bfcl_assistant_tool_message(tool_call),
                        }
                    )
                    execution = apply_bfcl_tool_call(record, state.runtime_state, tool_call)
                    tool_result_payload = build_bfcl_tool_result_payload(
                        tool_call,
                        ok=execution.success,
                        output=execution.result,
                        error=execution.error,
                    )
                    state.prompt_messages.append(
                        {
                            "role": "user",
                            "content": build_bfcl_tool_result_message(
                                tool_result_payload,
                                current_state_snapshot=state.runtime_state.current_state,
                                previous_state_snapshot=state_before_tool,
                            ),
                        }
                    )
                    if not execution.matched_expectation:
                        state.tool_errors += 1
                    tool_results.append(
                        {
                            "name": tool_call.name,
                            "arguments": dict(tool_call.arguments),
                            "success": execution.success,
                            "matched_expectation": execution.matched_expectation,
                            "result": execution.result,
                            "error": execution.error,
                            "state_snapshot": dict(execution.state_snapshot),
                        }
                    )
                    if state.tool_errors >= max_tool_errors:
                        state.termination_reason = "too_many_errors"
                        state.error = "too many BFCL tool execution errors"
                        break

                trace_entry["tool_results"] = tool_results
                if tool_results:
                    first_result = tool_results[0]
                    trace_entry["matched_expectation"] = first_result.get("matched_expectation")
                    trace_entry["tool_success"] = first_result.get("success")
                    trace_entry["tool_result"] = first_result.get("result")
                    trace_entry["tool_error"] = first_result.get("error")
                    trace_entry["state_snapshot"] = first_result.get("state_snapshot")
                trace.append(trace_entry)
                if state.termination_reason is not None:
                    break
                continue

            state.final_answer = outcome.final_answer.strip()
            state.prompt_messages.append(
                {
                    "role": "assistant",
                    "content": str(
                        outcome.trace_entry.get("branch_text")
                        or outcome.trace_entry.get("decision_text")
                        or state.final_answer
                    ).strip(),
                }
            )
            state.termination_reason = "agent_stop"
            trace.append(trace_entry)
            break

        if state.termination_reason is None:
            state.termination_reason = "max_steps"

    try:
        evaluation = evaluate_bfcl_v3_episode(
            record,
            state.runtime_state,
            state.final_answer,
            termination_reason=state.termination_reason,
            error=state.error,
        )
    except ValueError as exc:
        _LOG.warning("bfcl_v3 sample %s checker failed: %s", state.sample_index, exc)
        evaluation = _checker_failure_evaluation(
            reason=f"checker_error:{exc}",
            details={"checker_exception": str(exc)},
        )
        if state.termination_reason is None or state.termination_reason == "agent_stop":
            state.termination_reason = "checker_error"
        state.error = state.error or f"checker_error:{exc}"
    return _build_bfcl_v3_attempt_payload(
        state=state,
        record=record,
        run=run,
        sampling_payload=sampling_payload,
        evaluation=evaluation,
        trace=trace,
        history_max_chars=history_max_chars,
        prompt_max_chars=prompt_max_chars,
        candidate_router_config=candidate_router_config,
    )


def _run_one_bfcl_v3_attempt_scoreable(
    *,
    key: AttemptKey,
    record: BfclTaskRecord,
    run: ResolvedFunctionCallingRun,
    tool_sampling: Any,
    sampling_payload: Mapping[str, object],
    max_steps: int,
    max_tool_errors: int,
    history_max_chars: int,
    prompt_max_chars: int,
    prompt_style: str,
    tool_routing_config: ToolRoutingConfig,
    candidate_router_config: ParallelCandidateRouterConfig | None,
) -> dict[str, object]:
    try:
        return _run_one_bfcl_v3_attempt(
            key=key,
            record=record,
            run=run,
            tool_sampling=tool_sampling,
            sampling_payload=sampling_payload,
            max_steps=max_steps,
            max_tool_errors=max_tool_errors,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            prompt_style=prompt_style,
            tool_routing_config=tool_routing_config,
            candidate_router_config=candidate_router_config,
        )
    except Exception as exc:
        reason = f"inference_error:{type(exc).__name__}:{exc}"
        _LOG.exception("bfcl_v3 sample %s failed as scoreable zero: %s", key.sample_index, exc)
        state = _start_bfcl_episode(
            sample_index=key.sample_index,
            repeat_index=key.repeat_index,
            pass_index=key.pass_index,
            record=record,
        )
        state.termination_reason = "inference_error"
        state.error = reason
        evaluation = _checker_failure_evaluation(
            reason=reason,
            details={
                "runner_exception_type": type(exc).__name__,
                "runner_exception": str(exc),
            },
        )
        return _build_bfcl_v3_attempt_payload(
            state=state,
            record=record,
            run=run,
            sampling_payload=sampling_payload,
            evaluation=evaluation,
            trace=[
                {
                    "round_num": 0,
                    "termination_reason": "inference_error",
                    "error": reason,
                }
            ],
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            candidate_router_config=candidate_router_config,
        )


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
    tool_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    if tool_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    prompt_style = normalize_function_prompt_style(getattr(args, "prompt_style", None))
    tool_sampling = clamp_function_calling_sampling(tool_sampling, max(1, int(args.decision_max_tokens or 1024)))

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
    max_steps = _resolve_bfcl_v3_max_steps(getattr(args, "max_steps", None))
    max_tool_errors = _resolve_bfcl_v3_max_tool_errors(getattr(args, "max_tool_errors", None))
    history_max_chars = max(0, int(args.history_max_chars))
    prompt_max_chars = max(0, int(getattr(args, "prompt_max_chars", None) or BFCL_V3_DEFAULT_PROMPT_MAX_CHARS))
    tool_routing_config = tool_routing_config_from_args(args)
    candidate_router_config = _candidate_router_config_from_args(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, tool_sampling)]),
        tool_routing_config=tool_routing_config,
        candidate_router_config=candidate_router_config,
        prompt_max_chars=prompt_max_chars,
    )

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
        for state in probe_states:
            if has_bfcl_official_turns(state.record):
                additions = state.record.turns[0].tool_additions if state.record.turns else ()
                state.active_tools = _merge_bfcl_tools(state.active_tools, additions)
                state.system_prompt = build_bfcl_system_prompt(state.active_tools)
                turn_request = (
                    render_bfcl_turn_request(state.record.turns[0].messages)
                    if state.record.turns
                    else state.record.instruction.strip()
                )
                if not turn_request and additions:
                    turn_request = BFCL_ADDITIONAL_FUNCTION_PROMPT
                state.prompt_messages.append(
                    {
                        "role": "user",
                        "content": build_bfcl_user_block(turn_request),
                    }
                )
        if candidate_router_config is not None:
            for state in probe_states:
                _run_bfcl_generation_step(
                    state=state,
                    run=run,
                    tool_sampling=tool_sampling,
                    progress_suffix=f"probe sample {state.sample_index}",
                    prompt_style=prompt_style,
                    history_max_chars=history_max_chars,
                    prompt_max_chars=prompt_max_chars,
                    tool_routing_config=tool_routing_config,
                    candidate_router_config=candidate_router_config,
                )
        else:
            probe_routes = [
                _route_bfcl_tools(
                    state=state,
                    run=run,
                    tool_sampling=tool_sampling,
                    tool_routing_config=tool_routing_config,
                    progress_desc="BFCLV3-ToolRouter-Probe",
                    prompt_seed=sample_repeat_seed(state.sample_index, state.repeat_index, stage=10_001),
                )
                for state in probe_states
            ]
            prompt_rows = [
                _build_bfcl_prompt_with_budget(
                    system_prompt=build_bfcl_system_prompt(routed_tools),
                    messages=_bfcl_official_prompt_messages(state.prompt_messages),
                    history_max_chars=history_max_chars,
                    prompt_max_chars=prompt_max_chars,
                )
                for state, (_route, routed_tools) in zip(probe_states, probe_routes, strict=True)
            ]
            decision_prompts = [row[0] for row in prompt_rows]
            over_budget = [
                index
                for index, prompt in enumerate(decision_prompts)
                if _prompt_over_budget_error(prompt, prompt_max_chars=prompt_max_chars, label="BFCL v3 probe") is not None
            ]
            if over_budget:
                first_index = over_budget[0]
                raise ValueError(
                    f"BFCL v3 probe prompt over budget at index={first_index}: "
                    f"prompt_chars={len(decision_prompts[first_index])} prompt_max_chars={prompt_max_chars}"
                )
            constraints = (
                None
                if isinstance(run.engine, RemoteInferenceBackend)
                else [
                    build_bfcl_tool_call_constraint(
                        _bfcl_tools_with_control_functions(routed_tools),
                        prefilled_object=False,
                    )
                    for _route, routed_tools in probe_routes
                ]
            )
            run.engine.generate(
                decision_prompts,
                sampling=tool_sampling,
                batch_size=len(decision_prompts),
                progress_desc="BFCLV3-Probe-Decision",
                prompt_stop_suffixes=[list(BFCL_DECISION_STOP_SUFFIXES) for _ in decision_prompts],
                constraints=constraints,
                constraint_mode="off" if constraints is None else "strict",
                prompt_seeds=[
                    sample_repeat_seed(state.sample_index, state.repeat_index, stage=1)
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
                sample_workers = max(1, int(getattr(args, "sample_workers", 1) or 1))
                progress_last = [0.0]
                progress_bucket = [-1]

                def _progress(done: int, total: int) -> None:
                    now = monotonic()
                    bucket = int((done * 20) / max(1, total))
                    if done == total or bucket > progress_bucket[0] or now - progress_last[0] >= 2.0:
                        progress_last[0] = now
                        progress_bucket[0] = bucket
                        print(f"[bfcl_v3] {done}/{total} episodes done", file=sys.stderr, flush=True)

                run_episodes(
                    pending,
                    lambda item: _run_one_bfcl_v3_attempt_scoreable(
                        key=item[0],
                        record=item[1],
                        run=run,
                        tool_sampling=tool_sampling,
                        sampling_payload=sampling_payload,
                        max_steps=max_steps,
                        max_tool_errors=max_tool_errors,
                        history_max_chars=history_max_chars,
                        prompt_max_chars=prompt_max_chars,
                        prompt_style=prompt_style,
                        tool_routing_config=tool_routing_config,
                        candidate_router_config=candidate_router_config,
                    ),
                    max_workers=sample_workers,
                    on_result=writer.enqueue,
                    on_progress=_progress,
                    label="bfcl_v3 episode",
                    collect_results=False,
                )
            except Exception:  # noqa: BLE001
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
                is_cot=False,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.NO_COT.value),
                extra={
                    "cot_mode": CoTMode.NO_COT.value,
                    "history_max_chars": history_max_chars,
                    "prompt_max_chars": prompt_max_chars,
                    "candidate_router_mode": "parallel" if candidate_router_config is not None else "off",
                },
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"bfcl_v3 function-calling done: {len(completions_payloads)} samples")
    return 0
