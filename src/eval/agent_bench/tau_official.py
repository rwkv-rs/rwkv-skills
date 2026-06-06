from __future__ import annotations

"""Thin bridge to the official tau2/tau3 runtime package."""

import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from rwkv_agent_eval_plugin import agent_plugin_config_from_sources, route_agent_prompt_inputs

from src.eval.agent_bench.deps import import_module_with_auto_install
from src.eval.agent_bench.tasks import ensure_tau_v2_vendor_path
from src.eval.evaluators.common import StageRecord
from src.eval.function_calling.context_budget import normalize_rwkv_text, trim_message_history, truncate_text
from src.eval.function_calling.long_context_router import (
    LongContextRoutingConfig,
    compact_messages_for_prompt,
    compact_text_for_prompt,
    infer_long_context_query,
)
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_rwkv_json_call_prompt,
    extract_json_call_value_text,
)
from src.eval.function_calling.tool_router import ToolRoutingConfig, tool_name

_TAU_REWARD_TYPE_PREFIX = "RewardType."
RESPOND_TOOL_NAME = "respond"
DEFAULT_TAU_PROMPT_MAX_CHARS = 24576


@dataclass(slots=True)
class TauDomainInfo:
    policy: str
    tools: list[dict[str, Any]]


@dataclass(slots=True)
class TauOfficialEvaluation:
    reward: float
    is_passed: bool
    details: dict[str, Any]


class TauOfficialRuntime:
    def __init__(self, *, domain: str) -> None:
        ensure_tau_v2_vendor_path()
        registry_module = import_module_with_auto_install("tau2.registry", context="tau2 registry import")
        self.registry = getattr(registry_module, "registry")
        self.domain = str(domain)
        self._environment_constructor = self.registry.get_env_constructor(self.domain)
        message_module = import_module_with_auto_install("tau2.data_model.message", context="tau2 message import")
        self.AssistantMessage = getattr(message_module, "AssistantMessage")
        self.ToolCall = getattr(message_module, "ToolCall")
        self.ToolMessage = getattr(message_module, "ToolMessage")
        self.UserMessage = getattr(message_module, "UserMessage")
        simulation_module = import_module_with_auto_install(
            "tau2.data_model.simulation",
            context="tau2 simulation model import",
        )
        self.SimulationRun = getattr(simulation_module, "SimulationRun")
        self.TerminationReason = getattr(simulation_module, "TerminationReason")

    def load_task(self, payload: Mapping[str, Any]) -> Any:
        tasks_module = import_module_with_auto_install("tau2.data_model.tasks", context="tau2 Task model import")
        Task = getattr(tasks_module, "Task")
        return Task.model_validate(normalize_tau_official_task_payload(payload))

    def create_environment(
        self,
        *,
        solo_mode: bool = False,
        env_kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        kwargs = dict(env_kwargs or {})
        try:
            return self._environment_constructor(solo_mode=solo_mode, **kwargs)
        except TypeError:
            if solo_mode or kwargs:
                raise
            return self._environment_constructor()

    def initialize_environment(self, environment: Any, task: Any) -> list[Any]:
        initial_state = getattr(task, "initial_state", None)
        initialization_data = getattr(initial_state, "initialization_data", None) if initial_state is not None else None
        initialization_actions = (
            getattr(initial_state, "initialization_actions", None) if initial_state is not None else None
        )
        message_history = (
            deepcopy(getattr(initial_state, "message_history", None))
            if initial_state is not None and getattr(initial_state, "message_history", None) is not None
            else []
        )
        for message in message_history:
            if hasattr(message, "turn_idx"):
                message.turn_idx = None
        environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        environment.sync_tools()
        return list(message_history)

    def make_tool_call(self, name: str, arguments: Mapping[str, Any], *, call_id: str | None = None) -> Any:
        return self.ToolCall(
            id=call_id or f"call_{uuid.uuid4().hex[:12]}",
            name=str(name),
            arguments=dict(arguments),
            requestor="assistant",
        )

    def make_assistant_tool_message(self, tool_calls: Sequence[Any]) -> Any:
        return self.AssistantMessage(role="assistant", content=None, tool_calls=list(tool_calls))

    def make_assistant_text_message(self, content: str) -> Any:
        return self.AssistantMessage(role="assistant", content=str(content))

    def make_user_message(self, content: str) -> Any:
        return self.UserMessage(role="user", content=str(content))

    def make_simulation(self, *, task: Any, messages: Sequence[Any], termination_reason: str) -> Any:
        reason = getattr(self.TerminationReason, termination_reason)
        now = _tau_now()
        return self.SimulationRun(
            id=str(uuid.uuid4()),
            task_id=str(getattr(task, "id", "")),
            start_time=now,
            end_time=now,
            duration=0.0,
            termination_reason=reason,
            reward_info=None,
            user_cost=None,
            agent_cost=None,
            messages=list(messages),
            seed=None,
        )

    def evaluate(
        self,
        *,
        simulation: Any,
        task: Any,
        judge_model: Any | None = None,
        solo_mode: bool = False,
        env_kwargs: Mapping[str, Any] | None = None,
    ) -> TauOfficialEvaluation:
        if judge_model is not None:
            configure_tau_nl_assertions_judge(judge_model)
        evaluator_module = import_module_with_auto_install("tau2.evaluator.evaluator", context="tau2 evaluator import")
        EvaluationType = getattr(evaluator_module, "EvaluationType")
        evaluate_simulation = getattr(evaluator_module, "evaluate_simulation")
        evaluation_type = EvaluationType.ALL_WITH_NL_ASSERTIONS if _task_uses_nl_assertions(task) else EvaluationType.ALL
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=bool(solo_mode),
            domain=self.domain,
            env_kwargs=dict(env_kwargs or {}),
        )
        simulation.reward_info = reward_info
        details = _model_dump_safe(reward_info)
        details["termination_reason"] = str(getattr(simulation, "termination_reason", ""))
        return TauOfficialEvaluation(
            reward=float(getattr(reward_info, "reward", 0.0)),
            is_passed=float(getattr(reward_info, "reward", 0.0)) >= (1.0 - 1e-6),
            details=details,
        )

    def build_user(self, *, task: Any, environment: Any, user_model: Any | None) -> Any:
        if user_model is None:
            return StaticStopTauUser()
        user_module = import_module_with_auto_install("tau2.user.user_simulator", context="tau2 user simulator import")
        UserSimulator = getattr(user_module, "UserSimulator")
        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None
        return UserSimulator(
            tools=user_tools,
            instructions=str(getattr(task, "user_scenario", "")),
            llm=_tau_litellm_model_name(user_model),
            llm_args={
                "temperature": 0.0,
                "stream": False,
                "api_key": getattr(user_model, "api_key", None),
                "api_base": getattr(user_model, "base_url", None),
                **_tau_llm_timeout_args(),
            },
        )

    def build_orchestrator(
        self,
        *,
        agent: Any,
        user: Any,
        environment: Any,
        task: Any,
        max_steps: int,
        max_errors: int,
        seed: int,
        validate_communication: bool = True,
    ) -> Any:
        orchestrator_module = import_module_with_auto_install(
            "tau2.orchestrator.orchestrator",
            context="tau2 Orchestrator import",
        )
        Orchestrator = getattr(orchestrator_module, "Orchestrator")
        return Orchestrator(
            domain=self.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max(1, int(max_steps)),
            max_errors=max(1, int(max_errors)),
            seed=int(seed),
            solo_mode=False,
            validate_communication=bool(validate_communication),
        )


class RWKVTauOfficialAgent:
    """Official tau Agent interface backed by a repository inference engine."""

    def __init__(
        self,
        *,
        engine: Any,
        sampling: Any,
        tools: Sequence[Any],
        domain_policy: str,
        history_max_chars: int,
        prompt_max_chars: int = DEFAULT_TAU_PROMPT_MAX_CHARS,
        tool_routing_config: ToolRoutingConfig | None = None,
        long_context_routing_config: LongContextRoutingConfig | None = None,
        max_repeated_tool_calls: int = 2,
        generate_lock: Any | None = None,
    ) -> None:
        ensure_tau_v2_vendor_path()
        message_module = import_module_with_auto_install("tau2.data_model.message", context="tau2 message import")
        self._AssistantMessage = getattr(message_module, "AssistantMessage")
        self._ToolCall = getattr(message_module, "ToolCall")
        self._MultiToolMessage = getattr(message_module, "MultiToolMessage")
        self._ToolMessage = getattr(message_module, "ToolMessage")
        self._UserMessage = getattr(message_module, "UserMessage")
        self._engine = engine
        self._sampling = sampling
        self._tools = list(tools)
        self._tool_names = {tool_name(tool) for tool in self._tools if tool_name(tool)}
        self._current_tool_names = set(self._tool_names)
        self._domain_policy = str(domain_policy)
        self._history_max_chars = max(0, int(history_max_chars))
        self._prompt_max_chars = max(1024, int(prompt_max_chars))
        self._tool_routing_config = tool_routing_config or ToolRoutingConfig()
        self._long_context_routing_config = long_context_routing_config or LongContextRoutingConfig()
        self._max_repeated_tool_calls = max(1, int(max_repeated_tool_calls))
        self._generate_lock = generate_lock
        self._seed: int | None = None
        self._turn_index = 0
        self._tool_call_counts: dict[str, int] = {}
        self.stages: list[StageRecord] = []
        self.parse_errors: list[str] = []
        self.tool_routes: list[dict[str, Any]] = []

    def set_seed(self, seed: int) -> None:
        self._seed = int(seed)

    def get_init_state(self, message_history: list[Any] | None = None) -> list[Any]:
        return list(message_history or [])

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        content = getattr(message, "content", None)
        return isinstance(content, str) and "###STOP###" in content

    def stop(self, message: Any | None = None, state: list[Any] | None = None) -> None:
        del message, state

    def generate_next_message(self, message: Any, state: list[Any] | None) -> tuple[Any, list[Any]]:
        history = list(state or [])
        if message is not None:
            _append_tau_message(history, message, MultiToolMessage=self._MultiToolMessage)

        prompt_messages = _tau_messages_to_prompt_messages(
            history,
            ToolMessage=self._ToolMessage,
            UserMessage=self._UserMessage,
        )
        prompt = self._build_prompt(prompt_messages)
        prompt_seed = None if self._seed is None else int(self._seed) + self._turn_index
        generate_kwargs = {
            "sampling": self._sampling,
            "batch_size": 1,
            "progress_desc": "TauOfficial-Agent",
            "prompt_stop_suffixes": [list(JSON_CALL_STOP_SUFFIXES)],
            "prompt_seeds": [prompt_seed] if prompt_seed is not None else None,
            "preserve_prompt_whitespace": True,
        }
        generate_lock = self._generate_lock
        if generate_lock is None:
            outputs = self._engine.generate([prompt], **generate_kwargs)
        else:
            with generate_lock:
                outputs = self._engine.generate([prompt], **generate_kwargs)
        output = outputs[0] if outputs else None
        raw_text = output.text if output is not None else ""
        finish_reason = output.finish_reason if output is not None else "missing_output"
        try:
            name, arguments = parse_tau_agent_decision(raw_text)
            assistant_message = self._decision_to_assistant_message(
                name,
                arguments,
                prompt_messages=prompt_messages,
            )
        except Exception as exc:
            parse_error = str(exc)
            self.parse_errors.append(parse_error)
            assistant_message = self._AssistantMessage(
                role="assistant",
                content="I am unable to continue safely. ###STOP###",
            )
        self.stages.append(StageRecord(prompt=prompt, completion=raw_text, stop_reason=finish_reason))
        self._turn_index += 1
        history.append(assistant_message)
        return assistant_message, history

    def _build_prompt(self, prompt_messages: Sequence[Mapping[str, object]]) -> str:
        routed_inputs = route_agent_prompt_inputs(
            domain_policy=self._domain_policy,
            tools=self._tools,
            messages=prompt_messages,
            config=agent_plugin_config_from_sources(
                _agent_plugin_source_from_routing_configs(
                    self._tool_routing_config,
                    self._long_context_routing_config,
                )
            ),
            control_tool_names=(RESPOND_TOOL_NAME,),
        )
        prompt, emitted_tools, emitted_policy_chars, schema_mode, long_context_trace = build_budgeted_tau_prompt(
            domain_policy=routed_inputs.domain_policy,
            selected_tools=routed_inputs.selected_tools,
            messages=routed_inputs.messages,
            history_max_chars=self._history_max_chars,
            prompt_max_chars=self._prompt_max_chars,
            long_context_config=None,
        )
        self._current_tool_names = {tool_name(tool) for tool in emitted_tools if tool_name(tool)}
        if routed_inputs.tool_route is not None:
            route_trace = {"turn_index": self._turn_index, **routed_inputs.tool_route.trace_payload()}
        else:
            route_trace = {
                "turn_index": self._turn_index,
                "mode": self._tool_routing_config.mode,
                "routed": False,
                "reason": "disabled",
                "selected_names": [tool_name(tool) for tool in routed_inputs.selected_tools if tool_name(tool)],
                "total_tool_count": len(self._tools),
            }
        if len(emitted_tools) != len(routed_inputs.selected_tools):
            route_trace["emitted_names"] = [tool_name(tool) for tool in emitted_tools if tool_name(tool)]
            route_trace["system_budget_reduced"] = True
        if schema_mode != "full":
            route_trace["emitted_tool_schema_mode"] = schema_mode
            route_trace["system_budget_reduced"] = True
        if emitted_policy_chars < len(normalize_rwkv_text(self._domain_policy)):
            route_trace["emitted_policy_chars"] = int(emitted_policy_chars)
            route_trace["system_budget_reduced"] = True
        plugin_long_context_trace = routed_inputs.long_context_trace or long_context_trace
        if plugin_long_context_trace is not None:
            route_trace["long_context"] = plugin_long_context_trace
            if plugin_long_context_trace.get("compacted_message_count") or plugin_long_context_trace.get("policy_compacted"):
                route_trace["system_budget_reduced"] = True
        self.tool_routes.append(route_trace)
        return prompt

    def _decision_to_assistant_message(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        prompt_messages: Sequence[Mapping[str, object]],
    ) -> Any:
        del prompt_messages
        normalized_name, normalized_arguments = normalize_tau_decision(name, arguments)
        if normalized_name == RESPOND_TOOL_NAME:
            content = (
                normalized_arguments.get("content")
                or normalized_arguments.get("answer")
                or normalized_arguments.get("message")
                or "###STOP###"
            )
            return self._AssistantMessage(role="assistant", content=str(content))
        if normalized_name not in self._tool_names:
            raise ValueError(f"unknown tau tool name: {normalized_name}")
        if normalized_name not in self._current_tool_names:
            raise ValueError(f"tau tool name not in routed tool window: {normalized_name}")
        key = _tool_call_key(normalized_name, normalized_arguments)
        count = self._tool_call_counts.get(key, 0) + 1
        self._tool_call_counts[key] = count
        if count > self._max_repeated_tool_calls:
            return self._AssistantMessage(
                role="assistant",
                content="I already repeated the same tool call and cannot continue productively. ###STOP###",
            )
        return self._AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                self._ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=normalized_name,
                    arguments=normalized_arguments,
                    requestor="assistant",
                )
            ],
        )


class StaticStopTauUser:
    """Minimal no-LLM tau user for ticket-seeded lightweight tasks."""

    def __init__(self, *, stop_content: str = "###STOP###") -> None:
        self.stop_content = str(stop_content)
        message_module = import_module_with_auto_install("tau2.data_model.message", context="tau2 message import")
        base_module = import_module_with_auto_install("tau2.user.user_simulator_base", context="tau2 user base import")
        self._UserMessage = getattr(message_module, "UserMessage")
        self._UserState = getattr(base_module, "UserState")

    def get_init_state(self, message_history: list[Any] | None = None) -> Any:
        return self._UserState(system_messages=[], messages=list(message_history or []))

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        content = getattr(message, "content", "")
        return isinstance(content, str) and "###STOP###" in content

    def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        del message
        user_message = self._UserMessage(role="user", content=self.stop_content, cost=0.0)
        state.messages.append(user_message)
        return user_message, state

    def set_seed(self, seed: int) -> None:
        del seed

    def stop(self, message: Any | None = None, state: Any | None = None) -> None:
        del message, state


def _agent_plugin_source_from_routing_configs(
    tool_config: ToolRoutingConfig,
    long_context_config: LongContextRoutingConfig,
) -> dict[str, Any]:
    return {
        "agent_plugin_enabled": bool(tool_config.enabled or long_context_config.enabled),
        "tool_router_mode": tool_config.mode,
        "tool_router_max_tools": int(tool_config.max_tools),
        "tool_router_trigger_tool_count": int(tool_config.trigger_tool_count),
        "tool_router_trigger_catalog_chars": int(tool_config.trigger_catalog_chars),
        "tool_router_context_chars": int(tool_config.context_chars),
        "tool_router_description_chars": int(tool_config.description_chars),
        "tool_router_fallback_to_all_on_empty": bool(tool_config.fallback_to_all_on_empty),
        "long_context_router_mode": long_context_config.mode,
        "long_context_min_chars": int(long_context_config.min_chars),
        "long_context_chunk_chars": int(long_context_config.chunk_chars),
        "long_context_overlap_lines": int(long_context_config.overlap_lines),
        "long_context_max_evidence_chunks": int(long_context_config.max_evidence_chunks),
        "long_context_max_evidence_chars": int(long_context_config.max_evidence_chars),
        "long_context_query_chars": int(long_context_config.query_chars),
        "long_context_fallback_to_original_on_empty": bool(long_context_config.fallback_to_original_on_empty),
    }


def build_budgeted_tau_prompt(
    *,
    domain_policy: str,
    selected_tools: Sequence[Any],
    messages: Sequence[Mapping[str, object]],
    history_max_chars: int,
    prompt_max_chars: int,
    long_context_config: LongContextRoutingConfig | None = None,
) -> tuple[str, list[Any], int, str, dict[str, Any] | None]:
    tools = list(selected_tools)
    policy = normalize_rwkv_text(domain_policy)
    messages_for_prompt = list(messages)
    long_context_trace: dict[str, Any] | None = None
    if long_context_config is not None and long_context_config.enabled:
        query = infer_long_context_query(messages_for_prompt, config=long_context_config)
        message_route = compact_messages_for_prompt(
            messages_for_prompt,
            query=query,
            config=long_context_config,
        )
        policy_route = compact_text_for_prompt(
            policy,
            query=query,
            config=long_context_config,
            label="tau domain policy",
        )
        messages_for_prompt = message_route.messages
        policy = normalize_rwkv_text(policy_route.text)
        long_context_trace = {
            "mode": long_context_config.mode,
            "query_chars": len(query),
            "compacted_message_count": int(message_route.compacted_message_count),
            "message_original_chars": int(message_route.original_chars),
            "message_routed_chars": int(message_route.routed_chars),
            "message_reason": message_route.reason,
            "policy_compacted": bool(policy_route.compacted),
            "policy_original_chars": int(policy_route.original_chars),
            "policy_routed_chars": int(policy_route.routed_chars),
            "policy_reason": policy_route.reason,
        }
        if message_route.selected_chunk_ids:
            long_context_trace["message_selected_chunk_ids"] = {
                str(index): list(chunk_ids)
                for index, chunk_ids in sorted(message_route.selected_chunk_ids.items())
            }
        if policy_route.selected_chunk_ids:
            long_context_trace["policy_selected_chunk_ids"] = list(policy_route.selected_chunk_ids)
    policy_budgets = _policy_budget_candidates(policy, prompt_max_chars)
    schema_modes = ("full", "compact", "minimal")
    best_prompt = ""
    best_policy_chars = len(policy)
    best_schema_mode = "full"
    for policy_budget in policy_budgets:
        policy_view = truncate_text(policy, policy_budget)
        for schema_mode in schema_modes:
            system_prompt = build_tau_official_agent_system_prompt(policy_view, tools, tool_schema_mode=schema_mode)
            prompt = build_rwkv_json_call_prompt(system_prompt, messages_for_prompt, history_max_chars=history_max_chars)
            best_prompt = prompt
            best_policy_chars = len(policy_view)
            best_schema_mode = schema_mode
            if len(prompt) <= prompt_max_chars:
                return prompt, tools, len(policy_view), schema_mode, long_context_trace
            overflow = len(prompt) - prompt_max_chars
            trimmed_history = max(0, history_max_chars - overflow - 512)
            prompt = build_rwkv_json_call_prompt(
                system_prompt,
                trim_message_history(messages_for_prompt, max_chars=trimmed_history),
                history_max_chars=trimmed_history,
            )
            best_prompt = prompt
            if len(prompt) <= prompt_max_chars:
                return prompt, tools, len(policy_view), schema_mode, long_context_trace
    if len(best_prompt) <= prompt_max_chars:
        return best_prompt, tools, best_policy_chars, best_schema_mode, long_context_trace
    final_system = build_tau_official_agent_system_prompt(
        truncate_text(policy, min(policy_budgets[-1], 240)),
        tools,
        tool_schema_mode="minimal",
    )
    final_prompt = build_rwkv_json_call_prompt(final_system, [], history_max_chars=0)
    if len(final_prompt) > prompt_max_chars:
        raise ValueError(
            "tau prompt budget cannot fit routed tools without corrupting the prompt: "
            f"prompt_chars={len(final_prompt)} budget={prompt_max_chars} tools={len(tools)}"
        )
    return final_prompt, tools, min(policy_budgets[-1], 240), "minimal", long_context_trace


def build_tau_official_agent_system_prompt(
    domain_policy: str,
    tools: Sequence[Any],
    *,
    tool_schema_mode: str,
) -> str:
    if tool_schema_mode == "full":
        tool_schemas = [_normalize_tool_schema(tool) for tool in tools]
        json_kwargs: dict[str, Any] = {"ensure_ascii": False, "indent": 2}
    elif tool_schema_mode == "compact":
        tool_schemas = [_compact_tool_schema(_normalize_tool_schema(tool)) for tool in tools]
        json_kwargs = {"ensure_ascii": False, "separators": (",", ":")}
    elif tool_schema_mode == "minimal":
        tool_schemas = [_minimal_tool_schema(_normalize_tool_schema(tool)) for tool in tools]
        json_kwargs = {"ensure_ascii": False, "separators": (",", ":")}
    else:
        raise ValueError(f"unsupported tau tool schema mode: {tool_schema_mode}")
    tool_schemas.append(
        {
            "name": RESPOND_TOOL_NAME,
            "description": "Send a natural-language message to the user. Include ###STOP### when the task is complete.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        }
    )
    sections = [
        "You are the assistant in the official tau-bench simulation.",
        "Follow the domain policy exactly.",
        "Use a real tool call when you need information or need to change state.",
        "Use respond only when sending a message to the user.",
        "When the task is complete and no more tool calls are needed, use respond and include ###STOP### in the content.",
        "Return exactly one JSON function call object and no extra prose.",
        'JSON shape: {"name":"tool_name","arguments":{...}}',
        "Valid names are exactly the listed tool names plus respond.",
        "Never copy a Function output object; do not return requestor/ok/output as your decision.",
        "Do not invent ids/emails; copy IDs exactly, including #.",
        "Tools:",
        json.dumps(tool_schemas, **json_kwargs),
        "Policy:",
        normalize_rwkv_text(domain_policy),
    ]
    return normalize_rwkv_text("\n".join(sections))


def parse_tau_agent_decision(text: str) -> tuple[str, dict[str, Any]]:
    try:
        candidate = extract_json_call_value_text(text)
        raw_payload = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        raw_payload = _partial_tau_decision_payload(text, cause=exc)
    payload = _coerce_tau_decision_payload(raw_payload)
    return normalize_tau_decision(str(payload["name"]), dict(payload["arguments"]))


def normalize_tau_decision(name: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized_name = _strip_tau_requestor_prefix(name)
    if normalized_name == "final_answer":
        normalized_name = RESPOND_TOOL_NAME
    return normalized_name, dict(arguments)


def tau_trajectory_dump(messages: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in messages:
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                rows.append(dumped)
                continue
        row: dict[str, Any] = {
            "role": str(getattr(message, "role", "unknown")),
            "content": getattr(message, "content", ""),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            row["tool_calls"] = [
                {
                    "name": str(getattr(call, "name", "") or ""),
                    "arguments": dict(getattr(call, "arguments", {}) or {}),
                }
                for call in tool_calls
            ]
        rows.append(row)
    return rows


def tau_domain_info(
    domain: str,
    *,
    env_kwargs: Mapping[str, Any] | None = None,
) -> TauDomainInfo:
    runtime = TauOfficialRuntime(domain=domain)
    environment = runtime.create_environment(solo_mode=False, env_kwargs=env_kwargs)
    policy = str(environment.get_policy())
    tools = [_normalize_tool_schema(tool) for tool in environment.get_tools()]
    return TauDomainInfo(policy=policy, tools=tools)


def normalize_tau_official_task_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    criteria = normalized.get("evaluation_criteria")
    if not isinstance(criteria, Mapping):
        return normalized
    normalized_criteria = dict(criteria)
    reward_basis = normalized_criteria.get("reward_basis")
    if isinstance(reward_basis, Sequence) and not isinstance(reward_basis, (str, bytes)):
        normalized_criteria["reward_basis"] = [_normalize_tau_reward_type_value(value) for value in reward_basis]
    normalized["evaluation_criteria"] = normalized_criteria
    return normalized


def render_tau_user_prompt(task_payload: Mapping[str, Any]) -> str:
    ticket = task_payload.get("ticket")
    if isinstance(ticket, str) and ticket.strip():
        return ticket.strip()
    scenario = task_payload.get("user_scenario")
    if isinstance(scenario, Mapping):
        instructions = scenario.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            return instructions.strip()
        if instructions is not None:
            return str(instructions)
    description = task_payload.get("description")
    if isinstance(description, str):
        return description.strip()
    if description is not None:
        return str(description)
    return ""


def render_tau_messages(messages: Sequence[Any], *, max_chars: int = 8000) -> str:
    parts = [_render_tau_message(message) for message in messages]
    rendered = "\n\n".join(part for part in parts if part)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[-max(1, int(max_chars)) :]


def configure_tau_nl_assertions_judge(judge_model: Any) -> None:
    model_name = str(getattr(judge_model, "model_name", "") or "").strip()
    api_key = str(getattr(judge_model, "api_key", "") or "").strip()
    base_url = str(getattr(judge_model, "base_url", "") or "").strip()
    if not model_name or not api_key:
        return
    llm_args: dict[str, Any] = {
        "temperature": 0.0,
        "stream": False,
        "api_key": api_key,
        "response_format": {"type": "json_object"},
    }
    if base_url:
        llm_args["api_base"] = base_url.rstrip("/")
    timeout_s = _first_positive_float_env("RWKV_TAU_LLM_TIMEOUT_S", "RWKV_TAU_JUDGE_TIMEOUT_S", "RWKV_LLM_TIMEOUT_S")
    if timeout_s is not None:
        llm_args["timeout"] = timeout_s
    for module_name in ("tau2.config", "tau2.evaluator.evaluator_nl_assertions"):
        module = import_module_with_auto_install(module_name, context=f"tau2 NL assertion judge config: {module_name}")
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS", model_name)
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", dict(llm_args))


def _first_positive_float_env(*names: str) -> float | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = float(value.strip())
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


def _normalize_tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "openai_schema", None)
    if isinstance(schema, Mapping):
        function_schema = schema.get("function")
        if isinstance(function_schema, Mapping):
            return {
                "name": str(function_schema.get("name") or "").strip(),
                "description": str(function_schema.get("description") or ""),
                "parameters": dict(function_schema.get("parameters") or {}),
            }
        return dict(schema)
    if isinstance(tool, Mapping):
        return dict(tool)
    return {
        "name": str(getattr(tool, "name", "") or "").strip(),
        "description": str(getattr(tool, "short_desc", "") or tool),
        "parameters": {"type": "object", "properties": {}},
    }


def _compact_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required")
    if not isinstance(required, (list, tuple)):
        required = []
    return {
        "name": str(schema.get("name") or "").strip(),
        "description": truncate_text(normalize_rwkv_text(str(schema.get("description") or "")), 120),
        "parameters": {
            "type": "object",
            "properties": {str(key): _compact_parameter_schema(value) for key, value in properties.items()},
            "required": [str(item) for item in required],
        },
    }


def _minimal_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required")
    if not isinstance(required, (list, tuple)):
        required = []
    return {
        "name": str(schema.get("name") or "").strip(),
        "description": truncate_text(normalize_rwkv_text(str(schema.get("description") or "")), 64),
        "args": {str(key): _minimal_parameter_type(value) for key, value in properties.items()},
        "required": [str(item) for item in required],
    }


def _compact_parameter_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        return {"type": "string"}
    schema_type = schema.get("type") or "string"
    compact: dict[str, Any] = {"type": str(schema_type)}
    description = normalize_rwkv_text(str(schema.get("description") or "")).strip()
    if description:
        compact["description"] = truncate_text(description, 48)
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and len(enum) <= 12:
        compact["enum"] = [str(item) for item in enum]
    return compact


def _minimal_parameter_type(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return "string"
    schema_type = str(schema.get("type") or "string")
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and 0 < len(enum) <= 8:
        return f"{schema_type} enum={','.join(str(item) for item in enum)}"
    return schema_type


def _partial_tau_decision_payload(text: str, *, cause: Exception) -> dict[str, Any]:
    normalized = normalize_rwkv_text(text)
    start = normalized.find("{")
    if start < 0:
        raise ValueError(f"tau agent decision missing JSON object: {normalized}") from cause
    body = normalized[start:]
    name = _raw_decode_json_field(body, "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"tau agent decision missing recoverable name: {normalized}") from cause
    try:
        arguments = _raw_decode_json_field(body, "arguments")
    except ValueError:
        arguments = {}
    if arguments is None:
        arguments = {}
    return {"name": name, "arguments": arguments}


def _raw_decode_json_field(body: str, key: str) -> Any:
    match = re.search(rf'"{re.escape(key)}"\s*:', body)
    if match is None:
        if key == "arguments":
            return None
        raise ValueError(f"missing JSON field {key!r}")
    value_text = body[match.end() :].lstrip()
    if not value_text:
        raise ValueError(f"missing JSON value for field {key!r}")
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(value_text)
    return value


def _coerce_tau_decision_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("tau agent decision payload did not contain a function call")
        payload = payload[0]
    if not isinstance(payload, Mapping):
        raise ValueError("tau agent decision payload must be a JSON object")
    if "tool_calls" in payload:
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError("tau agent decision tool_calls payload must contain at least one tool call")
        first_call = tool_calls[0]
        if not isinstance(first_call, Mapping):
            raise ValueError("tau agent decision tool_calls item must be a JSON object")
        return _coerce_tau_decision_payload(first_call)

    function_payload = payload.get("function")
    if not isinstance(function_payload, Mapping):
        function_payload = payload.get("function_call")
    if isinstance(function_payload, Mapping):
        name = function_payload.get("name") or payload.get("name")
        arguments = function_payload.get("arguments", payload.get("arguments", {}))
    else:
        name = payload.get("name") or payload.get("action") or payload.get("tool_name") or payload.get("tool")
        if "arguments" in payload:
            arguments = payload.get("arguments", {})
        elif "action_input" in payload:
            arguments = payload.get("action_input", {})
        elif "input" in payload:
            arguments = payload.get("input", {})
        elif "parameters" in payload:
            arguments = payload.get("parameters", {})
        else:
            arguments = _tau_top_level_arguments(payload)

    name_text = str(name or "").strip()
    if not name_text:
        raise ValueError("tau agent decision missing name")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        raw_arguments = arguments.strip()
        if not raw_arguments:
            arguments = {}
        else:
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("tau agent decision string arguments must be a JSON object") from exc
    if not isinstance(arguments, Mapping):
        raise ValueError("tau agent decision arguments must be a JSON object")
    return {"name": name_text, "arguments": dict(arguments)}


def _tau_top_level_arguments(payload: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {
        "id",
        "type",
        "name",
        "action",
        "tool_name",
        "tool",
        "requestor",
        "role",
        "function",
        "function_call",
        "tool_calls",
    }
    return {str(key): value for key, value in payload.items() if key not in reserved}


def _tau_messages_to_prompt_messages(
    history: Sequence[Any],
    *,
    ToolMessage: Any,
    UserMessage: Any,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for message in history:
        role = str(getattr(message, "role", "") or "").strip().lower()
        if isinstance(message, ToolMessage) or role == "tool":
            rows.append({"role": "user", "content": _render_tau_tool_message(message)})
            continue
        if isinstance(message, UserMessage) or role == "user":
            content = str(getattr(message, "content", "") or "").strip()
            if content:
                rows.append({"role": "user", "content": content})
            continue
        if role == "assistant":
            content = str(getattr(message, "content", "") or "").strip()
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                rendered = "\n".join(_render_tau_tool_call(call) for call in tool_calls)
                if rendered:
                    rows.append({"role": "assistant", "content": rendered})
            elif content:
                rows.append({"role": "assistant", "content": content})
    return rows


def _append_tau_message(history: list[Any], message: Any, *, MultiToolMessage: Any) -> None:
    if isinstance(message, MultiToolMessage):
        history.extend(list(getattr(message, "tool_messages", []) or []))
    else:
        history.append(message)


def _render_tau_tool_message(message: Any) -> str:
    payload = {
        "requestor": str(getattr(message, "requestor", "assistant") or "assistant"),
        "ok": not bool(getattr(message, "error", False)),
        "output": getattr(message, "content", None),
    }
    return "Function output:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _render_tau_tool_call(tool_call: Any) -> str:
    return json.dumps(
        {
            "name": str(getattr(tool_call, "name", "") or ""),
            "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _policy_budget_candidates(policy: str, prompt_max_chars: int) -> list[int]:
    full = len(policy)
    candidates = [
        full,
        max(400, min(full, int(prompt_max_chars * 0.22))),
        max(350, min(full, 1600)),
        max(300, min(full, 1000)),
        max(240, min(full, 650)),
        max(180, min(full, 400)),
    ]
    out: list[int] = []
    for candidate in candidates:
        normalized = max(0, int(candidate))
        if normalized not in out:
            out.append(normalized)
    return out or [0]


def _strip_tau_requestor_prefix(name: str) -> str:
    text = str(name or "").strip()
    if "." not in text:
        return text
    prefix, rest = text.split(".", 1)
    if prefix in {"assistant", "user"} and rest.strip():
        return rest.strip()
    return text


def _tool_call_key(name: str, arguments: Mapping[str, Any]) -> str:
    return f"{name}:{json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"


def _tau_litellm_model_name(model_config: Any) -> str:
    model_name = str(getattr(model_config, "model_name", "") or "").strip()
    if not model_name or "/" in model_name:
        return model_name
    base_url = str(getattr(model_config, "base_url", "") or "").strip().rstrip("/")
    if "api.deepseek.com" in base_url and model_name.startswith("deepseek-"):
        return f"deepseek/{model_name}"
    return model_name


def _tau_llm_timeout_args() -> dict[str, float]:
    timeout_s = _first_positive_float_env("RWKV_TAU_LLM_TIMEOUT_S", "RWKV_TAU_USER_TIMEOUT_S", "RWKV_LLM_TIMEOUT_S")
    if timeout_s is None:
        return {}
    return {"timeout": timeout_s}


def _normalize_tau_reward_type_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_TAU_REWARD_TYPE_PREFIX):
        return value.removeprefix(_TAU_REWARD_TYPE_PREFIX)
    return value


def _task_uses_nl_assertions(task: Any) -> bool:
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is None:
        return False
    reward_basis = getattr(criteria, "reward_basis", None) or []
    return any(str(item).endswith("NL_ASSERTION") for item in reward_basis)


def _model_dump_safe(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        dumped = item.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if isinstance(item, dict):
        return dict(item)
    return {"value": str(item)}


def _render_tau_message(message: Any) -> str:
    role = str(getattr(message, "role", "") or message.__class__.__name__)
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        rows = []
        for call in tool_calls:
            rows.append(
                f"{getattr(call, 'name', '')}("
                f"{getattr(call, 'arguments', {})})"
            )
        return f"{role}: tool_calls=" + "; ".join(rows)
    if content is not None:
        return f"{role}: {content}"
    return str(message)


def _tau_now() -> str:
    utils_module = import_module_with_auto_install("tau2.utils.utils", context="tau2 timestamp")
    return str(getattr(utils_module, "get_now")())


__all__ = [
    "DEFAULT_TAU_PROMPT_MAX_CHARS",
    "RESPOND_TOOL_NAME",
    "RWKVTauOfficialAgent",
    "StaticStopTauUser",
    "TauDomainInfo",
    "TauOfficialEvaluation",
    "TauOfficialRuntime",
    "build_budgeted_tau_prompt",
    "build_tau_official_agent_system_prompt",
    "configure_tau_nl_assertions_judge",
    "normalize_tau_official_task_payload",
    "normalize_tau_decision",
    "parse_tau_agent_decision",
    "render_tau_messages",
    "render_tau_user_prompt",
    "tau_domain_info",
    "tau_trajectory_dump",
]
