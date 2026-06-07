from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.eval.agent_bench.deps import import_module_with_auto_install
from src.eval.agent_bench.tasks import ensure_tau_v2_vendor_path
from src.eval.env_config import normalize_openai_base_url
from src.eval.evaluators.common import StageRecord
from src.eval.function_calling.context_budget import normalize_rwkv_text, trim_message_history, truncate_text
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    apply_json_call_object_prefill,
    assistant_json_prefix,
    build_rwkv_json_call_prompt,
    extract_json_call_value_text,
)
from src.eval.function_calling.tool_router import ToolRoutingConfig, route_tools_for_prompt
from src.eval.long_doc_evidence import (
    LongDocEvidenceConfig,
    compact_long_text,
    compact_messages_for_long_context,
    infer_query_from_messages,
    long_doc_config_from_env,
)
from src.infer.backend import InferenceBackend
from src.infer.sampling import SamplingConfig

RESPOND_TOOL_NAME = "respond"
DEFAULT_TAU_PROMPT_MAX_CHARS = 24576
_TAU_REWARD_TYPE_PREFIX = "RewardType."
TAU_JSON_CALL_ASSISTANT_PREFIX = assistant_json_prefix(enable_think=False, prefill_object=True)


@dataclass(slots=True)
class TauOfficialEvaluation:
    reward: float
    is_passed: bool
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TauRetailProgressiveToolDisclosure:
    selected_tools: list[Any]
    trace: dict[str, Any]


class TauOfficialRuntime:
    """Thin dynamic bridge to the official tau2/tau3 runtime package."""

    def __init__(self, *, domain: str) -> None:
        ensure_tau_v2_vendor_path()
        registry_module = import_module_with_auto_install("tau2.registry", context="tau2 registry import")
        self.registry = getattr(registry_module, "registry")
        self.domain = str(domain)
        self._environment_constructor = self.registry.get_env_constructor(self.domain)

    def load_task(self, payload: Mapping[str, Any]) -> Any:
        tasks_module = import_module_with_auto_install("tau2.data_model.tasks", context="tau2 Task model import")
        Task = getattr(tasks_module, "Task")
        return Task.model_validate(normalize_tau_official_task_payload(payload))

    def create_environment(self, *, solo_mode: bool = False) -> Any:
        try:
            return self._environment_constructor(solo_mode=solo_mode)
        except TypeError:
            if solo_mode:
                raise
            return self._environment_constructor()

    def build_user(self, *, task: Any, environment: Any, user_model: Any | None, temperature: float = 0.0) -> Any:
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
                "temperature": _tau_openai_temperature(float(temperature)),
                "stream": False,
                "api_key": user_model.api_key,
                "api_base": user_model.base_url,
                **_tau_litellm_provider_args(user_model),
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
        seed: int | None,
        validate_communication: bool,
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
            seed=seed,
            solo_mode=False,
            validate_communication=bool(validate_communication),
        )

    def evaluate(self, *, simulation: Any, task: Any, judge_model: Any | None = None) -> TauOfficialEvaluation:
        if judge_model is not None:
            configure_tau_nl_assertions_judge(judge_model)
        evaluator_module = import_module_with_auto_install("tau2.evaluator.evaluator", context="tau2 evaluator import")
        EvaluationType = getattr(evaluator_module, "EvaluationType")
        evaluate_simulation = getattr(evaluator_module, "evaluate_simulation")
        evaluation_type = (
            EvaluationType.ALL_WITH_NL_ASSERTIONS
            if _task_uses_nl_assertions(task)
            else EvaluationType.ALL
        )
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=evaluation_type,
            solo_mode=False,
            domain=self.domain,
        )
        simulation.reward_info = reward_info
        details = _model_dump_safe(reward_info)
        details["termination_reason"] = str(getattr(simulation, "termination_reason", ""))
        return TauOfficialEvaluation(
            reward=float(getattr(reward_info, "reward", 0.0)),
            is_passed=float(getattr(reward_info, "reward", 0.0)) >= (1.0 - 1e-6),
            details=details,
        )


def configure_tau_nl_assertions_judge(judge_model: Any) -> None:
    model_name = _tau_litellm_model_name(judge_model)
    api_key = str(getattr(judge_model, "api_key", "") or "").strip()
    base_url = normalize_openai_base_url(getattr(judge_model, "base_url", None))
    if not model_name or not api_key:
        return
    llm_args: dict[str, Any] = {
        "temperature": _tau_openai_temperature(0.0),
        "stream": False,
        "api_key": api_key,
        "response_format": {"type": "json_object"},
    }
    if base_url:
        llm_args["api_base"] = base_url
    llm_args.update(_tau_litellm_provider_args(judge_model))
    llm_args.update(_tau_llm_timeout_args())
    for module_name in ("tau2.config", "tau2.evaluator.evaluator_nl_assertions"):
        module = import_module_with_auto_install(module_name, context=f"tau2 NL assertion judge config: {module_name}")
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS", model_name)
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", dict(llm_args))


def normalize_tau_official_task_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Convert known tau manifest wire formats to the vendored tau2 Task schema."""
    normalized = dict(payload)
    criteria = normalized.get("evaluation_criteria")
    if not isinstance(criteria, Mapping):
        return normalized

    normalized_criteria = dict(criteria)
    reward_basis = normalized_criteria.get("reward_basis")
    if isinstance(reward_basis, Sequence) and not isinstance(reward_basis, (str, bytes)):
        normalized_criteria["reward_basis"] = [
            _normalize_tau_reward_type_value(value) for value in reward_basis
        ]
    normalized["evaluation_criteria"] = normalized_criteria
    return normalized


def _normalize_tau_reward_type_value(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_TAU_REWARD_TYPE_PREFIX):
        return value.removeprefix(_TAU_REWARD_TYPE_PREFIX)
    return value


def _tau_litellm_model_name(model_config: Any) -> str:
    model_name = str(getattr(model_config, "model_name", "") or "").strip()
    if model_name.startswith("openai/"):
        return model_name.removeprefix("openai/")
    if not model_name or "/" in model_name:
        return model_name
    base_url = normalize_openai_base_url(getattr(model_config, "base_url", None)) or ""
    if "api.deepseek.com" in base_url and model_name.startswith("deepseek-"):
        return f"deepseek/{model_name}"
    return model_name


def _tau_litellm_provider_args(model_config: Any) -> dict[str, str]:
    model_name = str(getattr(model_config, "model_name", "") or "").strip()
    base_url = normalize_openai_base_url(getattr(model_config, "base_url", None)) or ""
    if "api.deepseek.com" in base_url:
        return {}
    if model_name.startswith("openai/") or (base_url and "/" not in model_name):
        return {"custom_llm_provider": "openai"}
    return {}


def _tau_openai_temperature(value: float) -> float:
    return max(0.001, float(value))


def _tau_llm_timeout_args() -> dict[str, float]:
    timeout_s = _first_positive_float_env(
        "RWKV_TAU_LLM_TIMEOUT_S",
        "RWKV_TAU_USER_TIMEOUT_S",
        "RWKV_LLM_TIMEOUT_S",
    )
    if timeout_s is None:
        return {}
    return {"timeout": timeout_s}


def _first_positive_float_env(*names: str) -> float | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        text = value.strip()
        if not text:
            continue
        try:
            parsed = float(text)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


class RWKVTauOfficialAgent:
    """Official tau Agent interface backed by the repository inference backend."""

    def __init__(
        self,
        *,
        engine: InferenceBackend,
        sampling: SamplingConfig,
        tools: Sequence[Any],
        domain_policy: str,
        domain: str | None = None,
        history_max_chars: int,
        prompt_max_chars: int = DEFAULT_TAU_PROMPT_MAX_CHARS,
        long_doc_config: LongDocEvidenceConfig | None = None,
        tool_routing_config: ToolRoutingConfig | None = None,
        retail_repeated_read_guard: bool = False,
        retail_tool_use_guard: bool = False,
        retail_progressive_tool_disclosure: bool = False,
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
        self._tools_by_name = {_tool_name(tool): tool for tool in self._tools if _tool_name(tool)}
        self._tool_names = set(self._tools_by_name)
        self._current_tool_names = set(self._tool_names)
        self._domain_policy = str(domain_policy)
        self._domain = str(domain or "").strip().lower()
        self._history_max_chars = max(0, int(history_max_chars))
        self._prompt_max_chars = max(1024, int(prompt_max_chars))
        self._long_doc_config = long_doc_config or long_doc_config_from_env("RWKV_TAU_LONG_DOC")
        self._tool_routing_config = tool_routing_config or ToolRoutingConfig()
        self._retail_repeated_read_guard = bool(retail_repeated_read_guard)
        self._retail_tool_use_guard = bool(retail_tool_use_guard)
        self._retail_progressive_tool_disclosure = bool(retail_progressive_tool_disclosure)
        self._seed: int | None = None
        self._turn_index = 0
        self.stages: list[StageRecord] = []
        self.parse_errors: list[str] = []
        self.tool_routes: list[dict[str, Any]] = []
        self.step_timings: list[dict[str, Any]] = []

    def set_seed(self, seed: int) -> None:
        self._seed = int(seed)

    def get_init_state(self, message_history: list[Any] | None = None) -> list[Any]:
        return list(message_history or [])

    def stop(self, message: Any | None = None, state: list[Any] | None = None) -> None:
        del message, state

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        content = getattr(message, "content", None)
        return isinstance(content, str) and "###STOP###" in content

    def generate_next_message(self, message: Any, state: list[Any] | None) -> tuple[Any, list[Any]]:
        step_started = time.perf_counter()
        history = list(state or [])
        if message is not None:
            _append_tau_message(history, message, MultiToolMessage=self._MultiToolMessage)

        prompt_messages = _tau_messages_to_prompt_messages(
            history,
            ToolMessage=self._ToolMessage,
            UserMessage=self._UserMessage,
        )
        prompt_build_started = time.perf_counter()
        prompt = self._build_prompt(prompt_messages)
        prompt_build_s = time.perf_counter() - prompt_build_started
        prompt_seed = None if self._seed is None else int(self._seed) + self._turn_index
        generation_started = time.perf_counter()
        outputs = self._engine.generate(
            [prompt],
            sampling=self._sampling,
            batch_size=1,
            progress_desc="TauOfficial-Agent",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
            prompt_seeds=[prompt_seed] if prompt_seed is not None else None,
        )
        generation_s = time.perf_counter() - generation_started
        output = outputs[0] if outputs else None
        raw_text = output.text if output is not None else ""
        finish_reason = output.finish_reason if output is not None else "missing_output"
        parse_error: str | None = None
        recovered = False
        parse_started = time.perf_counter()
        parse_text = _apply_tau_json_object_prefill(raw_text)
        parse_prefill_applied = parse_text != normalize_rwkv_text(raw_text)
        try:
            name, arguments = _parse_tau_agent_decision(parse_text)
            assistant_message = self._decision_to_assistant_message(
                name,
                arguments,
                prompt_messages=prompt_messages,
            )
        except Exception as exc:
            parse_error = str(exc)
            try:
                name, arguments = _recover_tau_agent_decision_from_text(
                    parse_text if parse_prefill_applied else raw_text,
                    prompt_messages,
                    available_tool_names=self._tool_names,
                )
                assistant_message = self._decision_to_assistant_message(
                    name,
                    arguments,
                    prompt_messages=prompt_messages,
                )
                parse_error = None
                recovered = True
            except Exception:
                self.parse_errors.append(parse_error)
                assistant_message = self._AssistantMessage(
                    role="assistant",
                    content="I am unable to continue safely. ###STOP###",
                )
        parse_s = time.perf_counter() - parse_started
        self.stages.append(
            StageRecord(
                prompt=prompt,
                completion=raw_text,
                stop_reason=finish_reason,
            )
        )
        self.step_timings.append(
            {
                "turn_index": int(self._turn_index),
                "prompt_chars": len(prompt),
                "completion_chars": len(raw_text),
                "prompt_build_s": prompt_build_s,
                "generation_s": generation_s,
                "parse_s": parse_s,
                "total_s": time.perf_counter() - step_started,
                "finish_reason": finish_reason,
                "format_prefill": "json_object_open",
                "parse_input_prefill_applied": parse_prefill_applied,
                "parse_input_chars": len(parse_text),
                "parse_recovered": recovered,
                "parse_error": parse_error,
            }
        )
        self._turn_index += 1
        history.append(assistant_message)
        return assistant_message, history

    def _build_prompt(self, prompt_messages: Sequence[Mapping[str, object]]) -> str:
        history_budget = self._history_max_chars
        long_doc_query = infer_query_from_messages(
            prompt_messages,
            skip_longer_than=max(1, int(self._long_doc_config.min_long_text_chars)),
        )
        long_doc_seed = None if self._seed is None else int(self._seed) + 20_000 + self._turn_index
        compacted_messages = compact_messages_for_long_context(
            prompt_messages,
            query=long_doc_query,
            config=self._long_doc_config,
            engine=self._engine,
            sampling=self._sampling,
            progress_desc="TauOfficial-LongDoc",
            prompt_seed=long_doc_seed,
        ).messages
        facts_message = _build_tau_tool_facts_message(prompt_messages, max_chars=1100)
        facts_text = facts_message["content"] if facts_message is not None else None
        if facts_message is not None:
            compacted_messages = [*compacted_messages, facts_message]
        policy_result = compact_long_text(
            self._domain_policy,
            query=long_doc_query,
            config=self._long_doc_config,
            label="domain_policy",
            engine=self._engine,
            sampling=self._sampling,
            progress_desc="TauOfficial-Policy",
            prompt_seed=None if long_doc_seed is None else long_doc_seed + 5_000,
        )
        domain_policy = policy_result.text
        tool_route = route_tools_for_prompt(
            self._tools,
            compacted_messages,
            config=self._tool_routing_config,
            engine=self._engine,
            sampling=self._sampling,
            control_tool_names=(RESPOND_TOOL_NAME,),
            progress_desc="TauOfficial-ToolRouter",
            prompt_seed=None if self._seed is None else int(self._seed) + 10_000 + self._turn_index,
        )
        selected_tools = list(tool_route.selected_tools)
        progressive_trace: dict[str, Any] | None = None
        if self._retail_progressive_tool_disclosure:
            progressive_result = _apply_tau_retail_progressive_tool_disclosure(
                self._tools,
                selected_tools,
                prompt_messages,
                max_tools=max(1, int(self._tool_routing_config.max_tools)),
            )
            selected_tools = progressive_result.selected_tools
            progressive_trace = progressive_result.trace
        (
            prompt,
            emitted_tools,
            emitted_policy_chars,
            emitted_tool_schema_mode,
        ) = self._build_budgeted_agent_prompt(
            domain_policy=domain_policy,
            facts_text=facts_text,
            selected_tools=selected_tools,
            messages=compacted_messages,
            history_budget=history_budget,
        )
        self._current_tool_names = {_tool_name(tool) for tool in emitted_tools if _tool_name(tool)}
        route_trace = {"turn_index": self._turn_index, **tool_route.trace_payload()}
        if progressive_trace is not None:
            route_trace["retail_progressive_tool_disclosure"] = progressive_trace
        if len(emitted_tools) != len(selected_tools):
            route_trace["emitted_names"] = [_tool_name(tool) for tool in emitted_tools if _tool_name(tool)]
            route_trace["system_budget_reduced"] = True
        if emitted_tool_schema_mode != "full":
            route_trace["emitted_tool_schema_mode"] = emitted_tool_schema_mode
            route_trace["system_budget_reduced"] = True
        if emitted_policy_chars < len(domain_policy):
            route_trace["emitted_policy_chars"] = int(emitted_policy_chars)
            route_trace["system_budget_reduced"] = True
        self.tool_routes.append(route_trace)
        return prompt

    def _build_budgeted_agent_prompt(
        self,
        *,
        domain_policy: str,
        facts_text: str | None,
        selected_tools: Sequence[Any],
        messages: Sequence[Mapping[str, object]],
        history_budget: int,
    ) -> tuple[str, list[Any], int, str]:
        tools = list(selected_tools)
        policy = normalize_rwkv_text(domain_policy)
        policy_budgets = _policy_budget_candidates(policy, self._prompt_max_chars)
        tool_schema_modes = ("full", "compact", "minimal")

        best_prompt = ""
        best_policy_chars = len(policy)
        best_tool_schema_mode = "full"
        for policy_budget in policy_budgets:
            policy_view = truncate_text(policy, policy_budget)
            for tool_schema_mode in tool_schema_modes:
                system_prompt = build_tau_official_agent_system_prompt(
                    policy_view,
                    tools,
                    domain=self._domain,
                    tool_schema_mode=tool_schema_mode,
                    facts_text=facts_text,
                )
                prompt = build_rwkv_json_call_prompt(
                    system_prompt,
                    messages,
                    history_max_chars=history_budget,
                    assistant_prefix=TAU_JSON_CALL_ASSISTANT_PREFIX,
                    single_user_turn=False,
                )
                best_prompt = prompt
                best_policy_chars = len(policy_view)
                best_tool_schema_mode = tool_schema_mode
                if len(prompt) <= self._prompt_max_chars:
                    return prompt, tools, len(policy_view), tool_schema_mode
                overflow = len(prompt) - self._prompt_max_chars
                trimmed_history_budget = max(0, history_budget - overflow - 512)
                prompt = build_rwkv_json_call_prompt(
                    system_prompt,
                    trim_message_history(messages, max_chars=trimmed_history_budget),
                    history_max_chars=trimmed_history_budget,
                    assistant_prefix=TAU_JSON_CALL_ASSISTANT_PREFIX,
                    single_user_turn=False,
                )
                best_prompt = prompt
                if len(prompt) <= self._prompt_max_chars:
                    return prompt, tools, len(policy_view), tool_schema_mode

        # Last resort: keep the prompt valid and under the hard cap by trimming history
        # and then the already-compacted policy. Do not drop routed tools here:
        # the official runtime must execute against the same tool window the
        # agent saw, otherwise a correct tool choice can be rejected locally.
        if len(best_prompt) <= self._prompt_max_chars:
            return best_prompt, tools, best_policy_chars, best_tool_schema_mode
        final_system = build_tau_official_agent_system_prompt(
            truncate_text(policy, min(policy_budgets[-1], 240)),
            tools,
            domain=self._domain,
            tool_schema_mode="minimal",
            facts_text=facts_text,
        )
        final_prompt = build_rwkv_json_call_prompt(
            final_system,
            [],
            history_max_chars=0,
            assistant_prefix=TAU_JSON_CALL_ASSISTANT_PREFIX,
            single_user_turn=False,
        )
        if len(final_prompt) > self._prompt_max_chars:
            raise ValueError(
                "tau prompt budget cannot fit routed tools without corrupting the prompt: "
                f"prompt_chars={len(final_prompt)} budget={self._prompt_max_chars} tools={len(tools)}"
            )
        return final_prompt, tools, min(policy_budgets[-1], 240), "minimal"

    def _decision_to_assistant_message(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        prompt_messages: Sequence[Mapping[str, object]] = (),
    ) -> Any:
        normalized_name, arguments = _normalize_tau_decision(name, arguments)
        if normalized_name == RESPOND_TOOL_NAME:
            content = (
                arguments.get("content")
                or arguments.get("answer")
                or arguments.get("message")
                or ""
            )
            content_text = str(content).strip()
            if not content_text:
                raise ValueError("empty tau respond content")
            replacement = _tau_respond_replacement_from_context(
                content_text,
                prompt_messages,
                available_tool_names=self._current_tool_names,
            )
            if replacement is not None:
                replacement_name, replacement_arguments = replacement
                return self._decision_to_assistant_message(
                    replacement_name,
                    replacement_arguments,
                    prompt_messages=prompt_messages,
                )
            return self._AssistantMessage(role="assistant", content=content_text)
        if normalized_name not in self._tool_names:
            raise ValueError(f"unknown tau tool name: {normalized_name}")
        normalized_name, arguments = _normalize_tau_tool_decision_from_context(
            normalized_name,
            arguments,
            prompt_messages,
            available_tool_names=self._tool_names,
            retail_repeated_read_guard=self._retail_repeated_read_guard,
            retail_tool_use_guard=self._retail_tool_use_guard,
        )
        if normalized_name == RESPOND_TOOL_NAME:
            content = (
                arguments.get("content")
                or arguments.get("answer")
                or arguments.get("message")
                or ""
            )
            content_text = str(content).strip()
            if not content_text:
                raise ValueError("empty tau respond content")
            return self._AssistantMessage(role="assistant", content=content_text)
        if normalized_name not in self._tool_names:
            raise ValueError(f"unknown tau tool name: {normalized_name}")
        if normalized_name not in self._current_tool_names:
            raise ValueError(f"tau tool name not in routed tool window: {normalized_name}")
        return self._AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                self._ToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}",
                    name=normalized_name,
                    arguments=dict(arguments),
                    requestor="assistant",
                )
            ],
        )


class StaticStopTauUser:
    """Minimal no-LLM tau user for ticket-seeded lightweight tasks."""

    def __init__(self, *, stop_content: str = "###STOP###") -> None:
        self.stop_content = str(stop_content)
        message_module = import_module_with_auto_install("tau2.data_model.message", context="tau2 message import")
        base_module = import_module_with_auto_install(
            "tau2.user.user_simulator_base",
            context="tau2 user base import",
        )
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


def build_tau_official_agent_system_prompt(
    domain_policy: str,
    tools: Sequence[Any],
    *,
    domain: str | None = None,
    tool_schema_mode: str = "full",
    facts_text: str | None = None,
) -> str:
    if tool_schema_mode == "compact":
        tool_schemas = [_compact_tool_schema(_normalize_tool_schema(tool)) for tool in tools]
    elif tool_schema_mode == "minimal":
        tool_schemas = [_minimal_tool_schema(_normalize_tool_schema(tool)) for tool in tools]
    elif tool_schema_mode == "full":
        tool_schemas = [_normalize_tool_schema(tool) for tool in tools]
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
    tool_json_kwargs: dict[str, Any] = {"ensure_ascii": False, "sort_keys": False}
    if tool_schema_mode == "full":
        tool_json_kwargs["indent"] = 2
    else:
        tool_json_kwargs["separators"] = (",", ":")
    sections = [
        "You are the assistant in the official tau-bench simulation.",
        "Follow the domain policy exactly.",
        "Use a real tool call when you need information or need to change state.",
        "Use respond only when sending a message to the user.",
        "When the task is complete and no more tool calls are needed, use respond and include ###STOP### in the content.",
        "Return exactly one JSON function call object and no extra prose.",
        'JSON shape: {"name":"tool_name","arguments":{...}}',
        'The prompt already opens a ```json block and the first {; continue with "name" and output only that JSON object.',
        "Earlier transcript turns may show <tool_call> and <tool_response> wrappers. For your next decision, do not output those wrappers.",
        "Do not write analysis, markdown, <think>, </think>, or any text before the JSON object.",
        "Valid names are exactly the listed tool names plus respond.",
        "Before every tool call, verify the name appears exactly in the Tools array below.",
        "Use only exact listed tool names; if no exact tool exists, respond instead of inventing wrapper or pseudo tools.",
        "Never copy a Function output object; do not return requestor/ok/output as your decision.",
        "Never invent ids/emails/phones; missing IDs must come from user text, prior tool outputs, lookup/list/read tools, or respond.",
        "Do not repeat successful reads; use outputs.",
        "Use schema argument keys; detail tools need exact IDs, list/find names first.",
    ]
    domain_name = str(domain or "").strip().lower()
    if domain_name == "telecom":
        sections.extend(
            [
                "Telecom: never invent lookup_customer/get_customer/update_line/reset_device/troubleshoot_* tools; use listed tools only.",
                'Telecom phone actions in policy text (run_speed_test/check_status_bar/check_network_status/toggle_data/toggle_roaming) are user device steps, not JSON tool names; use respond with plain device instructions and do not say "run the tool".',
                "Telecom IDs (customer_id,line_id,device_id,bill_id,plan_id) must be copied from user/tool outputs; do not guess C/L/D/B/P IDs.",
            ]
        )
    elif domain_name == "retail":
        sections.extend(
            [
                "Retail: order status comes from get_order_details.status; never call get_order_status/lookup_order/list_order_items.",
                "Retail product lookup: use list_all_product_types then get_product_details(product_id); never call search_product/find_product/filter_products.",
                "Retail: keep leading # on order IDs; product_id/item_id/user_id/payment_method_id must come from tool outputs.",
            ]
        )
    if facts_text:
        sections.extend(
            [
                "Known facts:",
                normalize_rwkv_text(facts_text),
                "Do not ask the user for values already listed in Known facts.",
            ]
        )
    sections.extend(
        [
            "Tools:",
            json.dumps(tool_schemas, **tool_json_kwargs),
            "Policy:",
            normalize_rwkv_text(domain_policy),
        ]
    )
    return normalize_rwkv_text("\n".join(sections))


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
    deduped: list[int] = []
    for value in candidates:
        normalized = max(0, int(value))
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped or [0]


def _parse_tau_agent_decision(text: str) -> tuple[str, dict[str, Any]]:
    try:
        candidate = extract_json_call_value_text(text)
        raw_payload = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        raw_payload = _partial_tau_decision_payload(text, cause=exc)
    payload = _coerce_tau_decision_payload(raw_payload)
    return _normalize_tau_decision(str(payload["name"]).strip(), dict(payload["arguments"]))


def _apply_tau_json_object_prefill(text: str) -> str:
    return apply_json_call_object_prefill(text)


def _recover_tau_agent_decision_from_text(
    text: str,
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]]:
    normalized = normalize_rwkv_text(text)
    try:
        name, arguments = _parse_tau_agent_decision(normalized)
    except Exception:
        embedded = _recover_embedded_tau_json_call(normalized)
        if embedded is not None:
            name, arguments = embedded
        else:
            natural = _recover_natural_language_tau_decision(
                normalized,
                prompt_messages,
                available_tool_names=available_tool_names,
            )
            if natural is None:
                raise
            return natural

    alias = _recover_tau_alias_decision(
        name,
        arguments,
        prompt_messages,
        available_tool_names=available_tool_names,
    )
    if alias is not None:
        return alias
    return name, arguments


def _recover_embedded_tau_json_call(text: str) -> tuple[str, dict[str, Any]] | None:
    normalized = normalize_rwkv_text(text)
    for start in [index for index, char in enumerate(normalized) if char == "{"]:
        candidate = normalized[start:]
        end = _leading_json_object_end(candidate)
        if end is None:
            continue
        try:
            payload = json.loads(candidate[:end])
            coerced = _coerce_tau_decision_payload(payload)
        except Exception:
            continue
        return _normalize_tau_decision(str(coerced["name"]).strip(), dict(coerced["arguments"]))
    return None


def _leading_json_object_end(text: str) -> int | None:
    decoder = json.JSONDecoder()
    try:
        _value, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None
    return int(end)


def _recover_tau_alias_decision(
    name: str,
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    normalized_name = _strip_tau_requestor_prefix(name).strip()
    normalized_arguments = dict(arguments)
    if normalized_name in {"ask_user", "ask", "request_user_info"}:
        content = (
            normalized_arguments.get("content")
            or normalized_arguments.get("message")
            or normalized_arguments.get("question")
            or _missing_identifier_question(prompt_messages)
            or "Could you provide the missing information so I can continue?"
        )
        return RESPOND_TOOL_NAME, {"content": str(content).strip()}
    if normalized_name in {"search_reservation", "find_reservation", "lookup_reservation", "get_booking"}:
        if "get_reservation_details" not in available_tool_names:
            return None
        requested_id = _requested_tau_reservation_id_from_user(prompt_messages)
        if requested_id:
            return "get_reservation_details", {"reservation_id": requested_id}
        return RESPOND_TOOL_NAME, {"content": "Could you provide your reservation ID so I can look it up?"}
    if normalized_name in {"search_order", "find_order", "lookup_order"}:
        if "get_order_details" not in available_tool_names:
            return None
        requested_order_id = _requested_tau_retail_order_id_from_user(prompt_messages)
        raw_order_id = str(normalized_arguments.get("order_id") or "").strip()
        if requested_order_id:
            return "get_order_details", {"order_id": requested_order_id}
        if raw_order_id:
            return "get_order_details", {"order_id": raw_order_id}
        return RESPOND_TOOL_NAME, {"content": "Could you provide the order ID so I can look it up?"}
    if normalized_name in {"search_product", "find_product", "filter_products", "lookup_product"}:
        recovered_product = _recover_tau_retail_product_search_decision(
            normalized_name,
            prompt_messages,
            available_tool_names=available_tool_names,
        )
        if recovered_product is not None:
            return recovered_product
        if "list_all_product_types" in available_tool_names:
            return "list_all_product_types", {}
    return None


def _recover_natural_language_tau_decision(
    text: str,
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    normalized = normalize_rwkv_text(text)
    lowered = normalized.lower()
    requested_reservation_id = _requested_tau_reservation_id_from_user(prompt_messages)
    if (
        "get_reservation_details" in available_tool_names
        and requested_reservation_id
        and (
            "get_reservation_details" in lowered
            or "reservation" in lowered
            or "booking" in lowered
            or "cancel" in lowered
        )
    ):
        return "get_reservation_details", {"reservation_id": requested_reservation_id}

    requested_order_id = _requested_tau_retail_order_id_from_user(prompt_messages)
    if "get_order_details" in available_tool_names and requested_order_id and ("order" in lowered or "get_order_details" in lowered):
        return "get_order_details", {"order_id": requested_order_id}

    retail_product_lookup = _recover_tau_retail_product_search_decision(
        normalized,
        prompt_messages,
        available_tool_names=available_tool_names,
    )
    if retail_product_lookup is not None:
        return retail_product_lookup

    user_id = _latest_tau_fact_value(prompt_messages, "user_id") or _first_regex_value(_TAU_USER_ID_RE, normalized)
    if "get_user_details" in available_tool_names and user_id and "get_user_details" in lowered:
        return "get_user_details", {"user_id": user_id}

    missing_question = _missing_identifier_question(prompt_messages, generated_text=normalized)
    if missing_question is not None:
        return RESPOND_TOOL_NAME, {"content": missing_question}
    return None


def _missing_identifier_question(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    generated_text: str = "",
) -> str | None:
    text = normalize_rwkv_text("\n".join([_tau_user_request_text(prompt_messages), generated_text])).lower()
    if re.search(r"\b(?:book|booking|reserve|reservation)\b", text) and re.search(
        r"\b(?:new|one-way|round trip|round-trip|from .+ to .+|for \d+ passengers?)\b",
        text,
    ):
        return "Could you provide your user ID so I can continue with the booking?"
    if "reservation" in text or "booking" in text or "flight" in text:
        if not _requested_tau_reservation_id_from_user(prompt_messages):
            return "Could you provide your reservation ID so I can look up the booking?"
    if "order" in text or "return" in text or "exchange" in text or "cancel" in text:
        if not _requested_tau_retail_order_id_from_user(prompt_messages):
            return "Could you provide the order ID so I can look it up?"
    if "user id" in text or "user_id" in text:
        return "Could you provide your user ID so I can continue?"
    if "email" in text:
        return "Could you provide the email address on your account so I can look it up?"
    return None


def _recover_tau_retail_product_search_decision(
    text: str,
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if not _available_tau_retail_tools(available_tool_names):
        return None
    normalized = normalize_rwkv_text(text)
    lowered = normalized.lower()
    mentions_product = bool(_TAU_RETAIL_PRODUCT_INTENT_RE.search(normalized)) or any(
        phrase in lowered
        for phrase in (
            "product type",
            "product category",
            "product catalog",
            "similar one",
            "similar product",
        )
    )
    if not mentions_product:
        return None

    if "get_product_details" in available_tool_names and _has_successful_tau_tool_name(
        "list_all_product_types",
        prompt_messages,
    ):
        catalog_product_id = _requested_tau_retail_product_id_from_catalog(prompt_messages)
        if catalog_product_id:
            return "get_product_details", {"product_id": catalog_product_id}

    if "list_all_product_types" not in available_tool_names:
        return None
    if any(
        phrase in lowered
        for phrase in (
            "search for product",
            "search products",
            "find a product",
            "find products",
            "look up product",
            "list product",
            "product type",
            "product category",
            "product catalog",
            "available product",
        )
    ):
        return "list_all_product_types", {}
    return None


def _first_regex_value(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return str(match.group(0)).strip()


def _partial_tau_decision_payload(text: str, *, cause: Exception) -> dict[str, Any]:
    """Recover a complete name/arguments pair from a runaway JSON object.

    RWKV sometimes emits a valid function call and then continues with an
    unfinished OpenAI-style id field. The tool execution contract only needs
    name and arguments; accepting those complete fields avoids turning a valid
    tool choice into a parse failure.
    """
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
        arguments = _partial_tau_arguments_payload(body)
    if arguments is None:
        arguments = {}
    return {"name": name, "arguments": arguments}


def _partial_tau_arguments_payload(body: str) -> dict[str, Any]:
    """Return an empty argument object when only the tool name is recoverable.

    Some RWKV generations emit a valid name and then truncate inside the
    arguments field. Keeping the recoverable tool name lets the downstream
    context normalizers fill exact IDs from previous tool outputs when they
    already have that information; otherwise the real environment still sees a
    malformed/empty call instead of a parser-level stop.
    """
    return {}


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


def _normalize_tau_decision(name: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized_name = _strip_tau_requestor_prefix(name)
    return normalized_name, dict(arguments)


_TAU_READ_TOOL_NAMES = {
    "get_customer_by_id",
    "get_customer_by_name",
    "get_customer_by_phone",
    "get_details_by_id",
    "get_flight_status",
    "get_order_details",
    "get_reservation_details",
    "get_user_details",
}
_TAU_USER_ID_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z][a-z0-9]*_\d+\b", re.IGNORECASE)
_TAU_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_TAU_GENERIC_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{3,}\b")
_TAU_FLIGHT_NUMBER_RE = re.compile(r"\b[A-Z]{2,4}\d{2,4}\b")
_TAU_HASH_ID_RE = re.compile(r"#[A-Za-z0-9][A-Za-z0-9_-]*")
_TAU_RETAIL_ORDER_ID_RE = re.compile(r"#?[A-Z]\d{7,}\b", re.IGNORECASE)
_TAU_NUMERIC_ID_RE = re.compile(r"\d{6,}")
_TAU_EXISTING_RESERVATION_ACTION_RE = re.compile(
    r"\b(?:cancel|canceling|cancelling|canceled|cancelled|cancellation|refund|change|reschedule|move|upgrade|"
    r"baggage|bag|bags|suitcase|suitcases|luggage|passenger|date of birth|dob|existing reservation)\b",
    re.IGNORECASE,
)
_TAU_RETAIL_EXCHANGE_INTENT_RE = re.compile(
    r"\b(?:exchange|replace|replacement|swap|change|different|wrong|size|color|colour|variant)\b",
    re.IGNORECASE,
)
_TAU_RETAIL_RETURN_INTENT_RE = re.compile(r"\b(?:return|refund)\b", re.IGNORECASE)
_TAU_RETAIL_CANCEL_INTENT_RE = re.compile(
    r"\b(?:cancel|cancellation|ordered by mistake|no longer needed)\b",
    re.IGNORECASE,
)
_TAU_RETAIL_ADDRESS_INTENT_RE = re.compile(
    r"\b(?:address|shipping|ship to|delivery address|mailing address|street|zip)\b",
    re.IGNORECASE,
)
_TAU_RETAIL_PAYMENT_INTENT_RE = re.compile(
    r"\b(?:payment|pay|card|credit card|gift card|paypal)\b",
    re.IGNORECASE,
)
_TAU_RETAIL_PRODUCT_INTENT_RE = re.compile(
    r"\b(?:product|item|option|variant|inventory|available|availability|size|color|colour|shirt|t-shirt|"
    r"tshirt|headphone|watch|keyboard|thermostat|camera|cleaner|vacuum)\b",
    re.IGNORECASE,
)
_TAU_RETAIL_BOOTSTRAP_TOOLS = (
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "list_all_product_types",
    "calculate",
    "transfer_to_human_agents",
)
_TAU_RETAIL_ORDER_WRITE_TOOLS = (
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
)
_TAU_RETAIL_DETAIL_TOOLS = ("get_product_details", "get_item_details")
_TAU_FACT_KEYS = {
    "account_status",
    "bill_ids",
    "customer_id",
    "date_of_birth",
    "dob",
    "email",
    "full_name",
    "item_id",
    "item_ids",
    "line_id",
    "line_ids",
    "order_id",
    "orders",
    "payment_method_id",
    "payment_methods",
    "phone_number",
    "product_id",
    "reservation_id",
    "reservations",
    "status",
    "user_id",
}


def _normalize_tau_tool_decision_from_context(
    name: str,
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
    retail_repeated_read_guard: bool = False,
    retail_tool_use_guard: bool = False,
) -> tuple[str, dict[str, Any]]:
    normalized_name = str(name or "").strip()
    normalized_arguments = dict(arguments)
    normalized_arguments = _preserve_hash_prefixed_ids_from_user_context(normalized_arguments, prompt_messages)
    if retail_tool_use_guard:
        retail_replacement = _tau_retail_tool_use_replacement(
            normalized_name,
            normalized_arguments,
            prompt_messages,
            available_tool_names=available_tool_names,
        )
        if retail_replacement is not None:
            normalized_name, normalized_arguments = retail_replacement
    cancel_replacement = _tau_cancel_intent_replacement_from_context(
        normalized_name,
        prompt_messages,
        available_tool_names=available_tool_names,
    )
    if cancel_replacement is not None:
        return cancel_replacement
    if normalized_name == "get_user_details":
        known_user_id = _latest_tau_fact_value(prompt_messages, "user_id")
        raw_user_id = str(normalized_arguments.get("user_id") or "").strip()
        if known_user_id and (raw_user_id != known_user_id or not _TAU_USER_ID_RE.fullmatch(raw_user_id)):
            normalized_arguments["user_id"] = known_user_id
    elif normalized_name in {"get_reservation_details", "cancel_reservation"}:
        raw_reservation_id = str(normalized_arguments.get("reservation_id") or "").strip().upper()
        if not raw_reservation_id or _TAU_USER_ID_RE.fullmatch(raw_reservation_id):
            requested_id = _requested_tau_reservation_id_from_user(prompt_messages)
            if requested_id:
                normalized_arguments["reservation_id"] = requested_id

    if normalized_name in _TAU_READ_TOOL_NAMES:
        has_same_observation = _has_successful_tau_tool_observation(
            normalized_name,
            normalized_arguments,
            prompt_messages,
        )
        has_prior_user_profile = (
            normalized_name == "get_user_details"
            and _has_successful_tau_tool_name("get_user_details", prompt_messages)
        )
        if has_same_observation or has_prior_user_profile:
            if retail_repeated_read_guard:
                guard_response = _tau_repeated_retail_read_guard_response(
                    normalized_name,
                    normalized_arguments,
                    prompt_messages,
                    available_tool_names=available_tool_names,
                )
                if guard_response is not None:
                    return guard_response
            context_response = _tau_readonly_reservation_response_from_context(prompt_messages)
            if context_response is not None:
                return context_response
            direct_action = _tau_direct_requested_reservation_action_from_context(
                prompt_messages,
                available_tool_names=available_tool_names,
            )
            if direct_action is not None:
                return direct_action
            replacement = _replacement_for_repeated_tau_read(
                normalized_name,
                normalized_arguments,
                prompt_messages,
                available_tool_names=available_tool_names,
            )
            if replacement is not None:
                return replacement
            exhausted_response = _tau_exhausted_repeated_read_response(normalized_name, prompt_messages)
            if exhausted_response is not None:
                return exhausted_response
    return normalized_name, normalized_arguments


def _tau_respond_replacement_from_context(
    content: str,
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    normalized_content = normalize_rwkv_text(content)
    lowered_content = normalized_content.lower()
    if "get_user_details" in available_tool_names and re.search(r"\buser(?:\s+|_)id\b", lowered_content):
        known_user_id = _latest_tau_fact_value(prompt_messages, "user_id")
        if known_user_id:
            return "get_user_details", {"user_id": known_user_id}

    if re.search(r"\border(?:\s+|_)id\b", lowered_content):
        requested_order_id = _requested_tau_retail_order_id_from_user(prompt_messages)
        if (
            "get_order_details" in available_tool_names
            and requested_order_id
            and not _has_successful_tau_tool_name("get_order_details", prompt_messages)
        ):
            return "get_order_details", {"order_id": requested_order_id}
        retail_lookup = _tau_retail_identity_or_order_lookup_replacement(
            prompt_messages,
            available_tool_names=available_tool_names,
        )
        if retail_lookup is not None:
            return retail_lookup

    if (
        "list_all_product_types" in available_tool_names
        and re.search(r"\border(?:\s+|_)id\b", lowered_content)
        and not _requested_tau_retail_order_id_from_user(prompt_messages)
        and not _has_successful_tau_tool_name("list_all_product_types", prompt_messages)
    ):
        user_text = _tau_user_request_text(prompt_messages)
        if _TAU_RETAIL_PRODUCT_INTENT_RE.search(user_text):
            return "list_all_product_types", {}
    return None


def _tau_retail_tool_use_replacement(
    name: str,
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if not _available_tau_retail_tools(available_tool_names):
        return None
    if name in {"get_order_details", "get_user_details", "get_product_details", "list_all_product_types"}:
        action_recovery = _tau_retail_action_or_confirmation_from_context(
            prompt_messages,
            available_tool_names=available_tool_names,
        )
        if action_recovery is not None:
            return action_recovery
    if name == "get_product_details" and "get_product_details" in available_tool_names:
        product_id = str(arguments.get("product_id") or "").strip()
        if _TAU_NUMERIC_ID_RE.fullmatch(product_id):
            parent_product_id = _tau_retail_product_id_for_item_id_from_context(prompt_messages, product_id)
            if parent_product_id and parent_product_id != product_id:
                product_id = parent_product_id
            if _has_successful_tau_tool_observation("get_product_details", {"product_id": product_id}, prompt_messages):
                next_product_id = _next_uninspected_tau_retail_product_id_from_catalog(prompt_messages)
                if next_product_id and next_product_id != product_id:
                    return "get_product_details", {"product_id": next_product_id}
                known_user_id = _latest_tau_fact_value(prompt_messages, "user_id")
                if (
                    known_user_id
                    and "get_user_details" in available_tool_names
                    and not _has_successful_tau_tool_name("get_user_details", prompt_messages)
                ):
                    return "get_user_details", {"user_id": known_user_id}
                return _tau_repeated_retail_product_response(prompt_messages, product_id)
            if parent_product_id and parent_product_id != str(arguments.get("product_id") or "").strip():
                return "get_product_details", {"product_id": parent_product_id}
        elif "list_all_product_types" in available_tool_names:
            catalog_product_id = _requested_tau_retail_product_id_from_catalog(prompt_messages)
            if catalog_product_id and "get_product_details" in available_tool_names:
                return "get_product_details", {"product_id": catalog_product_id}
            return "list_all_product_types", {}
    if name == "list_all_product_types" and "get_product_details" in available_tool_names:
        if _has_successful_tau_tool_name("list_all_product_types", prompt_messages):
            catalog_product_id = _requested_tau_retail_product_id_from_catalog(prompt_messages)
            if catalog_product_id:
                return "get_product_details", {"product_id": catalog_product_id}
    if name == "get_order_details":
        order_id = str(arguments.get("order_id") or "").strip()
        if _TAU_RETAIL_ORDER_ID_RE.fullmatch(order_id):
            return None
        requested_order_id = _requested_tau_retail_order_id_from_user(prompt_messages)
        if requested_order_id:
            return "get_order_details", {"order_id": requested_order_id}
        retail_lookup = _tau_retail_identity_or_order_lookup_replacement(
            prompt_messages,
            available_tool_names=available_tool_names,
        )
        if retail_lookup is not None:
            return retail_lookup
        if (
            "list_all_product_types" in available_tool_names
            and _TAU_RETAIL_PRODUCT_INTENT_RE.search(_tau_user_request_text(prompt_messages))
            and not _has_successful_tau_tool_name("list_all_product_types", prompt_messages)
        ):
            return "list_all_product_types", {}
        return RESPOND_TOOL_NAME, {
            "content": (
                "I need a valid order ID, or your email/name and ZIP code, before I can look up the order. "
                "I can also check product availability if you tell me the product names."
            )
        }
    return None


def _requested_tau_retail_product_id_from_catalog(prompt_messages: Sequence[Mapping[str, object]]) -> str | None:
    user_text = _tau_user_request_text(prompt_messages).lower()
    if not user_text:
        return None
    best: tuple[int, str] | None = None
    for tool_name, _args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if tool_name != "list_all_product_types" or not ok or not isinstance(output, Mapping):
            continue
        for raw_name, raw_product_id in output.items():
            product_name = str(raw_name or "").strip()
            product_id = str(raw_product_id or "").strip()
            if not product_name or not _TAU_NUMERIC_ID_RE.fullmatch(product_id):
                continue
            position = _tau_retail_product_name_position(product_name, user_text)
            if position is not None and (best is None or position < best[0]):
                best = (position, product_id)
        if best is not None:
            return best[1]
    return None


def _next_uninspected_tau_retail_product_id_from_catalog(
    prompt_messages: Sequence[Mapping[str, object]],
) -> str | None:
    user_text = _tau_user_request_text(prompt_messages).lower()
    if not user_text:
        return None
    observed_ids = _successful_tau_retail_product_ids(prompt_messages)
    candidates: list[tuple[int, str]] = []
    for tool_name, _args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if tool_name != "list_all_product_types" or not ok or not isinstance(output, Mapping):
            continue
        for raw_name, raw_product_id in output.items():
            product_name = str(raw_name or "").strip()
            product_id = str(raw_product_id or "").strip()
            if not product_name or not _TAU_NUMERIC_ID_RE.fullmatch(product_id) or product_id in observed_ids:
                continue
            position = _tau_retail_product_name_position(product_name, user_text)
            if position is not None:
                candidates.append((position, product_id))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
    return None


def _successful_tau_retail_product_ids(prompt_messages: Sequence[Mapping[str, object]]) -> set[str]:
    product_ids: set[str] = set()
    for tool_name, args, ok, output in _iter_tau_tool_observations(prompt_messages):
        if tool_name != "get_product_details" or not ok:
            continue
        product_id = ""
        if isinstance(output, Mapping):
            product_id = str(output.get("product_id") or "").strip()
        if not product_id:
            product_id = str(args.get("product_id") or "").strip()
        if _TAU_NUMERIC_ID_RE.fullmatch(product_id):
            product_ids.add(product_id)
    return product_ids


def _tau_retail_product_id_for_item_id_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
    item_id: str,
) -> str | None:
    target_item_id = str(item_id or "").strip()
    if not _TAU_NUMERIC_ID_RE.fullmatch(target_item_id):
        return None
    for tool_name, args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if not ok or not isinstance(output, Mapping):
            continue
        if tool_name == "get_product_details":
            product_id = str(output.get("product_id") or args.get("product_id") or "").strip()
            if _TAU_NUMERIC_ID_RE.fullmatch(product_id) and _tau_retail_variants_include_item_id(
                output.get("variants"),
                target_item_id,
            ):
                return product_id
        if tool_name == "get_order_details":
            product_id = _tau_retail_order_product_id_for_item_id(output, target_item_id)
            if product_id:
                return product_id
    return None


def _tau_retail_order_product_id_for_item_id(output: Mapping[str, Any], item_id: str) -> str | None:
    items = output.get("items")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, Mapping):
            continue
        observed_item_id = str(item.get("item_id") or "").strip()
        product_id = str(item.get("product_id") or "").strip()
        if observed_item_id == item_id and _TAU_NUMERIC_ID_RE.fullmatch(product_id):
            return product_id
    return None


def _tau_retail_variants_include_item_id(variants: Any, item_id: str) -> bool:
    if isinstance(variants, Mapping):
        for raw_variant_id, raw_variant in variants.items():
            if str(raw_variant_id or "").strip() == item_id:
                return True
            if isinstance(raw_variant, Mapping) and str(raw_variant.get("item_id") or "").strip() == item_id:
                return True
        return False
    if isinstance(variants, list):
        for raw_variant in variants:
            if isinstance(raw_variant, Mapping) and str(raw_variant.get("item_id") or "").strip() == item_id:
                return True
    return False


def _tau_repeated_retail_product_response(
    prompt_messages: Sequence[Mapping[str, object]],
    product_id: str,
) -> tuple[str, dict[str, Any]]:
    if _TAU_RETAIL_EXCHANGE_INTENT_RE.search(_tau_user_request_text(prompt_messages)):
        content = (
            "I already have the product variants for that item. Please confirm the exact exchange item "
            "and payment method before I proceed."
        )
    elif _TAU_RETAIL_RETURN_INTENT_RE.search(_tau_user_request_text(prompt_messages)):
        content = "I already have those product details. Please provide the order ID or confirm the return items."
    else:
        content = f"I already have the product details for product {product_id}. ###STOP###"
    return RESPOND_TOOL_NAME, {"content": content}


def _tau_retail_identity_or_order_lookup_replacement(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if _requested_tau_retail_order_id_from_user(prompt_messages):
        return None
    identity_lookup = _tau_retail_identity_lookup_from_user(
        prompt_messages,
        available_tool_names=available_tool_names,
    )
    if identity_lookup is not None:
        return identity_lookup

    known_user_id = _latest_tau_fact_value(prompt_messages, "user_id")
    if (
        known_user_id
        and "get_user_details" in available_tool_names
        and not _has_successful_tau_tool_name("get_user_details", prompt_messages)
    ):
        return "get_user_details", {"user_id": known_user_id}

    next_order_id = _next_uninspected_tau_retail_order_id(prompt_messages)
    if next_order_id and "get_order_details" in available_tool_names:
        return "get_order_details", {"order_id": next_order_id}
    return None


def _tau_retail_identity_lookup_from_user(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    user_text = _tau_user_request_text(prompt_messages)
    if "find_user_id_by_email" in available_tool_names:
        email_match = _TAU_EMAIL_RE.search(user_text)
        if email_match is not None:
            return "find_user_id_by_email", {"email": email_match.group(0)}
    if "find_user_id_by_name_zip" in available_tool_names:
        identity = _requested_tau_retail_name_zip_from_user(user_text)
        if identity is not None:
            return "find_user_id_by_name_zip", identity
    return None


def _requested_tau_retail_name_zip_from_user(user_text: str) -> dict[str, str] | None:
    zip_match = re.search(r"\b\d{5}(?:-\d{4})?\b", user_text)
    if zip_match is None:
        return None
    zip_code = zip_match.group(0)
    name_patterns = (
        r"\b(?:you are|i am|i'm|my name is|name is)\s+([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)\b",
        r"\b([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)\s+(?:in|at|with)\s+(?:zip|zip code|postal code)\b",
    )
    for pattern in name_patterns:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match is not None:
            return {
                "first_name": _title_name_part(match.group(1)),
                "last_name": _title_name_part(match.group(2)),
                "zip": zip_code,
            }

    prefix = user_text[max(0, zip_match.start() - 100) : zip_match.start()]
    name_matches = re.findall(r"\b([A-Z][A-Za-z'-]+)\s+([A-Z][A-Za-z'-]+)\b", prefix)
    for first_name, last_name in reversed(name_matches):
        if first_name.lower() in {"order", "email", "name", "code", "zip"}:
            continue
        return {"first_name": _title_name_part(first_name), "last_name": _title_name_part(last_name), "zip": zip_code}
    return None


def _title_name_part(value: str) -> str:
    text = str(value or "").strip()
    return text[:1].upper() + text[1:].lower() if text else text


def _tau_retail_action_or_confirmation_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    user_text = _tau_user_request_text(prompt_messages)
    if _TAU_RETAIL_EXCHANGE_INTENT_RE.search(user_text) and "exchange_delivered_order_items" in available_tool_names:
        args = _tau_retail_exchange_write_args_from_context(prompt_messages)
        if args is not None:
            return _tau_retail_confirm_or_write(
                "exchange_delivered_order_items",
                args,
                prompt_messages,
                label="exchange",
            )
    if _TAU_RETAIL_RETURN_INTENT_RE.search(user_text) and "return_delivered_order_items" in available_tool_names:
        args = _tau_retail_return_write_args_from_context(prompt_messages)
        if args is not None:
            return _tau_retail_confirm_or_write(
                "return_delivered_order_items",
                args,
                prompt_messages,
                label="return",
            )
    if _TAU_RETAIL_CANCEL_INTENT_RE.search(user_text) and "cancel_pending_order" in available_tool_names:
        args = _tau_retail_cancel_write_args_from_context(prompt_messages)
        if args is not None:
            return _tau_retail_confirm_or_write(
                "cancel_pending_order",
                args,
                prompt_messages,
                label="cancel",
            )
    return None


def _tau_retail_confirm_or_write(
    tool_name: str,
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> tuple[str, dict[str, Any]]:
    args = dict(arguments)
    if _tau_latest_user_confirms(prompt_messages):
        return tool_name, args
    return RESPOND_TOOL_NAME, {"content": _tau_retail_confirmation_message(label=label, arguments=args)}


def _tau_latest_user_confirms(prompt_messages: Sequence[Mapping[str, object]]) -> bool:
    for message in reversed(prompt_messages):
        role = _normalized_tau_message_role(message.get("role"))
        content = str(message.get("content") or "").strip()
        if not content or "Function output:" in content or content.startswith("Known facts"):
            continue
        if role == "assistant":
            return False
        if role != "user":
            continue
        lowered = normalize_rwkv_text(content).lower()
        if re.search(r"\b(?:yes|yeah|yep|confirm|confirmed|proceed|go ahead|do it|please proceed)\b", lowered):
            return True
        if re.search(r"\b(?:no|stop|do not|don't|cancel that)\b", lowered):
            return False
        return False
    return False


def _tau_retail_confirmation_message(*, label: str, arguments: Mapping[str, Any]) -> str:
    compact_args = json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if label == "exchange":
        return f"Please confirm this exchange before I proceed: {compact_args}"
    if label == "return":
        return f"Please confirm this return before I proceed: {compact_args}"
    if label == "cancel":
        return f"Please confirm this cancellation before I proceed: {compact_args}"
    return f"Please confirm before I proceed: {compact_args}"


def _tau_retail_exchange_write_args_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
) -> dict[str, Any] | None:
    order = _latest_tau_retail_action_order(prompt_messages)
    if not order or str(order.get("status") or "").strip().lower() != "delivered":
        return None
    order_id = str(order.get("order_id") or "").strip()
    if not order_id:
        return None
    requested_items = _requested_tau_retail_order_items(order, _tau_user_request_text(prompt_messages))
    if not requested_items:
        return None
    item_ids: list[str] = []
    new_item_ids: list[str] = []
    user_text = _tau_user_request_text(prompt_messages)
    for item in requested_items:
        item_id = str(item.get("item_id") or "").strip()
        product_id = str(item.get("product_id") or "").strip()
        if not item_id or not product_id:
            return None
        product = _latest_successful_tau_retail_product_observation(prompt_messages, product_id=product_id)
        if product is None:
            return None
        new_item_id = _tau_retail_best_exchange_variant_id(product, current_item_id=item_id, user_text=user_text)
        if not new_item_id:
            return None
        item_ids.append(item_id)
        new_item_ids.append(new_item_id)
    payment_method_id = _tau_retail_payment_method_id_from_context(prompt_messages, order=order)
    if not payment_method_id:
        return None
    return {
        "order_id": order_id,
        "item_ids": item_ids,
        "new_item_ids": new_item_ids,
        "payment_method_id": payment_method_id,
    }


def _tau_retail_return_write_args_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
) -> dict[str, Any] | None:
    order = _latest_tau_retail_action_order(prompt_messages)
    if not order or str(order.get("status") or "").strip().lower() != "delivered":
        return None
    order_id = str(order.get("order_id") or "").strip()
    if not order_id:
        return None
    items = _requested_tau_retail_order_items(
        order,
        _tau_user_request_text(prompt_messages),
        all_items_when_unspecified=True,
    )
    item_ids = [str(item.get("item_id") or "").strip() for item in items if isinstance(item, Mapping)]
    item_ids = [item_id for item_id in item_ids if item_id]
    payment_method_id = _tau_retail_payment_method_id_from_context(prompt_messages, order=order)
    if not item_ids or not payment_method_id:
        return None
    return {"order_id": order_id, "item_ids": item_ids, "payment_method_id": payment_method_id}


def _tau_retail_cancel_write_args_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
) -> dict[str, Any] | None:
    order = _latest_tau_retail_action_order(prompt_messages)
    if not order or str(order.get("status") or "").strip().lower() != "pending":
        return None
    order_id = str(order.get("order_id") or "").strip()
    if not order_id:
        return None
    user_text = _tau_user_request_text(prompt_messages).lower()
    if "ordered by mistake" in user_text or "mistake" in user_text:
        reason = "ordered by mistake"
    elif "no longer needed" in user_text or "don't need" in user_text or "do not need" in user_text:
        reason = "no longer needed"
    else:
        return None
    return {"order_id": order_id, "reason": reason}


def _tau_retail_product_name_position(product_name: str, user_text_lower: str) -> int | None:
    normalized_name = product_name.lower().strip()
    candidates = {
        normalized_name,
        normalized_name.replace("-", " "),
        normalized_name.replace(" ", "-"),
        normalized_name.replace("-", "").replace(" ", ""),
    }
    if normalized_name.endswith("s"):
        singular = normalized_name[:-1]
        candidates.update(
            {
                singular,
                singular.replace("-", " "),
                singular.replace(" ", "-"),
                singular.replace("-", "").replace(" ", ""),
            }
        )
    positions = [user_text_lower.find(candidate) for candidate in candidates if candidate and candidate in user_text_lower]
    compact_user_text = user_text_lower.replace("-", "").replace(" ", "")
    positions.extend(
        compact_user_text.find(candidate)
        for candidate in candidates
        if candidate and candidate in compact_user_text
    )
    return min(positions) if positions else None


def _requested_tau_retail_order_id_from_user(prompt_messages: Sequence[Mapping[str, object]]) -> str | None:
    user_text = _tau_user_request_text(prompt_messages)
    matches = [
        value if value.startswith("#") else f"#{value}"
        for match in _TAU_RETAIL_ORDER_ID_RE.finditer(user_text)
        for value in [match.group(0).upper()]
    ]
    deduped: list[str] = []
    for value in matches:
        normalized = value.upper()
        if normalized not in {existing.upper() for existing in deduped}:
            deduped.append(value)
    if len(deduped) == 1:
        return deduped[0]
    return None


def _latest_tau_retail_action_order(prompt_messages: Sequence[Mapping[str, object]]) -> Mapping[str, Any] | None:
    requested_order_id = _requested_tau_retail_order_id_from_user(prompt_messages)
    order = _latest_successful_tau_retail_order_observation(prompt_messages, order_id=requested_order_id)
    if order is not None:
        return order
    for tool_name, _args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if tool_name == "get_order_details" and ok and isinstance(output, Mapping):
            return output
    return None


def _tau_retail_order_ids_from_context(prompt_messages: Sequence[Mapping[str, object]]) -> list[str]:
    candidates: list[str] = []
    requested_order_id = _requested_tau_retail_order_id_from_user(prompt_messages)
    if requested_order_id:
        candidates.append(requested_order_id)
    for tool_name, _args, ok, output in _iter_tau_tool_observations(prompt_messages):
        if not ok or not isinstance(output, Mapping):
            continue
        if tool_name == "get_order_details":
            order_id = str(output.get("order_id") or "").strip()
            if order_id:
                candidates.append(order_id)
        elif tool_name == "get_user_details":
            orders = output.get("orders")
            if isinstance(orders, (list, tuple)):
                candidates.extend(str(order_id).strip() for order_id in orders if str(order_id).strip())
    return _dedupe_tau_retail_order_ids(candidates)


def _next_uninspected_tau_retail_order_id(prompt_messages: Sequence[Mapping[str, object]]) -> str | None:
    candidates = _tau_retail_order_ids_from_context(prompt_messages)
    if not candidates:
        return None
    inspected = {
        _normalize_tau_retail_order_id(str(args.get("order_id") or "").strip())
        for tool_name, args in _iter_prior_tau_tool_calls(prompt_messages)
        if tool_name == "get_order_details"
    }
    for order_id in candidates:
        normalized = _normalize_tau_retail_order_id(order_id)
        if normalized and normalized not in inspected:
            return order_id
    return None


def _dedupe_tau_retail_order_ids(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = _normalize_tau_retail_order_id(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(value if str(value).strip().startswith("#") else normalized)
    return out


def _normalize_tau_retail_order_id(value: str) -> str:
    text = str(value or "").strip().upper()
    if text and not text.startswith("#") and _TAU_RETAIL_ORDER_ID_RE.fullmatch(text):
        text = f"#{text}"
    return text


def _requested_tau_retail_order_items(
    order: Mapping[str, Any],
    user_text: str,
    *,
    all_items_when_unspecified: bool = False,
) -> list[Mapping[str, Any]]:
    raw_items = order.get("items")
    if not isinstance(raw_items, list):
        return []
    items = [item for item in raw_items if isinstance(item, Mapping)]
    if not items:
        return []
    text = str(user_text or "").lower()
    if all_items_when_unspecified and re.search(r"\b(?:all|everything|all things|all items)\b", text):
        return items
    matches: list[Mapping[str, Any]] = []
    for item in items:
        name = str(item.get("name") or "").strip()
        if name and _tau_retail_product_name_position(name, text) is not None:
            matches.append(item)
    if matches:
        return matches
    if len(items) == 1:
        return items
    return items if all_items_when_unspecified else []


def _latest_successful_tau_retail_product_observation(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    product_id: str,
) -> Mapping[str, Any] | None:
    requested = str(product_id or "").strip()
    if not requested:
        return None
    for tool_name, args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if tool_name != "get_product_details" or not ok or not isinstance(output, Mapping):
            continue
        observed_id = str(output.get("product_id") or args.get("product_id") or "").strip()
        if observed_id == requested:
            return output
    return None


def _tau_retail_best_exchange_variant_id(
    product: Mapping[str, Any],
    *,
    current_item_id: str,
    user_text: str,
) -> str | None:
    variants = product.get("variants")
    if not isinstance(variants, Mapping):
        return None
    candidates: list[tuple[int, str]] = []
    fallback: list[str] = []
    current = str(current_item_id or "").strip()
    user_tokens = _tau_retail_relevant_tokens(user_text)
    for raw_variant_id, raw_variant in variants.items():
        variant_id = str(raw_variant_id or "").strip()
        if not variant_id or variant_id == current or not isinstance(raw_variant, Mapping):
            continue
        item_id = str(raw_variant.get("item_id") or variant_id).strip()
        if item_id == current:
            continue
        if raw_variant.get("available") is False:
            continue
        fallback.append(item_id)
        options = raw_variant.get("options")
        option_text = ""
        if isinstance(options, Mapping):
            option_text = " ".join(
                f"{str(key).lower()} {str(value).lower()}" for key, value in options.items()
            )
        score = sum(1 for token in user_tokens if token in option_text)
        if score > 0:
            candidates.append((score, item_id))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][1]
    return fallback[0] if len(fallback) == 1 else None


def _tau_retail_relevant_tokens(text: str) -> set[str]:
    stop = {
        "order",
        "item",
        "items",
        "product",
        "products",
        "exchange",
        "return",
        "refund",
        "similar",
        "with",
        "instead",
        "compatible",
    }
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", str(text or ""))
        if len(token) >= 3 and token.lower() not in stop
    }
    return tokens


def _tau_retail_payment_method_id_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    order: Mapping[str, Any] | None = None,
) -> str | None:
    if order is not None:
        payment_id = _tau_retail_payment_method_id_from_order(order)
        if payment_id:
            return payment_id
    for tool_name, _args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if not ok or not isinstance(output, Mapping):
            continue
        if tool_name == "get_order_details":
            payment_id = _tau_retail_payment_method_id_from_order(output)
            if payment_id:
                return payment_id
        if tool_name == "get_user_details":
            payment_methods = output.get("payment_methods")
            if isinstance(payment_methods, Mapping):
                for payment_id in payment_methods:
                    text = str(payment_id or "").strip()
                    if text:
                        return text
    payment_fact = _latest_tau_fact_value(prompt_messages, "payment_method_id")
    if payment_fact:
        return payment_fact
    payment_methods_fact = _latest_tau_fact_value(prompt_messages, "payment_methods")
    if payment_methods_fact:
        return payment_methods_fact.split(",", 1)[0].strip()
    return None


def _tau_retail_payment_method_id_from_order(order: Mapping[str, Any]) -> str | None:
    payment_history = order.get("payment_history")
    if not isinstance(payment_history, list):
        return None
    for payment in payment_history:
        if not isinstance(payment, Mapping):
            continue
        payment_id = str(payment.get("payment_method_id") or "").strip()
        if payment_id:
            return payment_id
    return None


def _tau_repeated_retail_read_guard_response(
    name: str,
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if not _available_tau_retail_tools(available_tool_names):
        return None
    if name not in {"get_order_details", "get_user_details"}:
        return None
    if not _has_successful_tau_tool_observation(name, arguments, prompt_messages):
        return None
    if name == "get_order_details":
        action_recovery = _tau_retail_action_or_confirmation_from_context(
            prompt_messages,
            available_tool_names=available_tool_names,
        )
        if action_recovery is not None:
            return action_recovery
        known_user_id = _latest_tau_fact_value(prompt_messages, "user_id")
        if (
            known_user_id
            and "get_user_details" in available_tool_names
            and not _has_successful_tau_tool_name("get_user_details", prompt_messages)
        ):
            return "get_user_details", {"user_id": known_user_id}
        next_order_id = _next_uninspected_tau_retail_order_id(prompt_messages)
        if next_order_id and "get_order_details" in available_tool_names:
            return "get_order_details", {"order_id": next_order_id}
        if (
            "list_all_product_types" in available_tool_names
            and _TAU_RETAIL_PRODUCT_INTENT_RE.search(_tau_user_request_text(prompt_messages))
            and not _has_successful_tau_tool_name("list_all_product_types", prompt_messages)
        ):
            return "list_all_product_types", {}
    if name == "get_user_details":
        next_order_id = _next_uninspected_tau_retail_order_id(prompt_messages)
        if next_order_id and "get_order_details" in available_tool_names:
            return "get_order_details", {"order_id": next_order_id}
    return (
        RESPOND_TOOL_NAME,
        {
            "content": (
                "I already have the successful tool output for that record and cannot safely continue "
                "by repeating the same read. ###STOP###"
            )
        },
    )


def _available_tau_retail_tools(available_tool_names: set[str]) -> bool:
    retail_markers = {
        "get_order_details",
        "exchange_delivered_order_items",
        "return_delivered_order_items",
        "list_all_product_types",
        *_TAU_RETAIL_DETAIL_TOOLS,
    }
    return bool(retail_markers.intersection(available_tool_names))


def _apply_tau_retail_progressive_tool_disclosure(
    tools: Sequence[Any],
    selected_tools: Sequence[Any],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    max_tools: int,
) -> _TauRetailProgressiveToolDisclosure:
    tools_by_name = {_tool_name(tool): tool for tool in tools if _tool_name(tool)}
    selected_names = [_tool_name(tool) for tool in selected_tools if _tool_name(tool)]
    if not _available_tau_retail_tools(set(tools_by_name)):
        return _TauRetailProgressiveToolDisclosure(
            selected_tools=list(selected_tools),
            trace={
                "enabled": True,
                "applied": False,
                "reason": "non_retail_tools",
                "selected_names_before": selected_names,
                "selected_names_after": selected_names,
            },
        )

    allowed_names, phase = _tau_retail_progressive_allowed_tool_names(
        prompt_messages,
        available_tool_names=set(tools_by_name),
    )
    allowed_set = set(allowed_names)
    ranked_names = _dedupe_tau_tool_names(
        [
            *allowed_names,
            *(name for name in selected_names if name in allowed_set),
        ]
    )
    limit = max(1, int(max_tools))
    selected = [tools_by_name[name] for name in ranked_names[:limit] if name in tools_by_name]
    if not selected:
        selected = list(selected_tools)
    after_names = [_tool_name(tool) for tool in selected if _tool_name(tool)]
    return _TauRetailProgressiveToolDisclosure(
        selected_tools=selected,
        trace={
            "enabled": True,
            "applied": True,
            "phase": phase,
            "allowed_names": [name for name in allowed_names if name in tools_by_name],
            "selected_names_before": selected_names,
            "selected_names_after": after_names,
        },
    )


def _tau_retail_progressive_allowed_tool_names(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[list[str], str]:
    user_text = _tau_user_request_text(prompt_messages)
    requested_order_id = _requested_tau_retail_order_id_from_user(prompt_messages)
    order = _latest_successful_tau_retail_order_observation(
        prompt_messages,
        order_id=requested_order_id,
    )
    has_order = order is not None
    has_user = _has_successful_tau_tool_name("get_user_details", prompt_messages)
    has_catalog = _has_successful_tau_tool_name("list_all_product_types", prompt_messages)
    has_product = _has_successful_tau_tool_name("get_product_details", prompt_messages)
    has_product_id = bool(_latest_tau_fact_value(prompt_messages, "product_id"))
    has_item_id = bool(_latest_tau_fact_value(prompt_messages, "item_id"))
    status = str((order.get("status") if isinstance(order, Mapping) else "") or "").strip().lower()
    product_intent = bool(_TAU_RETAIL_PRODUCT_INTENT_RE.search(user_text))
    exchange_intent = bool(_TAU_RETAIL_EXCHANGE_INTENT_RE.search(user_text))
    return_intent = bool(_TAU_RETAIL_RETURN_INTENT_RE.search(user_text))
    cancel_intent = bool(_TAU_RETAIL_CANCEL_INTENT_RE.search(user_text))
    address_intent = bool(_TAU_RETAIL_ADDRESS_INTENT_RE.search(user_text))
    payment_intent = bool(_TAU_RETAIL_PAYMENT_INTENT_RE.search(user_text))
    lookup_tools = _tau_retail_user_lookup_tool_names(user_text)

    allowed: list[str] = []
    if requested_order_id and not has_order:
        allowed.append("get_order_details")
        phase = "order_lookup"
    elif not has_order and not has_catalog and product_intent:
        allowed.append("list_all_product_types")
        allowed.extend(lookup_tools)
        phase = "product_catalog_lookup"
    elif not has_order and has_catalog and product_intent:
        allowed.append("get_product_details")
        allowed.extend(lookup_tools)
        phase = "product_detail_lookup"
    elif not has_order and not has_user:
        allowed.extend(lookup_tools)
        if product_intent:
            allowed.append("list_all_product_types")
        phase = "user_lookup"
    elif has_order and not has_user:
        allowed.append("get_user_details")
        if product_intent or exchange_intent or has_product_id:
            allowed.append("get_product_details" if has_catalog or has_product_id else "list_all_product_types")
        phase = "order_known_user_lookup"
    else:
        phase = "action_or_detail"
        if product_intent or exchange_intent or has_product_id or has_catalog:
            if not has_catalog and not has_product_id:
                allowed.append("list_all_product_types")
            else:
                allowed.append("get_product_details")
        if has_product or has_item_id:
            allowed.append("get_item_details")
        if has_order:
            if "pending" in status:
                if cancel_intent:
                    allowed.append("cancel_pending_order")
                if address_intent:
                    allowed.append("modify_pending_order_address")
                if exchange_intent or product_intent:
                    allowed.append("modify_pending_order_items")
                if payment_intent:
                    allowed.append("modify_pending_order_payment")
            elif status == "delivered":
                if return_intent:
                    allowed.append("return_delivered_order_items")
                if exchange_intent or product_intent:
                    allowed.append("exchange_delivered_order_items")
        if address_intent and has_user:
            allowed.append("modify_user_address")
        allowed.append("get_user_details")

    allowed.extend(("calculate", "transfer_to_human_agents"))
    if not allowed:
        allowed.extend(_TAU_RETAIL_BOOTSTRAP_TOOLS)
    return [name for name in _dedupe_tau_tool_names(allowed) if name in available_tool_names], phase


def _latest_successful_tau_retail_order_observation(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    order_id: str | None = None,
) -> Mapping[str, Any] | None:
    requested = str(order_id or "").strip().upper()
    for tool_name, args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if tool_name != "get_order_details" or not ok or not isinstance(output, Mapping):
            continue
        observed_id = str(output.get("order_id") or args.get("order_id") or "").strip().upper()
        if requested and observed_id != requested:
            continue
        return output
    return None


def _tau_retail_explicit_user_lookup_tool_names(user_text: str) -> list[str]:
    text = str(user_text or "")
    names: list[str] = []
    if _TAU_EMAIL_RE.search(text):
        names.append("find_user_id_by_email")
    if re.search(r"\b\d{5}(?:-\d{4})?\b", text):
        names.append("find_user_id_by_name_zip")
    return names


def _tau_retail_user_lookup_tool_names(user_text: str) -> list[str]:
    return _tau_retail_explicit_user_lookup_tool_names(user_text)


def _dedupe_tau_tool_names(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        name = str(value or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _preserve_hash_prefixed_ids_from_user_context(
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    """Restore a leading # only when the user supplied the exact same ID token."""
    user_text = _tau_user_request_text(prompt_messages)
    hash_ids = {match.group(0)[1:]: match.group(0) for match in _TAU_HASH_ID_RE.finditer(user_text)}
    if not hash_ids:
        return dict(arguments)

    def preserve(value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped in hash_ids:
                return hash_ids[stripped]
            return value
        if isinstance(value, list):
            return [preserve(item) for item in value]
        if isinstance(value, tuple):
            return [preserve(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): preserve(child) for key, child in value.items()}
        return value

    return {str(key): preserve(value) for key, value in dict(arguments).items()}


def _tau_cancel_intent_replacement_from_context(
    selected_name: str,
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if "cancel_reservation" not in available_tool_names and "get_reservation_details" not in available_tool_names:
        return None
    requested_id = _requested_tau_reservation_id_from_user(prompt_messages)
    if not requested_id:
        return None
    user_text = _tau_user_request_text(prompt_messages)
    if not re.search(
        r"\b(?:cancel|canceling|cancelling|canceled|cancelled|cancellation)\b",
        user_text,
        re.IGNORECASE,
    ):
        return None
    completed_response = _tau_completed_cancel_response_from_context(prompt_messages, requested_id)
    if completed_response is not None:
        return completed_response
    has_reservation_details = bool(
        _latest_successful_tau_reservation_observation(prompt_messages, reservation_id=requested_id)
    )
    if not has_reservation_details and selected_name != "get_reservation_details":
        return "get_reservation_details", {"reservation_id": requested_id}
    if has_reservation_details and selected_name != "cancel_reservation" and "cancel_reservation" in available_tool_names:
        return "cancel_reservation", {"reservation_id": requested_id}
    return None


def _tau_completed_cancel_response_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
    reservation_id: str,
) -> tuple[str, dict[str, Any]] | None:
    requested = str(reservation_id or "").strip().upper()
    if not requested:
        return None
    for tool_name, args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if not ok or not isinstance(output, Mapping):
            continue
        observed_id = str(output.get("reservation_id") or args.get("reservation_id") or "").strip().upper()
        if observed_id != requested:
            continue
        if tool_name == "cancel_reservation" or str(output.get("status") or "").strip().lower() == "cancelled":
            return (
                RESPOND_TOOL_NAME,
                {
                    "content": (
                        f"Reservation {requested} has been cancelled, and the refund/payment reversal "
                        "has been recorded. ###STOP###"
                    )
                },
            )
    return None


def _tau_direct_requested_reservation_action_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    requested_id = _requested_tau_reservation_id_from_user(prompt_messages)
    if not requested_id:
        return None
    if not _latest_successful_tau_reservation_observation(prompt_messages, reservation_id=requested_id):
        return None
    user_text = _tau_user_request_text(prompt_messages)
    if "cancel_reservation" in available_tool_names and re.search(
        r"\b(?:cancel|canceling|cancelling|canceled|cancelled|cancellation)\b",
        user_text,
        re.IGNORECASE,
    ):
        return "cancel_reservation", {"reservation_id": requested_id}
    return None


def _tau_readonly_reservation_response_from_context(
    prompt_messages: Sequence[Mapping[str, object]],
) -> tuple[str, dict[str, Any]] | None:
    user_text = _tau_user_request_text(prompt_messages)
    if not re.search(r"\b(?:baggage|bag|bags|suitcase|suitcases|luggage)\b", user_text, re.IGNORECASE):
        return None
    requested_id = _requested_tau_reservation_id_from_user(prompt_messages)
    reservation = _latest_successful_tau_reservation_observation(prompt_messages, reservation_id=requested_id)
    if not reservation:
        return None
    reservation_id = str(reservation.get("reservation_id") or requested_id or "").strip().upper()
    total_baggages = reservation.get("total_baggages")
    if total_baggages is None:
        return None
    nonfree_baggages = reservation.get("nonfree_baggages")
    content = f"Reservation {reservation_id} allows {total_baggages} total suitcases."
    if nonfree_baggages is not None:
        content += f" Nonfree suitcases: {nonfree_baggages}."
    return RESPOND_TOOL_NAME, {"content": f"{content} ###STOP###"}


def _latest_successful_tau_reservation_observation(
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    reservation_id: str | None = None,
) -> Mapping[str, Any] | None:
    requested = str(reservation_id or "").strip().upper()
    for tool_name, _args, ok, output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if tool_name != "get_reservation_details" or not ok or not isinstance(output, Mapping):
            continue
        observed_id = str(output.get("reservation_id") or "").strip().upper()
        if requested and observed_id != requested:
            continue
        return output
    return None


def _tau_exhausted_repeated_read_response(
    name: str,
    prompt_messages: Sequence[Mapping[str, object]],
) -> tuple[str, dict[str, Any]] | None:
    if name not in {"get_user_details", "get_reservation_details"}:
        return None
    if not _tau_context_wants_existing_reservation_action(prompt_messages):
        return None
    if not _tau_reservation_ids_from_context(prompt_messages):
        return None
    if _next_uninspected_tau_reservation_id(prompt_messages):
        return None
    return (
        RESPOND_TOOL_NAME,
        {
            "content": (
                "I have already checked the available reservation records and cannot safely identify "
                "another matching action. ###STOP###"
            )
        },
    )


def _replacement_for_repeated_tau_read(
    name: str,
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    available_tool_names: set[str],
) -> tuple[str, dict[str, Any]] | None:
    if (
        name == "get_user_details"
        and "get_reservation_details" in available_tool_names
        and _tau_context_wants_existing_reservation_action(prompt_messages)
    ):
        next_reservation_id = _next_uninspected_tau_reservation_id(prompt_messages)
        if next_reservation_id:
            return "get_reservation_details", {"reservation_id": next_reservation_id}
    if name == "get_reservation_details" and "get_reservation_details" in available_tool_names:
        current_id = str(arguments.get("reservation_id") or "").strip().upper()
        next_reservation_id = _next_uninspected_tau_reservation_id(prompt_messages)
        if next_reservation_id and next_reservation_id != current_id:
            return "get_reservation_details", {"reservation_id": next_reservation_id}
    return None


def _tau_context_wants_existing_reservation_action(prompt_messages: Sequence[Mapping[str, object]]) -> bool:
    text = _tau_user_request_text(prompt_messages)
    lowered = text.lower()
    if not _TAU_EXISTING_RESERVATION_ACTION_RE.search(text):
        return False
    return (
        bool(_requested_tau_reservation_id_from_user(prompt_messages))
        or bool(_tau_reservation_ids_from_context(prompt_messages))
        or "reservation" in lowered
        or "booking" in lowered
        or "flight" in lowered
    )


def _requested_tau_reservation_id_from_user(prompt_messages: Sequence[Mapping[str, object]]) -> str | None:
    user_text = _tau_user_request_text(prompt_messages)
    labeled_match = re.search(
        r"\b(?:reservation(?:_id| id)?|booking(?:_id| id)?|confirmation(?:\s*(?:number|no\.?|id|code))?)"
        r"\s*(?:is|:|#)?\s*((?=[A-Z0-9]*\d)[A-Z0-9]{6})\b",
        user_text,
        flags=re.IGNORECASE,
    )
    if labeled_match:
        return labeled_match.group(1).upper()
    generic_match = re.search(r"\b(?=[A-Z0-9]{6}\b)(?=[A-Z0-9]*\d)[A-Z0-9]{6}\b", user_text)
    if generic_match and not _TAU_FLIGHT_NUMBER_RE.fullmatch(generic_match.group(0).upper()):
        return generic_match.group(0).upper()
    return None


def _next_uninspected_tau_reservation_id(prompt_messages: Sequence[Mapping[str, object]]) -> str | None:
    candidates = _tau_reservation_ids_from_context(prompt_messages)
    if not candidates:
        return None
    inspected = {
        str(args.get("reservation_id") or "").strip().upper()
        for tool_name, args in _iter_prior_tau_tool_calls(prompt_messages)
        if tool_name == "get_reservation_details"
    }
    for reservation_id in candidates:
        if reservation_id and reservation_id not in inspected:
            return reservation_id
    return None


def _tau_reservation_ids_from_context(prompt_messages: Sequence[Mapping[str, object]]) -> list[str]:
    candidates: list[str] = []
    requested = _requested_tau_reservation_id_from_user(prompt_messages)
    if requested:
        candidates.append(requested)
    for message in prompt_messages:
        content = str(message.get("content") or "")
        payload = _parse_tau_function_output_payload(content)
        if not payload or not bool(payload.get("ok", True)):
            continue
        output = _parse_tau_tool_output(payload.get("output"))
        if isinstance(output, Mapping):
            reservation_id = str(output.get("reservation_id") or "").strip().upper()
            if reservation_id:
                candidates.append(reservation_id)
            reservations = output.get("reservations")
            if isinstance(reservations, (list, tuple)):
                candidates.extend(str(item).strip().upper() for item in reservations if str(item).strip())
    return _dedupe_tau_ids(candidates)


def _tau_user_request_text(prompt_messages: Sequence[Mapping[str, object]]) -> str:
    parts: list[str] = []
    for message in prompt_messages:
        if _normalized_tau_message_role(message.get("role")) != "user":
            continue
        content = str(message.get("content") or "")
        if not content or "Function output:" in content or content.startswith("Known facts"):
            continue
        parts.append(content)
    return normalize_rwkv_text("\n".join(parts))


def _normalized_tau_message_role(value: object) -> str:
    text = str(value or "").strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if text in {"agent", "assistant"}:
        return "assistant"
    if text in {"env", "environment", "tool"}:
        return "tool"
    if text == "user":
        return "user"
    return text


def _iter_tau_tool_observations(prompt_messages: Sequence[Mapping[str, object]]) -> list[tuple[str, dict[str, Any], bool, Any]]:
    observations: list[tuple[str, dict[str, Any], bool, Any]] = []
    pending: list[tuple[str, dict[str, Any]]] = []
    for message in prompt_messages:
        role = _normalized_tau_message_role(message.get("role"))
        content = str(message.get("content") or "").strip()
        if role == "assistant" and content:
            try:
                tool_name, arguments = _parse_tau_agent_decision(content)
            except Exception:
                continue
            if tool_name != RESPOND_TOOL_NAME:
                pending.append((tool_name, arguments))
            continue
        if role != "user" or not content:
            continue
        payload = _parse_tau_function_output_payload(content)
        if not payload or not pending:
            continue
        tool_name, arguments = pending.pop(0)
        observations.append((tool_name, arguments, bool(payload.get("ok", True)), _parse_tau_tool_output(payload.get("output"))))
    return observations


def _iter_prior_tau_tool_calls(prompt_messages: Sequence[Mapping[str, object]]) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    for message in prompt_messages:
        if _normalized_tau_message_role(message.get("role")) != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        try:
            tool_name, arguments = _parse_tau_agent_decision(content)
        except Exception:
            continue
        calls.append((tool_name, arguments))
    return calls


def _has_successful_tau_tool_observation(
    name: str,
    arguments: Mapping[str, Any],
    prompt_messages: Sequence[Mapping[str, object]],
) -> bool:
    target_name = str(name or "").strip()
    target_arguments = _canonical_tau_arguments(arguments)
    for tool_name, observed_args, ok, _output in reversed(_iter_tau_tool_observations(prompt_messages)):
        if ok and tool_name == target_name and _canonical_tau_arguments(observed_args) == target_arguments:
            return True
    return False


def _has_successful_tau_tool_name(name: str, prompt_messages: Sequence[Mapping[str, object]]) -> bool:
    target_name = str(name or "").strip()
    return any(ok and tool_name == target_name for tool_name, _args, ok, _output in _iter_tau_tool_observations(prompt_messages))


def _canonical_tau_arguments(arguments: Mapping[str, Any]) -> str:
    try:
        return json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(
            {str(key): str(value) for key, value in dict(arguments).items()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def _dedupe_tau_ids(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = str(value or "").strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _build_tau_tool_facts_message(messages: Sequence[Mapping[str, object]], *, max_chars: int = 1200) -> dict[str, str] | None:
    facts = _extract_tau_tool_facts(messages, max_facts=18)
    if not facts:
        return None
    lines = [
        "Known facts from previous tool outputs.",
        "Use these exact values; do not ask for them again or invent replacements.",
    ]
    lines.extend(f"- {key}: {value}" for key, value in facts)
    return {"role": "user", "content": truncate_text("\n".join(lines), max_chars)}


def _latest_tau_fact_value(messages: Sequence[Mapping[str, object]], key: str) -> str | None:
    target = str(key)
    for fact_key, fact_value in _extract_tau_tool_facts(messages, max_facts=64):
        if fact_key == target:
            return fact_value
    return None


def _extract_tau_tool_facts(
    messages: Sequence[Mapping[str, object]],
    *,
    max_facts: int,
) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    seen: set[str] = set()
    for message in reversed(messages):
        content = str(message.get("content") or "")
        payload = _parse_tau_function_output_payload(content)
        if not payload or not bool(payload.get("ok", True)):
            continue
        output = _parse_tau_tool_output(payload.get("output"))
        for key, value in _iter_tau_fact_items(output):
            normalized_key = str(key).strip()
            normalized_value = _format_tau_fact_value(value)
            if not normalized_key or not normalized_value:
                continue
            seen_key = f"{normalized_key}:{normalized_value}"
            if seen_key in seen:
                continue
            facts.append((normalized_key, normalized_value))
            seen.add(seen_key)
            if len(facts) >= max(1, int(max_facts)):
                return facts
    return facts


def _parse_tau_function_output_payload(content: str) -> dict[str, Any] | None:
    tagged = re.search(r"(?is)<tool_response>\s*(.*?)\s*</tool_response>", content)
    if tagged is not None:
        payload_text = tagged.group(1).strip()
        if not payload_text:
            return None
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    marker = "Function output:"
    if marker not in content:
        return None
    payload_text = content.split(marker, 1)[1].strip()
    if not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_tau_tool_output(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text and text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _iter_tau_fact_items(value: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []

    def visit(current: Any, *, depth: int) -> None:
        if depth > 2:
            return
        if isinstance(current, str):
            text = current.strip()
            if _TAU_USER_ID_RE.fullmatch(text):
                items.append(("user_id", text))
            return
        if isinstance(current, Mapping):
            item_summary = _format_tau_item_fact(current)
            if item_summary:
                items.append(("item", item_summary))
                return
            for raw_key, raw_value in current.items():
                key = str(raw_key)
                if key in _TAU_FACT_KEYS and _is_tau_fact_value(raw_value):
                    items.append((key, raw_value))
                if isinstance(raw_value, (Mapping, list)):
                    visit(raw_value, depth=depth + 1)
            return
        if isinstance(current, list):
            for item in current[:5]:
                visit(item, depth=depth + 1)

    visit(value, depth=0)
    return items


def _format_tau_item_fact(value: Mapping[str, Any]) -> str | None:
    name = str(value.get("name") or "").strip()
    product_id = str(value.get("product_id") or "").strip()
    item_id = str(value.get("item_id") or "").strip()
    if not name or (not product_id and not item_id):
        return None
    parts = [name]
    if product_id:
        parts.append(f"product_id={product_id}")
    if item_id:
        parts.append(f"item_id={item_id}")
    options = value.get("options")
    if isinstance(options, Mapping):
        option_parts = [
            f"{str(key).strip()}={str(option_value).strip()}"
            for key, option_value in options.items()
            if str(key).strip() and str(option_value).strip()
        ]
        if option_parts:
            parts.append("options: " + ", ".join(option_parts[:5]))
    return truncate_text(" | ".join(parts), 180)


def _is_tau_fact_value(value: Any) -> bool:
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip() != ""
    if isinstance(value, list):
        return any(isinstance(item, (str, int, float, bool)) and str(item).strip() for item in value)
    if isinstance(value, Mapping):
        return any(str(key).strip() for key in value)
    return False


def _format_tau_fact_value(value: Any) -> str:
    if isinstance(value, Mapping):
        keys = [str(key) for key in value.keys() if str(key).strip()]
        return ", ".join(keys[:8])
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value[:8] if str(item).strip())
    return str(value).strip()


def _tau_messages_to_prompt_messages(
    history: Sequence[Any],
    *,
    ToolMessage: Any,
    UserMessage: Any,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for message in history:
        role = _normalized_tau_message_role(getattr(message, "role", ""))
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
    return "<tool_response>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_response>"


def _render_tau_tool_call(tool_call: Any) -> str:
    payload = json.dumps(
        {
            "name": str(getattr(tool_call, "name", "") or ""),
            "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"<tool_call>\n{payload}\n</tool_call>"


def _normalize_tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "openai_schema", None)
    if isinstance(schema, Mapping):
        schema = dict(schema)
        function_schema = schema.get("function")
        if isinstance(function_schema, Mapping):
            normalized = dict(function_schema)
            if "parameters" not in normalized and "arguments" in normalized:
                normalized["parameters"] = normalized.pop("arguments")
            return normalized
        return schema
    if isinstance(tool, Mapping):
        return dict(tool)
    return {"name": _tool_name(tool), "description": str(tool)}


def _compact_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    name = str(schema.get("name") or "").strip()
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = schema.get("arguments")
    if not isinstance(parameters, Mapping):
        parameters = {}

    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    compact_properties: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        compact_properties[str(prop_name)] = _compact_parameter_schema(prop_schema)

    required = parameters.get("required")
    if not isinstance(required, (list, tuple)):
        required = []

    compact: dict[str, Any] = {
        "name": name,
        "description": truncate_text(normalize_rwkv_text(str(schema.get("description") or "")), 120),
        "parameters": {
            "type": "object",
            "properties": compact_properties,
            "required": [str(item) for item in required],
        },
    }
    return compact


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
    items = schema.get("items")
    if isinstance(items, Mapping):
        compact["items"] = _compact_parameter_schema(items)
    return compact


def _minimal_tool_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    name = str(schema.get("name") or "").strip()
    description = truncate_text(normalize_rwkv_text(str(schema.get("description") or "")), 48)
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = schema.get("arguments")
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required")
    if not isinstance(required, (list, tuple)):
        required = []
    required_names = [str(item) for item in required]
    selected_property_names = required_names or [str(prop_name) for prop_name in list(properties)[:8]]
    selected_properties = {
        prop_name: properties[prop_name]
        for prop_name in selected_property_names
        if prop_name in properties
    }
    return {
        "name": name,
        "description": description,
        "args": {
            str(prop_name): _minimal_parameter_type(prop_schema)
            for prop_name, prop_schema in selected_properties.items()
        },
        "required": required_names,
    }


def _minimal_parameter_type(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return "string"
    schema_type = str(schema.get("type") or "string")
    enum = schema.get("enum")
    if isinstance(enum, (list, tuple)) and 0 < len(enum) <= 8:
        return f"{schema_type} enum={','.join(str(item) for item in enum)}"
    return schema_type


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if name:
        return str(name)
    schema = getattr(tool, "openai_schema", None)
    if isinstance(schema, Mapping):
        function_schema = schema.get("function")
        if isinstance(function_schema, Mapping) and function_schema.get("name"):
            return str(function_schema["name"])
        if schema.get("name"):
            return str(schema["name"])
    if isinstance(tool, Mapping) and tool.get("name"):
        return str(tool["name"])
    return ""


def _strip_tau_requestor_prefix(name: str) -> str:
    text = str(name or "").strip()
    if "." not in text:
        return text
    prefix, rest = text.split(".", 1)
    if prefix in {"assistant", "user"} and rest.strip():
        return rest.strip()
    return text


def _task_uses_nl_assertions(task: Any) -> bool:
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is None:
        return False
    reward_basis = {str(item) for item in (getattr(criteria, "reward_basis", []) or [])}
    return bool(getattr(criteria, "nl_assertions", None)) or any("NL_ASSERTION" in item for item in reward_basis)


def _model_dump_safe(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": str(value)}


__all__ = [
    "DEFAULT_TAU_PROMPT_MAX_CHARS",
    "RESPOND_TOOL_NAME",
    "RWKVTauOfficialAgent",
    "StaticStopTauUser",
    "TauOfficialEvaluation",
    "TauOfficialRuntime",
    "build_tau_official_agent_system_prompt",
    "configure_tau_nl_assertions_judge",
]
