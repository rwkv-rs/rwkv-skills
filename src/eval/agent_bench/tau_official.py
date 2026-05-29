from __future__ import annotations

import json
import os
import re
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


@dataclass(slots=True)
class TauOfficialEvaluation:
    reward: float
    is_passed: bool
    details: dict[str, Any]


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
        return Task.model_validate(dict(payload))

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
                "temperature": float(temperature),
                "api_key": user_model.api_key,
                "api_base": user_model.base_url,
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
        "temperature": 0.0,
        "api_key": api_key,
        "response_format": {"type": "json_object"},
    }
    if base_url:
        llm_args["api_base"] = base_url
    llm_args.update(_tau_llm_timeout_args())
    for module_name in ("tau2.config", "tau2.evaluator.evaluator_nl_assertions"):
        module = import_module_with_auto_install(module_name, context=f"tau2 NL assertion judge config: {module_name}")
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS", model_name)
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", dict(llm_args))


def _tau_litellm_model_name(model_config: Any) -> str:
    model_name = str(getattr(model_config, "model_name", "") or "").strip()
    if not model_name or "/" in model_name:
        return model_name
    base_url = normalize_openai_base_url(getattr(model_config, "base_url", None)) or ""
    if "api.deepseek.com" in base_url and model_name.startswith("deepseek-"):
        return f"deepseek/{model_name}"
    return model_name


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
        history_max_chars: int,
        prompt_max_chars: int = DEFAULT_TAU_PROMPT_MAX_CHARS,
        long_doc_config: LongDocEvidenceConfig | None = None,
        tool_routing_config: ToolRoutingConfig | None = None,
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
        self._history_max_chars = max(0, int(history_max_chars))
        self._prompt_max_chars = max(1024, int(prompt_max_chars))
        self._long_doc_config = long_doc_config or long_doc_config_from_env("RWKV_TAU_LONG_DOC")
        self._tool_routing_config = tool_routing_config or ToolRoutingConfig()
        self._seed: int | None = None
        self._turn_index = 0
        self.stages: list[StageRecord] = []
        self.parse_errors: list[str] = []
        self.tool_routes: list[dict[str, Any]] = []

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
        outputs = self._engine.generate(
            [prompt],
            sampling=self._sampling,
            batch_size=1,
            progress_desc="TauOfficial-Agent",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
            prompt_seeds=[prompt_seed] if prompt_seed is not None else None,
        )
        output = outputs[0] if outputs else None
        raw_text = output.text if output is not None else ""
        finish_reason = output.finish_reason if output is not None else "missing_output"
        parse_error: str | None = None
        try:
            name, arguments = _parse_tau_agent_decision(raw_text)
            assistant_message = self._decision_to_assistant_message(name, arguments)
        except Exception as exc:
            parse_error = str(exc)
            self.parse_errors.append(parse_error)
            assistant_message = self._AssistantMessage(
                role="assistant",
                content="I am unable to continue safely. ###STOP###",
            )
        self.stages.append(
            StageRecord(
                prompt=prompt,
                completion=raw_text,
                stop_reason=finish_reason,
            )
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
        (
            prompt,
            emitted_tools,
            emitted_policy_chars,
            emitted_tool_schema_mode,
        ) = self._build_budgeted_agent_prompt(
            domain_policy=domain_policy,
            selected_tools=tool_route.selected_tools,
            messages=compacted_messages,
            history_budget=history_budget,
        )
        self._current_tool_names = {_tool_name(tool) for tool in emitted_tools if _tool_name(tool)}
        route_trace = {"turn_index": self._turn_index, **tool_route.trace_payload()}
        if len(emitted_tools) != len(tool_route.selected_tools):
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
                    tool_schema_mode=tool_schema_mode,
                )
                prompt = build_rwkv_json_call_prompt(
                    system_prompt,
                    messages,
                    history_max_chars=history_budget,
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
            tool_schema_mode="minimal",
        )
        final_prompt = build_rwkv_json_call_prompt(final_system, [], history_max_chars=0)
        if len(final_prompt) > self._prompt_max_chars:
            raise ValueError(
                "tau prompt budget cannot fit routed tools without corrupting the prompt: "
                f"prompt_chars={len(final_prompt)} budget={self._prompt_max_chars} tools={len(tools)}"
            )
        return final_prompt, tools, min(policy_budgets[-1], 240), "minimal"

    def _decision_to_assistant_message(self, name: str, arguments: Mapping[str, Any]) -> Any:
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
    tool_schema_mode: str = "full",
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
    return normalize_rwkv_text(
        "\n".join(
            [
                "You are the assistant in the official tau-bench simulation.",
                "Follow the domain policy exactly.",
                "Use a real tool call when you need information or need to change state.",
                "Use respond only when sending a message to the user.",
                "When the task is complete and no more tool calls are needed, use respond and include ###STOP### in the content.",
                "Return exactly one JSON function call object and no extra prose.",
                'JSON shape: {"name":"tool_name","arguments":{...}}',
                "Valid names are exactly the listed tool names plus respond.",
                "Never invent wrapper or pseudo tools such as think, thought, search_flights, get_flights, search_bookings, get_user_bookings, or airline_agent_tool.",
                "Never copy a Function output object; do not return requestor/ok/output as your decision.",
                "Use only argument keys declared by the selected tool schema.",
                "Tools:",
                json.dumps(tool_schemas, **tool_json_kwargs),
                "Policy:",
                normalize_rwkv_text(domain_policy),
            ]
        )
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
    except ValueError as exc:
        raise ValueError(f"tau agent decision missing recoverable arguments: {normalized}") from exc
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


def _normalize_tau_decision(name: str, arguments: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    normalized_name = _strip_tau_requestor_prefix(name)
    return normalized_name, dict(arguments)


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
    description = truncate_text(normalize_rwkv_text(str(schema.get("description") or "")), 64)
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
    return {
        "name": name,
        "description": description,
        "args": {
            str(prop_name): _minimal_parameter_type(prop_schema)
            for prop_name, prop_schema in properties.items()
        },
        "required": [str(item) for item in required],
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
