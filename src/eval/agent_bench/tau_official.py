from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.eval.agent_bench.deps import import_module_with_auto_install
from src.eval.agent_bench.tasks import ensure_tau_v2_vendor_path
from src.eval.env_config import normalize_openai_base_url
from src.eval.evaluators.common import StageRecord
from src.eval.function_calling.context_budget import normalize_rwkv_text, trim_message_history
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_rwkv_json_call_prompt,
    extract_json_call_value_text,
)
from src.infer.backend import InferenceBackend
from src.infer.sampling import SamplingConfig

RESPOND_TOOL_NAME = "respond"
DEFAULT_TAU_PROMPT_MAX_CHARS = 24576
TAU_RESPOND_ALIASES = {
    "done",
    "final",
    "final_answer",
    "final_response",
    "stop",
}
TAU_USER_REQUEST_ALIASES = {
    "ask_for_confirmation": "Please confirm how you would like to proceed.",
    "ask_for_reservation_id": "Could you please provide the reservation ID?",
    "ask_for_user_id": "Could you please provide your user ID?",
    "request_confirmation": "Please confirm how you would like to proceed.",
    "request_reservation_id": "Could you please provide the reservation ID?",
    "request_user_id": "Could you please provide your user ID?",
}
TAU_TOOL_NAME_ALIASES = {
    "get_reservations_by_user_id": "get_user_details",
    "get_user_reservations": "get_user_details",
    "list_user_reservations": "get_user_details",
}


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

    def build_user(self, *, task: Any, environment: Any, user_model: Any, temperature: float = 0.0) -> Any:
        user_module = import_module_with_auto_install("tau2.user.user_simulator", context="tau2 user simulator import")
        UserSimulator = getattr(user_module, "UserSimulator")
        try:
            user_tools = environment.get_user_tools()
        except Exception:
            user_tools = None
        return UserSimulator(
            tools=user_tools,
            instructions=str(getattr(task, "user_scenario", "")),
            llm=user_model.model_name,
            llm_args={
                "temperature": float(temperature),
                "api_key": user_model.api_key,
                "api_base": user_model.base_url,
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
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                reward_info = evaluate_simulation(
                    simulation=simulation,
                    task=task,
                    evaluation_type=evaluation_type,
                    solo_mode=False,
                    domain=self.domain,
                )
                break
            except Exception as exc:  # Official NL assertion judges can return empty/non-JSON content.
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 + attempt)
        else:
            details = {
                "evaluation_error": str(last_error or "unknown tau evaluation error"),
                "termination_reason": str(getattr(simulation, "termination_reason", "")),
            }
            return TauOfficialEvaluation(reward=0.0, is_passed=False, details=details)
        simulation.reward_info = reward_info
        details = _model_dump_safe(reward_info)
        details["termination_reason"] = str(getattr(simulation, "termination_reason", ""))
        return TauOfficialEvaluation(
            reward=float(getattr(reward_info, "reward", 0.0)),
            is_passed=float(getattr(reward_info, "reward", 0.0)) >= (1.0 - 1e-6),
            details=details,
        )


def configure_tau_nl_assertions_judge(judge_model: Any) -> None:
    model_name = str(getattr(judge_model, "model_name", "") or "").strip()
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
    for module_name in ("tau2.config", "tau2.evaluator.evaluator_nl_assertions"):
        module = import_module_with_auto_install(module_name, context=f"tau2 NL assertion judge config: {module_name}")
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS", model_name)
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS_ARGS", dict(llm_args))


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
        self._tool_names = {_tool_name(tool) for tool in self._tools if _tool_name(tool)}
        self._system_prompt = build_tau_official_agent_system_prompt(domain_policy, self._tools)
        self._history_max_chars = max(0, int(history_max_chars))
        self._prompt_max_chars = max(4096, int(prompt_max_chars))
        self._seed: int | None = None
        self._turn_index = 0
        self.stages: list[StageRecord] = []
        self.parse_errors: list[str] = []

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
        prompt = build_rwkv_json_call_prompt(
            self._system_prompt,
            prompt_messages,
            history_max_chars=history_budget,
        )
        if len(prompt) <= self._prompt_max_chars or history_budget <= 0:
            return prompt
        overflow = len(prompt) - self._prompt_max_chars
        history_budget = max(0, history_budget - overflow - 512)
        return build_rwkv_json_call_prompt(
            self._system_prompt,
            trim_message_history(prompt_messages, max_chars=history_budget),
            history_max_chars=history_budget,
        )

    def _decision_to_assistant_message(self, name: str, arguments: Mapping[str, Any]) -> Any:
        normalized_name, arguments = _normalize_tau_decision(name, arguments)
        arguments = _normalize_tau_arguments(normalized_name, arguments)
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


def build_tau_official_agent_system_prompt(domain_policy: str, tools: Sequence[Any]) -> str:
    tool_schemas = [_normalize_tool_schema(tool) for tool in tools]
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
                "Tools:",
                json.dumps(tool_schemas, ensure_ascii=False, indent=2, sort_keys=False),
                "Policy:",
                normalize_rwkv_text(domain_policy),
            ]
        )
    )


def _parse_tau_agent_decision(text: str) -> tuple[str, dict[str, Any]]:
    candidate = extract_json_call_value_text(text)
    payload = _coerce_tau_decision_payload(json.loads(candidate))
    return _normalize_tau_decision(str(payload["name"]).strip(), dict(payload["arguments"]))


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
    normalized_arguments = dict(arguments)
    if normalized_name in TAU_USER_REQUEST_ALIASES:
        content = _first_text_field(
            normalized_arguments,
            ("content", "message", "question", "prompt", "summary", "answer"),
        )
        normalized_arguments = {"content": content or TAU_USER_REQUEST_ALIASES[normalized_name]}
        normalized_name = RESPOND_TOOL_NAME
    elif normalized_name in TAU_RESPOND_ALIASES:
        normalized_name = RESPOND_TOOL_NAME
        if "content" not in normalized_arguments:
            content = _first_text_field(normalized_arguments, ("answer", "message", "summary"))
            normalized_arguments["content"] = content or "###STOP###"
        if "###STOP###" not in str(normalized_arguments.get("content") or ""):
            normalized_arguments["content"] = f"{normalized_arguments['content']} ###STOP###"
    else:
        normalized_name = TAU_TOOL_NAME_ALIASES.get(normalized_name, normalized_name)
    return normalized_name, normalized_arguments


def _normalize_tau_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if name == "transfer_to_human_agents":
        summary = normalized.get("summary")
        for key in ("content", "message", "answer"):
            if isinstance(summary, str) and summary.strip():
                break
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                summary = value.strip()
        normalized = {"summary": str(summary or "").strip()}
    return normalized


def _first_text_field(arguments: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
    "TauOfficialEvaluation",
    "TauOfficialRuntime",
    "build_tau_official_agent_system_prompt",
    "configure_tau_nl_assertions_judge",
]
