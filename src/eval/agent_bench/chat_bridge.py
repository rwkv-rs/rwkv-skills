from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from src.infer.backend import InferenceBackend
from src.infer.sampling import SamplingConfig

PromptProfile = Literal["tau_v1", "tau_v2", "legacy"]


@dataclass(slots=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ChatResult:
    content: str
    tool_calls: list[ParsedToolCall]
    finish_reason: str
    parse_error: str | None
    raw_text: str
    prompt: str


class RWKVChatBridge:
    """Bridge RWKV text generation to a chat-completion-like tool calling API."""

    def __init__(
        self,
        *,
        engine: InferenceBackend,
        default_sampling: SamplingConfig,
        prompt_profile: PromptProfile = "tau_v1",
        extra_system_instruction: str | None = None,
    ) -> None:
        self._engine = engine
        self._default_sampling = default_sampling
        self._prompt_profile = prompt_profile
        self._extra_system_instruction = (extra_system_instruction or "").strip()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools_schema: Sequence[dict[str, Any]] | None = None,
        *,
        sampling: SamplingConfig | None = None,
        tool_choice: str | None = None,
        prompt_profile: PromptProfile | None = None,
    ) -> ChatResult:
        return self.chat_many(
            [messages],
            [tools_schema or ()],
            sampling=sampling,
            tool_choice=tool_choice,
            prompt_profile=prompt_profile,
        )[0]

    def chat_many(
        self,
        message_batches: Sequence[Sequence[dict[str, Any]]],
        tools_schemas: Sequence[Sequence[dict[str, Any]] | None] | None = None,
        *,
        sampling: SamplingConfig | None = None,
        tool_choice: str | None = None,
        prompt_profile: PromptProfile | None = None,
    ) -> list[ChatResult]:
        if not message_batches:
            return []
        if tools_schemas is None:
            tools_schemas = [()] * len(message_batches)
        if len(tools_schemas) != len(message_batches):
            raise ValueError("tools_schemas 长度必须与 message_batches 一致")

        profile = prompt_profile or self._prompt_profile
        prompts = [
            self._render_prompt(
                messages,
                tools_schema=tools_schema or (),
                tool_choice=tool_choice,
                profile=profile,
            )
            for messages, tools_schema in zip(message_batches, tools_schemas)
        ]
        outputs = self._engine.generate(
            prompts,
            sampling=sampling or self._default_sampling,
            batch_size=len(prompts),
            progress_desc="AgentBench",
        )

        by_index = {int(output.prompt_index): output for output in outputs}
        results: list[ChatResult] = []
        for idx, (prompt, tools_schema) in enumerate(zip(prompts, tools_schemas)):
            output = by_index.get(idx)
            if output is None:
                results.append(
                    ChatResult(
                        content="",
                        tool_calls=[],
                        finish_reason="error",
                        parse_error="rwkv_generate_missing_output",
                        raw_text="",
                        prompt=prompt,
                    )
                )
                continue

            raw_text = (output.text or "").strip()
            tool_calls, content, parse_error = _parse_generation(
                raw_text,
                tool_names=_extract_tool_names(tools_schema or ()),
            )
            results.append(
                ChatResult(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=output.finish_reason,
                    parse_error=parse_error,
                    raw_text=raw_text,
                    prompt=prompt,
                )
            )
        return results

    def _render_prompt(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools_schema: Sequence[dict[str, Any]],
        tool_choice: str | None,
        profile: PromptProfile,
    ) -> str:
        if profile == "tau_v1":
            return self._render_tau_v1(messages, tools_schema=tools_schema, tool_choice=tool_choice)
        if profile == "tau_v2":
            return self._render_tau_v2(messages, tools_schema=tools_schema, tool_choice=tool_choice)
        return self._render_legacy(messages, tools_schema=tools_schema, tool_choice=tool_choice)

    def _render_tau_v1(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools_schema: Sequence[dict[str, Any]],
        tool_choice: str | None,
    ) -> str:
        """Render prompt in tau-bench v1 style: system is domain policy/wiki."""
        lines: list[str] = []

        # Extract system message (domain policy)
        system_content = _extract_system_content(messages)
        if system_content:
            lines.append(system_content)
            lines.append("")

        # Tool calling guidance (minimal, since RWKV needs format hints)
        lines.append("When you need to call a tool, output JSON: {\"name\": \"<tool_name>\", \"arguments\": {...}}")
        lines.append("When you want to respond to the user, just write your response directly.")
        if tool_choice == "required":
            lines.append("A tool call is required for this turn.")
        elif tool_choice == "none":
            lines.append("Do not call any tools for this turn.")

        # Available tools
        if tools_schema:
            lines.append("")
            lines.append("Available tools:")
            for tool in tools_schema:
                lines.append(_serialize_tool(tool))

        # Extra instruction if provided
        if self._extra_system_instruction:
            lines.append("")
            lines.append(self._extra_system_instruction)

        # Conversation (skip system message)
        lines.append("")
        for message in messages:
            if message.get("role") == "system":
                continue
            lines.append(_format_message_tau_v1(message))

        lines.append("")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _render_tau_v2(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools_schema: Sequence[dict[str, Any]],
        tool_choice: str | None,
    ) -> str:
        """Render prompt in tau2-bench style: <instructions> + <policy> structure."""
        lines: list[str] = []

        # Extract system message (domain policy)
        system_content = _extract_system_content(messages)

        # Instructions block
        instruction_text = (
            "You are a customer service agent that helps the user according to the <policy> provided below.\n"
            "In each turn you can either:\n"
            "- Send a message to the user.\n"
            "- Make a tool call as JSON: {\"name\": \"<tool_name>\", \"arguments\": {...}}\n"
            "You cannot do both at the same time."
        )
        if tool_choice == "required":
            instruction_text += "\nA tool call is required for this turn."
        elif tool_choice == "none":
            instruction_text += "\nDo not call any tools for this turn."

        lines.append("<instructions>")
        lines.append(instruction_text)
        lines.append("</instructions>")

        # Policy block
        lines.append("<policy>")
        lines.append(system_content if system_content else "Follow standard customer service guidelines.")
        lines.append("</policy>")

        # Available tools
        if tools_schema:
            lines.append("")
            lines.append("<tools>")
            for tool in tools_schema:
                lines.append(_serialize_tool(tool))
            lines.append("</tools>")

        # Extra instruction if provided
        if self._extra_system_instruction:
            lines.append("")
            lines.append(self._extra_system_instruction)

        # Conversation (skip system message)
        lines.append("")
        lines.append("<conversation>")
        for message in messages:
            if message.get("role") == "system":
                continue
            lines.append(_format_message_tau_v2(message))
        lines.append("</conversation>")

        lines.append("")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _render_legacy(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools_schema: Sequence[dict[str, Any]],
        tool_choice: str | None,
    ) -> str:
        """Legacy prompt format (original strict JSON protocol)."""
        lines: list[str] = []
        lines.append("You are an AI support agent. You must follow tool definitions exactly.")
        if self._extra_system_instruction:
            lines.append(self._extra_system_instruction)

        lines.append("")
        lines.append("Output format (strict JSON only, no markdown):")
        lines.append('{"type":"tool_call","name":"<tool_name>","arguments":{...}}')
        lines.append('{"type":"response","content":"<message to user>"}')
        if tool_choice == "required":
            lines.append("A tool call is required for this turn.")
        elif tool_choice == "none":
            lines.append("Tool calls are not allowed for this turn.")

        if tools_schema:
            lines.append("")
            lines.append("Available tools:")
            for idx, tool in enumerate(tools_schema, start=1):
                lines.append(f"[{idx}] {_serialize_tool(tool)}")

        lines.append("")
        lines.append("Conversation history:")
        for idx, message in enumerate(messages, start=1):
            lines.append(f"[{idx}] {_serialize_message(message)}")

        lines.append("")
        lines.append("Now output one JSON object only.")
        return "\n".join(lines)


def _extract_system_content(messages: Sequence[dict[str, Any]]) -> str:
    """Extract system message content from messages."""
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
    return ""


def _format_message_tau_v1(message: dict[str, Any]) -> str:
    """Format a single message for tau_v1 style."""
    role = str(message.get("role", "assistant"))
    content = message.get("content")

    if role == "user":
        return f"User: {content or ''}"
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return f"Assistant: [tool_call] {json.dumps(tool_calls, ensure_ascii=False)}"
        return f"Assistant: {content or ''}"
    if role == "tool":
        tool_name = message.get("name", "unknown")
        return f"Tool ({tool_name}): {content or ''}"
    return f"{role}: {content or ''}"


def _format_message_tau_v2(message: dict[str, Any]) -> str:
    """Format a single message for tau_v2 style."""
    role = str(message.get("role", "assistant"))
    content = message.get("content")

    if role == "user":
        return f"<user>{content or ''}</user>"
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls:
            return f"<assistant><tool_call>{json.dumps(tool_calls, ensure_ascii=False)}</tool_call></assistant>"
        return f"<assistant>{content or ''}</assistant>"
    if role == "tool":
        tool_name = message.get("name", "unknown")
        return f"<tool name=\"{tool_name}\">{content or ''}</tool>"
    return f"<{role}>{content or ''}</{role}>"


def _serialize_message(message: dict[str, Any]) -> str:
    role = str(message.get("role", "assistant"))
    content = message.get("content")
    content_str = "" if content is None else str(content)
    if role == "assistant" and message.get("tool_calls"):
        tool_calls = message.get("tool_calls")
        return f"assistant tool_calls={json.dumps(tool_calls, ensure_ascii=False)}"
    if role == "tool":
        tool_name = message.get("name")
        tool_call_id = message.get("tool_call_id")
        return (
            f"tool name={tool_name!s} id={tool_call_id!s} "
            f"content={json.dumps(content_str, ensure_ascii=False)}"
        )
    return f"{role}: {json.dumps(content_str, ensure_ascii=False)}"


def _serialize_tool(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if isinstance(function, dict):
        name = function.get("name", "")
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        payload = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        return json.dumps(payload, ensure_ascii=False)
    return json.dumps(tool, ensure_ascii=False)


def _extract_tool_names(tools_schema: Sequence[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in tools_schema:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _parse_generation(raw_text: str, *, tool_names: set[str]) -> tuple[list[ParsedToolCall], str, str | None]:
    text = raw_text.strip()
    if not text:
        return [], "", "empty_output"

    for candidate in _iter_json_candidates(text):
        parsed = _loads_json(candidate)
        if parsed is None:
            continue
        tool_calls, content = _interpret_parsed_json(parsed, tool_names=tool_names)
        if tool_calls or content:
            return tool_calls, content, None

    fallback_call = _parse_function_call_fallback(text, tool_names=tool_names)
    if fallback_call is not None:
        return [fallback_call], "", "fallback:function_call_pattern"

    return [], text, "fallback:plain_text_response"


def _iter_json_candidates(text: str) -> Iterable[str]:
    seen: set[str] = set()

    def push(value: str) -> Iterable[str]:
        candidate = value.strip()
        if not candidate:
            return
        if candidate in seen:
            return
        seen.add(candidate)
        yield candidate

    yield from push(text)

    for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
        yield from push(match.group(1))

    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.DOTALL | re.IGNORECASE):
        yield from push(match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        yield from push(text[start : end + 1])


def _loads_json(candidate: str) -> Any | None:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _interpret_parsed_json(parsed: Any, *, tool_names: set[str]) -> tuple[list[ParsedToolCall], str]:
    if isinstance(parsed, list):
        if not parsed:
            return [], ""
        # Sometimes model outputs a list with a single call object.
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        return [], ""

    direct_calls = parsed.get("tool_calls")
    if isinstance(direct_calls, list) and direct_calls:
        calls = _parse_tool_call_list(direct_calls, tool_names=tool_names)
        if calls:
            return calls, ""

    nested_call = parsed.get("tool_call")
    if isinstance(nested_call, dict):
        calls = _parse_tool_call_list([nested_call], tool_names=tool_names)
        if calls:
            return calls, ""

    action_type = str(parsed.get("type", "")).lower()
    if action_type in {"tool_call", "function_call", "call_tool"}:
        name = parsed.get("name") or parsed.get("tool_name")
        arguments = parsed.get("arguments") if "arguments" in parsed else parsed.get("kwargs")
        tool_call = _build_tool_call(name, arguments, tool_names=tool_names)
        if tool_call is not None:
            return [tool_call], ""

    name = parsed.get("name")
    if isinstance(name, str):
        arguments = parsed.get("arguments") if "arguments" in parsed else parsed.get("kwargs")
        tool_call = _build_tool_call(name, arguments, tool_names=tool_names)
        if tool_call is not None:
            return [tool_call], ""

    content = (
        parsed.get("content")
        or parsed.get("response")
        or parsed.get("assistant_response")
        or parsed.get("message")
    )
    if content is None:
        return [], ""
    return [], str(content)


def _parse_tool_call_list(items: Sequence[Any], *, tool_names: set[str]) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        name = item.get("name")
        arguments: Any = item.get("arguments") if "arguments" in item else item.get("kwargs")
        call_id = item.get("id")
        if isinstance(function, dict):
            name = function.get("name", name)
            if arguments is None:
                arguments = function.get("arguments")
        tool_call = _build_tool_call(name, arguments, tool_names=tool_names, call_id=call_id)
        if tool_call is not None:
            calls.append(tool_call)
    return calls


def _build_tool_call(
    name: Any,
    arguments: Any,
    *,
    tool_names: set[str],
    call_id: Any | None = None,
) -> ParsedToolCall | None:
    if not isinstance(name, str) or not name.strip():
        return None
    normalized_name = name.strip()
    if tool_names and normalized_name not in tool_names:
        return None

    args_dict: dict[str, Any]
    if isinstance(arguments, dict):
        args_dict = dict(arguments)
    elif isinstance(arguments, str) and arguments.strip():
        loaded = _loads_json(arguments)
        args_dict = dict(loaded) if isinstance(loaded, dict) else {}
    else:
        args_dict = {}

    if isinstance(call_id, str) and call_id:
        tool_call_id = call_id
    else:
        tool_call_id = f"call_{uuid.uuid4().hex[:10]}"

    return ParsedToolCall(id=tool_call_id, name=normalized_name, arguments=args_dict)


def _parse_function_call_fallback(text: str, *, tool_names: set[str]) -> ParsedToolCall | None:
    if not tool_names:
        return None
    ordered_names = sorted(tool_names, key=len, reverse=True)
    for name in ordered_names:
        pattern = re.compile(rf"\b{re.escape(name)}\s*\((\{{.*\}})\)", flags=re.DOTALL)
        match = pattern.search(text)
        if not match:
            continue
        args = _loads_json(match.group(1))
        if isinstance(args, dict):
            return ParsedToolCall(
                id=f"call_{uuid.uuid4().hex[:10]}",
                name=name,
                arguments=args,
            )
    return None


__all__ = [
    "PromptProfile",
    "ParsedToolCall",
    "ChatResult",
    "RWKVChatBridge",
]
