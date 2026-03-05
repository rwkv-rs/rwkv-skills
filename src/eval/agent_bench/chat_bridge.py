from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.infer.engine import InferenceEngine
from src.infer.sampling import SamplingConfig


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
        engine: InferenceEngine,
        default_sampling: SamplingConfig,
        extra_system_instruction: str | None = None,
    ) -> None:
        self._engine = engine
        self._default_sampling = default_sampling
        self._extra_system_instruction = (extra_system_instruction or "").strip()

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        tools_schema: Sequence[dict[str, Any]] | None = None,
        *,
        sampling: SamplingConfig | None = None,
        tool_choice: str | None = None,
    ) -> ChatResult:
        prompt = self._render_prompt(messages, tools_schema=tools_schema or (), tool_choice=tool_choice)
        outputs = self._engine.generate(
            [prompt],
            sampling=sampling or self._default_sampling,
            batch_size=1,
            progress_desc="AgentBench",
        )
        if not outputs:
            return ChatResult(
                content="",
                tool_calls=[],
                finish_reason="error",
                parse_error="rwkv_generate_empty",
                raw_text="",
                prompt=prompt,
            )
        output = outputs[0]
        raw_text = (output.text or "").strip()
        tool_calls, content, parse_error = _parse_generation(
            raw_text,
            tool_names=_extract_tool_names(tools_schema or ()),
        )
        return ChatResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=output.finish_reason,
            parse_error=parse_error,
            raw_text=raw_text,
            prompt=prompt,
        )

    def _render_prompt(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools_schema: Sequence[dict[str, Any]],
        tool_choice: str | None,
    ) -> str:
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
    "ParsedToolCall",
    "ChatResult",
    "RWKVChatBridge",
]
