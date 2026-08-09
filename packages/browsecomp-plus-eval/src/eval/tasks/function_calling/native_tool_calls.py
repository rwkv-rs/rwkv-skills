from __future__ import annotations

"""OpenAI chat-tools bridge for function-calling runners."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.eval.tasks.function_calling.tool_call_contract import (
    coerce_tool_call_payloads,
    normalize_tool_schema,
)

TOOL_CALL_IO_OPENAI_TOOLS = "openai-tools"


@dataclass(frozen=True, slots=True)
class NativeToolCallDecision:
    prompt: str
    completion: str
    finish_reason: str
    decoded_calls: list[dict[str, Any]]
    parse_error: str | None
    trace: dict[str, Any]


def openai_tool_schema(tool: Mapping[str, Any]) -> dict[str, Any]:
    if str(tool.get("type") or "") == "function" and isinstance(tool.get("function"), Mapping):
        return {"type": "function", "function": dict(tool["function"])}  # type: ignore[index]
    schema = normalize_tool_schema(tool)
    name = str(schema.get("name") or "").strip()
    if not name:
        raise ValueError("tool schema missing function name")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": str(schema.get("description") or ""),
            "parameters": dict(schema.get("parameters") or {"type": "object", "properties": {}}),
        },
    }


def openai_tool_schemas(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [openai_tool_schema(tool) for tool in tools]


def assistant_message_from_calls(
    calls: Sequence[Mapping[str, Any]],
    *,
    content: str | None = None,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": str(call.get("id") or f"call_{index}"),
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": json.dumps(dict(call.get("arguments") or {}), ensure_ascii=False, separators=(",", ":")),
                },
            }
            for index, call in enumerate(calls)
        ],
    }


def serialize_chat_payload(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def run_native_tool_call_decision(
    *,
    engine: object,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    sampling: Any,
    progress_desc: str,
    prompt_seed: int | None = None,
    tool_choice: object = "auto",
    parallel_tool_calls: bool | None = None,
    context_label: str = "native tool call",
) -> NativeToolCallDecision:
    generate_tool_calls = getattr(engine, "generate_tool_calls", None)
    if not callable(generate_tool_calls):
        raise NotImplementedError("inference backend does not support native chat tool calls")

    request_messages = [dict(message) for message in messages]
    request_tools = openai_tool_schemas(tools)
    outputs = generate_tool_calls(
        [request_messages],
        [request_tools],
        sampling=sampling,
        batch_size=1,
        progress_desc=progress_desc,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        prompt_seeds=[prompt_seed],
    )
    output = outputs[0]
    prompt = serialize_chat_payload(
        {
            "messages": request_messages,
            "tools": request_tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
    )
    assistant_message = dict(getattr(output, "raw_message", {}) or {})
    if not assistant_message:
        assistant_message = assistant_message_from_calls(
            [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in output.tool_calls],
            content=output.content or None,
        )
    completion = serialize_chat_payload(assistant_message)

    decoded_calls: list[dict[str, Any]] = []
    parse_error: str | None = None
    try:
        if output.tool_calls:
            parsed_calls = coerce_tool_call_payloads(
                {"tool_calls": [call.as_openai_tool_call() for call in output.tool_calls]},
                context_label=context_label,
            )
            decoded_calls = [call.layer_payload() for call in parsed_calls]
    except Exception as exc:  # noqa: BLE001 - native fallback parser errors are sample-level failures
        parse_error = str(exc)

    trace = {
        "tool_call_io": TOOL_CALL_IO_OPENAI_TOOLS,
        "request": {
            "messages": request_messages,
            "tools": request_tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        },
        "assistant_message": assistant_message,
        "response_source": str(getattr(output, "response_source", "")),
        "decoded_calls": decoded_calls,
        "parse_error": parse_error or "",
    }
    return NativeToolCallDecision(
        prompt=prompt,
        completion=completion,
        finish_reason=str(getattr(output, "finish_reason", "stop") or "stop"),
        decoded_calls=decoded_calls,
        parse_error=parse_error,
        trace=trace,
    )


__all__ = [
    "TOOL_CALL_IO_OPENAI_TOOLS",
    "NativeToolCallDecision",
    "openai_tool_schema",
    "openai_tool_schemas",
    "run_native_tool_call_decision",
    "serialize_chat_payload",
]
