from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .context_budget import normalize_rwkv_text, trim_message_history

RWKV_OFFICIAL_JSON_PROMPT_STYLE = "rwkv_official_json"
FUNCTION_PROMPT_STYLE_CHOICES = (RWKV_OFFICIAL_JSON_PROMPT_STYLE,)
DEFAULT_FUNCTION_PROMPT_STYLE = RWKV_OFFICIAL_JSON_PROMPT_STYLE

JSON_TOOL_CATALOG_FORMAT = "json"
FUNCTION_TOOL_CATALOG_FORMAT_CHOICES = (JSON_TOOL_CATALOG_FORMAT,)
DEFAULT_TOOL_CATALOG_FORMAT = JSON_TOOL_CATALOG_FORMAT

JSON_CALL_STOP_SUFFIXES = (
    "\n```",
    "```",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
)


def normalize_function_prompt_style(value: str | None) -> str:
    normalized = str(value or DEFAULT_FUNCTION_PROMPT_STYLE).strip().lower()
    if normalized not in FUNCTION_PROMPT_STYLE_CHOICES:
        raise ValueError(
            f"unsupported function prompt style {value!r}; "
            f"expected one of {', '.join(FUNCTION_PROMPT_STYLE_CHOICES)}"
        )
    return normalized


def normalize_tool_catalog_format(value: str | None) -> str:
    normalized = str(value or DEFAULT_TOOL_CATALOG_FORMAT).strip().lower()
    if normalized not in FUNCTION_TOOL_CATALOG_FORMAT_CHOICES:
        raise ValueError(
            f"unsupported function tool catalog format {value!r}; "
            f"expected one of {', '.join(FUNCTION_TOOL_CATALOG_FORMAT_CHOICES)}"
        )
    return normalized


def assistant_json_prefix() -> str:
    return "Assistant: ```json\n"


def render_json_function_call(name: str, arguments: Mapping[str, Any] | None = None) -> str:
    payload = {
        "name": str(name).strip(),
        "arguments": dict(arguments or {}),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_assistant_json_block(json_text: str) -> str:
    return f"{assistant_json_prefix()}{normalize_rwkv_text(json_text)}\n```"


def render_function_output_user_block(payload: Any) -> str:
    return "User: Function output:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_rwkv_json_call_prompt(
    system_prompt: str,
    messages: Sequence[Mapping[str, object]],
    *,
    history_max_chars: int,
) -> str:
    bounded_messages = trim_message_history(messages, max_chars=max(0, int(history_max_chars)))
    parts = [f"System: {normalize_rwkv_text(system_prompt)}"]
    for message in bounded_messages:
        role = str(message.get("role") or "").strip().lower()
        content = normalize_rwkv_text(str(message.get("content") or ""))
        if not content:
            continue
        if role == "assistant":
            if content.startswith("Assistant: "):
                parts.append(content)
            elif _looks_like_json_call(content):
                parts.append(render_assistant_json_block(_strip_json_fence(content)))
            else:
                parts.append(f"Assistant: {content}")
            continue
        if content.startswith("User: "):
            parts.append(content)
        else:
            parts.append(f"User: {content}")
    parts.append(assistant_json_prefix())
    return "\n\n".join(parts)


def extract_json_call_object_text(response: str) -> str:
    candidate = extract_json_call_value_text(response)
    if not candidate.startswith("{"):
        raise ValueError(f"model response must be a JSON function call object: {candidate}")
    return candidate


def extract_json_call_value_text(response: str) -> str:
    normalized = _strip_assistant_prefix(normalize_rwkv_text(response))
    normalized = _strip_json_fence(normalized)
    if not (normalized.startswith("{") or normalized.startswith("[")):
        raise ValueError(f"model response must be a JSON function call object or array: {normalized}")
    end = _find_leading_json_value_end(normalized)
    if end is None:
        raise ValueError(f"model response must be a complete JSON function call object or array: {normalized}")
    candidate = normalized[:end]
    trailing = normalize_rwkv_text(normalized[end:])
    trailing = _strip_json_fence(trailing) if trailing.startswith("```") else trailing
    if trailing and trailing != "```":
        raise ValueError(f"model response has extra text after JSON function call object or array: {trailing}")
    return candidate


def coerce_json_function_call_payload(payload: Any, *, context_label: str = "tool call") -> dict[str, Any]:
    calls = _coerce_json_function_call_payloads(payload, context_label=context_label)
    if not calls:
        raise ValueError(f"{context_label} payload did not contain a function call")
    return calls[0]


def coerce_json_function_call_payloads(payload: Any, *, context_label: str = "tool call") -> list[dict[str, Any]]:
    calls = _coerce_json_function_call_payloads(payload, context_label=context_label)
    if not calls:
        raise ValueError(f"{context_label} payload did not contain a function call")
    return calls


def _coerce_json_function_call_payloads(payload: Any, *, context_label: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        calls: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"{context_label} list item #{index} must be a JSON object")
            calls.append(_coerce_json_function_call_mapping(item, context_label=context_label))
        return calls
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context_label} payload must be a JSON object")
    if "tool_calls" in payload:
        tool_calls = payload.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError(f"{context_label} tool_calls payload must contain at least one tool call object")
        calls: list[dict[str, Any]] = []
        for index, item in enumerate(tool_calls):
            if not isinstance(item, Mapping):
                raise ValueError(f"{context_label} tool_calls item #{index} must be a JSON object")
            calls.append(_coerce_json_function_call_mapping(item, context_label=context_label))
        return calls
    return [_coerce_json_function_call_mapping(payload, context_label=context_label)]


def _coerce_json_function_call_mapping(payload: Mapping[str, Any], *, context_label: str) -> dict[str, Any]:
    function_payload = payload.get("function")
    if not isinstance(function_payload, Mapping):
        function_payload = payload.get("function_call")
    if isinstance(function_payload, Mapping):
        name = function_payload.get("name") or payload.get("name")
        arguments = function_payload.get("arguments", payload.get("arguments", {}))
    else:
        name = payload.get("name")
        arguments = payload.get("arguments", {})

    name_text = str(name or "").strip()
    if not name_text:
        raise ValueError(f"{context_label} missing name")
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
                raise ValueError(f"{context_label} string arguments must be a JSON object") from exc
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{context_label} arguments must be a JSON object")
    return {"name": name_text, "arguments": dict(arguments)}


def _looks_like_json_call(text: str) -> bool:
    stripped = normalize_rwkv_text(text)
    return stripped.startswith("{") or stripped.startswith("```")


def _strip_assistant_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("Assistant:"):
        return normalize_rwkv_text(stripped[len("Assistant:") :])
    return stripped


def _strip_json_fence(text: str) -> str:
    normalized = normalize_rwkv_text(text)
    if normalized.startswith("```"):
        lines = normalized.split("\n")
        if lines:
            first = lines[0].strip().lower()
            if first in {"```", "```json", "```javascript", "```js"}:
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                normalized = normalize_rwkv_text("\n".join(lines))
    if normalized.endswith("```"):
        normalized = normalize_rwkv_text(normalized[: -len("```")])
    return normalized


def _find_leading_json_value_end(text: str) -> int | None:
    if not text:
        return None
    opening = text[0]
    if opening not in "{[":
        return None
    expected_stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            expected_stack.append("}")
            continue
        if char == "[":
            expected_stack.append("]")
            continue
        if char in "}]":
            if not expected_stack or expected_stack[-1] != char:
                return None
            expected_stack.pop()
            if not expected_stack:
                return index + 1
            continue
    return None


__all__ = [
    "DEFAULT_FUNCTION_PROMPT_STYLE",
    "DEFAULT_TOOL_CATALOG_FORMAT",
    "FUNCTION_PROMPT_STYLE_CHOICES",
    "FUNCTION_TOOL_CATALOG_FORMAT_CHOICES",
    "JSON_CALL_STOP_SUFFIXES",
    "JSON_TOOL_CATALOG_FORMAT",
    "RWKV_OFFICIAL_JSON_PROMPT_STYLE",
    "assistant_json_prefix",
    "build_rwkv_json_call_prompt",
    "coerce_json_function_call_payload",
    "coerce_json_function_call_payloads",
    "extract_json_call_object_text",
    "extract_json_call_value_text",
    "normalize_function_prompt_style",
    "normalize_tool_catalog_format",
    "render_assistant_json_block",
    "render_function_output_user_block",
    "render_json_function_call",
]
