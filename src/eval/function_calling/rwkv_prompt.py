from __future__ import annotations

import ast
import json
import re
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

USER_HEADER = "User:"
ASSISTANT_HEADER = "Assistant:"
SYSTEM_HEADER = "System:"

JSON_CALL_OBJECT_PREFILL = "{"
JSON_CALL_OBJECT_PREFILL_FIELDS = frozenset(
    {
        "action",
        "function",
        "function_call",
        "name",
        "tool",
        "tool_calls",
        "tool_name",
        "type",
    }
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


def assistant_json_prefix(*, enable_think: bool = False, prefill_object: bool = True) -> str:
    suffix = "{" if prefill_object else ""
    if not enable_think:
        return f"Assistant: ```json\n{suffix}"
    return f"Assistant: <think>\n</think>\n```json\n{suffix}"


def render_system_block(content: str) -> str:
    return f"{SYSTEM_HEADER} {normalize_rwkv_text(content)}".rstrip(" ")


def render_user_block(content: str) -> str:
    rendered = _strip_role_prefix(content, USER_HEADER)
    rendered = _sanitize_embedded_role_headers(rendered)
    return f"{USER_HEADER} {rendered}".rstrip(" ")


def render_assistant_text_block(content: str) -> str:
    rendered = _strip_role_prefix(content, ASSISTANT_HEADER)
    return f"{ASSISTANT_HEADER} {rendered}".rstrip(" ")


def render_json_function_call(name: str, arguments: Mapping[str, Any] | None = None) -> str:
    payload = {
        "name": str(name).strip(),
        "arguments": dict(arguments or {}),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_assistant_json_block(json_text: str, *, assistant_prefix: str | None = None) -> str:
    prefix = assistant_prefix if assistant_prefix is not None else assistant_json_prefix()
    rendered = normalize_rwkv_text(json_text)
    if prefix.rstrip().endswith("{") and rendered.startswith("{"):
        rendered = rendered[1:]
    elif prefix.rstrip().endswith("{") and rendered.startswith("["):
        prefix = prefix[: prefix.rfind("{")]
    return f"{prefix}{rendered}\n```"


def render_function_output_user_block(payload: Any) -> str:
    return render_user_block("Function output:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def build_rwkv_json_call_prompt(
    system_prompt: str,
    messages: Sequence[Mapping[str, object]],
    *,
    history_max_chars: int,
    assistant_prefix: str | None = None,
    single_user_turn: bool = True,
) -> str:
    bounded_messages = trim_message_history(messages, max_chars=max(0, int(history_max_chars)))
    json_prefix = assistant_prefix if assistant_prefix is not None else assistant_json_prefix()
    parts = [render_system_block(system_prompt)]
    if single_user_turn:
        user_content = _render_single_user_turn_content(bounded_messages)
        if user_content:
            parts.append(render_user_block(user_content))
        parts.append(json_prefix)
        return "\n\n".join(parts)
    for message in bounded_messages:
        role = str(message.get("role") or "").strip().lower()
        content = normalize_rwkv_text(str(message.get("content") or ""))
        if not content:
            continue
        if role == "assistant":
            assistant_content = _strip_role_prefix(content, ASSISTANT_HEADER)
            if _looks_like_json_call(assistant_content):
                parts.append(render_assistant_json_block(_strip_json_fence(assistant_content), assistant_prefix=json_prefix))
            elif _looks_like_json_call(content):
                parts.append(render_assistant_json_block(_strip_json_fence(content), assistant_prefix=json_prefix))
            else:
                parts.append(render_assistant_text_block(content))
            continue
        parts.append(render_user_block(content))
    parts.append(json_prefix)
    return "\n\n".join(parts)


def extract_json_call_object_text(response: str) -> str:
    candidate = extract_json_call_value_text(response)
    if not candidate.startswith("{"):
        raise ValueError(f"model response must be a JSON function call object: {candidate}")
    return candidate


def extract_json_call_value_text(response: str) -> str:
    normalized = _strip_assistant_prefix(normalize_rwkv_text(response))
    normalized = _strip_leading_think_block(normalized)
    normalized = _strip_json_fence(normalized)
    agentic = _agentic_tool_call_to_json_text(normalized)
    if agentic is not None:
        return agentic
    normalized = apply_json_call_object_prefill(normalized)
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


def apply_json_call_object_prefill(response: str) -> str:
    normalized = normalize_rwkv_text(response)
    stripped = normalized.lstrip()
    if not stripped:
        return normalized
    if stripped.startswith(("{", "[", "```")):
        return normalized
    field_match = re.match(r'^"(?P<field>[A-Za-z_][A-Za-z0-9_]*)"\s*:', stripped)
    if field_match and field_match.group("field") in JSON_CALL_OBJECT_PREFILL_FIELDS:
        return JSON_CALL_OBJECT_PREFILL + stripped
    return normalized


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
    return stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("```")


def _render_single_user_turn_content(messages: Sequence[Mapping[str, str]]) -> str:
    normalized_messages: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = normalize_rwkv_text(str(message.get("content") or ""))
        if not content:
            continue
        if role == "assistant":
            content = _strip_role_prefix(content, ASSISTANT_HEADER)
            if _looks_like_json_call(content):
                content = _strip_json_fence(content)
        elif role == "user":
            content = _strip_role_prefix(content, USER_HEADER)
        normalized_messages.append({"role": role, "content": content})

    if not normalized_messages:
        return ""
    if len(normalized_messages) == 1 and normalized_messages[0]["role"] == "user":
        return _sanitize_embedded_role_headers(normalized_messages[0]["content"])
    return "Conversation transcript JSON:\n" + json.dumps(
        normalized_messages,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sanitize_embedded_role_headers(text: str) -> str:
    normalized = normalize_rwkv_text(text)
    replacements = {
        USER_HEADER: "User message:",
        ASSISTANT_HEADER: "Assistant message:",
        SYSTEM_HEADER: "System message:",
    }
    rendered: list[str] = []
    for line in normalized.split("\n"):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        for header, replacement in replacements.items():
            if stripped.startswith(header):
                stripped = replacement + stripped[len(header) :]
                break
        rendered.append(indent + stripped)
    return "\n".join(rendered)


def _strip_assistant_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("Assistant:"):
        return normalize_rwkv_text(stripped[len("Assistant:") :])
    return stripped


def _agentic_tool_call_to_json_text(text: str) -> str | None:
    normalized = normalize_rwkv_text(text)
    tagged = re.match(r"(?is)^<tool_call>\s*(.*?)\s*</tool_call>\s*$", normalized)
    if tagged is not None:
        return _agentic_value_to_json_text(tagged.group(1))

    react = _react_action_to_json_text(normalized)
    if react is not None:
        return react

    label_match = re.match(
        r"(?is)^(?:\*\*)?\s*(?:tool\s+call|assistant\s+action)\s*:\s*(?:\*\*)?\s*(?P<body>.+)$",
        normalized,
    )
    if label_match is None:
        return None
    return _agentic_value_to_json_text(label_match.group("body"))


def _agentic_value_to_json_text(value: str) -> str | None:
    normalized = _strip_json_fence(normalize_rwkv_text(value))
    normalized = apply_json_call_object_prefill(normalized)
    if normalized.startswith(("{", "[")):
        end = _find_leading_json_value_end(normalized)
        if end is not None:
            return normalized[:end]
    functional = _functional_tool_call_to_payload(normalized)
    if functional is None:
        return None
    return json.dumps(functional, ensure_ascii=False, separators=(",", ":"))


def _react_action_to_json_text(text: str) -> str | None:
    match = re.match(
        r"(?is)^Action\s*:\s*(?P<name>[A-Za-z_][\w.:-]*)\s*(?:\n|$)\s*Action\s*Input\s*:\s*(?P<arguments>.+)$",
        normalize_rwkv_text(text),
    )
    if match is None:
        return None
    arguments = _parse_function_arguments_text(match.group("arguments"))
    if arguments is None:
        return None
    return json.dumps(
        {"name": match.group("name"), "arguments": arguments},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _functional_tool_call_to_payload(text: str) -> dict[str, Any] | None:
    match = re.match(r"(?s)^(?P<name>[A-Za-z_][\w.:-]*)\s*\((?P<arguments>.*)\)\s*$", normalize_rwkv_text(text))
    if match is None:
        return None
    arguments = _parse_function_arguments_text(match.group("arguments"))
    if arguments is None:
        return None
    return {"name": match.group("name"), "arguments": arguments}


def _parse_function_arguments_text(text: str) -> dict[str, Any] | None:
    raw = normalize_rwkv_text(text)
    if not raw:
        return {}
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (SyntaxError, TypeError, ValueError):
                return None
        return dict(parsed) if isinstance(parsed, Mapping) else None
    try:
        parsed_call = ast.parse(f"_rwkv_tool_call({raw})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(parsed_call, ast.Call):
        return None
    arguments: dict[str, Any] = {}
    if parsed_call.args:
        if len(parsed_call.args) != 1:
            return None
        try:
            positional = ast.literal_eval(parsed_call.args[0])
        except (SyntaxError, TypeError, ValueError):
            return None
        if not isinstance(positional, Mapping):
            return None
        arguments.update(dict(positional))
    for keyword in parsed_call.keywords:
        if keyword.arg is None:
            return None
        try:
            arguments[str(keyword.arg)] = ast.literal_eval(keyword.value)
        except (SyntaxError, TypeError, ValueError):
            return None
    return arguments


def _strip_role_prefix(text: str, prefix: str) -> str:
    stripped = normalize_rwkv_text(text)
    if stripped.startswith(prefix):
        return normalize_rwkv_text(stripped[len(prefix) :])
    return stripped


def _strip_leading_think_block(text: str) -> str:
    normalized = normalize_rwkv_text(text)
    if not normalized.startswith("<think>"):
        return normalized
    close = normalized.find("</think>")
    if close < 0:
        return normalized
    return normalize_rwkv_text(normalized[close + len("</think>") :])


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
    "JSON_CALL_OBJECT_PREFILL",
    "JSON_TOOL_CATALOG_FORMAT",
    "RWKV_OFFICIAL_JSON_PROMPT_STYLE",
    "apply_json_call_object_prefill",
    "assistant_json_prefix",
    "build_rwkv_json_call_prompt",
    "coerce_json_function_call_payload",
    "coerce_json_function_call_payloads",
    "extract_json_call_object_text",
    "extract_json_call_value_text",
    "normalize_function_prompt_style",
    "normalize_tool_catalog_format",
    "render_assistant_json_block",
    "render_assistant_text_block",
    "render_function_output_user_block",
    "render_system_block",
    "render_user_block",
    "render_json_function_call",
]
