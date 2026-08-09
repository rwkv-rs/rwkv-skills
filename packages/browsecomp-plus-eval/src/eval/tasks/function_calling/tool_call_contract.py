"""Shared JSON tool-call parsing and schema-contract helpers.

Normal text output uses the strict ``{"name": ..., "arguments": {...}}``
contract. OpenAI native ``tool_calls`` responses are handled as an explicit,
separate envelope rather than as aliases for normal output.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.eval.tasks.function_calling.context_budget import normalize_rwkv_text
from src.eval.tasks.function_calling.rwkv_prompt import extract_json_call_value_text


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    raw_payload: dict[str, Any]

    def layer_payload(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


def parse_tool_call_text(
    text: str,
    *,
    context_label: str = "tool call",
    recover_partial: bool = True,
    allowed_metadata_keys: Sequence[str] = (),
) -> ParsedToolCall:
    calls = parse_tool_calls_text(
        text,
        context_label=context_label,
        recover_partial=recover_partial,
        allowed_metadata_keys=allowed_metadata_keys,
    )
    if not calls:
        raise ValueError(f"{context_label} payload did not contain a tool call")
    return calls[0]


def parse_tool_calls_text(
    text: str,
    *,
    context_label: str = "tool call",
    recover_partial: bool = True,
    allowed_metadata_keys: Sequence[str] = (),
) -> list[ParsedToolCall]:
    payload = load_tool_call_payload(text, context_label=context_label, recover_partial=recover_partial)
    return coerce_tool_call_payloads(
        payload,
        context_label=context_label,
        allowed_metadata_keys=allowed_metadata_keys,
    )


def load_tool_call_payload(
    text: str,
    *,
    context_label: str = "tool call",
    recover_partial: bool = True,
) -> Any:
    try:
        candidate = extract_json_call_value_text(text)
    except ValueError as exc:
        if not recover_partial or not _should_recover_partial_tool_payload(text):
            raise
        return partial_tool_call_payload(text, context_label=context_label, cause=exc)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, TypeError, ValueError):
            if not recover_partial or not _should_recover_partial_tool_payload(text):
                raise
            return partial_tool_call_payload(text, context_label=context_label, cause=exc)


def coerce_tool_call_payloads(
    payload: Any,
    *,
    context_label: str = "tool call",
    allowed_metadata_keys: Sequence[str] = (),
) -> list[ParsedToolCall]:
    if isinstance(payload, list):
        calls: list[ParsedToolCall] = []
        for index, item in enumerate(payload):
            if not isinstance(item, Mapping):
                raise ValueError(f"{context_label} list item #{index} must be a JSON object")
            calls.append(
                _coerce_tool_call_mapping(
                    item,
                    context_label=context_label,
                    allowed_metadata_keys=allowed_metadata_keys,
                )
            )
        return calls
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context_label} payload must be a JSON object")
    if "tool_calls" in payload:
        extra_envelope_keys = sorted(key for key in payload if key != "tool_calls")
        if extra_envelope_keys:
            raise ValueError(f"{context_label} native envelope has unsupported fields: {extra_envelope_keys}")
        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes, bytearray)) or not raw_calls:
            raise ValueError(f"{context_label} tool_calls payload must contain at least one tool call object")
        calls = []
        for index, item in enumerate(raw_calls):
            if not isinstance(item, Mapping):
                raise ValueError(f"{context_label} tool_calls item #{index} must be a JSON object")
            calls.append(_coerce_native_tool_call_mapping(item, context_label=context_label))
        return calls
    return [
        _coerce_tool_call_mapping(
            payload,
            context_label=context_label,
            allowed_metadata_keys=allowed_metadata_keys,
        )
    ]


def partial_tool_call_payload(text: str, *, context_label: str = "tool call", cause: Exception) -> dict[str, Any]:
    normalized = normalize_rwkv_text(text)
    stripped = normalized.lstrip()
    if stripped.startswith('"'):
        body = "{" + stripped
    else:
        start = normalized.find("{")
        if start < 0:
            raise ValueError(f"{context_label} response missing JSON object: {normalized}") from cause
        body = normalized[start:]
    name = _first_json_field(body, ("name",))
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{context_label} response missing recoverable name: {normalized}") from cause
    arguments = _first_json_field(body, ("arguments",))
    arguments = _coerce_arguments(arguments, context_label=context_label)
    payload: dict[str, Any] = {"name": name, "arguments": arguments}
    for key in ("confidence", "score", "evidence", "reason", "rationale"):
        try:
            payload[key] = _first_json_field(body, (key,))
        except ValueError:
            pass
    return payload


def _should_recover_partial_tool_payload(text: str) -> bool:
    stripped = normalize_rwkv_text(text).lstrip()
    if not stripped:
        return False
    return stripped.startswith(("Assistant:", "```", "{", "[", '"'))


def tool_name(tool: Any) -> str:
    schema = normalize_tool_schema(tool)
    return str(schema.get("name") or "").strip()


def normalize_tool_schema(tool: Any) -> dict[str, Any]:
    raw = getattr(tool, "openai_schema", tool)
    if callable(raw):
        raw = raw()
    if not isinstance(raw, Mapping):
        return {}
    schema: dict[str, Any]
    function_payload = raw.get("function")
    if isinstance(function_payload, Mapping):
        schema = dict(function_payload)
    else:
        schema = dict(raw)
    if "parameters" not in schema and isinstance(schema.get("arguments"), Mapping):
        schema["parameters"] = schema["arguments"]
    return schema


def required_arguments_by_tool_name(tools: Sequence[Any], *, extra_tools: Sequence[Mapping[str, Any]] = ()) -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for schema in _iter_tool_schemas(tools, extra_tools=extra_tools):
        name = str(schema.get("name") or "").strip()
        if not name:
            continue
        parameters = schema.get("parameters")
        if not isinstance(parameters, Mapping):
            required.setdefault(name, set())
            continue
        raw_required = parameters.get("required")
        if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes, bytearray)):
            required[name] = {str(item) for item in raw_required if str(item)}
        else:
            required.setdefault(name, set())
    return required


def allowed_arguments_by_tool_name(tools: Sequence[Any], *, extra_tools: Sequence[Mapping[str, Any]] = ()) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    for schema in _iter_tool_schemas(tools, extra_tools=extra_tools):
        name = str(schema.get("name") or "").strip()
        if not name:
            continue
        parameters = schema.get("parameters")
        if not isinstance(parameters, Mapping):
            continue
        properties = parameters.get("properties")
        if isinstance(properties, Mapping):
            allowed[name] = {str(key) for key in properties}
    return allowed


def validate_tool_call_name(call: ParsedToolCall, *, valid_names: set[str], context_label: str = "tool call") -> None:
    if call.name not in valid_names:
        raise ValueError(f"{context_label} name {call.name!r} not in valid tool names")


def validate_tool_call_required_arguments(
    call: ParsedToolCall,
    *,
    required_args_by_name: Mapping[str, set[str]],
    context_label: str = "tool call",
) -> None:
    required = required_args_by_name.get(call.name) or set()
    missing = sorted(key for key in required if key not in call.arguments or call.arguments.get(key) is None)
    if missing:
        raise ValueError(f"{context_label} {call.name!r} missing required arguments: {missing}")


def prune_tool_call_arguments(
    call: ParsedToolCall,
    *,
    allowed_args_by_name: Mapping[str, set[str]],
) -> ParsedToolCall:
    allowed = allowed_args_by_name.get(call.name)
    if allowed is None:
        return call
    pruned = {key: value for key, value in call.arguments.items() if key in allowed}
    if pruned == call.arguments:
        return call
    return ParsedToolCall(name=call.name, arguments=pruned, raw_payload=dict(call.raw_payload))


def normalize_tool_call_arguments(
    call: ParsedToolCall,
    *,
    aliases: Mapping[str, str],
) -> ParsedToolCall:
    normalized = normalize_argument_aliases(call.arguments, aliases=aliases)
    if normalized == call.arguments:
        return call
    return ParsedToolCall(name=call.name, arguments=normalized, raw_payload=dict(call.raw_payload))


def normalize_argument_aliases(value: Any, *, aliases: Mapping[str, str]) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key)
            normalized_key = aliases.get(key.lower(), key)
            normalized_value = normalize_argument_aliases(raw_item, aliases=aliases)
            if normalized_key in normalized and normalized[normalized_key] not in (None, "", [], {}):
                continue
            normalized[normalized_key] = normalized_value
        return normalized
    if isinstance(value, list):
        return [normalize_argument_aliases(item, aliases=aliases) for item in value]
    if isinstance(value, tuple):
        return [normalize_argument_aliases(item, aliases=aliases) for item in value]
    return value


def strip_requestor_prefix(name: str) -> str:
    if "." not in name:
        return name
    prefix, rest = name.split(".", 1)
    if prefix in {"assistant", "user"} and rest.strip():
        return rest.strip()
    return name


def _coerce_tool_call_mapping(
    payload: Mapping[str, Any],
    *,
    context_label: str,
    allowed_metadata_keys: Sequence[str],
) -> ParsedToolCall:
    raw = dict(payload)
    legacy_keys = sorted(
        key
        for key in (
            "tool",
            "tool_name",
            "parameters",
            "action",
            "action_input",
            "input",
            "candidate",
            "function",
            "function_call",
        )
        if key in raw
    )
    if legacy_keys:
        raise ValueError(f"{context_label} uses unsupported legacy fields: {legacy_keys}")
    allowed_keys = {"name", "arguments", *(str(key) for key in allowed_metadata_keys)}
    extra_keys = sorted(str(key) for key in raw if str(key) not in allowed_keys)
    if extra_keys:
        raise ValueError(f"{context_label} has unsupported extra fields: {extra_keys}")
    if "name" not in raw:
        raise ValueError(f"{context_label} missing name")
    if "arguments" not in raw:
        raise ValueError(f"{context_label} missing arguments")
    name = raw.get("name")
    arguments = raw.get("arguments")
    name_text = strip_requestor_prefix(str(name or "").strip())
    if not name_text:
        raise ValueError(f"{context_label} missing name")
    return ParsedToolCall(
        name=name_text,
        arguments=_coerce_arguments(arguments, context_label=context_label),
        raw_payload=raw,
    )


def _coerce_native_tool_call_mapping(payload: Mapping[str, Any], *, context_label: str) -> ParsedToolCall:
    raw = dict(payload)
    function_payload = raw.get("function")
    if not isinstance(function_payload, Mapping):
        raise ValueError(f"{context_label} native tool call missing function")
    if "name" not in function_payload or "arguments" not in function_payload:
        raise ValueError(f"{context_label} native function requires name and arguments")
    name_text = strip_requestor_prefix(str(function_payload.get("name") or "").strip())
    if not name_text:
        raise ValueError(f"{context_label} native function missing name")
    return ParsedToolCall(
        name=name_text,
        arguments=_coerce_arguments(function_payload.get("arguments"), context_label=context_label),
        raw_payload=raw,
    )


def _coerce_arguments(arguments: Any, *, context_label: str) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        raw_arguments = arguments.strip()
        if not raw_arguments:
            return {}
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            recovered = _recover_stringified_arguments(raw_arguments)
            if recovered is None:
                raise ValueError(f"{context_label} string arguments must be a JSON object") from exc
            arguments = recovered
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{context_label} arguments must be a JSON object")
    return dict(arguments)


_STRING_ARGUMENT_FIELD_RE = re.compile(
    r'"(?P<key>query|command|answer)"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"',
    re.IGNORECASE,
)


def _recover_stringified_arguments(raw_arguments: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(raw_arguments)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        trailing = raw_arguments[end:].strip()
        if not trailing or all(char in '"}]),' for char in trailing):
            return dict(value)

    match = _STRING_ARGUMENT_FIELD_RE.search(raw_arguments)
    if match is None:
        return None
    raw_value = match.group("value")
    if "..." in raw_value:
        return None
    try:
        value_text = json.loads(f'"{raw_value}"')
    except json.JSONDecodeError:
        return None
    return {match.group("key"): value_text}


def _first_json_field(body: str, keys: Sequence[str]) -> Any:
    errors: list[str] = []
    for key in keys:
        try:
            return _raw_decode_json_field(body, key)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(errors[0] if errors else "missing JSON field")


def _raw_decode_json_field(body: str, key: str) -> Any:
    marker = f'"{key}"'
    key_pos = body.find(marker)
    if key_pos < 0:
        raise ValueError(f"missing JSON field {key!r}")
    colon_pos = body.find(":", key_pos + len(marker))
    if colon_pos < 0:
        raise ValueError(f"missing JSON colon for field {key!r}")
    value_text = body[colon_pos + 1 :].lstrip()
    if not value_text:
        raise ValueError(f"missing JSON value for field {key!r}")
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(value_text)
    return value


def _iter_tool_schemas(
    tools: Sequence[Any],
    *,
    extra_tools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    schemas = [normalize_tool_schema(tool) for tool in tools]
    schemas.extend(dict(tool) for tool in extra_tools)
    return schemas


__all__ = [
    "ParsedToolCall",
    "allowed_arguments_by_tool_name",
    "coerce_tool_call_payloads",
    "load_tool_call_payload",
    "normalize_argument_aliases",
    "normalize_tool_call_arguments",
    "normalize_tool_schema",
    "parse_tool_call_text",
    "parse_tool_calls_text",
    "partial_tool_call_payload",
    "prune_tool_call_arguments",
    "required_arguments_by_tool_name",
    "strip_requestor_prefix",
    "tool_name",
    "validate_tool_call_name",
    "validate_tool_call_required_arguments",
]
