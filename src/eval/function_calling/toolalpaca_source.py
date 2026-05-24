from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.function_calling.context_budget import normalize_rwkv_text

_TOOLALPACA_REF_KEY = "__toolalpaca_ref__"
_TOOLALPACA_OPTIONAL_KEY = "__toolalpaca_optional__"
_TOOLALPACA_AUTH_PARAMS_BY_API = {
    "apilayer weatherstack": frozenset({"access_key"}),
    "wolframalpha": frozenset({"appid"}),
    "currencybeacon": frozenset({"api_key"}),
}


def load_toolalpaca_rows_from_source(path: str | Path, *, dataset_name: str) -> list[dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ToolAlpaca source must be a JSON array: {source}")

    rows: list[dict[str, Any]] = []
    for api_index, api_info in enumerate(payload):
        if not isinstance(api_info, Mapping):
            continue
        api_name = str(api_info.get("Name") or api_info.get("API") or f"api_{api_index}")
        if _toolalpaca_should_skip_api(dataset_name, api_name):
            continue
        instructions = _coerce_list(api_info.get("Instructions"))
        golden_answers = _coerce_list(api_info.get("Golden_Answers"))
        tools = toolalpaca_tools(api_info)
        metadata_base: dict[str, Any] = {
            "source_format": "official_toolalpaca",
            "api_name": api_name,
            "api_index": api_index,
            "source_path": str(source),
            "execution_backend": _toolalpaca_execution_backend(dataset_name),
        }
        server_url = _toolalpaca_api_server_url(api_info)
        if server_url:
            metadata_base["api_server_url"] = server_url
        for question_index, instruction in enumerate(instructions):
            if question_index >= len(golden_answers):
                continue
            instruction_text = str(instruction or "").strip()
            if not instruction_text:
                continue
            rows.append(
                {
                    "task_id": f"{dataset_name}__{_slug(api_name)}_{question_index:03d}",
                    "instruction": instruction_text,
                    "tools": tools,
                    "expected_tool_calls": normalize_toolalpaca_golden_answer(golden_answers[question_index]),
                    "metadata": {
                        **metadata_base,
                        "question_index": question_index,
                    },
                }
            )
    return rows


def normalize_toolalpaca_golden_answer(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("Action") or item.get("action") or "").strip()
        optional = False
        if action.lower().startswith("[optional]"):
            optional = True
            action = action.split("]", 1)[-1].strip()
        action_input = item.get("Action_Input", item.get("action_input", {}))
        arguments = parse_toolalpaca_action_input(action_input)
        if optional:
            arguments[_TOOLALPACA_OPTIONAL_KEY] = True
        calls.append(
            {
                "name": action,
                "arguments": dict(arguments),
                "argument_options": {key: [value] for key, value in dict(arguments).items()},
            }
        )
    return calls


def parse_toolalpaca_action_input(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(_normalize_toolalpaca_placeholders(dict(raw)))
    if not isinstance(raw, str):
        return {}
    text = raw.strip()
    if not text:
        return {}
    repaired_text = _repair_toolalpaca_action_input_text(text)
    for candidate in (text, repaired_text, _quote_unquoted_toolalpaca_refs(repaired_text)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            normalized = _normalize_toolalpaca_placeholders(parsed)
            return dict(normalized) if isinstance(normalized, Mapping) else {}
        return {}
    return _parse_loose_toolalpaca_object(text)


def toolalpaca_tools(api_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    api_name = str(api_info.get("Name") or api_info.get("API") or "")
    openapi_spec = _toolalpaca_openapi_spec(api_info)
    server_url = _openapi_server_url(openapi_spec)
    descriptions = api_info.get("Function_Description")
    projection = api_info.get("Function_Projection")
    tools: list[dict[str, Any]] = []
    if isinstance(descriptions, Mapping):
        for name, description in descriptions.items():
            name_text = str(name).strip()
            if not name_text or name_text == "components":
                continue
            method = ""
            path = ""
            if isinstance(projection, Mapping):
                projected = projection.get(name)
                if isinstance(projected, Sequence) and not isinstance(projected, (str, bytes, bytearray)):
                    path = str(projected[0]) if len(projected) > 0 else ""
                    method = str(projected[1]) if len(projected) > 1 else ""
            metadata: dict[str, Any] = {"path": path, "method": method, "api_name": api_name}
            if server_url:
                metadata["server_url"] = server_url
            operation = _toolalpaca_openapi_operation(openapi_spec, path, method)
            if operation:
                metadata["operation"] = dict(operation)
            tools.append(
                {
                    "name": name_text,
                    "description": normalize_rwkv_text(str(description or "")),
                    "parameters": _strip_toolalpaca_auth_parameters(
                        api_name,
                        _toolalpaca_parameters_from_description(str(description or "")),
                    ),
                    "metadata": metadata,
                }
            )
    if _toolalpaca_api_uses_action(api_info, "getDetails"):
        tools.append(
            {
                "name": "getDetails",
                "description": (
                    "Ask the user for critical missing information required to complete the request. "
                    'Parameters: {"Question": "Required. String. The specific question to ask the user."} '
                    "Output: User supplied details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Question": {
                            "type": "string",
                            "description": "Required. String. The specific question to ask the user.",
                        }
                    },
                    "required": ["Question"],
                },
                "metadata": {"tool_type": "toolalpaca_builtin", "api_name": api_name},
            }
        )
    return tools


def _toolalpaca_execution_backend(dataset_name: str) -> str:
    if dataset_name == "toolalpaca_eval_simulated":
        return "toolalpaca_simulator"
    if dataset_name == "toolalpaca_eval_real":
        return "toolalpaca_real_http"
    return "toolalpaca_synthetic"


def _toolalpaca_should_skip_api(dataset_name: str, api_name: str) -> bool:
    return dataset_name == "toolalpaca_eval_real" and api_name.strip().lower() in _TOOLALPACA_AUTH_PARAMS_BY_API


def _toolalpaca_api_server_url(api_info: Mapping[str, Any]) -> str:
    return _openapi_server_url(_toolalpaca_openapi_spec(api_info))


def _toolalpaca_openapi_spec(api_info: Mapping[str, Any]) -> Mapping[str, Any]:
    documentation = api_info.get("Documentation")
    if not isinstance(documentation, str) or not documentation.strip():
        return {}
    try:
        payload = json.loads(documentation)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _openapi_server_url(openapi_spec: Mapping[str, Any]) -> str:
    servers = openapi_spec.get("servers")
    if isinstance(servers, Sequence) and not isinstance(servers, (str, bytes, bytearray)):
        for server in servers:
            if isinstance(server, Mapping) and server.get("url"):
                return str(server.get("url") or "").strip()
    return ""


def _toolalpaca_openapi_operation(openapi_spec: Mapping[str, Any], path: str, method: str) -> Mapping[str, Any]:
    paths = openapi_spec.get("paths") if isinstance(openapi_spec.get("paths"), Mapping) else {}
    path_doc = paths.get(path) if isinstance(paths, Mapping) else {}
    operation = path_doc.get(str(method or "").lower()) if isinstance(path_doc, Mapping) else {}
    return operation if isinstance(operation, Mapping) else {}


def _strip_toolalpaca_auth_parameters(api_name: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    auth_params = _TOOLALPACA_AUTH_PARAMS_BY_API.get(api_name.strip().lower())
    if not auth_params:
        return dict(parameters)
    normalized = dict(parameters)
    properties = normalized.get("properties") if isinstance(normalized.get("properties"), Mapping) else {}
    normalized["properties"] = {
        str(key): value for key, value in dict(properties).items() if str(key) not in auth_params
    }
    normalized["required"] = [
        str(item) for item in _coerce_list(normalized.get("required")) if str(item) not in auth_params
    ]
    return normalized


@lru_cache(maxsize=32)
def load_toolalpaca_source_payload(source_path: str) -> tuple[Any, ...]:
    payload = json.loads(Path(source_path).read_text(encoding="utf-8"))
    return tuple(payload) if isinstance(payload, list) else ()


def load_toolalpaca_api_info_from_source(
    source_path: str,
    *,
    api_index: Any,
    api_name: str,
) -> Mapping[str, Any]:
    payload = load_toolalpaca_source_payload(source_path)
    try:
        index = int(api_index)
    except (TypeError, ValueError):
        index = -1
    if 0 <= index < len(payload) and isinstance(payload[index], Mapping):
        return payload[index]
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("Name") or item.get("API") or "")
        if name == api_name:
            return item
    return {}


def _repair_toolalpaca_action_input_text(text: str) -> str:
    return re.sub(
        r"\$\{([^,{}]+?\s+from\s+[^,{}]+?),\s*\"",
        lambda match: f"${{{match.group(1).strip()}}}, \"",
        text,
    )


def _quote_unquoted_toolalpaca_refs(text: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if text.startswith("${", index):
            end = text.find("}", index + 2)
            if end != -1:
                output.append(json.dumps(text[index : end + 1], ensure_ascii=False))
                index = end + 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def _normalize_toolalpaca_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"\$\{([^{}]+)\}", value.strip())
        if match:
            return {_TOOLALPACA_REF_KEY: match.group(1).strip()}
        return value
    if isinstance(value, list):
        return [_normalize_toolalpaca_placeholders(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalize_toolalpaca_placeholders(item) for key, item in value.items()}
    return value


def _parse_loose_toolalpaca_object(text: str) -> dict[str, Any]:
    body = text.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    pairs = _split_toolalpaca_top_level(body, delimiter=",")
    parsed: dict[str, Any] = {}
    for pair in pairs:
        if ":" not in pair:
            continue
        key_text, value_text = pair.split(":", 1)
        key = key_text.strip().strip('"').strip("'").strip()
        if not key:
            continue
        parsed[key] = _parse_loose_toolalpaca_value(value_text)
    return parsed


def _parse_loose_toolalpaca_value(text: str) -> Any:
    value = text.strip().rstrip(",").strip()
    if value.startswith("${"):
        end = value.find("}")
        if end != -1:
            return {_TOOLALPACA_REF_KEY: value[2:end].strip()}
    if value.startswith("[") and value.endswith("]"):
        return [_parse_loose_toolalpaca_value(item) for item in _split_toolalpaca_top_level(value[1:-1], delimiter=",")]
    if value.startswith("{") and value.endswith("}"):
        return _parse_loose_toolalpaca_object(value)
    try:
        return _normalize_toolalpaca_placeholders(json.loads(value))
    except json.JSONDecodeError:
        pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"').strip("'")


def _split_toolalpaca_top_level(text: str, *, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if text.startswith("${", index):
            end = text.find("}", index + 2)
            if end != -1:
                index = end + 1
                continue
        if char in "[{(":
            depth += 1
        elif char in "]})" and depth > 0:
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
        index += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _toolalpaca_api_uses_action(api_info: Mapping[str, Any], action_name: str) -> bool:
    for answer in _coerce_list(api_info.get("Golden_Answers")):
        for item in _coerce_list(answer):
            if not isinstance(item, Mapping):
                continue
            action = str(item.get("Action") or item.get("action") or "").strip()
            if action == action_name:
                return True
    return False


def _toolalpaca_parameters_from_description(description: str) -> dict[str, Any]:
    marker = "Parameters:"
    if marker not in description:
        return {"type": "object", "properties": {}, "required": []}
    after = description.split(marker, 1)[1]
    before_output = after.split("\nOutput:", 1)[0].strip()
    try:
        raw_params = json.loads(before_output)
    except json.JSONDecodeError:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raw_params = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for key, value in raw_params.items():
        description_text = str(value or "")
        value_lower = description_text.lower()
        param_type = "string"
        if value_lower.startswith("integer") or ". integer" in value_lower:
            param_type = "integer"
        elif value_lower.startswith("number") or value_lower.startswith("float") or ". float" in value_lower:
            param_type = "number"
        elif value_lower.startswith("boolean") or ". boolean" in value_lower:
            param_type = "boolean"
        properties[str(key)] = {"type": param_type, "description": description_text}
        if "required." in value_lower:
            required.append(str(key))
    return {"type": "object", "properties": properties, "required": required}


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _slug(value: str) -> str:
    rendered = []
    for char in str(value).lower():
        rendered.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(rendered).split("_") if part) or "item"


__all__ = [
    "_TOOLALPACA_OPTIONAL_KEY",
    "_TOOLALPACA_REF_KEY",
    "load_toolalpaca_api_info_from_source",
    "load_toolalpaca_rows_from_source",
    "load_toolalpaca_source_payload",
    "normalize_toolalpaca_golden_answer",
    "parse_toolalpaca_action_input",
    "toolalpaca_tools",
]
