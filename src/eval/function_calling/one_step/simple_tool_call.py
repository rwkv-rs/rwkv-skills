from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.function_calling.context_budget import normalize_rwkv_text, truncate_text
from src.eval.function_calling.rwkv_prompt import (
    build_rwkv_json_call_prompt,
    coerce_json_function_call_payloads,
    extract_json_call_value_text,
)
from src.eval.function_calling.toolalpaca_source import (
    load_toolalpaca_rows_from_source as _load_toolalpaca_rows_from_official_source,
)

_MAX_TOOL_DESCRIPTION_CHARS = 700
_MAX_TOOL_SCHEMA_CHARS = 1200


@dataclass(frozen=True, slots=True)
class ToolCallExpectation:
    name: str
    arguments: dict[str, Any]
    argument_options: dict[str, tuple[Any, ...]]


@dataclass(frozen=True, slots=True)
class SimpleToolCallRecord:
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_tool_calls: tuple[ToolCallExpectation, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SimpleToolCallEvaluation:
    reward: float
    is_passed: bool
    fail_reason: str
    details: dict[str, Any]


def load_simple_tool_call_manifest_records(
    path: str | Path,
) -> list[SimpleToolCallRecord]:
    target = Path(path)
    records: list[SimpleToolCallRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            records.append(normalize_simple_tool_call_manifest_row(payload, index=index, source_path=target))
    return records


def normalize_simple_tool_call_manifest_row(
    payload: Mapping[str, Any],
    *,
    index: int,
    source_path: str | Path | None = None,
) -> SimpleToolCallRecord:
    task_id = str(payload.get("task_id") or payload.get("id") or f"tool_call_{index:04d}")
    instruction = str(payload.get("instruction") or payload.get("question") or "").strip()
    if not instruction:
        raise ValueError(f"simple tool-call row {task_id!r} is missing instruction")
    metadata = dict(payload.get("metadata") or {})
    if source_path is not None:
        metadata.setdefault("manifest_path", str(Path(source_path)))
    return SimpleToolCallRecord(
        task_id=task_id,
        instruction=instruction,
        tools=tuple(_normalize_tool_schema(tool) for tool in _coerce_list(payload.get("tools"))),
        expected_tool_calls=tuple(
            _normalize_tool_expectation(item) for item in _coerce_list(payload.get("expected_tool_calls"))
        ),
        metadata=metadata,
    )


def load_bfcl_ast_rows_from_sources(
    question_path: str | Path,
    possible_answer_path: str | Path,
    *,
    category: str,
) -> list[dict[str, Any]]:
    questions = _read_json_or_jsonl_items(Path(question_path))
    answer_lookup = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_json_or_jsonl_items(Path(possible_answer_path))
        if isinstance(item, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(questions):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{category}_{index}")
        answer = answer_lookup.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        if not instruction:
            raise ValueError(f"BFCL row {task_id!r} is missing question content")
        ground_truth = _coerce_list(answer.get("ground_truth"))
        execution_result_type = _coerce_list(answer.get("execution_result_type"))
        is_exec = category.startswith("exec_")
        row = {
            "task_id": task_id,
            "instruction": instruction,
            "tools": [_normalize_tool_schema(tool) for tool in _coerce_list(item.get("function"))],
            "expected_tool_calls": _normalize_bfcl_ground_truth_calls(ground_truth),
            "metadata": {
                "source_format": "official_bfcl_v4_exec" if is_exec else "official_bfcl_v4_ast",
                "category": category,
                "source_path": str(Path(question_path)),
                "possible_answer_path": str(Path(possible_answer_path)),
                "execution_result_type": execution_result_type,
            },
        }
        if is_exec:
            row["expected_executable_calls"] = ground_truth
            row["execution_result_type"] = execution_result_type
        rows.append(row)
    return rows


def load_toolalpaca_rows_from_source(path: str | Path, *, dataset_name: str) -> list[dict[str, Any]]:
    rows = _load_toolalpaca_rows_from_official_source(path, dataset_name=dataset_name)
    dataset_kind = "simulated" if "simulated" in dataset_name else "real"
    for row in rows:
        row.setdefault("scorer", {"type": "toolalpaca_official", "dataset": dataset_kind})
        metadata = row.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata.setdefault("toolalpaca_dataset", dataset_kind)
            tools_by_name = {
                str(tool.get("name") or ""): dict(tool.get("metadata") or {})
                for tool in _coerce_list(row.get("tools"))
                if isinstance(tool, Mapping)
            }
            if tools_by_name:
                metadata.setdefault("toolalpaca_tool_metadata_by_name", tools_by_name)
            expected_calls = _coerce_list(row.get("expected_tool_calls"))
            first_expected = expected_calls[0] if expected_calls and isinstance(expected_calls[0], Mapping) else {}
            first_tool_metadata = tools_by_name.get(str(first_expected.get("name") or ""))
            if first_tool_metadata:
                for key in ("path", "method", "api_name", "server_url", "operation"):
                    if key in first_tool_metadata:
                        metadata.setdefault(key, first_tool_metadata[key])
    return rows


def build_simple_tool_call_prompt(record: SimpleToolCallRecord, *, history_max_chars: int) -> str:
    date_instructions = [
        "For dates and times, use only dates/times stated or implied by the conversation or function outputs; do not use the real current date.",
    ]
    if str(record.metadata.get("source_format") or "").strip() in {
        "official_api_bank",
        "official_apibank",
    }:
        date_instructions.append(
            "API-Bank date convention: if a month/day or relative date has no explicit year and the conversation does not state today's date, use year 2023."
        )
    system_lines = [
        "Tools:",
        _render_tool_catalog(record.tools),
        "Output JSON schema:",
        _render_output_schema(),
        "Return exactly one JSON value that validates against the schema.",
        "For one tool call, return one JSON object.",
        "For multiple required tool calls, return a JSON array containing every required call in execution order; do not stop after the first call.",
        "Each arguments object must contain only final argument values for that tool.",
        *date_instructions,
        'Do not copy tool schemas, descriptions, type/items/properties/required/default fields, or wrapper objects like {"type":...,"value":...} into arguments.',
        "Use only listed tool names.",
        "Return no prose, no markdown, and no extra text outside the JSON value.",
    ]
    system_prompt = normalize_rwkv_text("\n".join(system_lines))
    return build_rwkv_json_call_prompt(
        system_prompt,
        [{"role": "user", "content": normalize_rwkv_text(record.instruction)}],
        history_max_chars=history_max_chars,
    )


def decode_simple_tool_call_response(response: str) -> list[dict[str, Any]]:
    candidate = extract_json_call_value_text(response)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = _literal_from_ast(ast.parse(candidate, mode="eval").body)
    if payload == []:
        return []
    calls = coerce_json_function_call_payloads(payload, context_label="tool-call selection")
    return [{"name": str(call["name"]), "arguments": dict(call.get("arguments") or {})} for call in calls]


def evaluate_simple_tool_calls(
    record: SimpleToolCallRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    expected = list(record.expected_tool_calls)
    actual = [
        {
            "name": str(item.get("name") or ""),
            "arguments": dict(item.get("arguments") or {}),
        }
        for item in decoded_calls
    ]
    details: dict[str, Any] = {
        "expected_tool_calls": [_expectation_payload(item) for item in expected],
        "decoded_tool_calls": actual,
        "tool_count_ok": len(actual) == len(expected),
        "call_matches": [],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return SimpleToolCallEvaluation(
            reward=0.0,
            is_passed=False,
            fail_reason=parse_error,
            details=details,
        )

    max_len = max(len(expected), len(actual))
    passed_count = 0
    failure_bits: list[str] = []
    for index in range(max_len):
        if index >= len(expected):
            details["call_matches"].append({"index": index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(actual):
            details["call_matches"].append({"index": index, "ok": False, "reason": "missing_call"})
            failure_bits.append(f"call_{index}:missing_call")
            continue
        ok, reason = _call_matches_expectation(actual[index], expected[index])
        details["call_matches"].append({"index": index, "ok": ok, "reason": reason})
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")

    denominator = max(1, len(expected))
    reward = passed_count / denominator
    is_passed = len(actual) == len(expected) and passed_count == len(expected)
    if not expected:
        is_passed = len(actual) == 0
        reward = 1.0 if is_passed else 0.0
    return SimpleToolCallEvaluation(
        reward=float(reward),
        is_passed=bool(is_passed),
        fail_reason="; ".join(failure_bits),
        details=details,
    )


def _normalize_tool_expectation(raw: Any) -> ToolCallExpectation:
    if not isinstance(raw, Mapping):
        return ToolCallExpectation(name="unknown_tool", arguments={}, argument_options={})
    name = str(raw.get("name") or raw.get("tool_name") or raw.get("function_name") or "").strip()
    arguments = raw.get("arguments")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            arguments = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    raw_options = raw.get("argument_options")
    argument_options: dict[str, tuple[Any, ...]] = {}
    if isinstance(raw_options, Mapping):
        for key, value in raw_options.items():
            options = tuple(_coerce_list(value) or [value])
            argument_options[str(key)] = options
    for key, value in arguments.items():
        argument_options.setdefault(str(key), (value,))
    return ToolCallExpectation(
        name=name or "unknown_tool",
        arguments=dict(arguments),
        argument_options=argument_options,
    )


def _call_matches_expectation(actual: Mapping[str, Any], expected: ToolCallExpectation) -> tuple[bool, str]:
    actual_name = str(actual.get("name") or "").strip()
    if actual_name != expected.name:
        return False, f"name_mismatch(expected={expected.name}, actual={actual_name})"
    arguments = actual.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments_not_object"
    actual_arguments = dict(arguments)
    for key, options in expected.argument_options.items():
        if key not in actual_arguments:
            if any(_is_absent_option(option) for option in options):
                continue
            return False, f"missing_argument({key})"
        actual_value = actual_arguments[key]
        if not any(_value_matches(actual_value, option) for option in options):
            return False, f"argument_mismatch({key})"
    for key, value in actual_arguments.items():
        if key not in expected.argument_options and not _is_absent_option(value):
            return False, f"unexpected_argument({key})"
    return True, "ok"


def _value_matches(actual: Any, expected: Any) -> bool:
    if _is_absent_option(expected):
        return _is_absent_option(actual)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(actual, str) and not isinstance(expected, str):
        parsed = _try_parse_json_scalar(actual)
        if parsed is not actual:
            return _value_matches(parsed, expected)
    if isinstance(expected, str) and not isinstance(actual, str):
        parsed = _try_parse_json_scalar(expected)
        if parsed is not expected:
            return _value_matches(actual, parsed)
    if isinstance(actual, str) and isinstance(expected, str):
        return normalize_rwkv_text(actual).strip() == normalize_rwkv_text(expected).strip()
    return actual == expected


def _is_absent_option(value: Any) -> bool:
    return value is None or value == "" or value == {} or value == []


def _try_parse_json_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return value
    if text[0] not in '[{"-0123456789tfn':
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _expectation_payload(expectation: ToolCallExpectation) -> dict[str, Any]:
    return {
        "name": expectation.name,
        "arguments": dict(expectation.arguments),
        "argument_options": {key: list(value) for key, value in expectation.argument_options.items()},
    }


def _normalize_bfcl_ground_truth_calls(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            name, arguments = _parse_python_call(item)
            calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "argument_options": {key: [value] for key, value in arguments.items()},
                }
            )
            continue
        if isinstance(item, Mapping):
            if "name" in item:
                call = _normalize_tool_expectation(item)
                calls.append(_expectation_payload(call))
                continue
            if len(item) != 1:
                continue
            name, argument_options = next(iter(item.items()))
            if not isinstance(argument_options, Mapping):
                argument_options = {}
            canonical_arguments = {
                str(key): _canonical_option_value(_coerce_list(value) or [value])
                for key, value in argument_options.items()
            }
            calls.append(
                {
                    "name": str(name),
                    "arguments": canonical_arguments,
                    "argument_options": {
                        str(key): list(_coerce_list(value) or [value]) for key, value in argument_options.items()
                    },
                }
            )
    return calls


def _canonical_option_value(options: Sequence[Any]) -> Any:
    for option in options:
        if not _is_absent_option(option):
            return option
    return options[0] if options else None


def _parse_python_call(text: str) -> tuple[str, dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"BFCL ground-truth expression is not a function call: {text}")
    name = _render_ast_call_name(parsed.body.func)
    arguments: dict[str, Any] = {}
    for keyword in parsed.body.keywords:
        if keyword.arg is None:
            continue
        arguments[keyword.arg] = _literal_from_ast(keyword.value)
    return name, arguments


def _render_ast_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _render_ast_call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in {"true", "True"}:
            return True
        if node.id in {"false", "False"}:
            return False
        if node.id in {"null", "None"}:
            return None
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.BinOp):
        left = _literal_from_ast(node.left)
        right = _literal_from_ast(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
    return ast.literal_eval(node)


def _normalize_toolalpaca_golden_answer(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("Action") or item.get("action") or "").strip()
        action_input = item.get("Action_Input", item.get("action_input", {}))
        arguments: Any = action_input
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments.strip() or "{}")
                arguments = parsed if isinstance(parsed, Mapping) else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        calls.append(
            {
                "name": action,
                "arguments": dict(arguments),
                "argument_options": {key: [value] for key, value in dict(arguments).items()},
            }
        )
    return calls


def _toolalpaca_tools(api_info: Mapping[str, Any]) -> list[dict[str, Any]]:
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
            tools.append(
                {
                    "name": name_text,
                    "description": _toolalpaca_description_summary(str(description or "")),
                    "parameters": _toolalpaca_parameters_from_description(str(description or "")),
                    "metadata": {"path": path, "method": method},
                }
            )
    return tools


def _sanitize_toolalpaca_authentication(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): "***" for key in raw}


def _toolalpaca_description_summary(description: str) -> str:
    text = normalize_rwkv_text(description)
    for marker in ("\nParameters:", "\nOutput:", " Parameters:", " Output:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return normalize_rwkv_text(text)


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


def _render_bfcl_question(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    turns = _coerce_list(raw)
    parts: list[str] = []
    for turn in turns:
        messages = _coerce_list(turn)
        for message in messages:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "user").strip().lower() or "user"
                content = str(message.get("content") or "").strip()
                if content:
                    parts.append(f"{role.title()}: {content}")
            elif str(message or "").strip():
                parts.append(str(message).strip())
    return "\n".join(parts).strip()


def _normalize_tool_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "name": "unknown_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    parameters = source.get("parameters") or {
        "type": "object",
        "properties": {},
        "required": [],
    }
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = dict(parameters)
    if str(parameters.get("type") or "").lower() == "dict":
        parameters["type"] = "object"
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    normalized = {
        "name": str(source.get("name") or raw.get("name") or "unknown_tool"),
        "description": str(source.get("description") or raw.get("description") or ""),
        "parameters": parameters,
    }
    metadata = raw.get("metadata") or source.get("metadata")
    if isinstance(metadata, Mapping):
        normalized["metadata"] = dict(metadata)
    return normalized


def _render_tool_catalog(tools: Sequence[Mapping[str, Any]]) -> str:
    rendered_tools: list[dict[str, Any]] = []
    for tool in tools:
        parameters = tool.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}, "required": []}
        raw_properties = parameters.get("properties")
        rendered_arguments: Any = dict(raw_properties) if isinstance(raw_properties, Mapping) else dict(parameters)
        rendered_schema = json.dumps(rendered_arguments, ensure_ascii=False, sort_keys=True)
        if len(rendered_schema) > _MAX_TOOL_SCHEMA_CHARS:
            rendered_arguments = {
                "_truncated": True,
                "preview": truncate_text(rendered_schema, _MAX_TOOL_SCHEMA_CHARS),
            }
        rendered_tools.append(
            {
                "name": str(tool.get("name") or ""),
                "description": truncate_text(
                    normalize_rwkv_text(str(tool.get("description") or "")),
                    _MAX_TOOL_DESCRIPTION_CHARS,
                ),
                "arguments": rendered_arguments,
                **(
                    {"required": list(parameters.get("required") or [])}
                    if isinstance(parameters.get("required"), list) and parameters.get("required")
                    else {}
                ),
            }
        )
    return json.dumps(rendered_tools, ensure_ascii=False, indent=2, sort_keys=False)


def _render_output_schema() -> str:
    tool_call_schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"},
        },
    }
    schema = {
        "oneOf": [
            tool_call_schema,
            {
                "type": "array",
                "items": tool_call_schema,
                "minItems": 1,
            },
        ]
    }
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=False)


def _read_json_or_jsonl_items(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError(f"unsupported JSON payload: {path}")


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
    "SimpleToolCallEvaluation",
    "SimpleToolCallRecord",
    "ToolCallExpectation",
    "build_simple_tool_call_prompt",
    "decode_simple_tool_call_response",
    "evaluate_simple_tool_calls",
    "load_bfcl_ast_rows_from_sources",
    "load_simple_tool_call_manifest_records",
    "load_toolalpaca_rows_from_source",
    "normalize_simple_tool_call_manifest_row",
]
