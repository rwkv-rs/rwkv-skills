from __future__ import annotations

"""ComplexFuncBench one-step subset adapter.

The full official benchmark is multi-turn and may use external Booking API,
embedding, and LLM comparison components.  This adapter exposes a deterministic
first-tool-turn subset through the existing one-step function-calling runner.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.one_step.simple_tool_call import (
    SimpleToolCallEvaluation,
)

OFFICIAL_COMPLEXFUNC_SOURCE = "zai-org/ComplexFuncBench"
DEFAULT_COMPLEXFUNC_SUBSET_SIZE = 100


def load_complexfuncbench_subset_rows_from_source(
    path: str | Path,
    *,
    dataset_name: str = "complexfuncbench_subset",
    max_rows: int = DEFAULT_COMPLEXFUNC_SUBSET_SIZE,
    max_instruction_chars: int = 24000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(_read_json_or_jsonl_items(Path(path))):
        if not isinstance(item, Mapping):
            continue
        converted = _convert_official_row(
            item,
            index=index,
            dataset_name=dataset_name,
            source_path=Path(path),
            max_instruction_chars=max_instruction_chars,
        )
        if converted is None:
            continue
        rows.append(converted)
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def evaluate_complexfuncbench_subset_calls(
    record: FunctionCallTaskRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    expected = [_normalize_call(item) for item in record.expected_tool_calls]
    actual = [_normalize_call(item) for item in decoded_calls]
    details: dict[str, Any] = {
        "official_complexfuncbench_source": OFFICIAL_COMPLEXFUNC_SOURCE,
        "subset_mode": "first_tool_turn",
        "expected_tool_calls": expected,
        "decoded_tool_calls": actual,
        "parse_error": parse_error or "",
        "call_matches": [],
    }
    if parse_error:
        return SimpleToolCallEvaluation(0.0, False, parse_error, details)

    failure_bits: list[str] = []
    correct_count = 0
    max_len = max(len(expected), len(actual))
    for index in range(max_len):
        if index >= len(expected):
            details["call_matches"].append({"index": index, "ok": False, "reason": "unexpected_extra_call"})
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(actual):
            details["call_matches"].append({"index": index, "ok": False, "reason": "missing_call"})
            failure_bits.append(f"call_{index}:missing_call")
            continue
        ok, reason = _call_matches(actual[index], expected[index], record.tools)
        details["call_matches"].append({"index": index, "ok": ok, "reason": reason})
        if ok:
            correct_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")

    denominator = max(1, len(expected))
    details["correct_call_num"] = correct_count
    details["total_call_num"] = len(expected)
    details["call_accuracy"] = correct_count / denominator
    passed = len(actual) == len(expected) and correct_count == len(expected)
    return SimpleToolCallEvaluation(
        reward=float(correct_count / denominator),
        is_passed=bool(passed),
        fail_reason="; ".join(failure_bits),
        details=details,
    )


def uses_complexfuncbench_scorer(record: FunctionCallTaskRecord) -> bool:
    scorer_type = str((record.scorer or {}).get("type") or "")
    if scorer_type == "complexfuncbench_subset":
        return True
    source_format = str((record.metadata or {}).get("source_format") or "")
    return source_format == "official_complexfuncbench" and str(record.task_id or "").startswith("complexfuncbench_")


def _convert_official_row(
    item: Mapping[str, Any],
    *,
    index: int,
    dataset_name: str,
    source_path: Path,
    max_instruction_chars: int,
) -> dict[str, Any] | None:
    conversations = _list_of_dicts(item.get("conversations"))
    functions = _list_of_dicts(item.get("functions"))
    tool_turn_index = _first_assistant_tool_turn(conversations)
    if tool_turn_index is None:
        return None
    instruction = _render_history(conversations[:tool_turn_index])
    if not instruction or len(instruction) > max_instruction_chars:
        return None
    tool_calls = _normalize_tool_calls(conversations[tool_turn_index].get("function_call"))
    if not tool_calls:
        return None
    official_id = str(item.get("id") or item.get("task_id") or index)
    return {
        "task_id": f"{dataset_name}__{official_id}",
        "instruction": instruction,
        "messages": _messages_from_history(conversations[:tool_turn_index]),
        "tools": [_normalize_tool_schema(tool) for tool in functions],
        "expected_tool_calls": tool_calls,
        "env": {"type": "simple_tool_call"},
        "scorer": {"type": "complexfuncbench_subset", "mode": "first_tool_turn"},
        "metadata": {
            "source_format": "official_complexfuncbench",
            "official_source": OFFICIAL_COMPLEXFUNC_SOURCE,
            "source_path": str(source_path),
            "official_id": official_id,
            "subset_mode": "first_tool_turn",
            "original_turn_index": tool_turn_index,
            "category": item.get("category") or item.get("type") or "",
        },
    }


def _first_assistant_tool_turn(conversations: Sequence[Mapping[str, Any]]) -> int | None:
    for index, turn in enumerate(conversations):
        if str(turn.get("role") or "").lower() == "assistant" and turn.get("function_call") is not None:
            return index
    return None


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        call = _normalize_call(item)
        if call["name"]:
            calls.append(
                {
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "argument_options": {key: [value] for key, value in call["arguments"].items()},
                }
            )
    return calls


def _normalize_call(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {"name": "", "arguments": {}}
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
    return {"name": name, "arguments": dict(arguments)}


def _call_matches(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    actual_name = str(actual.get("name") or "")
    expected_name = str(expected.get("name") or "")
    if actual_name != expected_name:
        return False, f"name_mismatch(expected={expected_name}, actual={actual_name})"
    arguments = actual.get("arguments")
    if not isinstance(arguments, Mapping):
        return False, "arguments_not_object"
    format_error = _format_error(actual_name, arguments, tools)
    if format_error:
        return False, format_error
    expected_arguments = expected.get("arguments")
    if not isinstance(expected_arguments, Mapping):
        expected_arguments = {}
    actual_arguments = dict(arguments)
    if sorted(actual_arguments) != sorted(expected_arguments):
        missing = sorted(set(expected_arguments) - set(actual_arguments))
        extra = sorted(set(actual_arguments) - set(expected_arguments))
        if missing:
            return False, f"missing_argument({missing[0]})"
        if extra:
            return False, f"unexpected_argument({extra[0]})"
    for key, expected_value in expected_arguments.items():
        if not _value_matches(actual_arguments.get(key), expected_value):
            return False, f"argument_mismatch({key})"
    return True, "ok"


def _format_error(name: str, arguments: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]) -> str:
    tool = next((item for item in tools if str(item.get("name") or "") == name), None)
    if not isinstance(tool, Mapping):
        return ""
    parameters = tool.get("parameters")
    if not isinstance(parameters, Mapping):
        return ""
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required")
    if not isinstance(required, list):
        required = []
    for key in required:
        if str(key) not in arguments:
            return f"missing_argument({key})"
    for key, value in arguments.items():
        schema = properties.get(key)
        if not isinstance(schema, Mapping):
            return f"unexpected_argument({key})"
        expected_type = str(schema.get("type") or "").lower()
        if expected_type and not _type_matches(value, expected_type):
            return f"argument_type_mismatch({key})"
    return ""


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type in {"string"}:
        return isinstance(value, str)
    if expected_type in {"number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type in {"integer"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type in {"boolean", "bool"}:
        return isinstance(value, bool)
    if expected_type in {"array", "list"}:
        return isinstance(value, list)
    if expected_type in {"object", "dict"}:
        return isinstance(value, Mapping)
    return True


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9
    if isinstance(actual, str) and isinstance(expected, str):
        if "," in actual or "," in expected:
            actual_parts = [part.strip() for part in actual.split(",")]
            expected_parts = [part.strip() for part in expected.split(",")]
            if set(actual_parts) == set(expected_parts):
                return True
        return " ".join(actual.split()) == " ".join(expected.split())
    return actual == expected


def _normalize_tool_schema(raw: Mapping[str, Any]) -> dict[str, Any]:
    parameters = raw.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    return {
        "name": str(raw.get("name") or "unknown_tool"),
        "description": str(raw.get("description") or ""),
        "parameters": dict(parameters),
    }


def _render_history(history: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for turn in history:
        role = str(turn.get("role") or "user").lower()
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if not content:
            continue
        if role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role in {"tool", "function"}:
            parts.append(f"Tool: {content}")
        elif role == "system":
            parts.append(f"System: {content}")
        else:
            parts.append(f"User: {content}")
    return "\n".join(parts).strip()


def _messages_from_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in history:
        role = str(turn.get("role") or "user").lower()
        content = str(turn.get("content") or turn.get("text") or "")
        if role in {"assistant", "system", "tool", "function", "user"}:
            messages.append({"role": role, "content": content})
    return messages


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


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    return []


__all__ = [
    "DEFAULT_COMPLEXFUNC_SUBSET_SIZE",
    "OFFICIAL_COMPLEXFUNC_SOURCE",
    "evaluate_complexfuncbench_subset_calls",
    "load_complexfuncbench_subset_rows_from_source",
    "uses_complexfuncbench_scorer",
]
