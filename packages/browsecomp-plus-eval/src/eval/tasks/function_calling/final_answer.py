from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.eval.tasks.function_calling.context_budget import normalize_rwkv_text
from src.eval.tasks.function_calling.rwkv_prompt import (
    build_rwkv_json_call_prompt,
)
from src.eval.tasks.function_calling.tool_call_contract import coerce_tool_call_payloads, load_tool_call_payload

FINAL_ANSWER_TOOL_NAME = "final_answer"
FINAL_ANSWER_CALL_ID = "final_answer"
DEFAULT_FINAL_ANSWER_KEYS = ("answer", "final_answer", "response", "choice", "letter", "prediction")


@dataclass(frozen=True, slots=True)
class FinalAnswerCall:
    answer: str
    call: dict[str, Any]
    call_id: str = FINAL_ANSWER_CALL_ID


def final_answer_tool_schema(*, answer_description: str = "The concise final answer.") -> dict[str, Any]:
    return {
        "name": FINAL_ANSWER_TOOL_NAME,
        "description": "Finish the benchmark task with the final answer.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "description": answer_description}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }


def final_answer_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["name", "arguments", "id"],
        "additionalProperties": False,
        "properties": {
            "name": {"const": FINAL_ANSWER_TOOL_NAME},
            "id": {"type": "string"},
            "arguments": {
                "type": "object",
                "required": ["answer"],
                "additionalProperties": False,
                "properties": {"answer": {"type": "string"}},
            },
        },
    }


def render_final_answer_call(answer: str, *, call_id: str = FINAL_ANSWER_CALL_ID) -> str:
    payload = {
        "name": FINAL_ANSWER_TOOL_NAME,
        "arguments": {"answer": str(answer or "")},
        "id": str(call_id or FINAL_ANSWER_CALL_ID),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def render_final_answer_json_block(answer: str, *, call_id: str = FINAL_ANSWER_CALL_ID) -> str:
    return f"```json\n{render_final_answer_call(answer, call_id=call_id)}\n```"


def build_final_answer_json_call_prompt(
    instruction: str,
    *,
    answer_description: str = "The concise final answer.",
    history_max_chars: int,
    extra_system_lines: Sequence[str] = (),
) -> str:
    system_prompt = normalize_rwkv_text(
        "\n".join(
            [
                "Tools:",
                json.dumps([final_answer_tool_schema(answer_description=answer_description)], ensure_ascii=False, indent=2),
                "Output JSON schema:",
                json.dumps(final_answer_output_schema(), ensure_ascii=False, indent=2),
                "Return exactly one RWKV JSON function-call object in the assistant json block.",
                f"Use only {FINAL_ANSWER_TOOL_NAME}.",
                f'Use id "{FINAL_ANSWER_CALL_ID}" unless the benchmark explicitly provides another id.',
                "Return no prose, no markdown, and no extra text outside the JSON value.",
                *[str(line) for line in extra_system_lines],
            ]
        )
    )
    return build_rwkv_json_call_prompt(
        system_prompt,
        [{"role": "user", "content": normalize_rwkv_text(instruction)}],
        history_max_chars=history_max_chars,
    )


def parse_final_answer_call(
    response: str,
    *,
    answer_keys: Sequence[str] = DEFAULT_FINAL_ANSWER_KEYS,
    context_label: str = "final answer",
) -> FinalAnswerCall:
    payload = load_tool_call_payload(response, context_label=context_label, recover_partial=True)
    calls = coerce_tool_call_payloads(payload, context_label=context_label, allowed_metadata_keys=("id",))
    for call in calls:
        if call.name != FINAL_ANSWER_TOOL_NAME:
            continue
        arguments = dict(call.arguments)
        answer = _extract_answer(arguments, answer_keys=answer_keys)
        if answer is None:
            raise ValueError(f"{context_label} final_answer call missing answer")
        call_id = _find_final_answer_call_id(call.raw_payload) or _find_final_answer_call_id(payload) or FINAL_ANSWER_CALL_ID
        return FinalAnswerCall(
            answer=answer,
            call={"name": FINAL_ANSWER_TOOL_NAME, "arguments": arguments, "id": call_id},
            call_id=call_id,
        )
    names = ", ".join(call.name for call in calls) or "<none>"
    raise ValueError(f"{context_label} must call final_answer, got {names}")


def _loads_json_or_literal(candidate: str) -> Any:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return ast.literal_eval(candidate)


def _extract_answer(arguments: Mapping[str, Any], *, answer_keys: Sequence[str]) -> str | None:
    for key in answer_keys:
        if key in arguments:
            return _stringify_answer(arguments.get(key))
    if len(arguments) == 1:
        return _stringify_answer(next(iter(arguments.values())))
    return None


def _find_final_answer_call_id(value: Any) -> str:
    if isinstance(value, Mapping):
        name = _mapping_call_name(value)
        if name == FINAL_ANSWER_TOOL_NAME:
            call_id = str(value.get("id") or value.get("call_id") or "").strip()
            if call_id:
                return call_id
        tool_calls = value.get("tool_calls")
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes, bytearray)):
            for item in tool_calls:
                call_id = _find_final_answer_call_id(item)
                if call_id:
                    return call_id
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            call_id = _find_final_answer_call_id(item)
            if call_id:
                return call_id
    return ""


def _mapping_call_name(value: Mapping[str, Any]) -> str:
    name = value.get("name")
    function_payload = value.get("function")
    if isinstance(function_payload, Mapping):
        name = function_payload.get("name") or name
    function_call_payload = value.get("function_call")
    if isinstance(function_call_payload, Mapping):
        name = function_call_payload.get("name") or name
    return str(name or "").strip()


def _stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_rwkv_text(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "DEFAULT_FINAL_ANSWER_KEYS",
    "FINAL_ANSWER_CALL_ID",
    "FINAL_ANSWER_TOOL_NAME",
    "FinalAnswerCall",
    "build_final_answer_json_call_prompt",
    "final_answer_output_schema",
    "final_answer_tool_schema",
    "parse_final_answer_call",
    "render_final_answer_call",
    "render_final_answer_json_block",
]
