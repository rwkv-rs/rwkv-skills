from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import ParseError, ToolCall


@dataclass(frozen=True, slots=True)
class ToolCallParseResult:
    calls: list[ToolCall] = field(default_factory=list)
    raw_text: str = ""
    parse_error: ParseError | None = None

    @property
    def ok(self) -> bool:
        return self.parse_error is None

    def call_payloads(self) -> list[dict[str, Any]]:
        return [call.as_dict() for call in self.calls]


def parse_json_tool_calls(response: str, *, context_label: str = "tool-call selection") -> ToolCallParseResult:
    from src.eval.function_calling.rwkv_prompt import (
        coerce_json_function_call_payloads,
        extract_json_call_value_text,
    )

    raw_text = str(response or "")
    try:
        candidate = extract_json_call_value_text(raw_text)
        import json

        payload = json.loads(candidate)
        if payload == []:
            return ToolCallParseResult(calls=[], raw_text=raw_text)
        calls = coerce_json_function_call_payloads(payload, context_label=context_label)
        return ToolCallParseResult(
            calls=[ToolCall.from_mapping(call) for call in calls],
            raw_text=raw_text,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolCallParseResult(
            calls=[],
            raw_text=raw_text,
            parse_error=ParseError(message=str(exc), raw_text=raw_text),
        )
