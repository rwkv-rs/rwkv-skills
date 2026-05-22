from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import ToolCall


@dataclass(frozen=True, slots=True)
class ToolAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool_call(cls, call: ToolCall) -> "ToolAction":
        return cls(
            name=call.name,
            arguments=dict(call.arguments),
            call_id=call.call_id,
            raw=dict(call.raw),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ToolAction":
        return cls.from_tool_call(ToolCall.from_mapping(payload))

    def as_dict(self) -> dict[str, Any]:
        return tool_action_payload(self)


def tool_action_payload(action: ToolAction) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": action.name,
        "arguments": dict(action.arguments),
    }
    if action.call_id is not None:
        payload["id"] = action.call_id
    return payload
