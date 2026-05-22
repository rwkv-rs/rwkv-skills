from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

FunctionCallingSubtype = Literal["one_step", "agent"]


@dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ToolSchema":
        function = payload.get("function") if isinstance(payload.get("function"), Mapping) else None
        source = function or payload
        parameters = source.get("parameters")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}, "required": []}
        metadata = payload.get("metadata")
        return cls(
            name=str(source.get("name") or payload.get("name") or "unknown_tool"),
            description=str(source.get("description") or payload.get("description") or ""),
            parameters=dict(parameters),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ToolCall":
        arguments = payload.get("arguments")
        if not isinstance(arguments, Mapping):
            arguments = {}
        call_id = payload.get("id") or payload.get("call_id")
        return cls(
            name=str(payload.get("name") or ""),
            arguments=dict(arguments),
            call_id=str(call_id) if call_id is not None else None,
            raw=dict(payload),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "arguments": dict(self.arguments),
        }
        if self.call_id is not None:
            payload["id"] = self.call_id
        return payload


@dataclass(frozen=True, slots=True)
class ParseError:
    message: str
    kind: str = "parse_error"
    raw_text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "raw_text": self.raw_text,
        }
