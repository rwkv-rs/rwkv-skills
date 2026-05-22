from __future__ import annotations

from typing import Any, Mapping, Sequence

from .types import ToolSchema


def normalize_tool_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return ToolSchema(
            name="unknown_tool",
            parameters={"type": "object", "properties": {}, "required": []},
        ).as_dict()
    schema = ToolSchema.from_mapping(raw)
    payload = schema.as_dict()
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    parameters = dict(parameters)
    if str(parameters.get("type") or "").lower() == "dict":
        parameters["type"] = "object"
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    payload["parameters"] = parameters
    return payload


def normalize_tool_schemas(raw_tools: Sequence[Any]) -> list[dict[str, Any]]:
    return [normalize_tool_schema(tool) for tool in raw_tools]
