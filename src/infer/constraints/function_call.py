from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .json import JsonObjectConstraint, extract_root_object_schema_hints
from .text import LiteralChoiceConstraint, LiteralConstraint


@dataclass(frozen=True, slots=True)
class SingleFunctionCallConstraintSpec:
    tool_names: tuple[str, ...]
    tool_schemas: dict[str, Mapping[str, Any]]
    prefix_literal: str = '{"name":"'
    middle_literal: str = '","arguments":'
    suffix_literal: str = "}"


@dataclass(slots=True)
class SingleFunctionCallConstraint:
    spec: SingleFunctionCallConstraintSpec
    _phase: str = "prefix"
    _prefix_part: LiteralConstraint | None = None
    _tool_name_part: LiteralChoiceConstraint | None = None
    _middle_part: LiteralConstraint | None = None
    _json_part: JsonObjectConstraint | None = None
    _suffix_part: LiteralConstraint | None = None
    _selected_tool_name: str | None = None

    def __post_init__(self) -> None:
        if self._prefix_part is None:
            self._prefix_part = LiteralConstraint(self.spec.prefix_literal)
        if self._tool_name_part is None:
            self._tool_name_part = LiteralChoiceConstraint(tuple(self.spec.tool_names))
        if self._middle_part is None:
            self._middle_part = LiteralConstraint(self.spec.middle_literal)
        if self._suffix_part is None:
            self._suffix_part = LiteralConstraint(self.spec.suffix_literal)
        if self._json_part is None and self._phase in {"json", "suffix", "done"}:
            self._json_part = self._build_json_part(self._selected_tool_name)

    def clone(self) -> "SingleFunctionCallConstraint":
        return SingleFunctionCallConstraint(
            self.spec,
            _phase=self._phase,
            _prefix_part=None if self._prefix_part is None else self._prefix_part.clone(),
            _tool_name_part=None if self._tool_name_part is None else self._tool_name_part.clone(),
            _middle_part=None if self._middle_part is None else self._middle_part.clone(),
            _json_part=None if self._json_part is None else self._json_part.clone(),
            _suffix_part=None if self._suffix_part is None else self._suffix_part.clone(),
            _selected_tool_name=self._selected_tool_name,
        )

    def feed_text(self, text: str) -> bool:
        for char in text:
            if self._phase == "prefix":
                if self._prefix_part is None or not self._prefix_part.feed_text(char):
                    return False
                if self._prefix_part.is_complete():
                    self._phase = "tool_name"
                continue
            if self._phase == "tool_name":
                if self._tool_name_part is None or not self._tool_name_part.feed_text(char):
                    return False
                if self._tool_name_part.is_complete():
                    self._selected_tool_name = self._tool_name_part.matched_choice()
                    self._json_part = self._build_json_part(self._selected_tool_name)
                    self._phase = "middle"
                continue
            if self._phase == "middle":
                if self._middle_part is None or not self._middle_part.feed_text(char):
                    return False
                if self._middle_part.is_complete():
                    self._phase = "json"
                continue
            if self._phase == "json":
                if self._json_part is None or not self._json_part.feed_text(char):
                    return False
                if self._json_part.is_complete():
                    self._phase = "suffix"
                continue
            if self._phase == "suffix":
                if self._suffix_part is None or not self._suffix_part.feed_text(char):
                    return False
                if self._suffix_part.is_complete():
                    self._phase = "done"
                continue
            return False
        return True

    def allowed_first_bytes(self) -> set[int] | None:
        if self._phase == "prefix" and self._prefix_part is not None:
            return self._prefix_part.allowed_first_bytes()
        if self._phase == "tool_name" and self._tool_name_part is not None:
            return self._tool_name_part.allowed_first_bytes()
        if self._phase == "middle" and self._middle_part is not None:
            return self._middle_part.allowed_first_bytes()
        if self._phase == "json" and self._json_part is not None:
            return self._json_part.allowed_first_bytes()
        if self._phase == "suffix" and self._suffix_part is not None:
            return self._suffix_part.allowed_first_bytes()
        return set()

    def is_complete(self) -> bool:
        return self._phase == "done"

    def finish_reason(self) -> str:
        return "constraint_stop"

    def _build_json_part(self, tool_name: str | None) -> JsonObjectConstraint:
        schema = None if tool_name is None else self.spec.tool_schemas.get(tool_name)
        allowed_root_keys, required_root_keys, additional_properties = extract_root_object_schema_hints(schema)
        return JsonObjectConstraint(
            allowed_root_keys=allowed_root_keys,
            required_root_keys=required_root_keys,
            root_additional_properties=additional_properties,
        )


def build_single_function_call_constraint(spec: SingleFunctionCallConstraintSpec) -> SingleFunctionCallConstraint:
    return SingleFunctionCallConstraint(spec)


def build_bfcl_tool_call_constraint(tools: Sequence[Mapping[str, Any]]) -> SingleFunctionCallConstraint:
    tool_names = tuple(str(tool.get("name") or "").strip() for tool in tools if str(tool.get("name") or "").strip())
    tool_schemas = {
        str(tool.get("name") or "").strip(): dict(tool.get("parameters") or {})
        for tool in tools
        if str(tool.get("name") or "").strip()
    }
    return build_single_function_call_constraint(
        SingleFunctionCallConstraintSpec(
            tool_names=tool_names,
            tool_schemas=tool_schemas,
        )
    )


__all__ = [
    "SingleFunctionCallConstraint",
    "SingleFunctionCallConstraintSpec",
    "build_bfcl_tool_call_constraint",
    "build_single_function_call_constraint",
]
