from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence


_JSON_WHITESPACE = frozenset(" \t\r\n")
_ANY_VALUE_KINDS = frozenset({"string", "number", "integer", "boolean", "null", "object", "array"})
_NUMBER_START_CHARS = frozenset("-0123456789")
_VALUE_START_BY_KIND = {
    "string": frozenset({'"'}),
    "number": _NUMBER_START_CHARS,
    "integer": _NUMBER_START_CHARS,
    "boolean": frozenset({"t", "f"}),
    "null": frozenset({"n"}),
    "object": frozenset({"{"}),
    "array": frozenset({"["}),
}
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_ESCAPE_CHARS = frozenset('"\\/bfnrtu')
_TYPE_ALIASES = {
    "dict": "object",
    "list": "array",
    "float": "number",
    "int": "integer",
    "bool": "boolean",
    "str": "string",
}


@dataclass(slots=True)
class _ContainerFrame:
    kind: str
    mode: str


@dataclass(slots=True)
class _JsonNumberState:
    integer_only: bool = False
    stage: str = "start"

    def clone(self) -> "_JsonNumberState":
        return _JsonNumberState(integer_only=self.integer_only, stage=self.stage)

    def feed_char(self, char: str) -> bool:
        if self.stage == "start":
            if char == "-":
                self.stage = "after_minus"
                return True
            if char == "0":
                self.stage = "zero"
                return True
            if char.isdigit() and char != "0":
                self.stage = "int"
                return True
            return False
        if self.stage == "after_minus":
            if char == "0":
                self.stage = "zero"
                return True
            if char.isdigit() and char != "0":
                self.stage = "int"
                return True
            return False
        if self.stage == "zero":
            if char == "." and not self.integer_only:
                self.stage = "after_dot"
                return True
            if char in "eE" and not self.integer_only:
                self.stage = "after_exp"
                return True
            return False
        if self.stage == "int":
            if char.isdigit():
                return True
            if char == "." and not self.integer_only:
                self.stage = "after_dot"
                return True
            if char in "eE" and not self.integer_only:
                self.stage = "after_exp"
                return True
            return False
        if self.stage == "after_dot":
            if char.isdigit():
                self.stage = "frac"
                return True
            return False
        if self.stage == "frac":
            if char.isdigit():
                return True
            if char in "eE":
                self.stage = "after_exp"
                return True
            return False
        if self.stage == "after_exp":
            if char in "+-":
                self.stage = "after_exp_sign"
                return True
            if char.isdigit():
                self.stage = "exp"
                return True
            return False
        if self.stage == "after_exp_sign":
            if char.isdigit():
                self.stage = "exp"
                return True
            return False
        if self.stage == "exp":
            return char.isdigit()
        return False

    def can_end(self) -> bool:
        return self.stage in {"zero", "int", "frac", "exp"}

    def allowed_first_bytes(self, delimiter_chars: set[str]) -> set[int]:
        allowed = set(ord(char) for char in delimiter_chars)
        if self.stage in {"start", "after_minus"}:
            allowed.update(ord(char) for char in _NUMBER_START_CHARS)
            allowed.discard(ord("-"))
            if self.stage == "start":
                allowed.add(ord("-"))
            return allowed
        if self.stage == "zero":
            if not self.integer_only:
                allowed.update({ord("."), ord("e"), ord("E")})
            return allowed
        if self.stage == "int":
            allowed.update(ord(char) for char in "0123456789")
            if not self.integer_only:
                allowed.update({ord("."), ord("e"), ord("E")})
            return allowed
        if self.stage == "after_dot":
            return set(ord(char) for char in "0123456789")
        if self.stage == "frac":
            allowed.update(ord(char) for char in "0123456789")
            allowed.update({ord("e"), ord("E")})
            return allowed
        if self.stage == "after_exp":
            return set(ord(char) for char in "0123456789+-")
        if self.stage == "after_exp_sign":
            return set(ord(char) for char in "0123456789")
        if self.stage == "exp":
            allowed.update(ord(char) for char in "0123456789")
            return allowed
        return allowed


@dataclass(slots=True)
class JsonObjectConstraint:
    allowed_root_keys: dict[str, frozenset[str]] = field(default_factory=dict)
    required_root_keys: frozenset[str] = frozenset()
    root_additional_properties: bool = True
    _started: bool = False
    _complete: bool = False
    _frames: list[_ContainerFrame] = field(default_factory=list)
    _string_role: str | None = None
    _string_buffer: str = ""
    _escape_active: bool = False
    _unicode_digits_remaining: int = 0
    _literal_target: str | None = None
    _literal_kind: str | None = None
    _literal_pos: int = 0
    _number_state: _JsonNumberState | None = None
    _current_root_key: str | None = None
    _completed_root_keys: set[str] = field(default_factory=set)

    def clone(self) -> "JsonObjectConstraint":
        return JsonObjectConstraint(
            allowed_root_keys=dict(self.allowed_root_keys),
            required_root_keys=frozenset(self.required_root_keys),
            root_additional_properties=bool(self.root_additional_properties),
            _started=bool(self._started),
            _complete=bool(self._complete),
            _frames=[_ContainerFrame(frame.kind, frame.mode) for frame in self._frames],
            _string_role=self._string_role,
            _string_buffer=self._string_buffer,
            _escape_active=bool(self._escape_active),
            _unicode_digits_remaining=int(self._unicode_digits_remaining),
            _literal_target=self._literal_target,
            _literal_kind=self._literal_kind,
            _literal_pos=int(self._literal_pos),
            _number_state=None if self._number_state is None else self._number_state.clone(),
            _current_root_key=self._current_root_key,
            _completed_root_keys=set(self._completed_root_keys),
        )

    def feed_text(self, text: str) -> bool:
        index = 0
        while index < len(text):
            char = text[index]
            if self._string_role is not None:
                if not self._feed_string_char(char):
                    return False
                index += 1
                continue
            if self._literal_target is not None:
                if not self._feed_literal_char(char):
                    return False
                index += 1
                continue
            if self._number_state is not None:
                if self._number_state.feed_char(char):
                    index += 1
                    continue
                if char not in self._delimiter_chars_after_value() or not self._number_state.can_end():
                    return False
                self._number_state = None
                if not self._complete_value():
                    return False
                continue
            if self._complete:
                if char in _JSON_WHITESPACE:
                    index += 1
                    continue
                return False
            if not self._started:
                if char in _JSON_WHITESPACE:
                    index += 1
                    continue
                if char != "{":
                    return False
                self._started = True
                self._frames.append(_ContainerFrame(kind="object", mode="key_or_end"))
                index += 1
                continue
            frame = self._frames[-1]
            if frame.kind == "object":
                if frame.mode == "key_or_end":
                    if char in _JSON_WHITESPACE:
                        index += 1
                        continue
                    if char == "}":
                        if not self._can_close_root_object():
                            return False
                        if not self._close_container("object"):
                            return False
                        index += 1
                        continue
                    if char != '"':
                        return False
                    self._string_role = "key"
                    self._string_buffer = ""
                    index += 1
                    continue
                if frame.mode == "colon":
                    if char in _JSON_WHITESPACE:
                        index += 1
                        continue
                    if char != ":":
                        return False
                    frame.mode = "value"
                    index += 1
                    continue
                if frame.mode == "value":
                    if char in _JSON_WHITESPACE:
                        index += 1
                        continue
                    if not self._start_value(char):
                        return False
                    index += 1
                    continue
                if frame.mode == "comma_or_end":
                    if char in _JSON_WHITESPACE:
                        index += 1
                        continue
                    if char == ",":
                        frame.mode = "key_or_end"
                        index += 1
                        continue
                    if char == "}":
                        if not self._can_close_root_object():
                            return False
                        if not self._close_container("object"):
                            return False
                        index += 1
                        continue
                    return False
            if frame.kind == "array":
                if frame.mode == "value_or_end":
                    if char in _JSON_WHITESPACE:
                        index += 1
                        continue
                    if char == "]":
                        if not self._close_container("array"):
                            return False
                        index += 1
                        continue
                    if not self._start_value(char):
                        return False
                    index += 1
                    continue
                if frame.mode == "comma_or_end":
                    if char in _JSON_WHITESPACE:
                        index += 1
                        continue
                    if char == ",":
                        frame.mode = "value_or_end"
                        index += 1
                        continue
                    if char == "]":
                        if not self._close_container("array"):
                            return False
                        index += 1
                        continue
                    return False
            return False
        return True

    def allowed_first_bytes(self) -> set[int] | None:
        if self._complete:
            return set()
        if self._string_role is not None:
            return self._allowed_string_first_bytes()
        if self._literal_target is not None:
            if self._literal_pos >= len(self._literal_target):
                return set()
            return set(self._literal_target[self._literal_pos].encode("utf-8"))
        if self._number_state is not None:
            return self._number_state.allowed_first_bytes(self._delimiter_chars_after_value())
        if not self._started:
            return set(ord(char) for char in "{ \t\r\n")
        if self._complete:
            return set(ord(char) for char in _JSON_WHITESPACE)
        frame = self._frames[-1]
        if frame.kind == "object":
            if frame.mode == "key_or_end":
                chars = set(_JSON_WHITESPACE)
                chars.add('"')
                if self._can_close_root_object():
                    chars.add("}")
                return set(ord(char) for char in chars)
            if frame.mode == "colon":
                return set(ord(char) for char in (_JSON_WHITESPACE | {":"}))
            if frame.mode == "value":
                chars = set(_JSON_WHITESPACE)
                chars.update(self._value_start_chars(self._allowed_value_kinds()))
                return set(ord(char) for char in chars)
            if frame.mode == "comma_or_end":
                return set(ord(char) for char in (_JSON_WHITESPACE | {",", "}"}))
        if frame.kind == "array":
            if frame.mode == "value_or_end":
                chars = set(_JSON_WHITESPACE)
                chars.add("]")
                chars.update(self._value_start_chars(_ANY_VALUE_KINDS))
                return set(ord(char) for char in chars)
            if frame.mode == "comma_or_end":
                return set(ord(char) for char in (_JSON_WHITESPACE | {",", "]"}))
        return None

    def is_complete(self) -> bool:
        return self._complete

    def finish_reason(self) -> str:
        return "constraint_stop"

    def _allowed_value_kinds(self) -> frozenset[str]:
        if not self._frames:
            return frozenset({"object"})
        frame = self._frames[-1]
        if frame.kind != "object" or frame.mode != "value" or len(self._frames) != 1 or self._current_root_key is None:
            return _ANY_VALUE_KINDS
        kinds = self.allowed_root_keys.get(self._current_root_key)
        if not kinds or "any" in kinds:
            return _ANY_VALUE_KINDS
        return frozenset(kinds)

    def _value_start_chars(self, kinds: frozenset[str]) -> set[str]:
        chars: set[str] = set()
        expanded = _ANY_VALUE_KINDS if "any" in kinds else kinds
        for kind in expanded:
            chars.update(_VALUE_START_BY_KIND.get(kind, frozenset()))
        return chars

    def _start_value(self, char: str) -> bool:
        kinds = self._allowed_value_kinds()
        if char == '"' and '"' in self._value_start_chars(kinds):
            self._string_role = "value"
            self._string_buffer = ""
            return True
        if char == "{" and "{" in self._value_start_chars(kinds):
            self._frames.append(_ContainerFrame(kind="object", mode="key_or_end"))
            return True
        if char == "[" and "[" in self._value_start_chars(kinds):
            self._frames.append(_ContainerFrame(kind="array", mode="value_or_end"))
            return True
        if char in _NUMBER_START_CHARS and any(kind in kinds for kind in ("number", "integer")):
            self._number_state = _JsonNumberState(integer_only="integer" in kinds and "number" not in kinds)
            return self._number_state.feed_char(char)
        if char == "t" and "boolean" in kinds:
            self._literal_target = "true"
            self._literal_kind = "boolean"
            self._literal_pos = 1
            return True
        if char == "f" and "boolean" in kinds:
            self._literal_target = "false"
            self._literal_kind = "boolean"
            self._literal_pos = 1
            return True
        if char == "n" and "null" in kinds:
            self._literal_target = "null"
            self._literal_kind = "null"
            self._literal_pos = 1
            return True
        return False

    def _feed_string_char(self, char: str) -> bool:
        if self._unicode_digits_remaining > 0:
            if char not in _HEX_CHARS:
                return False
            self._unicode_digits_remaining -= 1
            return True
        if self._escape_active:
            if char not in _ESCAPE_CHARS:
                return False
            self._escape_active = False
            if char == "u":
                self._unicode_digits_remaining = 4
            return True
        if char == "\\":
            self._escape_active = True
            return True
        if char == '"':
            if self._string_role == "key":
                if len(self._frames) == 1:
                    if not self.root_additional_properties and self.allowed_root_keys:
                        if self._string_buffer not in self.allowed_root_keys:
                            return False
                    if self._string_buffer in self._completed_root_keys:
                        return False
                    self._current_root_key = self._string_buffer
                self._frames[-1].mode = "colon"
            else:
                if not self._complete_value():
                    return False
            self._string_role = None
            self._string_buffer = ""
            self._escape_active = False
            self._unicode_digits_remaining = 0
            return True
        if ord(char) < 0x20:
            return False
        self._string_buffer += char
        if self._string_role == "key" and len(self._frames) == 1 and not self.root_additional_properties and self.allowed_root_keys:
            if not any(
                key not in self._completed_root_keys and key.startswith(self._string_buffer)
                for key in self.allowed_root_keys
            ):
                return False
        return True

    def _allowed_string_first_bytes(self) -> set[int] | None:
        if self._unicode_digits_remaining > 0:
            return set(ord(char) for char in _HEX_CHARS)
        if self._escape_active:
            return set(ord(char) for char in _ESCAPE_CHARS)
        if self._string_role == "key" and len(self._frames) == 1 and not self.root_additional_properties and self.allowed_root_keys:
            chars: set[str] = {'"', "\\"}
            for key in self.allowed_root_keys:
                if key in self._completed_root_keys or not key.startswith(self._string_buffer):
                    continue
                if len(key) > len(self._string_buffer):
                    chars.add(key[len(self._string_buffer)])
                else:
                    chars.add('"')
            return set(ord(char) for char in chars)
        return None

    def _feed_literal_char(self, char: str) -> bool:
        if self._literal_target is None:
            return False
        if self._literal_pos >= len(self._literal_target):
            return False
        if char != self._literal_target[self._literal_pos]:
            return False
        self._literal_pos += 1
        if self._literal_pos >= len(self._literal_target):
            self._literal_target = None
            self._literal_pos = 0
            self._literal_kind = None
            return self._complete_value()
        return True

    def _complete_value(self) -> bool:
        if not self._frames:
            self._complete = True
            return True
        frame = self._frames[-1]
        if frame.kind == "object":
            if frame.mode != "value":
                return False
            if len(self._frames) == 1 and self._current_root_key is not None:
                self._completed_root_keys.add(self._current_root_key)
                self._current_root_key = None
            frame.mode = "comma_or_end"
            return True
        if frame.kind == "array":
            if frame.mode != "value_or_end":
                return False
            frame.mode = "comma_or_end"
            return True
        return False

    def _delimiter_chars_after_value(self) -> set[str]:
        chars = set(_JSON_WHITESPACE)
        if not self._frames:
            return chars
        frame = self._frames[-1]
        if frame.kind == "object" and frame.mode == "value":
            chars.update({",", "}"})
            return chars
        if frame.kind == "array" and frame.mode == "value_or_end":
            chars.update({",", "]"})
            return chars
        return chars

    def _close_container(self, kind: str) -> bool:
        if not self._frames or self._frames[-1].kind != kind:
            return False
        self._frames.pop()
        if not self._frames:
            self._complete = True
            return True
        return self._complete_value()

    def _can_close_root_object(self) -> bool:
        if len(self._frames) != 1 or self._frames[0].kind != "object":
            return True
        if self._current_root_key is not None:
            return False
        return self.required_root_keys.issubset(self._completed_root_keys)


def extract_root_object_schema_hints(schema: Mapping[str, Any] | None) -> tuple[dict[str, frozenset[str]], frozenset[str], bool]:
    if not isinstance(schema, Mapping):
        return {}, frozenset(), True
    properties = schema.get("properties")
    raw_required = schema.get("required")
    additional_properties = schema.get("additionalProperties", True)
    allowed_root_keys: dict[str, frozenset[str]] = {}
    if isinstance(properties, Mapping):
        for key, value in properties.items():
            allowed_root_keys[str(key)] = _schema_value_kinds(value)
    required = frozenset(
        str(item)
        for item in raw_required
        if isinstance(raw_required, Sequence) and not isinstance(raw_required, (str, bytes, bytearray))
        for item in [item]
        if str(item)
    )
    return allowed_root_keys, required, bool(additional_properties)


def _schema_value_kinds(schema: Any) -> frozenset[str]:
    if not isinstance(schema, Mapping):
        return frozenset({"any"})
    if "enum" in schema and isinstance(schema.get("enum"), Sequence) and not isinstance(schema.get("enum"), (str, bytes, bytearray)):
        enum_kinds: set[str] = set()
        for item in schema.get("enum", ()):
            enum_kinds.add(_value_kind_for_instance(item))
        return frozenset(enum_kinds or {"any"})
    union: set[str] = set()
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        union.add(_TYPE_ALIASES.get(raw_type, raw_type))
    elif isinstance(raw_type, Sequence) and not isinstance(raw_type, (str, bytes, bytearray)):
        for item in raw_type:
            if isinstance(item, str):
                union.add(_TYPE_ALIASES.get(item, item))
    for key in ("anyOf", "oneOf", "allOf"):
        items = schema.get(key)
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
            for item in items:
                union.update(_schema_value_kinds(item))
    if "properties" in schema:
        union.add("object")
    if "items" in schema:
        union.add("array")
    if not union:
        return frozenset({"any"})
    normalized = {kind for kind in union if kind in _ANY_VALUE_KINDS or kind == "any"}
    return frozenset(normalized or {"any"})


def _value_kind_for_instance(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    return "any"


__all__ = [
    "JsonObjectConstraint",
    "extract_root_object_schema_hints",
]
