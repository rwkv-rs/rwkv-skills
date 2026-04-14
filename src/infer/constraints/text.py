from __future__ import annotations

from dataclasses import dataclass, field

from .base import DecodeConstraint


@dataclass(slots=True)
class LiteralConstraint:
    literal: str
    _offset: int = 0

    def clone(self) -> "LiteralConstraint":
        return LiteralConstraint(self.literal, self._offset)

    def feed_text(self, text: str) -> bool:
        target = self.literal[self._offset :]
        if not target.startswith(text):
            return False
        self._offset += len(text)
        return self._offset <= len(self.literal)

    def allowed_first_bytes(self) -> set[int] | None:
        if self.is_complete():
            return set()
        next_char = self.literal[self._offset]
        return set(next_char.encode("utf-8"))

    def is_complete(self) -> bool:
        return self._offset >= len(self.literal)

    def finish_reason(self) -> str:
        return "constraint_stop"


@dataclass(slots=True)
class LiteralChoiceConstraint:
    choices: tuple[str, ...]
    _prefix: str = ""
    _active: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self._active:
            self._active = tuple(str(choice) for choice in self.choices if str(choice))

    def clone(self) -> "LiteralChoiceConstraint":
        return LiteralChoiceConstraint(self.choices, self._prefix, self._active)

    def feed_text(self, text: str) -> bool:
        prefix = self._prefix + text
        active = tuple(choice for choice in self._active if choice.startswith(prefix))
        if not active:
            return False
        self._prefix = prefix
        self._active = active
        return True

    def allowed_first_bytes(self) -> set[int] | None:
        if self.is_complete():
            return set()
        allowed: set[int] = set()
        prefix_len = len(self._prefix)
        for choice in self._active:
            if prefix_len >= len(choice):
                continue
            allowed.update(choice[prefix_len].encode("utf-8"))
        return allowed

    def is_complete(self) -> bool:
        return any(choice == self._prefix for choice in self._active) and not any(
            len(choice) > len(self._prefix) and choice.startswith(self._prefix)
            for choice in self._active
        )

    def matched_choice(self) -> str | None:
        if not self.is_complete():
            return None
        return self._prefix

    def finish_reason(self) -> str:
        return "constraint_stop"


@dataclass(slots=True)
class SequenceConstraint:
    parts: tuple[DecodeConstraint, ...]
    _index: int = 0

    def clone(self) -> "SequenceConstraint":
        return SequenceConstraint(tuple(part.clone() for part in self.parts), self._index)

    def feed_text(self, text: str) -> bool:
        for char in text:
            if self._index >= len(self.parts):
                return False
            part = self.parts[self._index]
            if not part.feed_text(char):
                return False
            if part.is_complete():
                self._index += 1
        return True

    def allowed_first_bytes(self) -> set[int] | None:
        if self.is_complete():
            return set()
        return self.parts[self._index].allowed_first_bytes()

    def is_complete(self) -> bool:
        return self._index >= len(self.parts)

    def finish_reason(self) -> str:
        return "constraint_stop"


@dataclass(slots=True)
class PlainTextConstraint:
    forbidden_substrings: tuple[str, ...] = ()
    _window: str = ""
    _max_forbidden_len: int = 0

    def __post_init__(self) -> None:
        if self._max_forbidden_len <= 0:
            self._max_forbidden_len = max((len(item) for item in self.forbidden_substrings if item), default=0)

    def clone(self) -> "PlainTextConstraint":
        return PlainTextConstraint(
            forbidden_substrings=self.forbidden_substrings,
            _window=self._window,
            _max_forbidden_len=self._max_forbidden_len,
        )

    def feed_text(self, text: str) -> bool:
        for char in text:
            self._window = (self._window + char)[-max(self._max_forbidden_len, 1) :]
            if any(forbidden and forbidden in self._window for forbidden in self.forbidden_substrings):
                return False
        return True

    def allowed_first_bytes(self) -> set[int] | None:
        return None

    def is_complete(self) -> bool:
        return False

    def finish_reason(self) -> str:
        return "constraint_stop"


__all__ = [
    "LiteralChoiceConstraint",
    "LiteralConstraint",
    "PlainTextConstraint",
    "SequenceConstraint",
]
