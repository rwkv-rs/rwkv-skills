from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class FunctionCallEvent:
    type: str
    role: str | None = None
    content: str | None = None
    name: str | None = None
    arguments: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def event_payloads(events: Iterable[FunctionCallEvent]) -> list[dict[str, Any]]:
    return [asdict(event) for event in events]
