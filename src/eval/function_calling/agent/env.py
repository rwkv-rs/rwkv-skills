from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from src.eval.function_calling.common.action import ToolAction


@dataclass(frozen=True, slots=True)
class AgentObservation:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    observation: AgentObservation
    done: bool = False
    score: float | None = None
    success: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


class FunctionCallingEnv(Protocol):
    def reset(self) -> AgentObservation:
        ...

    def step(self, action: ToolAction) -> AgentStepResult:
        ...
