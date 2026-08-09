from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HumanUser:
    """Simple terminal-backed user for manual runs."""

    total_cost: float = 0.0

    def reset(self, instruction: str) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")

    def get_total_cost(self) -> float:
        return self.total_cost


__all__ = [
    "HumanUser",
]
