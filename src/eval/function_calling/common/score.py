from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class FunctionCallScore:
    success: bool
    reward: float
    fail_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "reward": float(self.reward),
            "fail_reason": self.fail_reason,
            "details": dict(self.details),
        }
