from __future__ import annotations

"""Shared dataclasses for function-call benchmark tasks."""

from dataclasses import dataclass, field
from typing import Any

from .base import JsonlDataset, RecordBase


@dataclass(slots=True)
class FunctionCallTaskRecord(RecordBase):
    task_id: str
    instruction: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    expected_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    scorer: dict[str, Any] = field(default_factory=dict)
    tools: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    max_steps: int | None = None
    time_limit_s: float | None = None


class FunctionCallTaskDataset(JsonlDataset[FunctionCallTaskRecord]):
    """Immutable dataset wrapper for function-call benchmark tasks."""


__all__ = [
    "FunctionCallTaskRecord",
    "FunctionCallTaskDataset",
]
