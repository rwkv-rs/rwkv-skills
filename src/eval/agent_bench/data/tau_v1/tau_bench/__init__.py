# Copyright Sierra

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "Env":
        return getattr(import_module("tau_bench.envs.base"), "Env")
    if name == "Agent":
        return getattr(import_module("tau_bench.agents.base"), "Agent")
    raise AttributeError(name)


__all__ = ["Env", "Agent"]
