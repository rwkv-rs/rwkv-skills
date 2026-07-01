# Copyright Sierra

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "MockAirlineDomainEnv":
        return getattr(import_module("tau_bench.envs.airline.env"), "MockAirlineDomainEnv")
    raise AttributeError(name)


__all__ = ["MockAirlineDomainEnv"]
