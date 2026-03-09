# Copyright Sierra

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    if name == "MockRetailDomainEnv":
        return getattr(import_module("tau_bench.envs.retail.env"), "MockRetailDomainEnv")
    raise AttributeError(name)


__all__ = ["MockRetailDomainEnv"]
