from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MAP = {
    "RWKVChatBridge": ("src.eval.agent_bench.chat_bridge", "RWKVChatBridge"),
    "ChatResult": ("src.eval.agent_bench.chat_bridge", "ChatResult"),
    "ParsedToolCall": ("src.eval.agent_bench.chat_bridge", "ParsedToolCall"),
    "StageRecord": ("src.eval.agent_bench.runtime", "StageRecord"),
    "EpisodeResult": ("src.eval.agent_bench.runtime", "EpisodeResult"),
    "run_tau_v1_episode": ("src.eval.agent_bench.runtime", "run_tau_v1_episode"),
    "run_tau_v2_episode": ("src.eval.agent_bench.runtime", "run_tau_v2_episode"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "RWKVChatBridge",
    "ChatResult",
    "ParsedToolCall",
    "StageRecord",
    "EpisodeResult",
    "run_tau_v1_episode",
    "run_tau_v2_episode",
]
