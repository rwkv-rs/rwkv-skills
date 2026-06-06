"""RWKV ecosystem helpers for long-context multi-turn agent benchmarks.

The package is intentionally limited to agent-eval prompt preparation. It does
not own model loading, scheduler state, database writes, or benchmark scoring.
"""

from .config import (
    AGENT_PLUGIN_MODE_CHOICES,
    AgentEvalPluginConfig,
    agent_plugin_config_from_sources,
    agent_plugin_config_to_payload,
)
from .prompt import AgentPromptRouteResult, route_agent_prompt_inputs

__all__ = [
    "AGENT_PLUGIN_MODE_CHOICES",
    "AgentEvalPluginConfig",
    "AgentPromptRouteResult",
    "agent_plugin_config_from_sources",
    "agent_plugin_config_to_payload",
    "route_agent_prompt_inputs",
]
