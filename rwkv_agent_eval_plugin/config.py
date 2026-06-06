from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from lexical_chunk_router import LongDocConfig, ToolRouterConfig

AgentPluginMode = Literal["off", "lexical"]
AGENT_PLUGIN_MODE_CHOICES: tuple[str, ...] = ("off", "lexical")
_DEFAULT_TOOL_ROUTER = ToolRouterConfig()
_DEFAULT_LONG_CONTEXT = LongDocConfig()


@dataclass(frozen=True, slots=True)
class AgentEvalPluginConfig:
    """Configuration for RWKV multi-turn agent prompt preparation.

    `enabled` is the TOML/CLI gate. When enabled, lexical tool routing and
    long-context routing default to on unless the benchmark explicitly sets the
    existing router mode fields to `off`.
    """

    enabled: bool = False
    tool_router_mode: AgentPluginMode = "off"
    long_context_router_mode: AgentPluginMode = "off"
    tool_router: ToolRouterConfig = field(default_factory=lambda: ToolRouterConfig(enabled=False))
    long_context: LongDocConfig = field(default_factory=lambda: LongDocConfig(enabled=False))
    long_context_query_chars: int = 1200
    long_context_fallback_to_original_on_empty: bool = True
    prompt_max_chars: int | None = None
    history_max_chars: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "tool_router_mode": self.tool_router_mode,
            "long_context_router_mode": self.long_context_router_mode,
            "tool_router": {
                "enabled": bool(self.tool_router.enabled),
                "max_tools": int(self.tool_router.max_tools),
                "trigger_tool_count": int(self.tool_router.trigger_tool_count),
                "trigger_catalog_chars": int(self.tool_router.trigger_catalog_chars),
                "context_chars": int(self.tool_router.context_chars),
                "description_chars": int(self.tool_router.description_chars),
                "fallback_to_all_on_empty": bool(self.tool_router.fallback_to_all_on_empty),
                "enable_domain_hints": bool(self.tool_router.enable_domain_hints),
            },
            "long_context": {
                "enabled": bool(self.long_context.enabled),
                "min_long_text_chars": int(self.long_context.min_long_text_chars),
                "max_chunk_chars": int(self.long_context.max_chunk_chars),
                "overlap_lines": int(self.long_context.overlap_lines),
                "max_evidence_chunks": int(self.long_context.max_evidence_chunks),
                "max_evidence_chars": int(self.long_context.max_evidence_chars),
                "query_chars": int(self.long_context_query_chars),
                "fallback_to_original_on_empty": bool(self.long_context_fallback_to_original_on_empty),
            },
            "prompt_max_chars": self.prompt_max_chars,
            "history_max_chars": self.history_max_chars,
        }


def agent_plugin_config_to_payload(config: AgentEvalPluginConfig | None = None) -> dict[str, Any]:
    return (config or AgentEvalPluginConfig()).to_payload()


def agent_plugin_config_from_sources(
    args: Any | None = None,
    benchmark_config: Any | None = None,
    *,
    enabled_default: bool = False,
) -> AgentEvalPluginConfig:
    """Build plugin config from argparse/TOML-like sources.

    The function accepts mappings, argparse namespaces, or arbitrary config
    objects so sibling RWKV benchmark projects can reuse it without adopting the
    rwkv-skills config classes.
    """

    enabled = _read_bool(
        args,
        benchmark_config,
        names=("agent_plugin_enabled",),
        default=enabled_default,
    )
    default_mode: AgentPluginMode = "lexical" if enabled else "off"
    tool_mode = _read_mode(
        args,
        benchmark_config,
        names=("tool_router_mode",),
        default=default_mode,
    )
    long_mode = _read_mode(
        args,
        benchmark_config,
        names=(
            "long_context_router_mode",
            "long_doc_mode",
        ),
        default=default_mode,
    )

    tool_router = ToolRouterConfig(
        enabled=enabled and tool_mode != "off",
        max_tools=_read_int(
            args,
            benchmark_config,
            names=("tool_router_max_tools",),
            default=_DEFAULT_TOOL_ROUTER.max_tools,
            minimum=1,
        ),
        trigger_tool_count=_read_int(
            args,
            benchmark_config,
            names=("tool_router_trigger_tool_count",),
            default=_DEFAULT_TOOL_ROUTER.trigger_tool_count,
            minimum=1,
        ),
        trigger_catalog_chars=_read_int(
            args,
            benchmark_config,
            names=("tool_router_trigger_catalog_chars",),
            default=_DEFAULT_TOOL_ROUTER.trigger_catalog_chars,
            minimum=1,
        ),
        context_chars=_read_int(
            args,
            benchmark_config,
            names=("tool_router_context_chars",),
            default=_DEFAULT_TOOL_ROUTER.context_chars,
            minimum=1,
        ),
        description_chars=_read_int(
            args,
            benchmark_config,
            names=("tool_router_description_chars",),
            default=_DEFAULT_TOOL_ROUTER.description_chars,
            minimum=40,
        ),
        fallback_to_all_on_empty=_read_bool(
            args,
            benchmark_config,
            names=("tool_router_fallback_to_all_on_empty",),
            default=True,
        ),
        enable_domain_hints=_read_bool(
            args,
            benchmark_config,
            names=("tool_router_enable_domain_hints",),
            default=True,
        ),
    )
    long_context = LongDocConfig(
        enabled=enabled and long_mode != "off",
        max_chunk_chars=_read_int(
            args,
            benchmark_config,
            names=("long_context_chunk_chars", "long_doc_max_chars"),
            default=_DEFAULT_LONG_CONTEXT.max_chunk_chars,
            minimum=1,
        ),
        overlap_lines=_read_int(
            args,
            benchmark_config,
            names=("long_context_overlap_lines", "long_doc_overlap_lines"),
            default=_DEFAULT_LONG_CONTEXT.overlap_lines,
            minimum=0,
        ),
        min_long_text_chars=_read_int(
            args,
            benchmark_config,
            names=("long_context_min_chars", "long_doc_min_chars"),
            default=_DEFAULT_LONG_CONTEXT.min_long_text_chars,
            minimum=1,
        ),
        max_evidence_chunks=_read_int(
            args,
            benchmark_config,
            names=(
                "long_context_max_evidence_chunks",
                "long_doc_max_evidence_chunks",
            ),
            default=_DEFAULT_LONG_CONTEXT.max_evidence_chunks,
            minimum=1,
        ),
        max_evidence_chars=_read_int(
            args,
            benchmark_config,
            names=(
                "long_context_max_evidence_chars",
                "long_doc_max_evidence_chars",
            ),
            default=_DEFAULT_LONG_CONTEXT.max_evidence_chars,
            minimum=1,
        ),
    )
    return AgentEvalPluginConfig(
        enabled=enabled,
        tool_router_mode=tool_mode,
        long_context_router_mode=long_mode,
        tool_router=tool_router,
        long_context=long_context,
        long_context_query_chars=_read_int(
            args,
            benchmark_config,
            names=("long_context_query_chars", "long_doc_query_chars"),
            default=1200,
            minimum=1,
        ),
        long_context_fallback_to_original_on_empty=_read_bool(
            args,
            benchmark_config,
            names=("long_context_fallback_to_original_on_empty",),
            default=True,
        ),
        prompt_max_chars=_read_optional_int(args, benchmark_config, names=("prompt_max_chars",)),
        history_max_chars=_read_optional_int(args, benchmark_config, names=("history_max_chars",)),
    )


def _read_mode(
    *sources: Any,
    names: tuple[str, ...],
    default: AgentPluginMode,
) -> AgentPluginMode:
    for name in names:
        value = _first_value(sources, name)
        if value is None:
            continue
        mode = str(value).strip().lower()
        if mode not in AGENT_PLUGIN_MODE_CHOICES:
            raise ValueError(
                f"unsupported agent plugin mode {mode!r}; expected one of {', '.join(AGENT_PLUGIN_MODE_CHOICES)}"
            )
        return mode  # type: ignore[return-value]
    return default


def _read_bool(*sources: Any, names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        value = _first_value(sources, name)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
    return bool(default)


def _read_int(*sources: Any, names: tuple[str, ...], default: int, minimum: int) -> int:
    value = _read_optional_int(*sources, names=names)
    if value is None:
        return max(minimum, int(default))
    return max(minimum, int(value))


def _read_optional_int(*sources: Any, names: tuple[str, ...]) -> int | None:
    for name in names:
        value = _first_value(sources, name)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                continue
    return None


def _first_value(sources: tuple[Any, ...], name: str) -> Any | None:
    for source in sources:
        if source is None:
            continue
        if isinstance(source, Mapping):
            if name in source and source[name] is not None:
                return source[name]
            continue
        value = getattr(source, name, None)
        if value is not None:
            return value
    return None


__all__ = [
    "AGENT_PLUGIN_MODE_CHOICES",
    "AgentEvalPluginConfig",
    "AgentPluginMode",
    "agent_plugin_config_from_sources",
    "agent_plugin_config_to_payload",
]
