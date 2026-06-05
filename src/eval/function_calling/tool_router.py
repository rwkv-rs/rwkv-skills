from __future__ import annotations

"""Function-calling tool-window router wrapper.

This module is the single integration point for benchmark code. The standalone
``lexical_chunk_router`` package stays model-free, while callers use this
wrapper so future routing modes can be added without changing benchmark
adapters.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from lexical_chunk_router import tool_router as lexical_tool_router

ToolRouterMode = Literal["off", "lexical"]
TOOL_ROUTER_MODE_CHOICES: tuple[str, ...] = ("off", "lexical")

DEFAULT_TOOL_ROUTER_MAX_TOOLS = 12
DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT = 16
DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS = 6000
DEFAULT_TOOL_ROUTER_CONTEXT_CHARS = 5000
DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS = 240


@dataclass(frozen=True, slots=True)
class ToolRoutingConfig:
    mode: ToolRouterMode = "off"
    max_tools: int = DEFAULT_TOOL_ROUTER_MAX_TOOLS
    trigger_tool_count: int = DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT
    trigger_catalog_chars: int = DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS
    context_chars: int = DEFAULT_TOOL_ROUTER_CONTEXT_CHARS
    description_chars: int = DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS
    fallback_to_all_on_empty: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_tools": int(self.max_tools),
            "trigger_tool_count": int(self.trigger_tool_count),
            "trigger_catalog_chars": int(self.trigger_catalog_chars),
            "context_chars": int(self.context_chars),
            "description_chars": int(self.description_chars),
            "fallback_to_all_on_empty": bool(self.fallback_to_all_on_empty),
        }


@dataclass(frozen=True, slots=True)
class ToolRouteResult:
    selected_tools: list[Any]
    selected_names: tuple[str, ...]
    total_tool_count: int
    catalog_chars: int
    mode: ToolRouterMode
    routed: bool
    reason: str
    lexical_names: tuple[str, ...] = ()
    heuristic_names: tuple[str, ...] = ()

    def trace_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "routed": self.routed,
            "reason": self.reason,
            "selected_names": list(self.selected_names),
            "total_tool_count": int(self.total_tool_count),
            "catalog_chars": int(self.catalog_chars),
        }
        if self.lexical_names:
            payload["lexical_names"] = list(self.lexical_names)
        if self.heuristic_names:
            payload["heuristic_names"] = list(self.heuristic_names)
        return payload


def tool_routing_config_from_args(args: Any, *, base: ToolRoutingConfig | None = None) -> ToolRoutingConfig:
    cfg = base or ToolRoutingConfig()
    mode = _arg_str(args, "tool_router_mode", cfg.mode).lower()
    if mode not in TOOL_ROUTER_MODE_CHOICES:
        raise ValueError(
            f"unsupported tool router mode {mode!r}; expected one of {', '.join(TOOL_ROUTER_MODE_CHOICES)}"
        )
    return replace(
        cfg,
        mode=mode,  # type: ignore[arg-type]
        max_tools=_arg_int(args, "tool_router_max_tools", cfg.max_tools, minimum=1),
        trigger_tool_count=_arg_int(args, "tool_router_trigger_tool_count", cfg.trigger_tool_count, minimum=1),
        trigger_catalog_chars=_arg_int(
            args,
            "tool_router_trigger_catalog_chars",
            cfg.trigger_catalog_chars,
            minimum=1,
        ),
        context_chars=_arg_int(args, "tool_router_context_chars", cfg.context_chars, minimum=512),
        description_chars=_arg_int(args, "tool_router_description_chars", cfg.description_chars, minimum=40),
    )


def tool_routing_config_from_benchmark_config(config: Any | None) -> ToolRoutingConfig:
    if config is None:
        return ToolRoutingConfig()
    mode = str(getattr(config, "tool_router_mode", None) or "off").strip().lower()
    if mode not in TOOL_ROUTER_MODE_CHOICES:
        raise ValueError(
            f"unsupported tool router mode {mode!r}; expected one of {', '.join(TOOL_ROUTER_MODE_CHOICES)}"
        )
    return ToolRoutingConfig(
        mode=mode,  # type: ignore[arg-type]
        max_tools=_positive_int(getattr(config, "tool_router_max_tools", None), DEFAULT_TOOL_ROUTER_MAX_TOOLS),
        trigger_tool_count=_positive_int(
            getattr(config, "tool_router_trigger_tool_count", None),
            DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT,
        ),
        trigger_catalog_chars=_positive_int(
            getattr(config, "tool_router_trigger_catalog_chars", None),
            DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS,
        ),
        context_chars=_positive_int(getattr(config, "tool_router_context_chars", None), DEFAULT_TOOL_ROUTER_CONTEXT_CHARS),
        description_chars=_positive_int(
            getattr(config, "tool_router_description_chars", None),
            DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS,
        ),
    )


def tool_routing_config_to_payload(config: ToolRoutingConfig | None = None) -> dict[str, Any]:
    return (config or ToolRoutingConfig()).to_payload()


def route_tools_for_prompt(
    tools: Sequence[Any],
    messages: Sequence[Mapping[str, Any]],
    *,
    config: ToolRoutingConfig | None = None,
    control_tool_names: Sequence[str] = (),
) -> ToolRouteResult:
    cfg = config or ToolRoutingConfig()
    all_tools = list(tools)
    lexical_cfg = lexical_tool_router.ToolRouterConfig(
        enabled=cfg.enabled,
        max_tools=max(1, int(cfg.max_tools)),
        trigger_tool_count=max(1, int(cfg.trigger_tool_count)),
        trigger_catalog_chars=max(1, int(cfg.trigger_catalog_chars)),
        context_chars=max(1, int(cfg.context_chars)),
        description_chars=max(40, int(cfg.description_chars)),
        fallback_to_all_on_empty=bool(cfg.fallback_to_all_on_empty),
        enable_domain_hints=True,
    )
    route = lexical_tool_router.route_tools(
        all_tools,
        messages,
        config=lexical_cfg,
        control_tool_names=control_tool_names,
    )
    return ToolRouteResult(
        selected_tools=list(route.selected_tools),
        selected_names=tuple(route.selected_names),
        total_tool_count=int(route.total_tool_count),
        catalog_chars=int(route.catalog_chars),
        mode=cfg.mode,
        routed=bool(route.routed),
        reason=route.reason,
        lexical_names=tuple(route.lexical_names),
        heuristic_names=tuple(route.heuristic_names),
    )


def tool_catalog_chars(tools: Sequence[Any]) -> int:
    return lexical_tool_router.tool_catalog_chars(tools)


def tool_name(tool: Any) -> str:
    return lexical_tool_router.tool_name(tool)


def normalize_tool_schema(tool: Any) -> dict[str, Any]:
    return lexical_tool_router.normalize_tool_schema(tool)


def summarize_tool_for_router(tool: Any, *, description_chars: int = DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS) -> dict[str, Any]:
    return lexical_tool_router.summarize_tool(tool, description_chars=description_chars)


def _arg_str(args: Any, name: str, default: str) -> str:
    value = getattr(args, name, None)
    if value is None:
        return str(default)
    return str(value or default).strip()


def _arg_int(args: Any, name: str, default: int, *, minimum: int) -> int:
    value = getattr(args, name, None)
    if value is None:
        return max(minimum, int(default))
    return max(minimum, int(value))


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, int) and value > 0:
        return int(value)
    return int(default)


__all__ = [
    "DEFAULT_TOOL_ROUTER_CONTEXT_CHARS",
    "DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS",
    "DEFAULT_TOOL_ROUTER_MAX_TOOLS",
    "DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS",
    "DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT",
    "TOOL_ROUTER_MODE_CHOICES",
    "ToolRouteResult",
    "ToolRouterMode",
    "ToolRoutingConfig",
    "normalize_tool_schema",
    "route_tools_for_prompt",
    "summarize_tool_for_router",
    "tool_catalog_chars",
    "tool_name",
    "tool_routing_config_from_args",
    "tool_routing_config_from_benchmark_config",
    "tool_routing_config_to_payload",
]
