from __future__ import annotations

"""Compatibility wrapper around the packaged RWKV long-context tool router."""

from dataclasses import dataclass
from typing import Any, Literal

from src.plugins.lexical_chunk_router import tool_router as _tool_router

DEFAULT_TOOL_ROUTER_CONTEXT_CHARS = _tool_router.DEFAULT_TOOL_ROUTER_CONTEXT_CHARS
DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS = _tool_router.DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS
DEFAULT_TOOL_ROUTER_MAX_TOKENS = _tool_router.DEFAULT_TOOL_ROUTER_MAX_TOKENS
DEFAULT_TOOL_ROUTER_MAX_TOOLS = _tool_router.DEFAULT_TOOL_ROUTER_MAX_TOOLS
DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS = _tool_router.DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS
DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT = _tool_router.DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT
TOOL_ROUTER_MODE_CHOICES = _tool_router.TOOL_ROUTER_MODE_CHOICES

ToolRouterMode = Literal["off", "lexical", "model"]
ToolRouteResult = _tool_router.ToolRouteResult


@dataclass(frozen=True, slots=True)
class ToolRoutingConfig:
    mode: ToolRouterMode = "off"
    max_tools: int = DEFAULT_TOOL_ROUTER_MAX_TOOLS
    trigger_tool_count: int = DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT
    trigger_catalog_chars: int = DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS
    context_chars: int = DEFAULT_TOOL_ROUTER_CONTEXT_CHARS
    max_tokens: int = DEFAULT_TOOL_ROUTER_MAX_TOKENS
    description_chars: int = DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS
    fallback_to_all_on_empty: bool = True
    enable_domain_hints: bool = True

    @property
    def active(self) -> bool:
        return self.mode != "off"

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


def tool_routing_config_from_args(args: Any) -> ToolRoutingConfig:
    mode = str(getattr(args, "tool_router_mode", "off") or "off").strip().lower()
    if mode not in TOOL_ROUTER_MODE_CHOICES:
        raise ValueError(
            f"unsupported tool router mode {mode!r}; expected one of {', '.join(TOOL_ROUTER_MODE_CHOICES)}"
        )
    return ToolRoutingConfig(
        mode=mode,  # type: ignore[arg-type]
        max_tools=_arg_int(args, "tool_router_max_tools", DEFAULT_TOOL_ROUTER_MAX_TOOLS, minimum=1),
        trigger_tool_count=_arg_int(
            args,
            "tool_router_trigger_tool_count",
            DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT,
            minimum=1,
        ),
        trigger_catalog_chars=_arg_int(
            args,
            "tool_router_trigger_catalog_chars",
            DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS,
            minimum=1,
        ),
        context_chars=_arg_int(args, "tool_router_context_chars", DEFAULT_TOOL_ROUTER_CONTEXT_CHARS, minimum=512),
        max_tokens=_arg_int(args, "tool_router_max_tokens", DEFAULT_TOOL_ROUTER_MAX_TOKENS, minimum=32),
        description_chars=_arg_int(
            args,
            "tool_router_description_chars",
            DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS,
            minimum=40,
        ),
    )


def route_tools_for_prompt(
    tools: list[Any] | tuple[Any, ...],
    messages: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    config: ToolRoutingConfig | None = None,
    engine: Any | None = None,
    sampling: Any | None = None,
    control_tool_names: tuple[str, ...] | list[str] = (),
    progress_desc: str = "ToolRouter",
    prompt_seed: int | None = None,
) -> ToolRouteResult:
    return _tool_router.route_tools(
        tools,
        messages,
        config=config,
        backend=engine,
        sampling=sampling,
        control_tool_names=control_tool_names,
        progress_desc=progress_desc,
        prompt_seed=prompt_seed,
    )


def build_tool_router_prompt(
    tools: list[Any] | tuple[Any, ...],
    *,
    context: str,
    config: ToolRoutingConfig | None = None,
) -> str:
    return _tool_router.build_tool_router_prompt(tools, context=context, config=config)


def parse_tool_router_response(text: str) -> list[str]:
    return _tool_router.parse_tool_router_response(text)


def render_tool_routing_context(messages: list[dict[str, Any]] | tuple[dict[str, Any], ...], *, max_chars: int) -> str:
    return _tool_router.render_tool_routing_context(messages, max_chars=max_chars)


def tool_catalog_chars(tools: list[Any] | tuple[Any, ...]) -> int:
    return _tool_router.tool_catalog_chars(tools)


def tool_name(tool: Any) -> str:
    return _tool_router.tool_name(tool)


def normalize_tool_schema(tool: Any) -> dict[str, Any]:
    return _tool_router.normalize_tool_schema(tool)


def summarize_tool_for_router(tool: Any, *, description_chars: int) -> dict[str, Any]:
    return _tool_router.summarize_tool(tool, description_chars=description_chars)


def _arg_int(args: Any, name: str, default: int, *, minimum: int) -> int:
    raw = getattr(args, name, None)
    if raw is None:
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    return max(int(minimum), value)


__all__ = [
    "DEFAULT_TOOL_ROUTER_CONTEXT_CHARS",
    "DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS",
    "DEFAULT_TOOL_ROUTER_MAX_TOKENS",
    "DEFAULT_TOOL_ROUTER_MAX_TOOLS",
    "DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS",
    "DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT",
    "TOOL_ROUTER_MODE_CHOICES",
    "ToolRouteResult",
    "ToolRouterMode",
    "ToolRoutingConfig",
    "build_tool_router_prompt",
    "normalize_tool_schema",
    "parse_tool_router_response",
    "render_tool_routing_context",
    "route_tools_for_prompt",
    "summarize_tool_for_router",
    "tool_catalog_chars",
    "tool_name",
    "tool_routing_config_from_args",
]
