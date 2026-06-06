from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lexical_chunk_router import (
    ToolRouteResult,
    compact_messages,
    compact_text,
    infer_query_from_messages,
    route_tools,
)

from .config import AgentEvalPluginConfig


@dataclass(frozen=True, slots=True)
class AgentPromptRouteResult:
    """Routed prompt inputs for a multi-turn agent benchmark."""

    domain_policy: str
    messages: list[dict[str, str]]
    selected_tools: list[Any]
    tool_route: ToolRouteResult | None
    long_context_trace: dict[str, Any] | None

    def trace_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.tool_route is not None:
            payload["tool_routing"] = self.tool_route.trace_payload()
        if self.long_context_trace is not None:
            payload["long_context"] = self.long_context_trace
        return payload


def route_agent_prompt_inputs(
    *,
    domain_policy: str,
    tools: Sequence[Any],
    messages: Sequence[Mapping[str, object]],
    config: AgentEvalPluginConfig,
    control_tool_names: Sequence[str] = (),
    query: str | None = None,
) -> AgentPromptRouteResult:
    """Prepare policy, history, and tool window for an agent prompt.

    The caller remains responsible for rendering the final prompt and enforcing
    the final prompt budget. This function only chooses lexical tool/evidence
    windows and returns trace data for benchmark records.
    """

    normalized_messages = _normalize_messages(messages)
    selected_tools = list(tools)
    tool_route: ToolRouteResult | None = None
    if config.enabled and config.tool_router.enabled:
        tool_route = route_tools(
            selected_tools,
            normalized_messages,
            config=config.tool_router,
            control_tool_names=control_tool_names,
        )
        selected_tools = list(tool_route.selected_tools)

    policy = str(domain_policy or "")
    long_context_trace: dict[str, Any] | None = None
    if config.enabled and config.long_context.enabled:
        resolved_query = query
        if resolved_query is None:
            resolved_query = infer_query_from_messages(
                normalized_messages,
                max_chars=max(1, int(config.long_context_query_chars)),
                skip_longer_than=max(1, int(config.long_context.min_long_text_chars)),
            )
        message_route = compact_messages(
            normalized_messages,
            query=resolved_query,
            config=config.long_context,
        )
        routed_messages: list[dict[str, str]] = []
        compacted_message_count = 0
        selected_by_message: dict[int, tuple[int, ...]] = {}
        for index, message in enumerate(message_route.messages):
            selected = message_route.selected_chunk_ids.get(index)
            if selected is not None and (selected or not config.long_context_fallback_to_original_on_empty):
                compacted_message_count += 1
                selected_by_message[index] = tuple(selected)
                routed_messages.append(dict(message))
                continue
            if selected is not None and config.long_context_fallback_to_original_on_empty:
                routed_messages.append(dict(normalized_messages[index]))
                continue
            routed_messages.append(dict(message))

        policy_route = compact_text(
            policy,
            query=resolved_query,
            config=config.long_context,
            label="agent domain policy",
        )
        policy_compacted = bool(policy_route.compacted)
        if policy_route.compacted and not policy_route.selected_chunk_ids and config.long_context_fallback_to_original_on_empty:
            routed_policy = policy
            policy_compacted = False
            policy_reason = "lexical_empty_original_fallback"
        else:
            routed_policy = policy_route.text
            policy_reason = policy_route.reason

        normalized_messages = routed_messages
        policy = routed_policy
        long_context_trace = {
            "mode": config.long_context_router_mode,
            "query_chars": len(str(resolved_query or "")),
            "compacted_message_count": int(compacted_message_count),
            "message_original_chars": _messages_chars(_normalize_messages(messages)),
            "message_routed_chars": _messages_chars(normalized_messages),
            "message_reason": "lexical" if compacted_message_count else "no_matching_long_messages",
            "policy_compacted": policy_compacted,
            "policy_original_chars": int(policy_route.original_chars),
            "policy_routed_chars": len(policy),
            "policy_reason": policy_reason,
        }
        if selected_by_message:
            long_context_trace["message_selected_chunk_ids"] = {
                str(index): list(chunk_ids) for index, chunk_ids in sorted(selected_by_message.items())
            }
        if policy_route.selected_chunk_ids and policy_compacted:
            long_context_trace["policy_selected_chunk_ids"] = list(policy_route.selected_chunk_ids)

    return AgentPromptRouteResult(
        domain_policy=policy,
        messages=normalized_messages,
        selected_tools=selected_tools,
        tool_route=tool_route,
        long_context_trace=long_context_trace,
    )


def _normalize_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role") or "user").strip().lower() or "user",
            "content": str(message.get("content") or ""),
        }
        for message in messages
        if str(message.get("content") or "")
    ]


def _messages_chars(messages: Sequence[Mapping[str, str]]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages)


__all__ = ["AgentPromptRouteResult", "route_agent_prompt_inputs"]
