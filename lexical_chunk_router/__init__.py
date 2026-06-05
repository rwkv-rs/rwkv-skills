"""Lexical long-context chunking and tool routing helpers.

This package is intentionally model-free. It can be copied into another
rwkv-skills-like project without pulling in eval runners, databases, or RWKV
inference code.
"""

from .long_doc import (
    EvidenceChunk,
    LongDocConfig,
    LongDocCompactionResult,
    LongDocMessageCompaction,
    TextChunk,
    chunk_text,
    compact_messages,
    compact_text,
    infer_query_from_messages,
    normalize_newlines,
    render_evidence_window,
    select_evidence_chunks,
)
from .tool_router import (
    ToolRouteResult,
    ToolRouterConfig,
    normalize_tool_schema,
    render_tool_routing_context,
    route_tools,
    route_tools_for_context,
    summarize_tool,
    tool_catalog_chars,
    tool_name,
)

__all__ = [
    "EvidenceChunk",
    "LongDocConfig",
    "LongDocCompactionResult",
    "LongDocMessageCompaction",
    "TextChunk",
    "ToolRouteResult",
    "ToolRouterConfig",
    "chunk_text",
    "compact_messages",
    "compact_text",
    "infer_query_from_messages",
    "normalize_newlines",
    "normalize_tool_schema",
    "render_evidence_window",
    "render_tool_routing_context",
    "route_tools",
    "route_tools_for_context",
    "select_evidence_chunks",
    "summarize_tool",
    "tool_catalog_chars",
    "tool_name",
]
