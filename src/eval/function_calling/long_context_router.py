from __future__ import annotations

"""Long-context routing wrapper for function-calling prompts.

Benchmark runners should import this module instead of calling
``lexical_chunk_router.long_doc`` directly. The standalone package stays
model-free; this wrapper owns rwkv-skills config parsing, trace payloads, and
fallback behavior.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from lexical_chunk_router import long_doc as lexical_long_doc

LongContextRouterMode = Literal["off", "lexical"]
LONG_CONTEXT_ROUTER_MODE_CHOICES: tuple[str, ...] = ("off", "lexical")

DEFAULT_LONG_CONTEXT_MIN_CHARS = 6000
DEFAULT_LONG_CONTEXT_CHUNK_CHARS = 1000
DEFAULT_LONG_CONTEXT_OVERLAP_LINES = 3
DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHUNKS = 4
DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHARS = 6000
DEFAULT_LONG_CONTEXT_QUERY_CHARS = 1200


@dataclass(frozen=True, slots=True)
class LongContextRoutingConfig:
    mode: LongContextRouterMode = "off"
    min_chars: int = DEFAULT_LONG_CONTEXT_MIN_CHARS
    chunk_chars: int = DEFAULT_LONG_CONTEXT_CHUNK_CHARS
    overlap_lines: int = DEFAULT_LONG_CONTEXT_OVERLAP_LINES
    max_evidence_chunks: int = DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHUNKS
    max_evidence_chars: int = DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHARS
    query_chars: int = DEFAULT_LONG_CONTEXT_QUERY_CHARS
    fallback_to_original_on_empty: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def to_payload(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "min_chars": int(self.min_chars),
            "chunk_chars": int(self.chunk_chars),
            "overlap_lines": int(self.overlap_lines),
            "max_evidence_chunks": int(self.max_evidence_chunks),
            "max_evidence_chars": int(self.max_evidence_chars),
            "query_chars": int(self.query_chars),
            "fallback_to_original_on_empty": bool(self.fallback_to_original_on_empty),
        }

    def to_lexical_config(self) -> lexical_long_doc.LongDocConfig:
        return lexical_long_doc.LongDocConfig(
            enabled=self.enabled,
            max_chunk_chars=max(1, int(self.chunk_chars)),
            overlap_lines=max(0, int(self.overlap_lines)),
            min_long_text_chars=max(1, int(self.min_chars)),
            max_evidence_chunks=max(1, int(self.max_evidence_chunks)),
            max_evidence_chars=max(1, int(self.max_evidence_chars)),
        )


@dataclass(frozen=True, slots=True)
class LongContextMessageRouteResult:
    messages: list[dict[str, str]]
    mode: LongContextRouterMode
    compacted_message_count: int
    selected_chunk_ids: dict[int, tuple[int, ...]]
    original_chars: int
    routed_chars: int
    reason: str

    def trace_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "compacted_message_count": int(self.compacted_message_count),
            "original_chars": int(self.original_chars),
            "routed_chars": int(self.routed_chars),
            "reason": self.reason,
        }
        if self.selected_chunk_ids:
            payload["selected_chunk_ids"] = {
                str(index): list(chunk_ids) for index, chunk_ids in sorted(self.selected_chunk_ids.items())
            }
        return payload


@dataclass(frozen=True, slots=True)
class LongContextTextRouteResult:
    text: str
    mode: LongContextRouterMode
    compacted: bool
    selected_chunk_ids: tuple[int, ...]
    original_chars: int
    routed_chars: int
    chunk_count: int
    reason: str

    def trace_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "compacted": bool(self.compacted),
            "original_chars": int(self.original_chars),
            "routed_chars": int(self.routed_chars),
            "chunk_count": int(self.chunk_count),
            "reason": self.reason,
        }
        if self.selected_chunk_ids:
            payload["selected_chunk_ids"] = list(self.selected_chunk_ids)
        return payload


def long_context_routing_config_from_args(
    args: Any,
    *,
    base: LongContextRoutingConfig | None = None,
) -> LongContextRoutingConfig:
    cfg = base or LongContextRoutingConfig()
    mode = _arg_str(args, "long_context_router_mode", cfg.mode).lower()
    if mode not in LONG_CONTEXT_ROUTER_MODE_CHOICES:
        raise ValueError(
            f"unsupported long context router mode {mode!r}; "
            f"expected one of {', '.join(LONG_CONTEXT_ROUTER_MODE_CHOICES)}"
        )
    return replace(
        cfg,
        mode=mode,  # type: ignore[arg-type]
        min_chars=_arg_int(args, "long_context_min_chars", cfg.min_chars, minimum=1),
        chunk_chars=_arg_int(args, "long_context_chunk_chars", cfg.chunk_chars, minimum=1),
        overlap_lines=_arg_int(args, "long_context_overlap_lines", cfg.overlap_lines, minimum=0),
        max_evidence_chunks=_arg_int(
            args,
            "long_context_max_evidence_chunks",
            cfg.max_evidence_chunks,
            minimum=1,
        ),
        max_evidence_chars=_arg_int(
            args,
            "long_context_max_evidence_chars",
            cfg.max_evidence_chars,
            minimum=1,
        ),
        query_chars=_arg_int(args, "long_context_query_chars", cfg.query_chars, minimum=1),
    )


def long_context_routing_config_from_benchmark_config(
    config: Any | None,
    *,
    fallback_mode: str = "off",
) -> LongContextRoutingConfig:
    mode = str(getattr(config, "long_context_router_mode", None) or fallback_mode or "off").strip().lower()
    if mode not in LONG_CONTEXT_ROUTER_MODE_CHOICES:
        raise ValueError(
            f"unsupported long context router mode {mode!r}; "
            f"expected one of {', '.join(LONG_CONTEXT_ROUTER_MODE_CHOICES)}"
        )
    return LongContextRoutingConfig(
        mode=mode,  # type: ignore[arg-type]
        min_chars=_positive_int(getattr(config, "long_context_min_chars", None), DEFAULT_LONG_CONTEXT_MIN_CHARS),
        chunk_chars=_positive_int(getattr(config, "long_context_chunk_chars", None), DEFAULT_LONG_CONTEXT_CHUNK_CHARS),
        overlap_lines=_non_negative_int(
            getattr(config, "long_context_overlap_lines", None),
            DEFAULT_LONG_CONTEXT_OVERLAP_LINES,
        ),
        max_evidence_chunks=_positive_int(
            getattr(config, "long_context_max_evidence_chunks", None),
            DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHUNKS,
        ),
        max_evidence_chars=_positive_int(
            getattr(config, "long_context_max_evidence_chars", None),
            DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHARS,
        ),
        query_chars=_positive_int(getattr(config, "long_context_query_chars", None), DEFAULT_LONG_CONTEXT_QUERY_CHARS),
    )


def long_context_routing_config_to_payload(config: LongContextRoutingConfig | None = None) -> dict[str, Any]:
    return (config or LongContextRoutingConfig()).to_payload()


def infer_long_context_query(
    messages: Sequence[Mapping[str, object]],
    *,
    config: LongContextRoutingConfig | None = None,
) -> str:
    cfg = config or LongContextRoutingConfig()
    return lexical_long_doc.infer_query_from_messages(
        messages,
        max_chars=max(1, int(cfg.query_chars)),
        skip_longer_than=max(1, int(cfg.min_chars)),
    )


def compact_messages_for_prompt(
    messages: Sequence[Mapping[str, object]],
    *,
    query: str | None = None,
    config: LongContextRoutingConfig | None = None,
) -> LongContextMessageRouteResult:
    cfg = config or LongContextRoutingConfig()
    normalized = _normalize_messages(messages)
    original_chars = _messages_chars(normalized)
    if not cfg.enabled:
        return LongContextMessageRouteResult(
            messages=normalized,
            mode=cfg.mode,
            compacted_message_count=0,
            selected_chunk_ids={},
            original_chars=original_chars,
            routed_chars=original_chars,
            reason="disabled",
        )

    result = lexical_long_doc.compact_messages(
        normalized,
        query=query,
        config=cfg.to_lexical_config(),
    )
    routed_messages: list[dict[str, str]] = []
    selected_chunk_ids: dict[int, tuple[int, ...]] = {}
    compacted_count = 0
    for index, message in enumerate(result.messages):
        selected = result.selected_chunk_ids.get(index)
        if selected is not None and (selected or not cfg.fallback_to_original_on_empty):
            compacted_count += 1
            selected_chunk_ids[index] = tuple(selected)
            routed_messages.append(dict(message))
            continue
        if selected is not None and cfg.fallback_to_original_on_empty:
            routed_messages.append(dict(normalized[index]))
            continue
        routed_messages.append(dict(message))

    routed_chars = _messages_chars(routed_messages)
    reason = "lexical" if compacted_count else "no_matching_long_messages"
    return LongContextMessageRouteResult(
        messages=routed_messages,
        mode=cfg.mode,
        compacted_message_count=compacted_count,
        selected_chunk_ids=selected_chunk_ids,
        original_chars=original_chars,
        routed_chars=routed_chars,
        reason=reason,
    )


def compact_text_for_prompt(
    text: str,
    *,
    query: str,
    config: LongContextRoutingConfig | None = None,
    label: str = "document",
) -> LongContextTextRouteResult:
    cfg = config or LongContextRoutingConfig()
    original = lexical_long_doc.normalize_newlines(text)
    if not cfg.enabled:
        return LongContextTextRouteResult(
            text=original,
            mode=cfg.mode,
            compacted=False,
            selected_chunk_ids=(),
            original_chars=len(original),
            routed_chars=len(original),
            chunk_count=0,
            reason="disabled",
        )

    result = lexical_long_doc.compact_text(
        original,
        query=query,
        config=cfg.to_lexical_config(),
        label=label,
    )
    if result.compacted and not result.selected_chunk_ids and cfg.fallback_to_original_on_empty:
        return LongContextTextRouteResult(
            text=original,
            mode=cfg.mode,
            compacted=False,
            selected_chunk_ids=(),
            original_chars=result.original_chars,
            routed_chars=len(original),
            chunk_count=result.chunk_count,
            reason="lexical_empty_original_fallback",
        )
    return LongContextTextRouteResult(
        text=result.text,
        mode=cfg.mode,
        compacted=bool(result.compacted),
        selected_chunk_ids=tuple(result.selected_chunk_ids),
        original_chars=int(result.original_chars),
        routed_chars=len(result.text),
        chunk_count=int(result.chunk_count),
        reason=result.reason,
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


def _non_negative_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(default)
    if isinstance(value, int) and value >= 0:
        return int(value)
    return int(default)


__all__ = [
    "DEFAULT_LONG_CONTEXT_CHUNK_CHARS",
    "DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHARS",
    "DEFAULT_LONG_CONTEXT_MAX_EVIDENCE_CHUNKS",
    "DEFAULT_LONG_CONTEXT_MIN_CHARS",
    "DEFAULT_LONG_CONTEXT_OVERLAP_LINES",
    "DEFAULT_LONG_CONTEXT_QUERY_CHARS",
    "LONG_CONTEXT_ROUTER_MODE_CHOICES",
    "LongContextMessageRouteResult",
    "LongContextRouterMode",
    "LongContextRoutingConfig",
    "LongContextTextRouteResult",
    "compact_messages_for_prompt",
    "compact_text_for_prompt",
    "infer_long_context_query",
    "long_context_routing_config_from_args",
    "long_context_routing_config_from_benchmark_config",
    "long_context_routing_config_to_payload",
]
