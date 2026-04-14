from __future__ import annotations

"""Shared prompt/history budgeting helpers for function-calling benchmarks."""

from collections.abc import Mapping, Sequence

DEFAULT_TOOL_SCHEMA_MAX_CHARS = 1200
DEFAULT_TOOL_RESULT_MAX_CHARS = 4000
DEFAULT_TOOL_ERROR_MAX_CHARS = 1000
DEFAULT_HISTORY_MAX_CHARS = 24000
_TRUNCATION_NOTICE = "[Earlier conversation history truncated]"


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def trim_history(history: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(history) <= max_chars:
        return history
    keep_tail = max_chars - len(_TRUNCATION_NOTICE) - 2
    if keep_tail <= 0:
        return history[-max_chars:]
    return f"{_TRUNCATION_NOTICE}\n\n{history[-keep_tail:]}"


def trim_message_history(
    messages: Sequence[Mapping[str, object]],
    *,
    max_chars: int,
    notice: str = _TRUNCATION_NOTICE,
) -> list[dict[str, str]]:
    normalized = [
        {
            "role": str(message.get("role") or "user").strip().lower() or "user",
            "content": str(message.get("content") or ""),
        }
        for message in messages
        if str(message.get("content") or "")
    ]
    if max_chars <= 0 or not normalized:
        return []

    total = 0
    kept_reversed: list[dict[str, str]] = []
    cut_index = -1
    for index in range(len(normalized) - 1, -1, -1):
        message = normalized[index]
        rendered_len = _rendered_message_len(message)
        if total + rendered_len > max_chars:
            cut_index = index
            break
        kept_reversed.append(message)
        total += rendered_len

    kept = list(reversed(kept_reversed))
    if cut_index < 0:
        return kept

    available = max_chars - total
    prefix: list[dict[str, str]] = []
    notice_message = {"role": "user", "content": notice}
    notice_len = _rendered_message_len(notice_message)
    truncated = None
    if available > notice_len:
        truncated = _fit_message_tail(normalized[cut_index], available - notice_len)
        if truncated is not None:
            prefix.append(notice_message)
    if truncated is None:
        truncated = _fit_message_tail(normalized[cut_index], available)
        if truncated is None and kept and total + notice_len <= max_chars:
            prefix.append(notice_message)
    if truncated is not None:
        prefix.append(truncated)
    return [*prefix, *kept]


def _rendered_message_len(message: Mapping[str, str]) -> int:
    return len(message["role"]) + len(message["content"]) + 4


def _fit_message_tail(message: Mapping[str, str], budget: int) -> dict[str, str] | None:
    role = str(message.get("role") or "user")
    content = str(message.get("content") or "")
    overhead = len(role) + 4
    if budget <= overhead:
        return None
    content_budget = budget - overhead
    if len(content) <= content_budget:
        return {"role": role, "content": content}
    return {"role": role, "content": content[-content_budget:]}


__all__ = [
    "DEFAULT_HISTORY_MAX_CHARS",
    "DEFAULT_TOOL_ERROR_MAX_CHARS",
    "DEFAULT_TOOL_RESULT_MAX_CHARS",
    "DEFAULT_TOOL_SCHEMA_MAX_CHARS",
    "trim_history",
    "trim_message_history",
    "truncate_text",
]
