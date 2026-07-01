from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

JSON_CALL_STOP_SUFFIXES = (
    "\n```",
    "```",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
)

USER_HEADER = "User:"
ASSISTANT_HEADER = "Assistant:"
SYSTEM_HEADER = "System:"
_TRUNCATION_NOTICE = "[Earlier conversation history truncated]"


@dataclass(frozen=True, slots=True)
class RouterGeneration:
    text: str
    finish_reason: str | None = None


class RwkvRouterBackend(Protocol):
    def generate(
        self,
        prompts: Sequence[str],
        *,
        sampling: Any | None = None,
        batch_size: int = 1,
        progress_desc: str = "RwkvRouter",
        prompt_stop_suffixes: Sequence[Sequence[str]] | None = None,
        prompt_seeds: Sequence[int] | None = None,
        show_progress: bool = False,
    ) -> Sequence[Any]:
        ...


def normalize_rwkv_text(text: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = normalized.strip()
    return re.sub(r"\n{2,}", "\n", normalized)


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


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


def assistant_json_prefix() -> str:
    return "Assistant: <think>\n</think>\n```json\n"


def render_system_block(content: str) -> str:
    return f"{SYSTEM_HEADER} {normalize_rwkv_text(content)}".rstrip(" ")


def render_user_block(content: str) -> str:
    rendered = _strip_role_prefix(content, USER_HEADER)
    return f"{USER_HEADER} {rendered}".rstrip(" ")


def render_assistant_text_block(content: str) -> str:
    rendered = _strip_role_prefix(content, ASSISTANT_HEADER)
    return f"{ASSISTANT_HEADER} {rendered}".rstrip(" ")


def render_assistant_json_block(json_text: str) -> str:
    return f"{assistant_json_prefix()}{normalize_rwkv_text(json_text)}\n```"


def build_rwkv_json_call_prompt(
    system_prompt: str,
    messages: Sequence[Mapping[str, object]],
    *,
    history_max_chars: int,
) -> str:
    bounded_messages = trim_message_history(messages, max_chars=max(0, int(history_max_chars)))
    parts = [render_system_block(system_prompt)]
    for message in bounded_messages:
        role = str(message.get("role") or "").strip().lower()
        content = normalize_rwkv_text(str(message.get("content") or ""))
        if not content:
            continue
        if role == "assistant":
            assistant_content = _strip_role_prefix(content, ASSISTANT_HEADER)
            if _looks_like_json_call(assistant_content):
                parts.append(render_assistant_json_block(_strip_json_fence(assistant_content)))
            elif _looks_like_json_call(content):
                parts.append(render_assistant_json_block(_strip_json_fence(content)))
            else:
                parts.append(render_assistant_text_block(content))
            continue
        parts.append(render_user_block(content))
    parts.append(assistant_json_prefix())
    return "\n\n".join(parts)


def extract_json_call_value_text(response: str) -> str:
    normalized = _strip_assistant_prefix(normalize_rwkv_text(response))
    normalized = _strip_leading_think_block(normalized)
    normalized = _strip_json_fence(normalized)
    if not (normalized.startswith("{") or normalized.startswith("[")):
        raise ValueError(f"model response must be a JSON function call object or array: {normalized}")
    end = _find_leading_json_value_end(normalized)
    if end is None:
        raise ValueError(f"model response must be a complete JSON function call object or array: {normalized}")
    candidate = normalized[:end]
    trailing = normalize_rwkv_text(normalized[end:])
    trailing = _strip_json_fence(trailing) if trailing.startswith("```") else trailing
    if trailing and trailing != "```":
        raise ValueError(f"model response has extra text after JSON function call object or array: {trailing}")
    return candidate


def extract_json_call_object_text(response: str) -> str:
    candidate = extract_json_call_value_text(response)
    if not candidate.startswith("{"):
        raise ValueError(f"model response must be a JSON function call object: {candidate}")
    return candidate


def clamp_router_sampling(sampling: Any, *, max_tokens: int) -> Any:
    clamp = getattr(sampling, "clamp", None)
    if callable(clamp):
        try:
            sampling = clamp(max(1, int(max_tokens)))
        except Exception:
            return sampling
    try:
        return replace(
            sampling,
            max_generate_tokens=max(1, int(max_tokens)),
            temperature=0.001,
            top_k=1,
            top_p=1.0,
            alpha_presence=0.0,
            alpha_frequency=0.0,
        )
    except Exception:
        return sampling


def json_call_stop_suffixes(count: int) -> list[list[str]]:
    return [list(JSON_CALL_STOP_SUFFIXES) for _ in range(max(0, int(count)))]


def generation_text(output: Any) -> str:
    return str(getattr(output, "text", "") or "")


def parse_json_value(text: str) -> Any:
    return json.loads(extract_json_call_value_text(text))


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


def _looks_like_json_call(text: str) -> bool:
    stripped = normalize_rwkv_text(text)
    return stripped.startswith("{") or stripped.startswith("```")


def _strip_assistant_prefix(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("Assistant:"):
        return normalize_rwkv_text(stripped[len("Assistant:") :])
    return stripped


def _strip_role_prefix(text: str, prefix: str) -> str:
    stripped = normalize_rwkv_text(text)
    if stripped.startswith(prefix):
        return normalize_rwkv_text(stripped[len(prefix) :])
    return stripped


def _strip_leading_think_block(text: str) -> str:
    normalized = normalize_rwkv_text(text)
    if not normalized.startswith("<think>"):
        return normalized
    close = normalized.find("</think>")
    if close < 0:
        return normalized
    return normalize_rwkv_text(normalized[close + len("</think>") :])


def _strip_json_fence(text: str) -> str:
    normalized = normalize_rwkv_text(text)
    if normalized.startswith("```"):
        lines = normalized.split("\n")
        if lines:
            first = lines[0].strip().lower()
            if first in {"```", "```json", "```javascript", "```js"}:
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                normalized = normalize_rwkv_text("\n".join(lines))
    if normalized.endswith("```"):
        normalized = normalize_rwkv_text(normalized[: -len("```")])
    return normalized


def _find_leading_json_value_end(text: str) -> int | None:
    if not text:
        return None
    opening = text[0]
    if opening not in "{[":
        return None
    expected_stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            expected_stack.append("}")
            continue
        if char == "[":
            expected_stack.append("]")
            continue
        if char in "}]":
            if not expected_stack or expected_stack[-1] != char:
                return None
            expected_stack.pop()
            if not expected_stack:
                return index + 1
            continue
    return None


__all__ = [
    "ASSISTANT_HEADER",
    "JSON_CALL_STOP_SUFFIXES",
    "RouterGeneration",
    "RwkvRouterBackend",
    "SYSTEM_HEADER",
    "USER_HEADER",
    "assistant_json_prefix",
    "build_rwkv_json_call_prompt",
    "clamp_router_sampling",
    "extract_json_call_object_text",
    "extract_json_call_value_text",
    "generation_text",
    "json_call_stop_suffixes",
    "normalize_rwkv_text",
    "parse_json_value",
    "render_assistant_json_block",
    "render_assistant_text_block",
    "render_system_block",
    "render_user_block",
    "trim_message_history",
    "truncate_text",
]
