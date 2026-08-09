from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from .rwkv import (
    build_rwkv_json_call_prompt,
    extract_json_call_value_text,
)

DEFAULT_LONG_DOC_MAX_CHARS = 1000
DEFAULT_LONG_DOC_OVERLAP_LINES = 3
DEFAULT_LONG_DOC_MIN_CHARS = 6000
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS = 4
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS = 6000
LongDocMode = Literal["lexical"]

_LATIN_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_CJK_SPAN_RE = re.compile("[\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]{2,}")


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: int
    text: str
    line_start: int
    line_end: int
    overlap_lines: int = 0

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": int(self.chunk_id),
            "char_count": self.char_count,
            "line_start": int(self.line_start),
            "line_end": int(self.line_end),
            "overlap_lines": int(self.overlap_lines),
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    chunk: TextChunk
    score: float


@dataclass(frozen=True, slots=True)
class LongDocConfig:
    enabled: bool = True
    mode: LongDocMode = "lexical"
    max_chunk_chars: int = DEFAULT_LONG_DOC_MAX_CHARS
    overlap_lines: int = DEFAULT_LONG_DOC_OVERLAP_LINES
    min_long_text_chars: int = DEFAULT_LONG_DOC_MIN_CHARS
    max_evidence_chunks: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS
    max_evidence_chars: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS


@dataclass(frozen=True, slots=True)
class LongDocCompactionResult:
    text: str
    original_chars: int
    chunk_count: int
    selected_chunk_ids: tuple[int, ...]
    compacted: bool
    reason: str = "lexical"
    mode: LongDocMode = "lexical"
    router_error: str | None = None

    def trace_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": self.mode,
            "reason": self.reason,
            "compacted": bool(self.compacted),
            "original_chars": int(self.original_chars),
            "chunk_count": int(self.chunk_count),
            "selected_chunk_ids": list(self.selected_chunk_ids),
        }
        if self.router_error:
            payload["error"] = self.router_error
        return payload


@dataclass(frozen=True, slots=True)
class LongDocMessageCompaction:
    messages: list[dict[str, str]]
    compacted_message_count: int
    selected_chunk_ids: dict[int, tuple[int, ...]]

    def trace_payload(self) -> dict[str, Any]:
        return {
            "compacted_message_count": int(self.compacted_message_count),
            "selected_chunk_ids": {
                int(index): list(chunk_ids) for index, chunk_ids in self.selected_chunk_ids.items()
            },
        }


def normalize_newlines(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_LONG_DOC_MAX_CHARS,
    overlap_lines: int = DEFAULT_LONG_DOC_OVERLAP_LINES,
    split_long_lines: bool = True,
) -> list[TextChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")

    numbered = _numbered_lines(normalize_newlines(text), max_chars=max_chars, split_long_lines=split_long_lines)
    base = _base_chunks(numbered, max_chars=max_chars)
    chunks: list[TextChunk] = []
    for chunk_id, (line_start, line_end, chunk_body) in enumerate(base):
        emitted_text = chunk_body
        effective_start = line_start
        effective_overlap = 0
        if chunk_id > 0 and overlap_lines:
            prev_start, prev_end, prev_text = base[chunk_id - 1]
            tail = prev_text.splitlines(keepends=True)[-overlap_lines:]
            effective_start = max(prev_start, prev_end - len(tail) + 1)
            emitted_text = "".join(tail) + chunk_body
            effective_overlap = len(tail)
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=emitted_text,
                line_start=effective_start,
                line_end=line_end,
                overlap_lines=effective_overlap,
            )
        )
    return chunks


def select_evidence_chunks(
    chunks: Sequence[TextChunk],
    query: str,
    *,
    max_chunks: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS,
    max_chars: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS,
) -> list[EvidenceChunk]:
    if not chunks or max_chunks <= 0 or max_chars <= 0:
        return []
    query_text = normalize_newlines(query).strip()
    terms = _query_terms(query_text)
    scored = [EvidenceChunk(chunk=chunk, score=_chunk_score(chunk.text, terms, query_text)) for chunk in chunks]
    if terms or query_text:
        scored = [item for item in scored if item.score > 0.0]
        if not scored:
            return []
    return _take_evidence_chunks(scored, max_chunks=max_chunks, max_chars=max_chars)


def build_long_doc_router_prompt(*, chunk: TextChunk, query: str) -> str:
    system_prompt = normalize_newlines(
        "\n".join(
            [
                "You decide whether one document chunk contains evidence needed for the next agent step.",
                "Prefer recall over precision. Mark relevant when the chunk may help choose a tool, fill arguments, or follow policy.",
                "Return exactly one JSON object with this shape:",
                '{"relevant":true,"score":3}',
                "Use score 0 for irrelevant, 1 for weakly relevant, 2 for useful, 3 for critical.",
                "Do not include reason, explanation, markdown, or any other fields.",
            ]
        )
    )
    user_text = normalize_newlines(
        "\n".join(
            [
                "Current task/context:",
                str(query or "").strip()[-1600:],
                "",
                f"Chunk {chunk.chunk_id} lines {chunk.line_start}-{chunk.line_end}:",
                chunk.text.strip(),
            ]
        )
    )
    return build_rwkv_json_call_prompt(system_prompt, [{"role": "user", "content": user_text}], history_max_chars=4096)


def parse_long_doc_router_response(text: str) -> tuple[bool, float]:
    raw_text = str(text or "")
    try:
        candidate = extract_json_call_value_text(raw_text)
        payload = json.loads(candidate)
    except (json.JSONDecodeError, ValueError) as exc:
        payload = _recover_partial_long_doc_router_payload(raw_text)
        if payload is None:
            raise exc
    if not isinstance(payload, Mapping):
        raise ValueError("long-doc router response must be a JSON object")
    relevant_raw = payload.get("relevant", payload.get("is_relevant", payload.get("selected", False)))
    if isinstance(relevant_raw, str):
        relevant = relevant_raw.strip().lower() in {"true", "yes", "y", "1", "relevant", "selected"}
    else:
        relevant = bool(relevant_raw)
    try:
        score = float(payload.get("score", 1.0 if relevant else 0.0))
    except (TypeError, ValueError):
        score = 1.0 if relevant else 0.0
    return relevant, max(0.0, min(score, 3.0))


def compact_text(
    text: str,
    *,
    query: str,
    config: LongDocConfig | None = None,
    label: str = "document",
    backend: Any | None = None,
    sampling: Any | None = None,
    progress_desc: str = "LongDocEvidence",
    prompt_seed: int | None = None,
) -> LongDocCompactionResult:
    cfg = config or LongDocConfig()
    if str(cfg.mode) != "lexical":
        raise ValueError("unsupported long-doc mode {!r}; expected lexical".format(cfg.mode))
    normalized = normalize_newlines(text)
    if not cfg.enabled:
        return LongDocCompactionResult(
            text=normalized,
            original_chars=len(normalized),
            chunk_count=0,
            selected_chunk_ids=(),
            compacted=False,
            reason="disabled",
            mode=cfg.mode,
        )
    if len(normalized) < max(1, int(cfg.min_long_text_chars)):
        return LongDocCompactionResult(
            text=normalized,
            original_chars=len(normalized),
            chunk_count=0,
            selected_chunk_ids=(),
            compacted=False,
            reason="below_min_chars",
            mode=cfg.mode,
        )
    chunks = chunk_text(
        normalized,
        max_chars=max(1, int(cfg.max_chunk_chars)),
        overlap_lines=max(0, int(cfg.overlap_lines)),
    )
    selected = select_evidence_chunks(
        chunks,
        query,
        max_chunks=max(1, int(cfg.max_evidence_chunks)),
        max_chars=max(1, int(cfg.max_evidence_chars)),
    )
    compacted = render_evidence_window(
        selected,
        label=label,
        original_chars=len(normalized),
        chunk_count=len(chunks),
        mode=cfg.mode,
        reason="lexical",
        error=None,
    )
    return LongDocCompactionResult(
        text=compacted,
        original_chars=len(normalized),
        chunk_count=len(chunks),
        selected_chunk_ids=tuple(item.chunk.chunk_id for item in selected),
        compacted=True,
        reason="lexical",
        mode=cfg.mode,
        router_error=None,
    )


def compact_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    query: str | None = None,
    config: LongDocConfig | None = None,
    backend: Any | None = None,
    sampling: Any | None = None,
    progress_desc: str = "LongDocEvidence",
    prompt_seed: int | None = None,
) -> LongDocMessageCompaction:
    cfg = config or LongDocConfig()
    if str(cfg.mode) != "lexical":
        raise ValueError("unsupported long-doc mode {!r}; expected lexical".format(cfg.mode))
    normalized = [
        {
            "role": str(message.get("role") or "user").strip().lower() or "user",
            "content": str(message.get("content") or ""),
        }
        for message in messages
        if str(message.get("content") or "")
    ]
    resolved_query = (
        query
        if query is not None
        else infer_query_from_messages(normalized, skip_longer_than=max(1, int(cfg.min_long_text_chars)))
    )
    compacted_messages: list[dict[str, str]] = []
    selected_by_message: dict[int, tuple[int, ...]] = {}
    compacted_count = 0
    for index, message in enumerate(normalized):
        result = compact_text(
            message["content"],
            query=resolved_query,
            config=cfg,
            label=f"message {index} role={message['role']}",
            backend=backend,
            sampling=sampling,
            progress_desc=progress_desc,
            prompt_seed=None if prompt_seed is None else int(prompt_seed) + index * 10_000,
        )
        if result.compacted:
            compacted_count += 1
            selected_by_message[index] = result.selected_chunk_ids
        compacted_messages.append({"role": message["role"], "content": result.text})
    return LongDocMessageCompaction(
        messages=compacted_messages,
        compacted_message_count=compacted_count,
        selected_chunk_ids=selected_by_message,
    )


def infer_query_from_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    max_chars: int = 1200,
    skip_longer_than: int | None = None,
) -> str:
    max_query_chars = max(1, int(max_chars))
    length_cap = None if skip_longer_than is None else max(1, int(skip_longer_than))
    for message in reversed(messages):
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = str(message.get("content") or "").strip()
        if length_cap is not None and len(content) >= length_cap:
            continue
        if content:
            return content[-max_query_chars:]
    for message in reversed(messages):
        content = str(message.get("content") or "").strip()
        if content:
            return content[-max_query_chars:]
    return ""


def render_evidence_window(
    selected: Sequence[EvidenceChunk],
    *,
    label: str,
    original_chars: int,
    chunk_count: int,
    mode: str = "lexical",
    reason: str = "lexical",
    error: str | None = None,
) -> str:
    header = (
        f"[Long document compacted: label={label}; original_chars={int(original_chars)}; "
        f"chunks={int(chunk_count)}; selected_chunks={len(selected)}; mode={mode}; reason={reason}]"
    )
    if error:
        header = header + f"\n[Long document router note: {error}]"
    if not selected:
        return header + "\n[No evidence chunk selected.]"
    parts = [header]
    for item in selected:
        chunk = item.chunk
        parts.append(
            f"[chunk {chunk.chunk_id} lines {chunk.line_start}-{chunk.line_end} score={item.score:.3f}]\n"
            f"{chunk.text.strip()}"
        )
    return "\n\n".join(parts)


def _recover_partial_long_doc_router_payload(text: str) -> dict[str, Any] | None:
    normalized = normalize_newlines(str(text or ""))
    relevant_match = re.search(
        r'"(?:relevant|is_relevant|selected)"\s*:\s*("?(?:true|false|yes|no|1|0|relevant|selected)"?)',
        normalized,
        flags=re.IGNORECASE,
    )
    score_match = re.search(r'"score"\s*:\s*"?(-?\d+(?:\.\d+)?)"?', normalized, flags=re.IGNORECASE)
    if relevant_match is None or score_match is None:
        return None
    relevant_token = relevant_match.group(1).strip().strip('"').lower()
    relevant = relevant_token in {"true", "yes", "1", "relevant", "selected"}
    try:
        score = float(score_match.group(1))
    except ValueError:
        return None
    return {"relevant": relevant, "score": score}


def _to_text_chunk(chunk: Any) -> TextChunk:
    if isinstance(chunk, TextChunk):
        return chunk
    if isinstance(chunk, Mapping):
        return TextChunk(
            chunk_id=int(chunk.get("chunk_id", 0)),
            text=str(chunk.get("text", "")),
            line_start=int(chunk.get("line_start", 0)),
            line_end=int(chunk.get("line_end", 0)),
            overlap_lines=int(chunk.get("overlap_lines", 0)),
        )
    return TextChunk(
        chunk_id=int(getattr(chunk, "chunk_id")),
        text=str(getattr(chunk, "text")),
        line_start=int(getattr(chunk, "line_start")),
        line_end=int(getattr(chunk, "line_end")),
        overlap_lines=int(getattr(chunk, "overlap_lines", 0)),
    )


def _numbered_lines(text: str, *, max_chars: int, split_long_lines: bool) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(keepends=True), start=1):
        if len(line) <= max_chars:
            lines.append((line_no, line))
            continue
        if not split_long_lines:
            raise ValueError(f"line {line_no} has {len(line)} chars > max_chars={max_chars}")
        for start in range(0, len(line), max_chars):
            lines.append((line_no, line[start : start + max_chars]))
    return lines


def _base_chunks(lines: Sequence[tuple[int, str]], *, max_chars: int) -> list[tuple[int, int, str]]:
    chunks: list[tuple[int, int, str]] = []
    current: list[tuple[int, str]] = []
    current_len = 0
    for line_no, line in lines:
        line_len = len(line)
        if line_len > max_chars:
            raise ValueError(f"line {line_no} has {line_len} chars > max_chars={max_chars}")
        if current and current_len + line_len > max_chars:
            chunks.append((current[0][0], current[-1][0], "".join(item[1] for item in current)))
            current = []
            current_len = 0
        current.append((line_no, line))
        current_len += line_len
    if current:
        chunks.append((current[0][0], current[-1][0], "".join(item[1] for item in current)))
    return chunks


def _take_evidence_chunks(
    scored: Sequence[EvidenceChunk],
    *,
    max_chunks: int,
    max_chars: int,
) -> list[EvidenceChunk]:
    ranked = list(scored)
    ranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    selected: list[EvidenceChunk] = []
    used_chars = 0
    for item in ranked:
        if len(selected) >= max_chunks:
            break
        chunk_len = item.chunk.char_count
        if selected and used_chars + chunk_len > max_chars:
            continue
        selected.append(item)
        used_chars += chunk_len
        if used_chars >= max_chars:
            break
    selected.sort(key=lambda item: item.chunk.chunk_id)
    return selected


def _query_terms(query: str) -> tuple[str, ...]:
    lowered = str(query or "").lower()
    terms = set(_LATIN_WORD_RE.findall(lowered))
    for span in _CJK_SPAN_RE.findall(str(query or "")):
        if len(span) <= 8:
            terms.add(span)
        for size in (2, 3, 4):
            if len(span) < size:
                continue
            for index in range(0, len(span) - size + 1):
                terms.add(span[index : index + size])
    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


def _chunk_score(text: str, terms: Sequence[str], query: str) -> float:
    lowered = str(text or "").lower()
    score = 0.0
    for term in terms:
        hits = lowered.count(term.lower())
        if hits:
            score += min(hits, 3) * max(1.0, len(term) / 2.0)
    query_text = normalize_newlines(query).strip().lower()
    if query_text and len(query_text) <= 200 and query_text in lowered:
        score += 100.0
    return score


__all__ = [
    "DEFAULT_LONG_DOC_MAX_CHARS",
    "DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS",
    "DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS",
    "DEFAULT_LONG_DOC_MIN_CHARS",
    "DEFAULT_LONG_DOC_OVERLAP_LINES",
    "EvidenceChunk",
    "LongDocConfig",
    "LongDocCompactionResult",
    "LongDocMode",
    "LongDocMessageCompaction",
    "TextChunk",
    "build_long_doc_router_prompt",
    "chunk_text",
    "compact_messages",
    "compact_text",
    "infer_query_from_messages",
    "normalize_newlines",
    "parse_long_doc_router_response",
    "render_evidence_window",
    "select_evidence_chunks",
]
