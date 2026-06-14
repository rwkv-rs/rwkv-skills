"""Long-document chunking and evidence-window helpers.

The first implementation intentionally stays lexical and deterministic. It is
designed to keep oversized document/tool messages out of an 8k context window
without introducing embedding or vector-search infrastructure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from lexical_chunk_router import long_doc as lexical_long_doc
from lexical_chunk_router.rwkv import clamp_router_sampling

ALLOWED_ANSWER_FORMATS = frozenset({"scalar_string", "scalar_number_string"})
LongDocEvidenceMode = Literal["lexical"]
LONG_DOC_MODE_CHOICES: tuple[str, ...] = ("off", "lexical")
DEFAULT_LONG_DOC_MAX_CHARS = 1000
DEFAULT_LONG_DOC_OVERLAP_LINES = 3
DEFAULT_LONG_DOC_MIN_CHARS = 6000
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS = 4
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS = 6000


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

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_id": int(self.chunk_id),
            "char_count": self.char_count,
            "line_start": int(self.line_start),
            "line_end": int(self.line_end),
            "overlap_lines": int(self.overlap_lines),
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class SelectedEvidenceChunk:
    chunk: TextChunk
    score: float


@dataclass(frozen=True, slots=True)
class LongDocEvidenceConfig:
    enabled: bool = True
    mode: LongDocEvidenceMode = "lexical"
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
    router_error: str | None = None


@dataclass(frozen=True, slots=True)
class LongDocMessageCompaction:
    messages: list[dict[str, str]]
    compacted_message_count: int
    selected_chunk_ids: dict[int, tuple[int, ...]]


def normalize_newlines(text: str) -> str:
    return lexical_long_doc.normalize_newlines(text)


def chunk_text_by_newline(
    text: str,
    *,
    max_chars: int = DEFAULT_LONG_DOC_MAX_CHARS,
    overlap_lines: int = DEFAULT_LONG_DOC_OVERLAP_LINES,
    split_long_lines: bool = True,
) -> list[TextChunk]:
    return [
        _to_eval_text_chunk(chunk)
        for chunk in lexical_long_doc.chunk_text(
            text,
            max_chars=max_chars,
            overlap_lines=overlap_lines,
            split_long_lines=split_long_lines,
        )
    ]


def summarize_chunks(
    *,
    input_name: str,
    text: str,
    chunks: Sequence[TextChunk],
    max_chars: int,
    overlap_lines: int,
) -> dict[str, Any]:
    normalized = normalize_newlines(text)
    lengths = [chunk.char_count for chunk in chunks]
    line_count = normalized.count("\n") + (0 if not normalized or normalized.endswith("\n") else 1)
    return {
        "input": input_name,
        "max_chars": int(max_chars),
        "overlap_lines": int(overlap_lines),
        "total_chars": len(normalized),
        "line_count": line_count,
        "chunk_count": len(chunks),
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "avg_chunk_chars": (sum(lengths) / len(lengths)) if lengths else 0,
    }


def match_positive_rule(text: str, rule: Mapping[str, Any]) -> bool:
    if not isinstance(rule, Mapping):
        raise ValueError(f"unsupported positive_rule: {rule!r}")
    if "all" in rule:
        terms = _coerce_rule_terms(rule["all"])
        return all(term in text for term in terms)
    if "any" in rule:
        terms = _coerce_rule_terms(rule["any"])
        return any(term in text for term in terms)
    nested = rule.get("not")
    if isinstance(nested, Mapping):
        return not match_positive_rule(text, nested)
    raise ValueError(f"unsupported positive_rule: {rule!r}")


def build_evidence_tasks(
    chunks: Sequence[TextChunk | Mapping[str, Any]],
    task_defs: Sequence[Mapping[str, Any]],
    *,
    chunk_source: str = "",
    allow_empty: bool = False,
    allow_answer_missing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_rows = [_chunk_payload(chunk) for chunk in chunks]
    chunks_by_id = {int(row["chunk_id"]): row for row in chunk_rows}
    output_tasks: list[dict[str, Any]] = []
    empty_positive: list[str] = []
    answer_missing: list[str] = []
    positive_counts: dict[str, int] = {}
    answer_hit_counts: dict[str, int] = {}

    for index, raw_task in enumerate(task_defs, start=1):
        task = dict(raw_task)
        validate_task_definition(task, index=index)
        task_id = str(task["id"])
        rule = task["positive_rule"]
        positive_chunks = [
            int(chunk["chunk_id"])
            for chunk in chunk_rows
            if match_positive_rule(str(chunk.get("text", "")), rule)
        ]
        positive_counts[task_id] = len(positive_chunks)
        if not positive_chunks:
            empty_positive.append(task_id)

        answer = str(task["answer"])
        answer_hits = [
            chunk_id
            for chunk_id in positive_chunks
            if answer in str(chunks_by_id[chunk_id].get("text", ""))
        ]
        answer_hit_counts[task_id] = len(answer_hits)
        if not answer_hits:
            answer_missing.append(task_id)

        output = dict(task)
        output["positive_chunks"] = positive_chunks
        output["null_rule"] = "chunk does not contain the positive_rule terms"
        output["chunking"] = {
            "source": chunk_source,
            "positive_chunks_recomputed_from": "positive_rule",
        }
        output_tasks.append(output)

    if empty_positive and not allow_empty:
        raise ValueError(f"tasks without positive chunks: {', '.join(empty_positive)}")
    if answer_missing and not allow_answer_missing:
        raise ValueError(f"tasks whose answer is missing from positive chunks: {', '.join(answer_missing)}")

    summary = {
        "chunk_count": len(chunk_rows),
        "task_count": len(output_tasks),
        "empty_positive_tasks": empty_positive,
        "answer_missing_from_positive_tasks": answer_missing,
        "positive_chunk_count": _count_summary(positive_counts),
        "answer_hit_count_in_positive_chunks": _count_summary(answer_hit_counts),
        "oracle_passed": not empty_positive and not answer_missing,
    }
    return output_tasks, summary


def validate_task_definition(row: Mapping[str, Any], *, index: int = 0) -> None:
    required = ("id", "question", "answer", "answer_format", "positive_rule")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"task #{index} missing required keys: {missing}")
    if row["answer_format"] not in ALLOWED_ANSWER_FORMATS:
        raise ValueError(f"task {row['id']!r} unsupported answer_format: {row['answer_format']!r}")
    if not isinstance(row["positive_rule"], Mapping):
        raise ValueError(f"task {row['id']!r} positive_rule must be an object")


def build_answer_or_null_prompt(*, chunk_text: str, question: str) -> str:
    return (
        'User: Answer the question using only the material. If the material explicitly supports the answer, '
        'return JSON {"answer":"..."}. Otherwise return JSON {"answer":"null"}.\n'
        f"Material:\n{normalize_newlines(chunk_text).strip()}\n"
        f"Question: {str(question).strip()}\n"
        "Assistant:"
    )


def parse_answer_or_null_response(text: str) -> tuple[str, bool]:
    raw = str(text or "").strip()
    if not raw:
        return "null", False
    if raw.startswith("```"):
        raw = _strip_json_fence(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "null", False
    if not isinstance(payload, Mapping) or "answer" not in payload:
        return "null", False
    answer = str(payload.get("answer") or "null").strip()
    return answer or "null", True


def select_relevant_chunks(
    chunks: Sequence[TextChunk],
    query: str,
    *,
    max_chunks: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS,
    max_chars: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS,
) -> list[SelectedEvidenceChunk]:
    selected = lexical_long_doc.select_evidence_chunks(
        chunks,
        query,
        max_chunks=max_chunks,
        max_chars=max_chars,
    )
    return [
        SelectedEvidenceChunk(chunk=_to_eval_text_chunk(item.chunk), score=item.score)
        for item in selected
    ]


def _take_evidence_chunks(
    scored: Sequence[SelectedEvidenceChunk],
    *,
    max_chunks: int,
    max_chars: int,
) -> list[SelectedEvidenceChunk]:
    ranked = list(scored)
    ranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    selected: list[SelectedEvidenceChunk] = []
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


def build_long_doc_evidence_router_prompt(*, chunk: TextChunk, query: str) -> str:
    return lexical_long_doc.build_long_doc_router_prompt(chunk=_to_plugin_text_chunk(chunk), query=query)


def parse_long_doc_evidence_router_response(text: str) -> tuple[bool, float]:
    return lexical_long_doc.parse_long_doc_router_response(text)


def _long_doc_router_sampling(sampling: Any, *, max_tokens: int) -> Any:
    return clamp_router_sampling(sampling, max_tokens=max_tokens)


def compact_long_text(
    text: str,
    *,
    query: str,
    config: LongDocEvidenceConfig | None = None,
    label: str = "document",
    engine: Any | None = None,
    sampling: Any | None = None,
    progress_desc: str = "LongDocEvidence",
    prompt_seed: int | None = None,
) -> LongDocCompactionResult:
    cfg = config or LongDocEvidenceConfig()
    if str(cfg.mode) not in LONG_DOC_MODE_CHOICES:
        raise ValueError(f"unsupported long-doc mode {cfg.mode!r}; expected one of {', '.join(LONG_DOC_MODE_CHOICES)}")
    normalized = normalize_newlines(text)
    if not cfg.enabled:
        return LongDocCompactionResult(
            text=normalized,
            original_chars=len(normalized),
            chunk_count=0,
            selected_chunk_ids=(),
            compacted=False,
        )
    if len(normalized) < max(1, int(cfg.min_long_text_chars)):
        return LongDocCompactionResult(
            text=normalized,
            original_chars=len(normalized),
            chunk_count=0,
            selected_chunk_ids=(),
            compacted=False,
        )
    result = lexical_long_doc.compact_text(
        normalized,
        query=query,
        config=_lexical_long_doc_config(cfg),
        label=label,
    )
    return LongDocCompactionResult(
        text=result.text,
        original_chars=result.original_chars,
        chunk_count=result.chunk_count,
        selected_chunk_ids=result.selected_chunk_ids,
        compacted=result.compacted,
    )


def compact_messages_for_long_context(
    messages: Sequence[Mapping[str, object]],
    *,
    query: str | None = None,
    config: LongDocEvidenceConfig | None = None,
    engine: Any | None = None,
    sampling: Any | None = None,
    progress_desc: str = "LongDocEvidence",
    prompt_seed: int | None = None,
) -> LongDocMessageCompaction:
    cfg = config or LongDocEvidenceConfig()
    if str(cfg.mode) not in LONG_DOC_MODE_CHOICES:
        raise ValueError(f"unsupported long-doc mode {cfg.mode!r}; expected one of {', '.join(LONG_DOC_MODE_CHOICES)}")
    result = lexical_long_doc.compact_messages(
        messages,
        query=query,
        config=_lexical_long_doc_config(cfg),
    )
    return LongDocMessageCompaction(
        messages=result.messages,
        compacted_message_count=result.compacted_message_count,
        selected_chunk_ids=result.selected_chunk_ids,
    )


def infer_query_from_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    max_chars: int = 1200,
    skip_longer_than: int | None = None,
) -> str:
    return lexical_long_doc.infer_query_from_messages(
        messages,
        max_chars=max_chars,
        skip_longer_than=skip_longer_than,
    )


def render_evidence_window(
    selected: Sequence[SelectedEvidenceChunk],
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


def long_doc_config_from_env(prefix: str = "RWKV_LONG_DOC") -> LongDocEvidenceConfig:
    mode = _env_choice(f"{prefix}_MODE", "lexical", ("lexical",))
    return LongDocEvidenceConfig(
        enabled=_env_bool(f"{prefix}_ENABLED", True),
        mode=mode,  # type: ignore[arg-type]
        max_chunk_chars=_env_int(f"{prefix}_MAX_CHARS", DEFAULT_LONG_DOC_MAX_CHARS),
        overlap_lines=_env_int(f"{prefix}_OVERLAP_LINES", DEFAULT_LONG_DOC_OVERLAP_LINES),
        min_long_text_chars=_env_int(f"{prefix}_MIN_CHARS", DEFAULT_LONG_DOC_MIN_CHARS),
        max_evidence_chunks=_env_int(f"{prefix}_MAX_EVIDENCE_CHUNKS", DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS),
        max_evidence_chars=_env_int(f"{prefix}_MAX_EVIDENCE_CHARS", DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS),
    )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _coerce_rule_terms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"positive_rule terms must be a list: {value!r}")
    return tuple(str(item) for item in value)


def _lexical_long_doc_config(config: LongDocEvidenceConfig) -> lexical_long_doc.LongDocConfig:
    return lexical_long_doc.LongDocConfig(
        enabled=bool(config.enabled),
        max_chunk_chars=max(1, int(config.max_chunk_chars)),
        overlap_lines=max(0, int(config.overlap_lines)),
        min_long_text_chars=max(1, int(config.min_long_text_chars)),
        max_evidence_chunks=max(1, int(config.max_evidence_chunks)),
        max_evidence_chars=max(1, int(config.max_evidence_chars)),
    )


def _to_eval_text_chunk(chunk: Any) -> TextChunk:
    if isinstance(chunk, TextChunk):
        return chunk
    return TextChunk(
        chunk_id=int(getattr(chunk, "chunk_id")),
        text=str(getattr(chunk, "text")),
        line_start=int(getattr(chunk, "line_start")),
        line_end=int(getattr(chunk, "line_end")),
        overlap_lines=int(getattr(chunk, "overlap_lines", 0)),
    )


def _to_plugin_text_chunk(chunk: Any) -> lexical_long_doc.TextChunk:
    if isinstance(chunk, lexical_long_doc.TextChunk):
        return chunk
    if isinstance(chunk, TextChunk):
        return lexical_long_doc.TextChunk(
            chunk_id=int(chunk.chunk_id),
            text=chunk.text,
            line_start=int(chunk.line_start),
            line_end=int(chunk.line_end),
            overlap_lines=int(chunk.overlap_lines),
        )
    if isinstance(chunk, Mapping):
        return lexical_long_doc.TextChunk(
            chunk_id=int(chunk.get("chunk_id", 0)),
            text=str(chunk.get("text", "")),
            line_start=int(chunk.get("line_start", 0)),
            line_end=int(chunk.get("line_end", 0)),
            overlap_lines=int(chunk.get("overlap_lines", 0)),
        )
    return lexical_long_doc.TextChunk(
        chunk_id=int(getattr(chunk, "chunk_id")),
        text=str(getattr(chunk, "text")),
        line_start=int(getattr(chunk, "line_start")),
        line_end=int(getattr(chunk, "line_end")),
        overlap_lines=int(getattr(chunk, "overlap_lines", 0)),
    )


def _chunk_payload(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, TextChunk):
        return chunk.to_json()
    to_dict = getattr(chunk, "to_dict", None)
    if callable(to_dict):
        payload = dict(to_dict())
        payload.setdefault("char_count", len(str(payload.get("text", ""))))
        return payload
    if all(hasattr(chunk, field) for field in ("chunk_id", "text", "line_start", "line_end")):
        return _to_eval_text_chunk(chunk).to_json()
    payload = dict(chunk)
    payload.setdefault("char_count", len(str(payload.get("text", ""))))
    return payload


def _count_summary(values: Mapping[str, int]) -> dict[str, Any]:
    counts = list(values.values())
    return {
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "avg": (sum(counts) / len(counts)) if counts else 0,
        "by_task": dict(values),
    }


def _strip_json_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json", "```js", "```javascript"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_choice(name: str, default: str, choices: Sequence[str]) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return str(default)
    normalized = raw.strip().lower()
    return normalized if normalized in set(choices) else str(default)


__all__ = [
    "ALLOWED_ANSWER_FORMATS",
    "LONG_DOC_MODE_CHOICES",
    "LongDocCompactionResult",
    "LongDocEvidenceConfig",
    "LongDocEvidenceMode",
    "LongDocMessageCompaction",
    "SelectedEvidenceChunk",
    "TextChunk",
    "build_long_doc_evidence_router_prompt",
    "build_answer_or_null_prompt",
    "build_evidence_tasks",
    "chunk_text_by_newline",
    "compact_long_text",
    "compact_messages_for_long_context",
    "infer_query_from_messages",
    "load_jsonl",
    "long_doc_config_from_env",
    "match_positive_rule",
    "normalize_newlines",
    "parse_long_doc_evidence_router_response",
    "parse_answer_or_null_response",
    "render_evidence_window",
    "select_relevant_chunks",
    "summarize_chunks",
    "validate_task_definition",
    "write_jsonl",
]
