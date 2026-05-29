"""Long-document chunking and evidence-window helpers.

The first implementation intentionally stays lexical and deterministic. It is
designed to keep oversized document/tool messages out of an 8k context window
without introducing embedding or vector-search infrastructure.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ALLOWED_ANSWER_FORMATS = frozenset({"scalar_string", "scalar_number_string"})
LongDocEvidenceMode = Literal["lexical", "model_parallel"]
LONG_DOC_MODE_CHOICES: tuple[str, ...] = ("off", "lexical", "model_parallel")
DEFAULT_LONG_DOC_MAX_CHARS = 1000
DEFAULT_LONG_DOC_OVERLAP_LINES = 3
DEFAULT_LONG_DOC_MIN_CHARS = 6000
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS = 4
DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS = 6000
DEFAULT_LONG_DOC_MODEL_MAX_TOKENS = 96
DEFAULT_LONG_DOC_MODEL_PARALLEL_BATCH_SIZE = 8

_LATIN_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_CJK_SPAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]{2,}")


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
    model_max_tokens: int = DEFAULT_LONG_DOC_MODEL_MAX_TOKENS
    model_parallel_batch_size: int = DEFAULT_LONG_DOC_MODEL_PARALLEL_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class LongDocCompactionResult:
    text: str
    original_chars: int
    chunk_count: int
    selected_chunk_ids: tuple[int, ...]
    compacted: bool


@dataclass(frozen=True, slots=True)
class LongDocMessageCompaction:
    messages: list[dict[str, str]]
    compacted_message_count: int
    selected_chunk_ids: dict[int, tuple[int, ...]]


def normalize_newlines(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n")


def chunk_text_by_newline(
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

    numbered_lines = _numbered_lines(normalize_newlines(text), max_chars=max_chars, split_long_lines=split_long_lines)
    base = _base_chunks(numbered_lines, max_chars=max_chars)
    chunks: list[TextChunk] = []
    for chunk_id, (line_start, line_end, chunk_text) in enumerate(base):
        emitted_text = chunk_text
        effective_start = line_start
        effective_overlap = 0
        if chunk_id > 0 and overlap_lines:
            prev_start, prev_end, prev_text = base[chunk_id - 1]
            tail = prev_text.splitlines(keepends=True)[-overlap_lines:]
            effective_start = max(prev_start, prev_end - len(tail) + 1)
            emitted_text = "".join(tail) + chunk_text
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
    if not chunks or max_chunks <= 0 or max_chars <= 0:
        return []
    terms = _query_terms(query)
    scored = [
        SelectedEvidenceChunk(chunk=chunk, score=_chunk_score(chunk.text, terms, query))
        for chunk in chunks
    ]
    if terms or normalize_newlines(query).strip():
        positive_scored = [item for item in scored if item.score > 0.0]
        if not positive_scored:
            return []
        scored = positive_scored
    return _take_evidence_chunks(scored, max_chunks=max_chunks, max_chars=max_chars)


def select_relevant_chunks_model_parallel(
    chunks: Sequence[TextChunk],
    query: str,
    *,
    engine: Any | None,
    sampling: Any | None,
    max_chunks: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS,
    max_chars: int = DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS,
    max_tokens: int = DEFAULT_LONG_DOC_MODEL_MAX_TOKENS,
    batch_size: int = DEFAULT_LONG_DOC_MODEL_PARALLEL_BATCH_SIZE,
    progress_desc: str = "LongDocEvidence",
    prompt_seed: int | None = None,
) -> tuple[list[SelectedEvidenceChunk], str, str | None]:
    from src.eval.function_calling.rwkv_prompt import JSON_CALL_STOP_SUFFIXES

    if not chunks or max_chunks <= 0 or max_chars <= 0:
        return [], "model_parallel_empty", None
    if engine is None or sampling is None:
        fallback = select_relevant_chunks(chunks, query, max_chunks=max_chunks, max_chars=max_chars)
        return fallback, "model_parallel_missing_engine_lexical_fallback", "missing engine/sampling"

    prompts = [build_long_doc_evidence_router_prompt(chunk=chunk, query=query) for chunk in chunks]
    try:
        outputs = engine.generate(
            prompts,
            sampling=_long_doc_router_sampling(sampling, max_tokens=max_tokens),
            batch_size=min(len(prompts), max(1, int(batch_size))),
            progress_desc=progress_desc,
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
            prompt_seeds=None if prompt_seed is None else [int(prompt_seed) + index for index in range(len(prompts))],
            show_progress=False,
        )
    except Exception as exc:  # noqa: BLE001 - compaction falls back to the deterministic selector.
        fallback = select_relevant_chunks(chunks, query, max_chunks=max_chunks, max_chars=max_chars)
        return fallback, "model_parallel_error_lexical_fallback", str(exc)

    scored: list[SelectedEvidenceChunk] = []
    parse_errors: list[str] = []
    for chunk, output in zip(chunks, outputs, strict=False):
        raw_text = str(getattr(output, "text", "") or "")
        try:
            relevant, score = parse_long_doc_evidence_router_response(raw_text)
        except Exception as exc:  # noqa: BLE001 - one bad shard should not discard the whole message.
            relevant = False
            score = 0.0
            parse_errors.append(f"chunk {chunk.chunk_id}: {exc}")
        if relevant or score > 0.0:
            scored.append(SelectedEvidenceChunk(chunk=chunk, score=max(float(score), 1.0 if relevant else 0.0)))

    if scored:
        return _take_evidence_chunks(scored, max_chunks=max_chunks, max_chars=max_chars), "model_parallel", (
            "; ".join(parse_errors) if parse_errors else None
        )

    fallback = select_relevant_chunks(chunks, query, max_chunks=max_chunks, max_chars=max_chars)
    reason = "model_parallel_empty_lexical_fallback" if fallback else "model_parallel_empty"
    return fallback, reason, "; ".join(parse_errors) if parse_errors else None


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
    from src.eval.function_calling.rwkv_prompt import build_rwkv_json_call_prompt

    system_prompt = normalize_newlines(
        "\n".join(
            [
                "You decide whether one document chunk contains evidence needed for the next agent step.",
                "Prefer recall over precision. Mark relevant when the chunk may help choose a tool, fill arguments, or follow policy.",
                "Return exactly one JSON object with this shape:",
                '{"relevant":true,"score":3,"reason":"short"}',
                "Use score 0 for irrelevant, 1 for weakly relevant, 2 for useful, 3 for critical.",
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


def parse_long_doc_evidence_router_response(text: str) -> tuple[bool, float]:
    from src.eval.function_calling.rwkv_prompt import extract_json_call_value_text

    candidate = extract_json_call_value_text(str(text or ""))
    payload = json.loads(candidate)
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


def _long_doc_router_sampling(sampling: Any, *, max_tokens: int) -> Any:
    clamp = getattr(sampling, "clamp", None)
    if callable(clamp):
        try:
            return clamp(max(1, int(max_tokens)))
        except Exception:
            return sampling
    return sampling


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
    chunks = chunk_text_by_newline(
        normalized,
        max_chars=max(1, int(cfg.max_chunk_chars)),
        overlap_lines=max(0, int(cfg.overlap_lines)),
    )
    reason = "lexical"
    error: str | None = None
    if cfg.mode == "model_parallel":
        selected, reason, error = select_relevant_chunks_model_parallel(
            chunks,
            query,
            engine=engine,
            sampling=sampling,
            max_chunks=max(1, int(cfg.max_evidence_chunks)),
            max_chars=max(1, int(cfg.max_evidence_chars)),
            max_tokens=max(1, int(cfg.model_max_tokens)),
            batch_size=max(1, int(cfg.model_parallel_batch_size)),
            progress_desc=progress_desc,
            prompt_seed=prompt_seed,
        )
    else:
        selected = select_relevant_chunks(
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
        reason=reason,
        error=error,
    )
    return LongDocCompactionResult(
        text=compacted,
        original_chars=len(normalized),
        chunk_count=len(chunks),
        selected_chunk_ids=tuple(item.chunk.chunk_id for item in selected),
        compacted=True,
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
        content = message["content"]
        result = compact_long_text(
            content,
            query=resolved_query,
            config=cfg,
            label=f"message {index} role={message['role']}",
            engine=engine,
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
    if error:
        header += f"\n[Long document router note: {str(error)[:500]}]"
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
    mode = _env_choice(f"{prefix}_MODE", "lexical", ("lexical", "model_parallel"))
    return LongDocEvidenceConfig(
        enabled=_env_bool(f"{prefix}_ENABLED", True),
        mode=mode,  # type: ignore[arg-type]
        max_chunk_chars=_env_int(f"{prefix}_MAX_CHARS", DEFAULT_LONG_DOC_MAX_CHARS),
        overlap_lines=_env_int(f"{prefix}_OVERLAP_LINES", DEFAULT_LONG_DOC_OVERLAP_LINES),
        min_long_text_chars=_env_int(f"{prefix}_MIN_CHARS", DEFAULT_LONG_DOC_MIN_CHARS),
        max_evidence_chunks=_env_int(f"{prefix}_MAX_EVIDENCE_CHUNKS", DEFAULT_LONG_DOC_MAX_EVIDENCE_CHUNKS),
        max_evidence_chars=_env_int(f"{prefix}_MAX_EVIDENCE_CHARS", DEFAULT_LONG_DOC_MAX_EVIDENCE_CHARS),
        model_max_tokens=_env_int(f"{prefix}_MODEL_MAX_TOKENS", DEFAULT_LONG_DOC_MODEL_MAX_TOKENS),
        model_parallel_batch_size=_env_int(
            f"{prefix}_MODEL_PARALLEL_BATCH_SIZE",
            DEFAULT_LONG_DOC_MODEL_PARALLEL_BATCH_SIZE,
        ),
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


def _coerce_rule_terms(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"positive_rule terms must be a list: {value!r}")
    return tuple(str(item) for item in value)


def _chunk_payload(chunk: TextChunk | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(chunk, TextChunk):
        return chunk.to_json()
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
    "select_relevant_chunks_model_parallel",
    "summarize_chunks",
    "validate_task_definition",
    "write_jsonl",
]
