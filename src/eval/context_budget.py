from __future__ import annotations

"""Prompt-budget helpers for non-function-calling long-context benchmarks."""

import argparse
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from src.eval.long_doc_evidence import (
    LONG_DOC_MODE_CHOICES,
    LongDocEvidenceConfig,
    compact_long_text,
    normalize_newlines,
)


def middle_truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    notice = "\n[... middle truncated to fit prompt budget ...]\n"
    if max_chars <= len(notice) + 8:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - len(notice)
    return text[:head] + notice + text[-tail:]


def fuse_context_question(context: str, question: str) -> str:
    clean_context = normalize_newlines(context).strip()
    clean_question = normalize_newlines(question).lstrip()
    if not clean_context:
        return clean_question
    return f"Context:\n{clean_context}\n\nQuestion:\n{clean_question}"


def compose_context_question(
    context: str | None,
    question: str,
    *,
    long_doc_config: LongDocEvidenceConfig,
    label: str,
    query: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    if not long_doc_config.enabled:
        return question, None
    prompt, trace = build_budgeted_context_prompt(
        context=context,
        query=query if query is not None else question,
        render=lambda ctx: fuse_context_question(ctx, question) if ctx else question,
        long_doc_config=long_doc_config,
        prompt_max_chars=None,
        label=label,
    )
    return prompt, trace


def build_budgeted_context_prompt(
    *,
    context: str | None,
    query: str,
    render: Callable[[str], str],
    long_doc_config: LongDocEvidenceConfig,
    prompt_max_chars: int | None,
    label: str,
) -> tuple[str, dict[str, Any] | None]:
    normalized_context = normalize_newlines(context or "").strip()
    if not normalized_context or not long_doc_config.enabled:
        return render(""), None

    compaction = compact_long_text(normalized_context, query=query, config=long_doc_config, label=label)
    context_text = compaction.text
    prompt = render(context_text)
    trimmed_context_chars = 0
    max_chars = int(prompt_max_chars or 0)
    if 0 < max_chars < len(prompt):
        context_budget = max(0, max_chars - len(render("")) - 16)
        fitted = middle_truncate_text(context_text, context_budget)
        prompt = render(fitted)
        if len(prompt) > max_chars:
            fitted = middle_truncate_text(fitted, max(0, len(fitted) - (len(prompt) - max_chars) - 32))
            prompt = render(fitted)
        trimmed_context_chars = max(0, len(context_text) - len(fitted))

    trace = {
        "mode": long_doc_config.mode if long_doc_config.enabled else "off",
        "enabled": bool(long_doc_config.enabled),
        "original_context_chars": int(compaction.original_chars),
        "rendered_context_chars": max(0, len(context_text) - trimmed_context_chars),
        "trimmed_context_chars": trimmed_context_chars,
        "compacted": bool(compaction.compacted),
        "chunk_count": int(compaction.chunk_count),
        "selected_chunk_ids": list(compaction.selected_chunk_ids),
        "prompt_chars": len(prompt),
        "prompt_max_chars": max_chars,
    }
    if compaction.router_error:
        trace["router_error"] = compaction.router_error
    return prompt, trace


def add_long_doc_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--long-doc-mode", choices=LONG_DOC_MODE_CHOICES, default=None)
    parser.add_argument("--long-doc-max-chars", type=int, default=None)
    parser.add_argument("--long-doc-overlap-lines", type=int, default=None)
    parser.add_argument("--long-doc-min-chars", type=int, default=None)
    parser.add_argument("--long-doc-max-evidence-chunks", type=int, default=None)
    parser.add_argument("--long-doc-max-evidence-chars", type=int, default=None)


def resolve_long_doc_config(
    args: Any,
    benchmark_config: Any | None,
    *,
    default_mode: str = "off",
) -> LongDocEvidenceConfig:
    defaults = LongDocEvidenceConfig()

    def _int(cli_name: str, config_name: str, default: int, *, minimum: int = 1) -> int:
        for value in (getattr(args, cli_name, None), getattr(benchmark_config, config_name, None)):
            if value is None:
                continue
            try:
                return max(minimum, int(value))
            except (TypeError, ValueError):
                continue
        return default

    mode = str(
        getattr(args, "long_doc_mode", None)
        or getattr(benchmark_config, "long_context_router_mode", None)
        or default_mode
    ).strip().lower()
    return LongDocEvidenceConfig(
        enabled=mode == "lexical",
        mode="lexical",
        max_chunk_chars=_int("long_doc_max_chars", "long_context_chunk_chars", defaults.max_chunk_chars),
        overlap_lines=_int("long_doc_overlap_lines", "long_context_overlap_lines", defaults.overlap_lines, minimum=0),
        min_long_text_chars=_int("long_doc_min_chars", "long_context_min_chars", defaults.min_long_text_chars),
        max_evidence_chunks=_int("long_doc_max_evidence_chunks", "long_context_max_evidence_chunks", defaults.max_evidence_chunks),
        max_evidence_chars=_int("long_doc_max_evidence_chars", "long_context_max_evidence_chars", defaults.max_evidence_chars),
    )


def long_doc_config_payload(config: LongDocEvidenceConfig) -> dict[str, Any]:
    payload = asdict(config)
    if not config.enabled:
        payload["mode"] = "off"
    return payload


__all__ = [
    "add_long_doc_cli_args",
    "build_budgeted_context_prompt",
    "compose_context_question",
    "fuse_context_question",
    "long_doc_config_payload",
    "middle_truncate_text",
    "resolve_long_doc_config",
]
