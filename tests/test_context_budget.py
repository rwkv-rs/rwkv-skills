from __future__ import annotations

import argparse
from types import SimpleNamespace

from src.eval.context_budget import (
    add_long_doc_cli_args,
    build_budgeted_context_prompt,
    compose_context_question,
    middle_truncate_text,
    resolve_long_doc_config,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig

_SMALL_CONFIG = LongDocEvidenceConfig(
    enabled=True,
    max_chunk_chars=120,
    overlap_lines=0,
    min_long_text_chars=200,
    max_evidence_chunks=2,
    max_evidence_chars=400,
)


def _long_context(hit_line: str) -> str:
    noise = [f"noise row {idx} lorem ipsum filler" for idx in range(60)]
    noise.insert(40, hit_line)
    return "\n".join(noise)


def test_compose_context_question_preserves_question_without_context() -> None:
    question = "  Line 1\r\nLine 2?"

    rendered, trace = compose_context_question(
        None,
        question,
        long_doc_config=LongDocEvidenceConfig(enabled=False),
        label="unit",
    )

    assert rendered == question
    assert trace is None


def test_compose_context_question_selects_query_relevant_chunk() -> None:
    hit = "case77 answer blue supporting evidence"
    rendered, trace = compose_context_question(
        _long_context(hit),
        "What is the answer for case77?",
        long_doc_config=_SMALL_CONFIG,
        label="unit",
    )

    assert hit in rendered
    assert rendered.startswith("Context:")
    assert rendered.rstrip().endswith("What is the answer for case77?")
    assert trace is not None
    assert trace["compacted"] is True
    assert trace["selected_chunk_ids"]


def test_compose_context_question_ignores_context_when_disabled() -> None:
    rendered, trace = compose_context_question(
        "short document",
        "The question?",
        long_doc_config=LongDocEvidenceConfig(enabled=False),
        label="unit",
    )

    assert rendered == "The question?"
    assert trace is None


def test_build_budgeted_context_prompt_clamps_to_prompt_max_chars() -> None:
    prompt_max_chars = 500
    prompt, trace = build_budgeted_context_prompt(
        context=_long_context("needle"),
        query="needle",
        render=lambda ctx: f"PREFIX\n{ctx}\nSUFFIX",
        long_doc_config=LongDocEvidenceConfig(enabled=True, min_long_text_chars=100_000),
        prompt_max_chars=prompt_max_chars,
        label="unit",
    )

    assert len(prompt) <= prompt_max_chars
    assert trace is not None
    assert trace["prompt_chars"] == len(prompt)
    assert trace["trimmed_context_chars"] > 0


def test_middle_truncate_text_edges() -> None:
    assert middle_truncate_text("abc", 0) == ""
    assert middle_truncate_text("abc", 10) == "abc"
    assert middle_truncate_text("x" * 100, 20) == "x" * 20
    truncated = middle_truncate_text("a" * 200 + "b" * 200, 120)
    assert len(truncated) == 120
    assert "middle truncated" in truncated
    assert truncated.startswith("a") and truncated.endswith("b")


def test_resolve_long_doc_config_prefers_cli_over_config_over_default() -> None:
    parser = argparse.ArgumentParser()
    add_long_doc_cli_args(parser)
    args = parser.parse_args(["--long-doc-mode", "lexical", "--long-doc-min-chars", "111"])
    config = SimpleNamespace(
        long_context_router_mode="off",
        long_context_min_chars=999,
        long_context_max_evidence_chunks=7,
    )

    resolved = resolve_long_doc_config(args, config)

    assert resolved.enabled is True
    assert resolved.min_long_text_chars == 111  # CLI wins
    assert resolved.max_evidence_chunks == 7  # config fills CLI gaps
    assert resolved.max_chunk_chars == LongDocEvidenceConfig().max_chunk_chars  # default fills the rest


def test_resolve_long_doc_config_defaults_to_off() -> None:
    parser = argparse.ArgumentParser()
    add_long_doc_cli_args(parser)
    args = parser.parse_args([])

    assert resolve_long_doc_config(args, None).enabled is False
    assert resolve_long_doc_config(args, SimpleNamespace(long_context_router_mode="lexical"), default_mode="off").enabled is True


def test_resolve_long_doc_config_allows_zero_overlap_lines() -> None:
    parser = argparse.ArgumentParser()
    add_long_doc_cli_args(parser)
    args = parser.parse_args(["--long-doc-overlap-lines", "0"])

    assert resolve_long_doc_config(args, None).overlap_lines == 0
