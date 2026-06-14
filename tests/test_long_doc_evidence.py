from __future__ import annotations

import pytest

from src.infer.sampling import SamplingConfig
from src.eval.long_doc_evidence import (
    LongDocEvidenceConfig,
    TextChunk,
    build_evidence_tasks,
    build_long_doc_evidence_router_prompt,
    chunk_text_by_newline,
    compact_long_text,
    compact_messages_for_long_context,
    infer_query_from_messages,
    parse_answer_or_null_response,
    parse_long_doc_evidence_router_response,
    _long_doc_router_sampling,
)


def test_chunk_text_by_newline_preserves_overlap_lines() -> None:
    chunks = chunk_text_by_newline(
        "line1\nline2\nline3\nline4\n",
        max_chars=12,
        overlap_lines=1,
    )

    assert [chunk.line_start for chunk in chunks] == [1, 2]
    assert [chunk.line_end for chunk in chunks] == [2, 4]
    assert chunks[1].overlap_lines == 1
    assert chunks[1].text.startswith("line2\nline3\n")


def test_build_evidence_tasks_recomputes_positive_chunks_and_answer_hits() -> None:
    chunks = [
        TextChunk(chunk_id=0, text="alpha beta answer42\n", line_start=1, line_end=1),
        TextChunk(chunk_id=1, text="alpha only\n", line_start=2, line_end=2),
    ]

    tasks, summary = build_evidence_tasks(
        chunks,
        [
            {
                "id": "q1",
                "question": "Which answer belongs to alpha beta?",
                "answer": "answer42",
                "answer_format": "scalar_string",
                "positive_rule": {"all": ["alpha", "beta"]},
            }
        ],
        chunk_source="unit",
    )

    assert tasks[0]["positive_chunks"] == [0]
    assert tasks[0]["chunking"]["positive_chunks_recomputed_from"] == "positive_rule"
    assert summary["oracle_passed"] is True
    assert summary["answer_hit_count_in_positive_chunks"]["by_task"] == {"q1": 1}


def test_compact_long_text_selects_query_relevant_chunk() -> None:
    lines = [f"noise row {index:03d}" for index in range(40)]
    lines.insert(25, "case928 answer color blue verified")
    text = "\n".join(lines)

    result = compact_long_text(
        text,
        query="What color is case928?",
        config=LongDocEvidenceConfig(
            max_chunk_chars=120,
            overlap_lines=1,
            min_long_text_chars=100,
            max_evidence_chunks=1,
            max_evidence_chars=200,
        ),
        label="unit",
    )

    assert result.compacted is True
    assert result.chunk_count > 1
    assert "case928 answer color blue verified" in result.text
    assert "Long document compacted" in result.text


def test_compact_messages_uses_recent_short_user_query_before_long_tool_output() -> None:
    task = "Find invoice INV-42 status."
    long_tool_output = "\n".join(
        [f"unrelated ledger row {index:03d}" for index in range(30)]
        + ["invoice INV-42 status paid evidence"]
        + [f"archive row {index:03d}" for index in range(30)]
    )
    messages = [
        {"role": "user", "content": task},
        {"role": "assistant", "content": '{"name":"lookup_invoice","arguments":{"id":"INV-42"}}'},
        {"role": "user", "content": long_tool_output},
    ]

    result = compact_messages_for_long_context(
        messages,
        config=LongDocEvidenceConfig(
            max_chunk_chars=160,
            overlap_lines=1,
            min_long_text_chars=200,
            max_evidence_chunks=1,
            max_evidence_chars=240,
        ),
    )

    assert result.compacted_message_count == 1
    assert infer_query_from_messages(messages, skip_longer_than=200) == task
    assert "invoice INV-42 status paid evidence" in result.messages[-1]["content"]


def test_long_doc_compaction_rejects_removed_model_parallel_mode() -> None:
    with pytest.raises(ValueError, match="unsupported long-doc mode"):
        compact_long_text(
            "case-77 answer green\n" * 30,
            query="What is the answer for case-77?",
            config=LongDocEvidenceConfig(
                mode="model_parallel",  # type: ignore[arg-type]
                max_chunk_chars=120,
                min_long_text_chars=200,
            ),
        )


def test_long_doc_router_prompt_uses_compact_schema_without_reason_field() -> None:
    prompt = build_long_doc_evidence_router_prompt(
        chunk=TextChunk(chunk_id=2, text="case-77 answer green", line_start=4, line_end=4),
        query="What is the answer for case-77?",
    )

    assert '{"relevant":true,"score":3}' in prompt
    assert '"reason":"short"' not in prompt
    assert "Do not include reason" in prompt


def test_parse_long_doc_router_recovers_truncated_reason_payload() -> None:
    response = (
        '{\n'
        '  "relevant": true,\n'
        '  "score": 2,\n'
        '  "reason": "The chunk is useful but the model keeps explaining'
    )

    assert parse_long_doc_evidence_router_response(response) == (True, 2.0)


def test_parse_long_doc_router_rejects_partial_payload_without_score_value() -> None:
    with pytest.raises(ValueError):
        parse_long_doc_evidence_router_response('{"relevant":false,"score":')


def test_long_doc_router_sampling_uses_deterministic_router_settings() -> None:
    sampling = SamplingConfig(max_generate_tokens=256, temperature=0.3, top_k=500, top_p=0.4)

    routed = _long_doc_router_sampling(sampling, max_tokens=24)

    assert routed.max_generate_tokens == 24
    assert routed.temperature == 0.001
    assert routed.top_k == 1
    assert routed.top_p == 1.0
    assert routed.alpha_presence == 0.0
    assert routed.alpha_frequency == 0.0


def test_parse_answer_or_null_response_accepts_json_fence() -> None:
    assert parse_answer_or_null_response('```json\n{"answer":"blue"}\n```') == ("blue", True)
    assert parse_answer_or_null_response("not json") == ("null", False)
