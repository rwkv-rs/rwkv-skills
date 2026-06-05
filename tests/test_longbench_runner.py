from __future__ import annotations

from src.eval.function_calling.longbench import (
    LongBenchRecord,
    build_longbench_budgeted_prompt,
    normalize_longbench_answer,
    score_longbench_answer,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig


def test_longbench_answer_normalization_prefers_final_answer_line() -> None:
    assert normalize_longbench_answer("Reasoning...\nAnswer: Alice Smith") == "Alice Smith"


def test_longbench_score_uses_best_reference_f1() -> None:
    score = score_longbench_answer("Answer: Alice Smith", ["Bob", "Alice Smith"])

    assert score.exact_match is True
    assert score.f1 == 1.0
    assert score.passed is True
    assert score.best_reference == "Alice Smith"


def test_longbench_budgeted_prompt_compacts_long_context() -> None:
    record = LongBenchRecord(
        task_id="demo",
        dataset="hotpotqa",
        input="Where is the answer?",
        context="\n".join(["irrelevant row"] * 80 + ["The answer is Zurich."] + ["other row"] * 80),
        answers=("Zurich",),
    )

    prompt, trace = build_longbench_budgeted_prompt(
        record,
        long_doc_config=LongDocEvidenceConfig(
            enabled=True,
            mode="lexical",
            max_chunk_chars=200,
            overlap_lines=0,
            min_long_text_chars=300,
            max_evidence_chunks=2,
            max_evidence_chars=400,
        ),
        prompt_max_chars=1200,
    )

    assert "Long document compacted" in prompt
    assert trace["compacted"] is True
    assert trace["prompt_chars"] <= 1200
