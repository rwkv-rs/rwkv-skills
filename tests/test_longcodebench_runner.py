from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.eval.function_calling.longcodebench import (
    LongCodeQARecord,
    build_longcodeqa_budgeted_prompt,
    load_longcodeqa_rows_from_source,
    normalize_longcodeqa_answer,
    score_longcodeqa_answer,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig


def test_longcodeqa_answer_normalization_prefers_letter_answer() -> None:
    assert normalize_longcodeqa_answer("Reasoning\nAnswer: D", allowed_letters=("A", "B", "C", "D")) == "D"
    assert normalize_longcodeqa_answer("The answer is C.", allowed_letters=("A", "B", "C", "D")) == "C"
    assert normalize_longcodeqa_answer("<think>hidden</think>\n(B)", allowed_letters=("A", "B")) == "B"
    assert (
        normalize_longcodeqa_answer(
            'Assistant: <think>\n</think>\n```json\n{"answer":"B"}\n```',
            allowed_letters=("A", "B", "C", "D"),
        )
        == "B"
    )
    assert (
        normalize_longcodeqa_answer(
            "The correct answer is **A) replace(self, old, new, count=-1, /)**.",
            allowed_letters=("A", "B", "C", "D"),
        )
        == "A"
    )
    assert (
        normalize_longcodeqa_answer(
            "The main reason is **C) the highlighted option**.\nExplanation follows.",
            allowed_letters=("A", "B", "C", "D"),
        )
        == "C"
    )


def test_longcodeqa_score_uses_exact_letter() -> None:
    score = score_longcodeqa_answer("Final answer: C", "C", allowed_letters=("A", "B", "C", "D"))

    assert score.exact_match is True
    assert score.reward == 1.0
    assert score.prediction == "C"


def test_longcodeqa_default_prompt_preserves_official_prompt() -> None:
    repo_text = "Repository:\n[start of a.py]\nVALUE = 1\n[end of a.py]"
    question = "Question:\nWhat value is assigned?\nA) 0\nB) 1\n"
    official_prompt = f"Official instructions\nRepository: {repo_text}\n{question}"
    record = LongCodeQARecord(
        task_id="demo",
        prompt=official_prompt,
        repo_text=repo_text,
        question=question,
        correct_letter="B",
    )

    prompt, trace = build_longcodeqa_budgeted_prompt(
        record,
        long_doc_config=LongDocEvidenceConfig(enabled=False),
        prompt_max_chars=64,
    )

    assert prompt == official_prompt
    assert trace["mode"] == "off"
    assert trace["prompt_chars"] == len(official_prompt)


def test_longcodeqa_budgeted_prompt_replaces_repo_text_with_evidence() -> None:
    repo_text = "\n".join(
        ["Repository:", "[start of a.py]"]
        + [f"irrelevant_{index} = {index}" for index in range(80)]
        + ["def target_symbol():", "    return 'selected'"]
        + [f"tail_{index} = {index}" for index in range(80)]
    )
    question = "Question:\nWhich function returns selected?\nA) other\nB) target_symbol\n"
    official_prompt = f"Official instructions\nRepository: {repo_text}\n{question}"
    record = LongCodeQARecord(
        task_id="demo",
        prompt=official_prompt,
        repo_text=repo_text,
        question=question,
        correct_letter="B",
        context_bucket="32K",
    )

    prompt, trace = build_longcodeqa_budgeted_prompt(
        record,
        long_doc_config=LongDocEvidenceConfig(
            enabled=True,
            mode="lexical",
            max_chunk_chars=220,
            overlap_lines=0,
            min_long_text_chars=300,
            max_evidence_chunks=2,
            max_evidence_chars=450,
        ),
        prompt_max_chars=900,
    )

    assert "Long document compacted" in prompt
    assert "target_symbol" in prompt
    assert prompt.rstrip().endswith("Answer:")
    assert trace["compacted"] is True
    assert trace["replacement_found"] is True
    assert trace["prompt_chars"] <= 900


def test_load_longcodeqa_rows_from_zip_source(tmp_path: Path) -> None:
    archive = tmp_path / "LongCodeQA.zip"
    row = {
        "prompt_goal": "Answer with a letter.",
        "repo_text": "Repository:\n[start of a.py]\nVALUE = 1",
        "question": "Question:\nWhat is VALUE?\nA) 0\nB) 1\n",
        "prompt": "Answer with a letter.\nRepository: Repository:\n[start of a.py]\nVALUE = 1\nQuestion:\nWhat is VALUE?\nA) 0\nB) 1\n",
        "correct_letter": "B",
        "repo": "demo/repo",
        "is_hard": "No",
    }
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("LQA/32K.json", json.dumps([row]))

    [parsed] = load_longcodeqa_rows_from_source(archive)

    assert parsed["task_id"] == "longcodeqa_32k_00000"
    assert parsed["context_bucket"] == "32K"
    assert parsed["context_size"] == 32768
    assert parsed["correct_letter"] == "B"
    assert parsed["is_hard_label"] == "no"
