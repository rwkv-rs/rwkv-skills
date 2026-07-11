from __future__ import annotations

import json
import types
import zipfile
from pathlib import Path

from src.eval.tasks.function_calling import longcodebench as longcodebench_module
from src.eval.tasks.function_calling.longcodebench import (
    LongCodeQARecord,
    _run_longcodebench,
    build_longcodeqa_budgeted_prompt,
    load_longcodeqa_rows_from_source,
    normalize_longcodeqa_answer,
    parse_longcodeqa_final_answer_text,
    score_longcodeqa_answer,
)
from src.eval.tasks.function_calling.runner_common import FunctionCallingBenchmarkKind, ResolvedFunctionCallingRun
from src.eval.long_doc_evidence import LongDocEvidenceConfig
from src.infer.sampling import GenerationOutput, SamplingConfig


class _CollectingWriter:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def enqueue(self, payload: dict[str, object]) -> None:
        self.payloads.append(payload)


class _FakeRuntime:
    state = types.SimpleNamespace(is_terminal=lambda: True)

    def handle_attempt_stage_failure(self, *_args, **_kwargs) -> None:
        return None

    def fail_task(self, *_args, **_kwargs) -> None:
        return None


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


def test_parse_longcodeqa_final_answer_falls_back_to_raw_letter_json() -> None:
    answer, call, call_id, error = parse_longcodeqa_final_answer_text(
        '```json\n{"answer":"B"}\n```',
        allowed_letters=("A", "B", "C", "D"),
    )

    assert answer == "B"
    assert call == {}
    assert call_id == ""
    assert error == ""


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

    assert official_prompt in prompt
    assert '"name": "final_answer"' in prompt
    assert '"id": {' in prompt
    assert prompt.rstrip().endswith("Assistant: ```json\n{")
    assert trace["mode"] == "off"
    assert trace["output_format"] == "rwkv_final_answer_json_call"
    assert trace["prompt_chars"] == len(prompt)


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
        prompt_max_chars=3000,
    )

    assert "Long document compacted" in prompt
    assert "target_symbol" in prompt
    assert '"name": "final_answer"' in prompt
    assert prompt.rstrip().endswith("Assistant: ```json\n{")
    assert trace["compacted"] is True
    assert trace["replacement_found"] is True
    assert trace["prompt_chars"] <= 3000


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


def test_run_longcodebench_keeps_raw_completion_separate_from_sandbox_return(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "longcodebench_test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "lc-1",
                "prompt": "Answer with a letter.\nRepository: VALUE = 1\nQuestion:\nWhat is VALUE?\nA) 0\nB) 1",
                "repo_text": "VALUE = 1",
                "question": "Question:\nWhat is VALUE?\nA) 0\nB) 1",
                "correct_letter": "B",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_completion = '```json\n{"name":"final_answer","arguments":{"answer":"B"},"id":"call_lc"}\n```'

    class FakeEngine:
        def generate(self, prompts, **_kwargs):
            return [
                GenerationOutput(
                    prompt_index=index,
                    prompt=prompt,
                    token_ids=[],
                    text=raw_completion,
                    finish_reason="stop_token",
                )
                for index, prompt in enumerate(prompts)
            ]

    captured: dict[str, object] = {}

    def _fake_prepare_function_calling_run(**_kwargs):
        writer = _CollectingWriter()
        captured["writer"] = writer
        return types.SimpleNamespace(
            service=object(),
            runtime=_FakeRuntime(),
            writer=writer,
            task_id="task",
            skip_keys=frozenset(),
        )

    def _fake_finalize_function_calling_run(*, ctx, **_kwargs):
        return list(ctx.writer.payloads), [], {}

    monkeypatch.setattr(longcodebench_module, "resolve_sampling_config", lambda *_args, **_kwargs: SamplingConfig())
    monkeypatch.setattr(longcodebench_module, "prepare_function_calling_run", _fake_prepare_function_calling_run)
    monkeypatch.setattr(longcodebench_module, "finalize_function_calling_run", _fake_finalize_function_calling_run)

    rc = _run_longcodebench(
        types.SimpleNamespace(
            max_samples=1,
            avg_k=[1.0],
            answer_max_tokens=64,
            batch_size=1,
            prompt_max_chars=4000,
            long_doc_mode="off",
            long_doc_max_chars=1000,
            long_doc_overlap_lines=3,
            long_doc_min_chars=6000,
            long_doc_max_evidence_chunks=4,
            long_doc_max_evidence_chars=6000,
            db_write_queue=1,
            db_close_timeout_s=0.1,
            probe_only=False,
        ),
        ResolvedFunctionCallingRun(
            benchmark_kind=FunctionCallingBenchmarkKind.LONGCODEBENCH,
            dataset_path=dataset,
            dataset_slug="longcodebench_test",
            benchmark_name="longcodebench",
            dataset_split="test",
            model_name="demo-model",
            engine=FakeEngine(),
        ),
    )

    writer = captured["writer"]
    assert rc == 0
    assert isinstance(writer, _CollectingWriter)
    [payload] = writer.payloads
    sandbox_return = '{"name":"final_answer","arguments":{"answer":"B"},"id":"call_lc"}'
    assert payload["completion1"] == raw_completion
    assert payload["agent_info"]["prediction"] == "B"
    assert payload["agent_info"]["final_answer_call"] == sandbox_return
    assert payload["agent_info"]["decoded_final_answer_call"] == {
        "name": "final_answer",
        "arguments": {"answer": "B"},
        "id": "call_lc",
    }
    assert payload["agent_trace"][0]["raw_completion"] == raw_completion
    assert payload["agent_trace"][0]["sandbox_return"] == sandbox_return
