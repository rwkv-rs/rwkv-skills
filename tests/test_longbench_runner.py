from __future__ import annotations

import json
import types
from pathlib import Path

from src.eval.tasks.function_calling import longbench as longbench_module
from src.eval.tasks.function_calling.longbench import (
    LongBenchRecord,
    MAX_LONGBENCH_GENERATION_BATCH_SIZE,
    _run_longbench,
    _resolve_longbench_batch_size,
    build_longbench_budgeted_prompt,
    normalize_longbench_answer,
    score_longbench_answer,
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


def test_longbench_answer_normalization_prefers_final_answer_line() -> None:
    assert normalize_longbench_answer("Reasoning...\nAnswer: Alice Smith") == "Alice Smith"


def test_longbench_score_uses_best_reference_f1() -> None:
    score = score_longbench_answer("Answer: Alice Smith", ["Bob", "Alice Smith"])

    assert score.exact_match is True
    assert score.f1 == 1.0
    assert score.passed is True
    assert score.best_reference == "Alice Smith"


def test_longbench_batch_size_is_capped_for_long_context_generation() -> None:
    assert _resolve_longbench_batch_size(64) == MAX_LONGBENCH_GENERATION_BATCH_SIZE
    assert _resolve_longbench_batch_size(None) == MAX_LONGBENCH_GENERATION_BATCH_SIZE
    assert _resolve_longbench_batch_size(2) == 2


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
        prompt_max_chars=3000,
    )

    assert "Long document compacted" in prompt
    assert '"name": "final_answer"' not in prompt
    assert "Return no prose" not in prompt
    assert "```json" not in prompt
    assert prompt.rstrip().endswith("Assistant:")
    assert trace["compacted"] is True
    assert trace["output_format"] == "direct_final_answer"
    assert trace["prompt_chars"] <= 3000


def test_run_longbench_keeps_raw_completion_separate_from_sandbox_return(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "longbench_test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "lb-1",
                "dataset": "hotpotqa",
                "input": "Where is the answer?",
                "context": "The answer is Zurich.",
                "answers": ["Zurich"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_completion = '```json\n{"name":"final_answer","arguments":{"answer":"Zurich"},"id":"call_lb"}\n```'

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

    monkeypatch.setattr(longbench_module, "resolve_sampling_config", lambda *_args, **_kwargs: SamplingConfig())
    monkeypatch.setattr(longbench_module, "prepare_function_calling_run", _fake_prepare_function_calling_run)
    monkeypatch.setattr(longbench_module, "finalize_function_calling_run", _fake_finalize_function_calling_run)

    rc = _run_longbench(
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
            benchmark_kind=FunctionCallingBenchmarkKind.LONGBENCH,
            dataset_path=dataset,
            dataset_slug="longbench_test",
            benchmark_name="longbench",
            dataset_split="test",
            model_name="demo-model",
            engine=FakeEngine(),
        ),
    )

    writer = captured["writer"]
    assert rc == 0
    assert isinstance(writer, _CollectingWriter)
    [payload] = writer.payloads
    sandbox_return = '{"name":"final_answer","arguments":{"answer":"Zurich"},"id":"call_lb"}'
    assert payload["completion1"] == raw_completion
    assert payload["agent_info"]["prediction"] == "Zurich"
    assert payload["agent_info"]["final_answer_call"] == sandbox_return
    assert payload["agent_info"]["decoded_final_answer_call"] == {
        "name": "final_answer",
        "arguments": {"answer": "Zurich"},
        "id": "call_lb",
    }
    assert payload["agent_trace"][0]["raw_completion"] == raw_completion
    assert payload["agent_trace"][0]["sandbox_return"] == sandbox_return
    assert payload["agent_info"]["answer_extraction"] == "final_answer_json"


def test_run_longbench_normalizes_bare_answer_json_fragment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "longbench_test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "lb-1",
                "dataset": "hotpotqa",
                "input": "Where is the answer?",
                "context": "The answer is Zurich.",
                "answers": ["Zurich"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    raw_completion = '"answer": "Zurich"\n}'

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

    monkeypatch.setattr(longbench_module, "resolve_sampling_config", lambda *_args, **_kwargs: SamplingConfig())
    monkeypatch.setattr(longbench_module, "prepare_function_calling_run", _fake_prepare_function_calling_run)
    monkeypatch.setattr(longbench_module, "finalize_function_calling_run", _fake_finalize_function_calling_run)

    rc = _run_longbench(
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
            benchmark_kind=FunctionCallingBenchmarkKind.LONGBENCH,
            dataset_path=dataset,
            dataset_slug="longbench_test",
            benchmark_name="longbench",
            dataset_split="test",
            model_name="demo-model",
            engine=FakeEngine(),
        ),
    )

    writer = captured["writer"]
    assert rc == 0
    [payload] = writer.payloads
    assert payload["agent_info"]["prediction"] == "Zurich"
    assert payload["agent_info"]["parse_error"] == ""
    assert payload["agent_info"]["answer_extraction"] == "answer_json"
    assert payload["agent_info"]["final_answer_call"] == (
        '{"name":"final_answer","arguments":{"answer":"Zurich"},"id":"final_answer"}'
    )


def test_longbench_extracts_fenced_answer_json_after_echoed_assistant_prefix() -> None:
    for raw_completion in (
        'Assistant:\n```json\n{"answer": "Zurich"}\n```',
        'Assistant: ```json\n{"answer": "Zurich"}\n```',
    ):
        prediction, decoded_call, call_id, parse_error, extraction = longbench_module._extract_longbench_prediction(
            raw_completion
        )

        assert prediction == "Zurich"
        assert decoded_call == {}
        assert call_id == "final_answer"
        assert parse_error == ""
        assert extraction == "answer_json"
