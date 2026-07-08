from __future__ import annotations

import json
import types
from pathlib import Path

from src.eval.tasks.function_calling import browsecomp as browsecomp_module
from src.eval.tasks.function_calling.browsecomp import (
    BrowseCompJudgeConfig,
    BrowseCompJudgeOutcome,
    BrowseCompRecord,
    _run_browsecomp,
    judge_browsecomp_answers,
)
from src.eval.tasks.function_calling.runner_common import FunctionCallingBenchmarkKind, ResolvedFunctionCallingRun
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


def test_judge_browsecomp_answers_marks_failed_items_after_retries(monkeypatch) -> None:
    class FakeCompletions:
        def create(self, **_kwargs):
            raise RuntimeError("rate limited")

    class FakeClient:
        chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.OpenAI", lambda **_kwargs: FakeClient())

    outcomes = judge_browsecomp_answers(
        [(BrowseCompRecord(task_id="bc-1", question="q", answer="a", locale="en"), "response")],
        config=BrowseCompJudgeConfig(
            api_key="key",
            model="judge",
            max_workers=1,
            max_retries=1,
            backoff_base_s=0,
        ),
    )

    assert outcomes == [
        BrowseCompJudgeOutcome(
            is_passed=False,
            reason="judge failed after retries: browsecomp judge failed after retries: rate limited",
        )
    ]


def test_run_browsecomp_keeps_raw_completion_separate_from_sandbox_return(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "browsecomp_test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "task_id": "bc-1",
                "question": "Which city is the answer?",
                "answer": "Zurich",
                "locale": "en",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cot_completion = "Reasoning: the evidence points to Zurich.  \n\n"
    raw_answer_completion = '```json\n{"name":"final_answer","arguments":{"answer":"Zurich"},"id":"call_bc"}\n```'

    class FakeEngine:
        def generate(self, prompts, *, progress_desc, **_kwargs):
            text = raw_answer_completion if progress_desc == "BrowseComp-Answer" else cot_completion
            return [
                GenerationOutput(
                    prompt_index=index,
                    prompt=prompt,
                    token_ids=[],
                    text=text,
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

    def _fake_judge_browsecomp_answers(items, *, config):
        assert config.model == "judge"
        assert config.max_workers == 9
        assert [(record.task_id, answer) for record, answer in items] == [("bc-1", "Explanation: Zurich")]
        return [BrowseCompJudgeOutcome(is_passed=True, reason="matched")]

    monkeypatch.setattr(browsecomp_module, "resolve_sampling_config", lambda *_args, **_kwargs: SamplingConfig())
    monkeypatch.setattr(
        browsecomp_module,
        "resolve_judge_model_config",
        lambda: types.SimpleNamespace(api_key="", model_name="judge", base_url=None),
    )
    monkeypatch.setattr(browsecomp_module, "prepare_function_calling_run", _fake_prepare_function_calling_run)
    monkeypatch.setattr(browsecomp_module, "finalize_function_calling_run", _fake_finalize_function_calling_run)
    monkeypatch.setattr(browsecomp_module, "judge_browsecomp_answers", _fake_judge_browsecomp_answers)

    rc = _run_browsecomp(
        types.SimpleNamespace(
            max_samples=1,
            avg_k=[1.0],
            cot_max_tokens=64,
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
            judge_max_workers=9,
            probe_only=False,
        ),
        ResolvedFunctionCallingRun(
            benchmark_kind=FunctionCallingBenchmarkKind.BROWSECOMP,
            dataset_path=dataset,
            dataset_slug="browsecomp_test",
            benchmark_name="browsecomp",
            dataset_split="test",
            model_name="demo-model",
            engine=FakeEngine(),
        ),
    )

    writer = captured["writer"]
    assert rc == 0
    assert isinstance(writer, _CollectingWriter)
    [payload] = writer.payloads
    sandbox_return = '{"name":"final_answer","arguments":{"answer":"Zurich"},"id":"call_bc"}'
    assert payload["completion1"] == cot_completion
    assert str(payload["prompt2"]).startswith(str(payload["prompt1"]))
    assert payload["completion2"] == raw_answer_completion
    assert payload["agent_info"]["response"] == "Explanation: Zurich"
    assert payload["agent_info"]["final_answer_call"] == sandbox_return
    assert payload["agent_info"]["decoded_final_answer_call"] == {
        "name": "final_answer",
        "arguments": {"answer": "Zurich"},
        "id": "call_bc",
    }
    assert payload["agent_trace"][1]["raw_completion"] == raw_answer_completion
    assert payload["agent_trace"][1]["sandbox_return"] == sandbox_return
    assert payload["agent_trace"][1]["long_doc"]["prompt_delta_fallback"] is True
