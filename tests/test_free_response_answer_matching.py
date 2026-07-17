from __future__ import annotations

import re
import time

from src.eval.metrics import free_response as fr
from src.eval.metrics.free_response import (
    LLMJudge,
    LLMJudgeConfig,
    STRATEGY_GROUPS,
    build_grouped_metrics_payload,
    evaluate_free_response,
)


def _patch_math_verify(monkeypatch) -> None:
    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return [("gold", text.removeprefix("$\\boxed{").removesuffix("}$"))]
        if "requires_unclosed_repair" in text and "</think>" in text and "Therefore, the final answer is " in text:
            return [("pred", "7")]
        if "requires_truncated_repair" in text and "\nTherefore, the final answer is " in text:
            return [("pred", "7")]
        boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxes:
            return [("pred", boxes[-1])]
        return []

    def verify(gold, pred, *, strict: bool = False):
        _ = strict
        if not gold or not pred:
            return False
        return gold[-1][-1] == pred[-1][-1]

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))


def test_math_verify_timeout_marks_completion_wrong(monkeypatch) -> None:
    def parse(text: str):
        if text.startswith("$\\boxed{"):
            return [("gold", "7")]
        return [("pred", "8")]

    def verify(_gold, _pred, *, strict: bool = False):
        _ = strict
        time.sleep(1.0)
        return True

    monkeypatch.setattr(fr, "_load_math_verify", lambda: (parse, verify))
    monkeypatch.setenv("RWKV_MATH_VERIFY_TIMEOUT_S", "0.1")

    result = fr._math_verify("7", "Final answer: \\boxed{8}")

    assert not result.passed
    assert result.fail_reason == "math_verify_timeout"


def test_strategy_a_scores_raw_full_generation_and_tracks_stop_rate(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q1","answer":"7"}\n{"question":"q2","answer":"9"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q1\n\nAssistant: <think",
                "completion1": "<think>work</think>\nFinal answer: \\boxed{7}",
                "stop_reason1": "stop_token",
                "stats": {"truncated": False, "stop_detail": "token_0"},
            },
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 1,
                "repeat_index": 0,
                "prompt1": "User: q2\n\nAssistant: <think",
                "completion1": "<think>work</think>\nFinal answer: \\boxed{8}",
                "stop_reason1": "max_length",
                "stats": {"truncated": True, "stop_detail": "max_length"},
            },
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert set(evaluation.metrics_by_group) == set(STRATEGY_GROUPS)
    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_a"]["stop_rate"] == 0.5
    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, True), (1, 0, False)]
    assert len(evaluation.payloads) == 2
    assert all("eval_group" not in payload for payload in evaluation.payloads)

    metrics_payload, task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert metrics_payload["avg@1"] == 0.5
    assert metrics_payload["stop_rate"] == 0.5
    assert task_details == {}


def test_strategy_b_c_inherit_a_passes_and_only_rescore_a_failures(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q1","answer":"7"}\n{"question":"q2","answer":"9"}\n',
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q1\n\nAssistant: <think",
                "completion1": "</think>\nlegacy answer \\boxed{7}",
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "0",
                "stop_reason2": "stop_token",
            },
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 1,
                "repeat_index": 0,
                "prompt1": "User: q2\n\nAssistant: <think",
                "completion1": "</think>\nlegacy answer \\boxed{8}",
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "9",
                "stop_reason2": "stop_token",
            },
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.primary_group == "strategy_a"
    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group["strategy_b"] == [(0, 0, True), (1, 0, False)]
    assert evaluation.rows_by_group["strategy_c"] == [(0, 0, True), (1, 0, True)]
    assert evaluation.payloads[0]["answer"].endswith("7")

    metrics_payload, _task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert metrics_payload["avg@1"] == 0.5
    assert metrics_payload["strategy_metrics"]["strategy_c"]["avg@1"] == 1.0


def test_strategy_c_repairs_unclosed_think_for_scoring_only(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "requires_unclosed_repair",
                "stop_reason1": "max_length",
                "stats": {"truncated": True, "stop_detail": "max_length"},
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0


def test_strategy_c_repairs_truncated_answer_region(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "</think>\nrequires_truncated_repair",
                "stop_reason1": "max_length",
                "stats": {"truncated": True, "stop_detail": "max_length"},
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_b"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0


def test_two_stage_payload_uses_legacy_completion1_scoring(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "</think>\nwrong intermediate \\boxed{8}",
                "stop_reason1": "stop_token",
                "prompt2": "\nTherefore, the answer is \\(\\boxed{",
                "completion2": "7",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group["strategy_a"] == [(0, 0, False)]
    assert evaluation.primary_group == "strategy_a"
    assert evaluation.payloads[0]["answer"].endswith("8")


def test_judgement_label_dataset_scores_final_stage_without_math_verify(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fr, "_load_math_verify", lambda: None)
    dataset = tmp_path / "answer_judge.jsonl"
    dataset.write_text(
        (
            '{"question":"q","expected_answer":"reference text",'
            '"predicted_answer":"candidate text","expected_judgement":"Judgement: No"}\n'
        ),
        encoding="utf-8",
    )

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "answer_judge",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "User: q\n\nAssistant: <think",
                "completion1": "</think>\nJudgement: Yes",
                "stop_reason1": "stop_token",
                "prompt2": "\nReturn exactly one final label: ",
                "completion2": "Judgement: No",
                "stop_reason2": "stop_token",
            }
        ],
        dataset_path=dataset,
        judge=None,
    )

    assert evaluation.primary_group == "strategy_c"
    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.0
    assert evaluation.metrics_by_group["strategy_c"]["exact_accuracy"] == 1.0
    assert evaluation.rows_by_group["strategy_c"] == [(0, 0, True)]
    assert evaluation.payloads[0]["answer"] == "Judgement: No"

    metrics_payload, _task_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=(1,),
        avg_k=(1,),
        report_pass_k=(1,),
        report_avg_k=(1,),
    )
    assert metrics_payload["avg@1"] == 1.0
    assert metrics_payload["strategy_metrics"]["strategy_a"]["avg@1"] == 0.0


def test_a_judge_pass_skips_strategy_judge_and_inherits_correctness(monkeypatch, tmp_path) -> None:
    _patch_math_verify(monkeypatch)
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"q1","answer":"7"}\n{"question":"q2","answer":"9"}\n',
        encoding="utf-8",
    )
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key="k",
            model="m",
            max_workers=1,
            max_retries=0,
            backoff_base=0.0,
        )
    )
    judge.client = _CapturingJudgeClient("True")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "\\boxed{7}",
                "stop_reason1": "stop_token",
            },
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 1,
                "repeat_index": 0,
                "prompt1": "p",
                "completion1": "\\boxed{8}",
                "stop_reason1": "stop_token",
            },
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.metrics_by_group["strategy_a"]["exact_accuracy"] == 0.5
    assert evaluation.metrics_by_group["strategy_a"]["judge_accuracy"] == 1.0
    assert evaluation.metrics_by_group["strategy_b"]["judge_accuracy"] == 1.0
    assert evaluation.metrics_by_group["strategy_c"]["judge_accuracy"] == 1.0
    assert [row[2] for row in evaluation.rows_by_group["strategy_a"]] == [True, True]
    assert [row[2] for row in evaluation.rows_by_group["strategy_b"]] == [True, True]
    assert [row[2] for row in evaluation.rows_by_group["strategy_c"]] == [True, True]
    assert len(judge.client.prompts) == 1
    assert all("Question: q2" in prompt for prompt in judge.client.prompts)
    assert all("Student's Answer:" in prompt for prompt in judge.client.prompts)


class _CapturingJudgeClient:
    def __init__(self, response: str) -> None:
        self.prompts: list[str] = []
        self.chat = _CapturingChat(self, response)


class _CapturingChat:
    def __init__(self, client: _CapturingJudgeClient, response: str) -> None:
        self.completions = _CapturingCompletions(client, response)


class _CapturingCompletions:
    def __init__(self, client: _CapturingJudgeClient, response: str) -> None:
        self._client = client
        self._response = response

    def create(self, **kwargs):
        from types import SimpleNamespace

        self._client.prompts.append(kwargs["messages"][0]["content"])
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=self._response)),
            ]
        )
