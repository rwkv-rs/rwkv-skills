from __future__ import annotations

from src.eval.metrics.free_response import (
    LLMJudge,
    LLMJudgeConfig,
    _extract_answer_from_final_stage,
    _is_exact_match,
    _strip_thinking_for_answer,
    evaluate_free_response,
)


def test_exact_match_requires_extracted_answer_text() -> None:
    assert _is_exact_match("9", "9")
    assert not _is_exact_match(r"Therefore, the answer is \(\\boxed{9", "9")
    assert not _is_exact_match("100", "9")


def test_legacy_text_exact_match_is_not_case_folded() -> None:
    assert _is_exact_match("Evelyn", "Evelyn")
    assert not _is_exact_match("Briana", "Evelyn")
    assert not _is_exact_match("evelyn", "Evelyn")


def test_strip_thinking_ignores_hidden_reasoning() -> None:
    text = "<think>maybe 42</think>\nFinal answer: 7"
    assert _strip_thinking_for_answer(text) == "Final answer: 7"
    assert _strip_thinking_for_answer("<think>unfinished 42") == ""


def test_extract_answer_uses_final_prompt_boxed_brace_format() -> None:
    prompt = "\n</think>\nTherefore, the final answer is \\(\\boxed{"
    completion = r"C=\dfrac{\pi}{3}}\)."

    assert _extract_answer_from_final_stage(prompt, completion) == r"C=\dfrac{\pi}{3}"


def test_extract_answer_keeps_final_stage_text_when_wrapper_is_unclosed() -> None:
    prompt = "\n</think>\nTherefore, the final answer is \\(\\boxed{"

    assert _extract_answer_from_final_stage(prompt, r"-1-\sqrt{3") == r"-1-\sqrt{3"


def test_extract_answer_supports_original_latex_paren_stage_format() -> None:
    assert _extract_answer_from_final_stage("Final answer: \\(", r"x+1\).") == "x+1"


def test_extract_answer_keeps_stage_output_when_prompt_has_no_wrapper() -> None:
    assert _extract_answer_from_final_stage("Final answer:", "x=2") == "x=2"


def test_judge_receives_frontend_answer_not_reasoning(tmp_path) -> None:
    dataset = tmp_path / "free.jsonl"
    dataset.write_text('{"question":"q","answer":"7"}\n', encoding="utf-8")
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key="k",
            model="m",
            max_workers=1,
            max_retries=0,
            backoff_base=0.0,
        )
    )
    judge.client = _CapturingJudgeClient("False")

    evaluation = evaluate_free_response(
        [
            {
                "benchmark_name": "free",
                "dataset_split": "test",
                "sample_index": 0,
                "repeat_index": 0,
                "prompt1": "Final answer:",
                "completion1": "<think>the answer might be 42</think>\nFinal answer: 7",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.payloads[0]["answer"] == "Final answer: 7"
    assert "Student's Answer: Final answer: 7" in judge.client.prompts[0]
    assert "42" not in judge.client.prompts[0]


def test_judge_receives_extracted_final_stage_answer_for_structured_refs(tmp_path) -> None:
    dataset = tmp_path / "free.jsonl"
    dataset.write_text(
        '{"question":"Solve the inequality.","answer":"$\\\\{x|-2\\\\leq x < 1\\\\}$"}\n',
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
                "completion1": "reasoning",
                "prompt2": "\n</think>\nTherefore, the final answer is \\(\\boxed{",
                "completion2": "[-2,\\,1)}\\).",
            }
        ],
        dataset_path=dataset,
        judge=judge,
    )

    assert evaluation.payloads[0]["answer"] == "[-2,\\,1)"
    assert "Student's Answer: [-2,\\,1)" in judge.client.prompts[0]
    assert "Student's Answer: 1" not in judge.client.prompts[0]


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
