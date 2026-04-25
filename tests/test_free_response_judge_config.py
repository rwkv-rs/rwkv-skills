from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.eval.metrics.free_response import LLMJudge, LLMJudgeConfig


class LLMJudgeConfigTest(unittest.TestCase):
    def test_default_prompt_template_is_string(self) -> None:
        cfg = LLMJudgeConfig(api_key="k", model="m")
        self.assertIsInstance(cfg.prompt_template, str)
        self.assertIn("Only output 'True' or 'False'", cfg.prompt_template)

    def test_judge_accepts_strict_boolean_responses(self) -> None:
        judge = LLMJudge(
            LLMJudgeConfig(
                api_key="k",
                model="m",
                max_workers=1,
                max_retries=0,
                backoff_base=0.0,
            )
        )
        judge.client = _FakeJudgeClient(["True", "False"])

        result = judge.judge(
            [
                ("q1", "ref1", "pred1"),
                ("q2", "ref2", "pred2"),
            ]
        )

        self.assertEqual(result, [True, False])

    def test_judge_rejects_non_strict_boolean_responses(self) -> None:
        judge = LLMJudge(
            LLMJudgeConfig(
                api_key="k",
                model="m",
                max_workers=1,
                max_retries=0,
                backoff_base=0.0,
            )
        )
        judge.client = _FakeJudgeClient(["true"])

        self.assertEqual(judge.judge([("q1", "ref1", "pred1")]), [False])


class _FakeJudgeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_: object) -> SimpleNamespace:
        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=str(response))),
            ]
        )


if __name__ == "__main__":
    unittest.main()
