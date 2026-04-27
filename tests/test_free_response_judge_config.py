from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.eval.metrics.free_response import LLMJudge, LLMJudgeConfig
from src.eval.scheduler.jobs import (
    JOB_CATALOGUE,
    LLM_JUDGE_DATASET_SLUGS,
    MATH_DATASET_SLUGS,
    MATH_DATASET_SLUGS_FOR_FREE_RESPONSE,
    detect_job_from_dataset,
)


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

    def test_judge_accepts_boolean_after_think_block(self) -> None:
        judge = LLMJudge(
            LLMJudgeConfig(
                api_key="k",
                model="m",
                max_workers=1,
                max_retries=0,
                backoff_base=0.0,
            )
        )
        judge.client = _FakeJudgeClient(
            [
                "<think>\nreasoning may mention False or True\n</think>\n\nTrue",
                "<think>\nreasoning may mention True or False\n</think>\n\nFalse",
            ]
        )

        result = judge.judge(
            [
                ("q1", "ref1", "pred1"),
                ("q2", "ref2", "pred2"),
            ]
        )

        self.assertEqual(result, [True, False])
        self.assertIsNotNone(judge.last_run_stats)
        self.assertEqual(judge.last_run_stats.parsed_count, 2)
        self.assertEqual(judge.last_run_stats.invalid_output_count, 0)

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


class MathJudgeRoutingTest(unittest.TestCase):
    def test_all_math_datasets_route_to_free_response_judge(self) -> None:
        self.assertEqual(LLM_JUDGE_DATASET_SLUGS, MATH_DATASET_SLUGS)
        self.assertEqual(MATH_DATASET_SLUGS_FOR_FREE_RESPONSE, ())
        self.assertEqual(JOB_CATALOGUE["free_response"].dataset_slugs, ())
        self.assertEqual(
            JOB_CATALOGUE["free_response_judge"].dataset_slugs,
            MATH_DATASET_SLUGS,
        )
        for slug in MATH_DATASET_SLUGS:
            self.assertEqual(detect_job_from_dataset(slug, True), "free_response_judge")


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
