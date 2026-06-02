from __future__ import annotations

import unittest
import tomllib
from pathlib import Path
from types import SimpleNamespace

from src.eval.metrics.free_response import LLMJudge, LLMJudgeConfig
from src.eval.scheduler.dataset_utils import canonical_slug
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
        self.assertIn("semantically completely equivalent", cfg.prompt_template)
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

    def test_judge_rejects_boolean_after_think_block(self) -> None:
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

        self.assertEqual(result, [False, False])
        self.assertIsNotNone(judge.last_run_stats)
        self.assertEqual(judge.last_run_stats.parsed_count, 0)
        self.assertEqual(judge.last_run_stats.invalid_output_count, 2)

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

    def test_qwen3_judge_disables_thinking_in_vllm_chat_template(self) -> None:
        judge = LLMJudge(
            LLMJudgeConfig(
                api_key="k",
                model="Qwen/Qwen3-4B",
                max_workers=1,
                max_retries=0,
                backoff_base=0.0,
            )
        )
        fake = _FakeJudgeClient(["True"])
        judge.client = fake

        self.assertEqual(judge.judge([("q1", "ref1", "pred1")]), [True])
        self.assertEqual(
            fake.calls[0]["extra_body"],
            {"chat_template_kwargs": {"enable_thinking": False}},
        )


class MathJudgeRoutingTest(unittest.TestCase):
    def test_math_datasets_split_between_exact_and_judge(self) -> None:
        expected_judge = tuple(
            slug
            for slug in (
                canonical_slug("gsm8k_test"),
                canonical_slug("math_500_test"),
                canonical_slug("answer_judge_test"),
                canonical_slug("gaokao2023en_test"),
                canonical_slug("comp_math_24_25_test"),
                canonical_slug("minerva_math_test"),
                canonical_slug("amc23_test"),
                canonical_slug("olympiadbench_test"),
            )
            if slug in MATH_DATASET_SLUGS
        )
        expected_exact = tuple(slug for slug in MATH_DATASET_SLUGS if slug not in expected_judge)

        self.assertEqual(LLM_JUDGE_DATASET_SLUGS, expected_judge)
        self.assertEqual(MATH_DATASET_SLUGS_FOR_FREE_RESPONSE, expected_exact)
        self.assertEqual(JOB_CATALOGUE["free_response"].dataset_slugs, expected_exact)
        self.assertEqual(
            JOB_CATALOGUE["free_response_judge"].dataset_slugs,
            expected_judge,
        )

        for slug in expected_judge:
            self.assertEqual(detect_job_from_dataset(slug, True), "free_response_judge")
        for slug in expected_exact:
            self.assertEqual(detect_job_from_dataset(slug, True), "free_response")

    def test_college_math_routes_to_exact_free_response(self) -> None:
        self.assertIn(canonical_slug("college_math_test"), MATH_DATASET_SLUGS_FOR_FREE_RESPONSE)
        self.assertEqual(detect_job_from_dataset("college_math_test", True), "free_response")

    def test_free_response_generation_uses_only_eod_stop_token(self) -> None:
        data = tomllib.loads(Path("configs/_templates.toml").read_text())
        self.assertEqual(data["free_response_cot_default"]["stop_tokens"], [0])
        self.assertEqual(data["math_cot_default"]["stop_tokens"], [0])
        self.assertEqual(detect_job_from_dataset("olympiadbench_test", True), "free_response_judge")


class _FakeJudgeClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
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
