from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.eval.tasks.maths.common import JudgeMode
from src.eval.tasks.maths import runner as maths_runner
from src.infer.sampling import SamplingConfig


def test_judge_transport_contract_is_secret_free_and_endpoint_sensitive() -> None:
    first = maths_runner._judge_transport_contract(
        SimpleNamespace(
            base_url="https://judge-a.example/v1/",
            timeout_s=60.0,
            backoff_base=0.5,
            api_key="must-not-leak",
        )
    )
    second = maths_runner._judge_transport_contract(
        SimpleNamespace(
            base_url="https://judge-b.example/v1",
            timeout_s=60.0,
            backoff_base=0.5,
            api_key="different-secret",
        )
    )

    assert first["base_url_sha256"] != second["base_url_sha256"]
    assert "must-not-leak" not in repr(first)
    assert "api_key" not in first


def test_maths_runner_parser_accepts_judge_mode() -> None:
    args = maths_runner.parse_args(
        [
            "--dataset",
            "dataset.jsonl",
            "--judge-mode",
            "llm",
            "--max-tokens",
            "128",
            "--probe-only",
        ]
    )
    assert args.judge_mode == "llm"
    assert args.max_tokens == 128
    assert args.probe_only is True
    assert args.run_checker is False
    assert args.primary_only is False


def test_maths_runner_parser_long_doc_flags_default_to_none() -> None:
    args = maths_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert args.prompt_max_chars is None
    assert args.long_doc_mode is None
    assert args.long_doc_min_chars is None

    args = maths_runner.parse_args(
        ["--dataset", "dataset.jsonl", "--long-doc-mode", "lexical", "--prompt-max-chars", "4096"]
    )
    assert args.long_doc_mode == "lexical"
    assert args.prompt_max_chars == 4096


def test_maths_runner_checker_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_MATH_RUN_CHECKER", raising=False)
    args = maths_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert maths_runner._should_run_checker(args) is False

    args = maths_runner.parse_args(["--dataset", "dataset.jsonl", "--run-checker"])
    assert maths_runner._should_run_checker(args) is True

    monkeypatch.setenv("RWKV_MATH_RUN_CHECKER", "1")
    args = maths_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert maths_runner._should_run_checker(args) is True


def test_maths_runner_primary_only_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_MATH_PRIMARY_ONLY", raising=False)
    args = maths_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert maths_runner._should_score_primary_only(args) is False

    args = maths_runner.parse_args(["--dataset", "dataset.jsonl", "--primary-only"])
    assert maths_runner._should_score_primary_only(args) is True

    monkeypatch.setenv("RWKV_MATH_PRIMARY_ONLY", "true")
    args = maths_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert maths_runner._should_score_primary_only(args) is True


def test_judge_error_summaries_ignores_clean_groups() -> None:
    assert (
        maths_runner._judge_error_summaries(
            {
                "strategy_a": {
                    "total": 3,
                    "invalid_output_count": 0,
                    "request_error_count": 0,
                }
            }
        )
        == []
    )


def test_judge_error_summaries_reports_bad_groups() -> None:
    assert maths_runner._judge_error_summaries(
        {
            "strategy_b": {
                "total": 5,
                "invalid_output_count": 1,
                "request_error_count": 2,
            }
        }
    ) == ["strategy_b 3/5 (invalid_output=1, request_error=2)"]


def test_llm_judge_errors_refuse_to_write_score() -> None:
    stats = {
        "strategy_c": {
            "total": 64,
            "invalid_output_count": 1,
            "request_error_count": 2,
        }
    }

    with pytest.raises(RuntimeError, match="refusing to .* write a score"):
        maths_runner._require_clean_judge_results(JudgeMode.LLM, stats)


def test_non_llm_mode_does_not_raise_for_diagnostic_judge_stats() -> None:
    maths_runner._require_clean_judge_results(
        JudgeMode.EXACT,
        {
            "strategy_a": {
                "total": 1,
                "invalid_output_count": 1,
                "request_error_count": 0,
            }
        },
    )


def test_strategy_a_full_response_allows_think_close_without_dropping_other_bad_words() -> None:
    stage = maths_runner.MathStageConfig(
        cot_prompt_template="Assistant: <think",
        final_answer_template=None,
        cot_sampling=SamplingConfig(
            bad_words=("</think>", "forbidden-literal"),
            min_think_tokens=16,
        ),
        final_sampling=None,
    )

    resolved = maths_runner._allow_full_response_think_close(stage)

    assert resolved.cot_sampling.bad_words == ("forbidden-literal",)
    assert resolved.cot_sampling.min_think_tokens == 0
    # The staged B/C source config must remain unchanged.
    assert stage.cot_sampling.bad_words == ("</think>", "forbidden-literal")
    assert stage.cot_sampling.min_think_tokens == 16


def test_strategy_a_full_response_leaves_unrelated_sampling_unchanged() -> None:
    stage = maths_runner.MathStageConfig(
        cot_prompt_template="Assistant: <think",
        final_answer_template=None,
        cot_sampling=SamplingConfig(
            bad_words=("forbidden-literal",),
            min_think_tokens=7,
        ),
        final_sampling=None,
    )

    assert maths_runner._allow_full_response_think_close(stage) is stage
