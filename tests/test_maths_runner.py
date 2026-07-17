from __future__ import annotations

from src.eval.tasks.maths import runner as maths_runner


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
