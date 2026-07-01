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
