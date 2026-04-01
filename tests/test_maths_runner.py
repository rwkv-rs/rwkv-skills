from __future__ import annotations

from src.eval.maths import runner as maths_runner


def test_maths_runner_parser_accepts_judge_mode() -> None:
    args = maths_runner.parse_args(
        [
            "--model-path",
            "model.pth",
            "--dataset",
            "dataset.jsonl",
            "--judge-mode",
            "llm",
            "--probe-only",
        ]
    )
    assert args.judge_mode == "llm"
    assert args.probe_only is True
