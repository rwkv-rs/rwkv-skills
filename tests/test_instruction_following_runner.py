from __future__ import annotations

from src.eval.instruction_following import runner as instruction_following_runner


def test_instruction_following_runner_parser_accepts_core_flags() -> None:
    args = instruction_following_runner.parse_args(
        [
            "--model-path",
            "model.pth",
            "--dataset",
            "dataset.jsonl",
            "--enable-think",
        ]
    )
    assert args.enable_think is True
