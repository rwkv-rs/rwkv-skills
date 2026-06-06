from __future__ import annotations

import pytest

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


def test_instruction_following_runner_rejects_data_only_benchmarks() -> None:
    with pytest.raises(ValueError, match="does not have a rule-based instruction-following scorer"):
        instruction_following_runner._ensure_rule_based_dataset("flores200_devtest")
