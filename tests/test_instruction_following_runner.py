from __future__ import annotations

import pytest

from src.eval.instruction_following import runner as instruction_following_runner
from src.eval.metrics.instruction_following.metrics import evaluate_instruction_following


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


def test_ifbench_uses_official_rule_registry(tmp_path) -> None:
    dataset = tmp_path / "ifbench" / "test.jsonl"
    dataset.parent.mkdir()
    dataset.write_text(
        '{"key": 0, "prompt": "Answer without whitespace.", '
        '"instruction_id_list": ["format:no_whitespace"], '
        '"kwargs": [{"N": null}]}\n'
    )

    metrics = evaluate_instruction_following(
        [{"sample_index": 0, "repeat_index": 0, "completion1": "NoSpaces"}],
        dataset_path=dataset,
        dataset_slug="ifbench_test",
        strict=False,
    )

    assert metrics.samples == 1
    assert metrics.prompt_accuracy == 1.0
    assert metrics.instruction_accuracy == 1.0
    assert metrics.tier1_accuracy["format:no_whitespace"] == 1.0
