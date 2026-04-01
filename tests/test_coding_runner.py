from __future__ import annotations

from src.eval.coding import runner as coding_runner


def test_coding_runner_parser_accepts_benchmark_kind_and_cot_mode() -> None:
    args = coding_runner.parse_args(
        [
            "--model-path",
            "model.pth",
            "--dataset",
            "dataset.jsonl",
            "--benchmark-kind",
            "mbpp",
            "--cot-mode",
            "fake_cot",
            "--probe-only",
        ]
    )
    assert args.benchmark_kind == "mbpp"
    assert args.cot_mode == "fake_cot"
    assert args.probe_only is True
