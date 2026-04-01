from __future__ import annotations

from src.eval.knowledge import runner as knowledge_runner


def test_knowledge_runner_parser_accepts_all_modes() -> None:
    args = knowledge_runner.parse_args(
        [
            "--model-path",
            "model.pth",
            "--dataset",
            "dataset.jsonl",
            "--cot-mode",
            "cot",
            "--probe-only",
        ]
    )
    assert args.cot_mode == "cot"
    assert args.probe_only is True
