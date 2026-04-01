from __future__ import annotations

from src.bin import eval_instruction_following, instruction_following_runner


def test_instruction_following_runner_parser_accepts_compat_flags() -> None:
    args = instruction_following_runner.parse_args(
        [
            "--model-path",
            "model.pth",
            "--dataset",
            "dataset.jsonl",
            "--enable-think",
            "--no-param-search",
        ]
    )
    assert args.enable_think is True
    assert args.no_param_search is True


def test_eval_instruction_following_wrapper_forces_no_param_search(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 23

    monkeypatch.setattr(eval_instruction_following, "_instruction_following_main", fake_main)
    result = eval_instruction_following.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 23
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--no-param-search",
    ]
