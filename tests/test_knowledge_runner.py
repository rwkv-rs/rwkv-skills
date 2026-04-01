from __future__ import annotations

from src.bin import eval_multi_choice, eval_multi_choice_cot, eval_multi_choice_fake_cot, knowledge_runner


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


def test_eval_multi_choice_wrapper_forces_no_cot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 7

    monkeypatch.setattr(eval_multi_choice, "_knowledge_main", fake_main)
    result = eval_multi_choice.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 7
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--cot-mode",
        "no_cot",
    ]


def test_eval_multi_choice_fake_cot_wrapper_forces_fake_cot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 9

    monkeypatch.setattr(eval_multi_choice_fake_cot, "_knowledge_main", fake_main)
    result = eval_multi_choice_fake_cot.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 9
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--cot-mode",
        "fake_cot",
    ]


def test_eval_multi_choice_cot_wrapper_forces_cot(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 11

    monkeypatch.setattr(eval_multi_choice_cot, "_knowledge_main", fake_main)
    result = eval_multi_choice_cot.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 11
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--cot-mode",
        "cot",
        "--no-param-search",
    ]
