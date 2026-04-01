from __future__ import annotations

from src.bin import eval_free_response, eval_free_response_judge, maths_runner


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


def test_eval_free_response_wrapper_forces_exact_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 5

    monkeypatch.setattr(eval_free_response, "_maths_main", fake_main)
    result = eval_free_response.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 5
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--judge-mode",
        "exact",
    ]


def test_eval_free_response_judge_wrapper_forces_llm_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 6

    monkeypatch.setattr(eval_free_response_judge, "_maths_main", fake_main)
    result = eval_free_response_judge.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 6
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--judge-mode",
        "llm",
    ]
