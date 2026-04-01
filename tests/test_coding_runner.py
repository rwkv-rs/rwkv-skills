from __future__ import annotations

from src.bin import coding_runner, eval_code_human_eval, eval_code_livecodebench, eval_code_mbpp


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


def test_eval_code_human_eval_wrapper_forces_human_eval_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 13

    monkeypatch.setattr(eval_code_human_eval, "_coding_main", fake_main)
    result = eval_code_human_eval.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 13
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--benchmark-kind",
        "human_eval",
        "--cot-mode",
        "no_cot",
    ]


def test_eval_code_mbpp_wrapper_forces_mbpp_kind(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 17

    monkeypatch.setattr(eval_code_mbpp, "_coding_main", fake_main)
    result = eval_code_mbpp.main(["--model-path", "m.pth", "--dataset", "d.jsonl", "--cot-mode", "cot"])
    assert result == 17
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--cot-mode",
        "cot",
        "--benchmark-kind",
        "mbpp",
    ]


def test_eval_code_livecodebench_wrapper_forces_livecodebench_mode(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 19

    monkeypatch.setattr(eval_code_livecodebench, "_coding_main", fake_main)
    result = eval_code_livecodebench.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 19
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--benchmark-kind",
        "livecodebench",
        "--cot-mode",
        "cot",
    ]
