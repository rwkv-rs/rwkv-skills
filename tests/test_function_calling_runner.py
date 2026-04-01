from __future__ import annotations

from src.bin import (
    eval_function_browsecomp,
    eval_function_mcp_bench,
    eval_function_tau_bench,
    function_calling_runner,
)


def test_function_calling_runner_parser_accepts_benchmark_kind() -> None:
    args = function_calling_runner.parse_args(
        [
            "--dataset",
            "browsecomp_test.jsonl",
            "--benchmark-kind",
            "mcp_bench",
            "--model-path",
            "model.pth",
        ]
    )
    assert args.dataset == "browsecomp_test.jsonl"
    assert args.benchmark_kind == "mcp_bench"


def test_eval_function_browsecomp_wrapper_forces_browsecomp_kind(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 29

    monkeypatch.setattr(eval_function_browsecomp, "_function_calling_main", fake_main)
    result = eval_function_browsecomp.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 29
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--benchmark-kind",
        "browsecomp",
    ]


def test_eval_function_mcp_bench_wrapper_forces_mcp_kind(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 31

    monkeypatch.setattr(eval_function_mcp_bench, "_function_calling_main", fake_main)
    result = eval_function_mcp_bench.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 31
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
        "--benchmark-kind",
        "mcp_bench",
    ]


def test_eval_function_tau_bench_wrapper_preserves_auto_detection(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_main(argv):
        captured["argv"] = list(argv)
        return 37

    monkeypatch.setattr(eval_function_tau_bench, "_function_calling_main", fake_main)
    result = eval_function_tau_bench.main(["--model-path", "m.pth", "--dataset", "d.jsonl"])
    assert result == 37
    assert captured["argv"] == [
        "--model-path",
        "m.pth",
        "--dataset",
        "d.jsonl",
    ]
