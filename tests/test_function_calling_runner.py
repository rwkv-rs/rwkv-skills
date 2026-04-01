from __future__ import annotations

from src.eval.function_calling import runner as function_calling_runner


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
