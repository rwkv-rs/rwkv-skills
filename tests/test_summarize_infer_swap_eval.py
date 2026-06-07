from __future__ import annotations

from src.bin import summarize_infer_swap_eval


def test_summarize_infer_swap_eval_parse_args_accepts_probe_json() -> None:
    args = summarize_infer_swap_eval.parse_args(
        [
            "--model",
            "demo",
            "--datasets",
            "mcp_bench_test",
            "--probe-json",
            "/tmp/probe.json",
            "--output-json",
            "/tmp/summary.json",
        ]
    )

    assert args.model == "demo"
    assert args.datasets == ["mcp_bench_test"]
    assert args.probe_json == "/tmp/probe.json"
