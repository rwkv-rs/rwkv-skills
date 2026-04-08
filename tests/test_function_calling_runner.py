from __future__ import annotations

from pathlib import Path

from src.eval.evaluating import RunContext, RunMode
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


def test_function_calling_runner_can_infer_benchmark_kind_from_dataset_slug() -> None:
    assert (
        function_calling_runner._infer_benchmark_kind("browsecomp_test.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.BROWSECOMP
    )
    assert (
        function_calling_runner._infer_benchmark_kind("tau2_bench_airline_base.jsonl")
        is function_calling_runner.FunctionCallingBenchmarkKind.TAU2_BENCH
    )


def test_function_calling_runner_main_dispatches_to_internal_implementation(monkeypatch) -> None:
    called: list[str] = []
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.BROWSECOMP,
        dataset_path=Path("/tmp/browsecomp_test.jsonl"),
        dataset_slug="browsecomp_test",
        benchmark_name="browsecomp",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)
    monkeypatch.setattr(
        function_calling_runner,
        "_run_browsecomp",
        lambda _args, _run, *, run_context=None: called.append("browsecomp") or 0,
    )

    rc = function_calling_runner.main(["--dataset", "browsecomp_test.jsonl", "--model-path", "model.pth"])

    assert rc == 0
    assert called == ["browsecomp"]


def test_function_calling_runner_main_forwards_explicit_run_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    resolved = function_calling_runner.ResolvedFunctionCallingRun(
        benchmark_kind=function_calling_runner.FunctionCallingBenchmarkKind.MCP_BENCH,
        dataset_path=Path("/tmp/mcp_bench_test.jsonl"),
        dataset_slug="mcp_bench_test",
        benchmark_name="mcp_bench",
        dataset_split="test",
        model_name="demo-model",
        engine=None,  # type: ignore[arg-type]
    )
    run_context = RunContext(job_name="function_mcp_bench", run_mode=RunMode.RESUME)

    monkeypatch.setattr(function_calling_runner, "validate_inference_backend_args", lambda _args: None)
    monkeypatch.setattr(function_calling_runner, "_resolve_run", lambda _args: resolved)

    def _fake_run(_args, _run, *, run_context=None):
        captured["run_context"] = run_context
        return 0

    monkeypatch.setattr(function_calling_runner, "_run_mcp_bench", _fake_run)

    rc = function_calling_runner.main(
        ["--dataset", "mcp_bench_test.jsonl", "--model-path", "model.pth"],
        run_context=run_context,
    )

    assert rc == 0
    assert captured["run_context"] is run_context
