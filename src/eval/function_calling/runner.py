"""Unified function-calling runner entrypoint.

The benchmark-specific execution loops live in the sibling modules:
- browsecomp.py
- mcp_bench.py
- bfcl_v3_runner.py
- tau_runner.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.env_config import load_env_file
from src.eval.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS
from src.eval.function_calling.bfcl_v3_runner import _run_bfcl_v3
from src.eval.function_calling.browsecomp import _run_browsecomp
from src.eval.function_calling.mcp_bench import _run_mcp_bench
from src.eval.function_calling.runner_common import (
    FunctionCallingBenchmarkKind,
    ResolvedFunctionCallingRun,
)
from src.eval.function_calling.tau_runner import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOOL_ERRORS,
    _run_tau,
)
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    validate_inference_backend_args,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext, TaskSpec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV unified function-calling benchmark runner")
    parser.add_argument("--dataset", required=True, help="Prepared function-calling JSONL dataset path")
    parser.add_argument(
        "--benchmark-kind",
        choices=[kind.value for kind in FunctionCallingBenchmarkKind],
        default=FunctionCallingBenchmarkKind.AUTO.value,
        help="Explicit function-calling benchmark family (defaults to auto-detect from dataset slug)",
    )
    add_inference_backend_arguments(parser)
    parser.add_argument("--batch-size", type=int, help="Generation batch size for batched runners")
    parser.add_argument("--max-samples", type=int, help="Limit source task count before avg@k planning")
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        dest="avg_k",
        help="Override auto avg@k planning; function-calling runners accept exactly one explicit avg_k",
    )
    parser.add_argument("--db-write-queue", type=int, help="DB completion write queue max size")
    parser.add_argument("--db-close-timeout-s", type=float, default=30.0, help="DB close timeout")
    parser.add_argument("--probe-only", action="store_true", help="Run a minimal probe and skip scoring")
    parser.add_argument(
        "--history-max-chars",
        type=int,
        default=DEFAULT_HISTORY_MAX_CHARS,
        help="Clamp accumulated conversation/tool history length",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=2048, help="Clamp CoT generation length")
    parser.add_argument("--answer-max-tokens", type=int, default=1024, help="Clamp final answer generation length")
    parser.add_argument("--planning-max-tokens", type=int, default=2048, help="Clamp MCP planning generation length")
    parser.add_argument("--decision-max-tokens", type=int, help="Clamp tool/final-decision generation length")
    parser.add_argument("--final-max-tokens", type=int, default=3072, help="Clamp MCP final synthesis generation length")
    parser.add_argument("--max-rounds", type=int, default=20, help="Maximum MCP planning rounds per task")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum tau turns per task")
    parser.add_argument(
        "--max-tool-errors",
        type=int,
        default=DEFAULT_MAX_TOOL_ERRORS,
        help="Abort one tau task after this many tool-call errors",
    )
    return parser.parse_args(argv)


def _infer_benchmark_kind(dataset_arg: str) -> FunctionCallingBenchmarkKind:
    dataset_slug = infer_dataset_slug_from_path(dataset_arg)
    metadata = resolve_benchmark_metadata(dataset_slug)
    if metadata.field is not BenchmarkField.FUNCTION_CALLING:
        raise ValueError(f"dataset {dataset_slug!r} 不是 function-calling benchmark，无法用 function_calling runner 运行。")

    job_names = frozenset(metadata.scheduler_jobs)
    if "function_browsecomp" in job_names:
        return FunctionCallingBenchmarkKind.BROWSECOMP
    if "function_mcp_bench" in job_names:
        return FunctionCallingBenchmarkKind.MCP_BENCH
    if "function_bfcl_v3" in job_names:
        return FunctionCallingBenchmarkKind.BFCL_V3
    if "function_tau2_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU2_BENCH
    if "function_tau_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU_BENCH
    raise ValueError(f"dataset {dataset_slug!r} 没有已知的 function-calling scheduler job。")


def _resolve_run(args: argparse.Namespace) -> ResolvedFunctionCallingRun:
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    dataset_slug = infer_dataset_slug_from_path(str(dataset_path))
    detected_kind = _infer_benchmark_kind(str(dataset_path))
    requested_kind = FunctionCallingBenchmarkKind(args.benchmark_kind)
    if requested_kind is FunctionCallingBenchmarkKind.AUTO:
        benchmark_kind = detected_kind
    else:
        if requested_kind is not detected_kind:
            raise ValueError(
                f"dataset {dataset_slug!r} 对应 {detected_kind.value}，但收到了不匹配的 --benchmark-kind={requested_kind.value}"
            )
        benchmark_kind = requested_kind
    benchmark_name, dataset_split = split_benchmark_and_split(dataset_slug)
    engine = build_inference_backend_from_args(args)
    return ResolvedFunctionCallingRun(
        benchmark_kind=benchmark_kind,
        dataset_path=dataset_path,
        dataset_slug=dataset_slug,
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        model_name=engine.model_name,
        engine=engine,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    run_context: "RunContext | None" = None,
    task_spec: "TaskSpec | None" = None,
) -> int:
    del task_spec
    load_env_file(Path(".env"))
    args = parse_args(argv)
    validate_inference_backend_args(args)
    run = _resolve_run(args)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BROWSECOMP:
        return _run_browsecomp(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.MCP_BENCH:
        return _run_mcp_bench(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BFCL_V3:
        return _run_bfcl_v3(args, run, run_context=run_context)
    return _run_tau(args, run, run_context=run_context)


__all__ = [
    "FunctionCallingBenchmarkKind",
    "ResolvedFunctionCallingRun",
    "main",
    "parse_args",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
