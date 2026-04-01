from __future__ import annotations

"""Field-oriented function-calling runner aligned with rwkv-rs datasets."""

import argparse
import sys
from enum import Enum
from typing import Callable, Sequence

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path


class FunctionCallingBenchmarkKind(str, Enum):
    AUTO = "auto"
    BROWSECOMP = "browsecomp"
    MCP_BENCH = "mcp_bench"
    TAU_BENCH = "tau_bench"
    TAU2_BENCH = "tau2_bench"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV function-calling benchmark runner")
    parser.add_argument("--dataset", required=True, help="Prepared function-calling JSONL dataset path")
    parser.add_argument(
        "--benchmark-kind",
        choices=[kind.value for kind in FunctionCallingBenchmarkKind],
        default=FunctionCallingBenchmarkKind.AUTO.value,
        help="Explicit function-calling benchmark family (defaults to auto-detect from dataset slug)",
    )
    args, remaining = parser.parse_known_args(argv)
    args.remaining = remaining
    return args


def _strip_benchmark_kind(argv: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--benchmark-kind":
            skip_next = True
            continue
        if item.startswith("--benchmark-kind="):
            continue
        filtered.append(item)
    return filtered


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
    if "function_tau2_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU2_BENCH
    if "function_tau_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU_BENCH
    raise ValueError(f"dataset {dataset_slug!r} 没有已知的 function-calling scheduler job。")


def _resolve_main(kind: FunctionCallingBenchmarkKind) -> Callable[[Sequence[str] | None], int]:
    if kind is FunctionCallingBenchmarkKind.BROWSECOMP:
        from src.eval.function_calling.browsecomp_runner import main as target_main

        return target_main
    if kind is FunctionCallingBenchmarkKind.MCP_BENCH:
        from src.eval.function_calling.mcp_bench_runner import main as target_main

        return target_main
    from src.eval.function_calling.tau_runner import main as target_main

    return target_main


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_args)
    requested_kind = FunctionCallingBenchmarkKind(args.benchmark_kind)
    benchmark_kind = (
        _infer_benchmark_kind(args.dataset)
        if requested_kind is FunctionCallingBenchmarkKind.AUTO
        else requested_kind
    )
    forward_args = _strip_benchmark_kind(raw_args)
    target_main = _resolve_main(benchmark_kind)
    return target_main(forward_args)


__all__ = ["FunctionCallingBenchmarkKind", "main", "parse_args"]
