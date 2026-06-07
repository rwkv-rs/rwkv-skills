"""Unified function-calling runner entrypoint.

The benchmark-specific execution loops live in the sibling modules:
- browsecomp.py
- mcp_bench.py
- bfcl_v3_runner.py
- tau_runner.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.env_config import load_env_file
from src.eval.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS
from src.eval.function_calling.rwkv_prompt import (
    DEFAULT_FUNCTION_PROMPT_STYLE,
    DEFAULT_TOOL_CATALOG_FORMAT,
    FUNCTION_PROMPT_STYLE_CHOICES,
    FUNCTION_TOOL_CATALOG_FORMAT_CHOICES,
)
from src.eval.function_calling.tool_router import (
    DEFAULT_TOOL_ROUTER_CONTEXT_CHARS,
    DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS,
    DEFAULT_TOOL_ROUTER_MAX_TOKENS,
    DEFAULT_TOOL_ROUTER_MAX_TOOLS,
    DEFAULT_TOOL_ROUTER_PARALLEL_BATCH_SIZE,
    DEFAULT_TOOL_ROUTER_PARALLEL_CHUNK_TOOLS,
    DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS,
    DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT,
    TOOL_ROUTER_MODE_CHOICES,
)
from src.eval.function_calling.agentbench import _run_agentbench
from src.eval.function_calling.api_bank import _run_api_bank
from src.eval.function_calling.bfcl_ast import _run_bfcl_ast
from src.eval.function_calling.bfcl_exec import _run_bfcl_exec
from src.eval.function_calling.bfcl_v3_runner import _run_bfcl_v3
from src.eval.function_calling.browsecomp import _run_browsecomp
from src.eval.function_calling.browsecomp_plus import _run_browsecomp_plus
from src.eval.function_calling.complexfuncbench import _run_complexfuncbench
from src.eval.function_calling.longbench import _run_longbench
from src.eval.function_calling.longcodebench import _run_longcodebench
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
from src.eval.function_calling.toolalpaca import _run_toolalpaca
from src.eval.long_doc_evidence import (
    DEFAULT_LONG_DOC_MODEL_MAX_TOKENS,
    DEFAULT_LONG_DOC_MODEL_PARALLEL_BATCH_SIZE,
    LONG_DOC_MODE_CHOICES,
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
    parser.add_argument("--task-desc", help="Task description stored in the local evaluation DB")
    parser.add_argument(
        "--run-mode",
        choices=("auto", "new", "resume", "rerun"),
        help="Task persistence mode; mirrors RWKV_EVAL_RUN_MODE without requiring shell env injection",
    )
    parser.add_argument(
        "--tau-bench-root",
        help="Official tau2/tau3-bench repository root; mirrors RWKV_TAU3_BENCH_ROOT",
    )
    parser.add_argument(
        "--tau-llm-timeout-s",
        type=float,
        help="Timeout for official tau user/judge LLM calls; mirrors RWKV_TAU_LLM_TIMEOUT_S",
    )
    parser.add_argument("--user-model", help="Model name for official tau user simulator")
    parser.add_argument("--user-api-key", help="API key for official tau user simulator")
    parser.add_argument("--user-base-url", help="OpenAI-compatible base URL for official tau user simulator")
    parser.add_argument("--judge-model", help="Model name for official tau NL assertion judge")
    parser.add_argument("--judge-api-key", help="API key for official tau NL assertion judge")
    parser.add_argument("--judge-base-url", help="OpenAI-compatible base URL for official tau NL assertion judge")
    parser.add_argument("--judge-max-workers", type=int, help="Reserved for judge clients that support worker pools")
    parser.add_argument(
        "--disable-checker",
        action="store_true",
        help="Disable optional checker hooks; mirrors RWKV_SKILLS_DISABLE_CHECKER=1",
    )
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
        "--prompt-style",
        choices=FUNCTION_PROMPT_STYLE_CHOICES,
        default=DEFAULT_FUNCTION_PROMPT_STYLE,
        help="Function-calling prompt serialization style",
    )
    parser.add_argument(
        "--tool-catalog-format",
        choices=FUNCTION_TOOL_CATALOG_FORMAT_CHOICES,
        default=DEFAULT_TOOL_CATALOG_FORMAT,
        help="Function-calling tool catalog serialization format",
    )
    parser.add_argument(
        "--history-max-chars",
        type=int,
        default=DEFAULT_HISTORY_MAX_CHARS,
        help="Clamp accumulated conversation/tool history length",
    )
    parser.add_argument(
        "--prompt-max-chars",
        type=int,
        help="Hard prompt character budget for long-context agent runners (env-specific defaults may apply)",
    )
    parser.add_argument(
        "--long-doc-mode",
        choices=LONG_DOC_MODE_CHOICES,
        default="lexical",
        help="Long-message compaction mode for long-context agent runners",
    )
    parser.add_argument(
        "--tool-router-mode",
        choices=TOOL_ROUTER_MODE_CHOICES,
        default="off",
        help="Select a per-turn tool window before rendering long-context agent prompts",
    )
    parser.add_argument(
        "--tool-router-max-tools",
        type=int,
        default=DEFAULT_TOOL_ROUTER_MAX_TOOLS,
        help="Maximum environment tools exposed after tool routing",
    )
    parser.add_argument(
        "--tool-router-trigger-tool-count",
        type=int,
        default=DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT,
        help="Only route when the environment exposes at least this many tools",
    )
    parser.add_argument(
        "--tool-router-trigger-catalog-chars",
        type=int,
        default=DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS,
        help="Only route when the full tool catalog has at least this many serialized characters",
    )
    parser.add_argument(
        "--tool-router-context-chars",
        type=int,
        default=DEFAULT_TOOL_ROUTER_CONTEXT_CHARS,
        help="Recent conversation characters shown to the tool router",
    )
    parser.add_argument(
        "--tool-router-max-tokens",
        type=int,
        default=DEFAULT_TOOL_ROUTER_MAX_TOKENS,
        help="Generation token cap for model-based tool routing",
    )
    parser.add_argument(
        "--tool-router-description-chars",
        type=int,
        default=DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS,
        help="Description character cap per tool in the router catalog",
    )
    parser.add_argument(
        "--tool-router-parallel-chunk-tools",
        type=int,
        default=DEFAULT_TOOL_ROUTER_PARALLEL_CHUNK_TOOLS,
        help="Tool count per model_parallel router shard",
    )
    parser.add_argument(
        "--tool-router-parallel-batch-size",
        type=int,
        default=DEFAULT_TOOL_ROUTER_PARALLEL_BATCH_SIZE,
        help="Batch size for model_parallel router shard calls",
    )
    parser.add_argument("--long-doc-max-chars", type=int, default=1000, help="Long-document chunk max characters")
    parser.add_argument("--long-doc-overlap-lines", type=int, default=3, help="Long-document chunk overlap lines")
    parser.add_argument(
        "--long-doc-min-chars",
        type=int,
        default=6000,
        help="Only compact individual messages at or above this character count",
    )
    parser.add_argument(
        "--long-doc-max-evidence-chunks",
        type=int,
        default=4,
        help="Maximum selected chunks when compacting one long message",
    )
    parser.add_argument(
        "--long-doc-max-evidence-chars",
        type=int,
        default=6000,
        help="Maximum selected evidence characters when compacting one long message",
    )
    parser.add_argument(
        "--long-doc-model-max-tokens",
        type=int,
        default=DEFAULT_LONG_DOC_MODEL_MAX_TOKENS,
        help="Generation token cap for model_parallel long-document chunk routing",
    )
    parser.add_argument(
        "--long-doc-model-parallel-batch-size",
        type=int,
        default=DEFAULT_LONG_DOC_MODEL_PARALLEL_BATCH_SIZE,
        help="Batch size for model_parallel long-document chunk routing",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=2048, help="Clamp CoT generation length")
    parser.add_argument("--answer-max-tokens", type=int, default=1024, help="Clamp final answer generation length")
    parser.add_argument("--planning-max-tokens", type=int, default=2048, help="Clamp MCP planning generation length")
    parser.add_argument("--decision-max-tokens", type=int, help="Clamp tool/final-decision generation length")
    parser.add_argument("--final-max-tokens", type=int, default=3072, help="Clamp MCP final synthesis generation length")
    parser.add_argument("--max-rounds", type=int, default=20, help="Maximum MCP planning rounds per task")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum tau turns per task")
    parser.add_argument(
        "--skip-runtime-preflight",
        action="store_true",
        help="Skip benchmark-specific sandbox/runtime preflight checks.",
    )
    parser.add_argument(
        "--max-tool-errors",
        type=int,
        default=DEFAULT_MAX_TOOL_ERRORS,
        help="Abort one tau task after this many tool-call errors",
    )
    parser.add_argument(
        "--complexfuncbench-disable-response-eval",
        action="store_true",
        help="Disable official ComplexFuncBench GPT response evaluation and score only sandbox/tool-call matching.",
    )
    parser.add_argument(
        "--complexfuncbench-offline-compare",
        action="store_true",
        help="Disable official ComplexFuncBench RapidAPI/GPT equivalence checks and use offline rule/value matching.",
    )
    parser.add_argument(
        "--tau-retail-repeated-read-guard",
        action="store_true",
        help="Stop retail tau tasks when the agent repeats an already-successful read call.",
    )
    parser.add_argument(
        "--tau-retail-tool-use-guard",
        action="store_true",
        help="Route obvious retail ID/type mismatches to the safer read tool in tau ablations.",
    )
    parser.add_argument(
        "--tau-retail-progressive-tool-disclosure",
        action="store_true",
        help="Expose retail tau tools in procedural stages instead of showing read/detail/write tools at once.",
    )
    parser.add_argument(
        "--agentbench-controller-url",
        help="AgentBench/AgentRL controller API URL, default AGENTBENCH_CONTROLLER_URL or http://127.0.0.1:5020/api",
    )
    return parser.parse_args(argv)


def _infer_benchmark_kind(dataset_arg: str) -> FunctionCallingBenchmarkKind:
    dataset_slug = infer_dataset_slug_from_path(dataset_arg)
    metadata = resolve_benchmark_metadata(dataset_slug)
    if metadata.field is not BenchmarkField.FUNCTION_CALLING:
        raise ValueError(f"dataset {dataset_slug!r} 不是 function-calling benchmark，无法用 function_calling runner 运行。")

    job_names = frozenset(metadata.scheduler_jobs)
    if "function_browsecomp_plus" in job_names:
        return FunctionCallingBenchmarkKind.BROWSECOMP_PLUS
    if "function_browsecomp" in job_names:
        return FunctionCallingBenchmarkKind.BROWSECOMP
    if "function_longbench" in job_names:
        return FunctionCallingBenchmarkKind.LONGBENCH
    if "function_longcodebench" in job_names:
        return FunctionCallingBenchmarkKind.LONGCODEBENCH
    if "function_mcp_bench" in job_names:
        return FunctionCallingBenchmarkKind.MCP_BENCH
    if "function_api_bank" in job_names:
        return FunctionCallingBenchmarkKind.API_BANK
    if "function_agentbench" in job_names:
        return FunctionCallingBenchmarkKind.AGENTBENCH
    if "function_bfcl_ast" in job_names:
        return FunctionCallingBenchmarkKind.BFCL_AST
    if "function_bfcl_exec" in job_names:
        return FunctionCallingBenchmarkKind.BFCL_EXEC
    if "function_bfcl_v3" in job_names:
        return FunctionCallingBenchmarkKind.BFCL_V3
    if "function_toolalpaca" in job_names:
        return FunctionCallingBenchmarkKind.TOOLALPACA
    if "function_complexfuncbench" in job_names:
        return FunctionCallingBenchmarkKind.COMPLEXFUNCBENCH
    if "function_tau3_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU3_BENCH
    if "function_tau2_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU2_BENCH
    if "function_tau_bench" in job_names:
        return FunctionCallingBenchmarkKind.TAU_BENCH
    raise ValueError(f"dataset {dataset_slug!r} 没有已知的 function-calling scheduler job。")


def _resolve_run(args: argparse.Namespace) -> ResolvedFunctionCallingRun:
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False, record_stats=not bool(args.probe_only))
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
    _apply_runner_env_overrides(args)
    validate_inference_backend_args(args)
    run = _resolve_run(args)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BROWSECOMP_PLUS:
        return _run_browsecomp_plus(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BROWSECOMP:
        return _run_browsecomp(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.LONGBENCH:
        return _run_longbench(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.LONGCODEBENCH:
        return _run_longcodebench(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.MCP_BENCH:
        return _run_mcp_bench(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.API_BANK:
        return _run_api_bank(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.AGENTBENCH:
        return _run_agentbench(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BFCL_V3:
        return _run_bfcl_v3(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BFCL_AST:
        return _run_bfcl_ast(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.BFCL_EXEC:
        return _run_bfcl_exec(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.TOOLALPACA:
        return _run_toolalpaca(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.COMPLEXFUNCBENCH:
        return _run_complexfuncbench(args, run, run_context=run_context)
    return _run_tau(args, run, run_context=run_context)


def _apply_runner_env_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "task_desc", None):
        os.environ["RWKV_TASK_DESC"] = str(args.task_desc)
    if getattr(args, "run_mode", None):
        os.environ["RWKV_EVAL_RUN_MODE"] = str(args.run_mode)
    if getattr(args, "tau_bench_root", None):
        os.environ["RWKV_TAU3_BENCH_ROOT"] = str(args.tau_bench_root)
    timeout_s = getattr(args, "tau_llm_timeout_s", None)
    if timeout_s is not None:
        os.environ["RWKV_TAU_LLM_TIMEOUT_S"] = str(float(timeout_s))
    if bool(getattr(args, "disable_checker", False)):
        os.environ["RWKV_SKILLS_DISABLE_CHECKER"] = "1"


__all__ = [
    "FunctionCallingBenchmarkKind",
    "ResolvedFunctionCallingRun",
    "main",
    "parse_args",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
