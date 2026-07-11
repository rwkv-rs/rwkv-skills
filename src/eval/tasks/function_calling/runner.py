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
from src.eval.tasks.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS
from src.eval.tasks.function_calling.rwkv_prompt import (
    DEFAULT_FUNCTION_PROMPT_STYLE,
    DEFAULT_TOOL_CATALOG_FORMAT,
    FUNCTION_PROMPT_STYLE_CHOICES,
    FUNCTION_TOOL_CATALOG_FORMAT_CHOICES,
)
from src.eval.tasks.function_calling.tool_router import (
    DEFAULT_TOOL_ROUTER_CONTEXT_CHARS,
    DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS,
    DEFAULT_TOOL_ROUTER_MAX_TOKENS,
    DEFAULT_TOOL_ROUTER_MAX_TOOLS,
    DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS,
    DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT,
    TOOL_ROUTER_MODE_CHOICES,
)
from src.eval.tasks.function_calling.parallel_candidate_router import ParallelCandidateRouterConfig
from src.eval.tasks.function_calling.agent_loop import _run_agent_loop
from src.eval.tasks.function_calling.agentbench import _run_agentbench
from src.eval.tasks.function_calling.api_bank import _run_api_bank
from src.eval.tasks.function_calling.bfcl_ast import _run_bfcl_ast
from src.eval.tasks.function_calling.bfcl_exec import _run_bfcl_exec
from src.eval.tasks.function_calling.bfcl_v3_runner import _run_bfcl_v3
from src.eval.tasks.function_calling.browsecomp import _run_browsecomp
from src.eval.tasks.function_calling.browsecomp_plus import _run_browsecomp_plus
from src.eval.tasks.function_calling.complexfuncbench import _run_complexfuncbench
from src.eval.tasks.function_calling.longbench import _run_longbench
from src.eval.tasks.function_calling.longcodebench import _run_longcodebench
from src.eval.tasks.function_calling.mcp_bench import _run_mcp_bench
from src.eval.tasks.function_calling.runner_common import (
    FunctionCallingBenchmarkKind,
    ResolvedFunctionCallingRun,
)
from src.eval.tasks.function_calling.simple_tool_call import DEFAULT_TOOL_CALL_IO, _run_simple_tool_call
from src.eval.tasks.function_calling.tau_runner import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOOL_ERRORS,
    _run_tau,
)
from src.eval.tasks.function_calling.toolalpaca import _run_toolalpaca
from src.eval.long_doc_evidence import LONG_DOC_MODE_CHOICES
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    require_completion_style_remote_protocol,
    validate_inference_backend_args,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext, TaskSpec


_SAMPLE_WORKER_ENABLED_KINDS = frozenset(
    {
        FunctionCallingBenchmarkKind.AGENT_LOOP,
        FunctionCallingBenchmarkKind.BFCL_V3,
        FunctionCallingBenchmarkKind.BROWSECOMP_PLUS,
        FunctionCallingBenchmarkKind.MCP_BENCH,
    }
)
_PARALLEL_CANDIDATE_ROUTER_DEFAULTS = ParallelCandidateRouterConfig()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV unified function-calling benchmark runner")
    parser.add_argument("--dataset", required=True, help="Prepared function-calling JSONL dataset path")
    parser.add_argument("--task-desc", help="Task description stored in the local evaluation DB")
    parser.add_argument(
        "--run-mode",
        choices=("auto", "new", "resume", "rerun", "fresh"),
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
    parser.add_argument("--judge-max-workers", type=int, help="Max concurrent workers for BrowseComp judge clients")
    parser.add_argument(
        "--browsecomp-plus-judge-mode",
        choices=("inline", "defer", "judge"),
        help=(
            "BrowseComp-Plus judge mode: inline scores during the run, defer stores judge_pending "
            "completions without a score, judge scores a previously deferred task."
        ),
    )
    parser.add_argument(
        "--browsecomp-plus-judge-task-id",
        help="Existing BrowseComp-Plus task_id to score when --browsecomp-plus-judge-mode=judge.",
    )
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
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=8,
        help=(
            "Concurrent episode workers for remote-safe function-calling runners; "
            "未实现 episode 并发的 benchmark 或本地模式会自动回退为 1"
        ),
    )
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
        "--tool-call-io",
        choices=("native", "rwkv-json"),
        default=os.environ.get("RWKV_TOOL_CALL_IO", DEFAULT_TOOL_CALL_IO),
        help=(
            "Tool-call decision transport for simple tool-call benchmarks. "
            "rwkv-json prompts the model to continue a JSON call and parses it locally; "
            "native uses OpenAI chat tools/tool_calls and must be enabled explicitly."
        ),
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
        "--candidate-router-mode",
        choices=("off", "auto", "parallel"),
        default=None,
        help=(
            "Candidate-layer router mode; auto is used by generic agent tool-call benchmarks for long contexts, "
            "parallel always splits the tool table into shards before aggregation"
        ),
    )
    parser.add_argument(
        "--candidate-router-chunk-tools",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.chunk_tools,
        help="Tool count per parallel candidate-router shard",
    )
    parser.add_argument(
        "--candidate-router-batch-size",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.batch_size,
        help="Generation batch size for candidate-router shard calls",
    )
    parser.add_argument(
        "--candidate-router-context-chars",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.context_chars,
        help="Conversation characters shown to each candidate-router shard",
    )
    parser.add_argument(
        "--candidate-router-prompt-max-chars",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.prompt_max_chars,
        help="Hard prompt character budget for each candidate-router shard prompt",
    )
    parser.add_argument(
        "--candidate-router-candidate-max-tokens",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.candidate_max_tokens,
        help="Generation token cap for each candidate-router shard",
    )
    parser.add_argument(
        "--candidate-router-aggregate-max-tokens",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.aggregate_max_tokens,
        help="Generation token cap for candidate-router aggregation",
    )
    parser.add_argument(
        "--candidate-router-max-candidates",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.max_candidates,
        help="Maximum candidate calls considered by the aggregator",
    )
    parser.add_argument(
        "--candidate-router-tool-schema-mode",
        choices=("minimal", "compact", "full"),
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.tool_schema_mode,
        help="Tool schema verbosity used inside candidate-router shard prompts",
    )
    parser.add_argument(
        "--candidate-router-evidence-chars",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.evidence_chars,
        help="Maximum evidence characters retained per candidate-router candidate",
    )
    parser.add_argument(
        "--candidate-router-policy-chars",
        type=int,
        default=_PARALLEL_CANDIDATE_ROUTER_DEFAULTS.policy_chars,
        help="Policy/system prompt excerpt characters shown to candidate-router prompts",
    )
    parser.add_argument(
        "--disable-candidate-router-grounding",
        action="store_true",
        help="Disable candidate-router identifier grounding checks",
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
    parser.add_argument(
        "--agent-loop-command-timeout-s",
        type=float,
        default=60.0,
        help="Per-command timeout for agent-loop shell sandbox executors",
    )
    parser.add_argument(
        "--agent-loop-max-output-chars",
        type=int,
        default=8000,
        help="Character cap for tool outputs fed back into the agent-loop prompt",
    )
    parser.add_argument(
        "--agent-loop-workspace-root",
        help="Root directory for agent-loop subprocess sandbox workspaces (default: system temp)",
    )
    return parser.parse_args(argv)


def _infer_benchmark_kind(dataset_arg: str) -> FunctionCallingBenchmarkKind:
    dataset_slug = infer_dataset_slug_from_path(dataset_arg)
    metadata = resolve_benchmark_metadata(dataset_slug)
    if metadata.field is not BenchmarkField.FUNCTION_CALLING:
        raise ValueError(f"dataset {dataset_slug!r} 不是 function-calling benchmark，无法用 function_calling runner 运行。")

    job_names = frozenset(metadata.scheduler_jobs)
    if "function_agent_loop" in job_names:
        return FunctionCallingBenchmarkKind.AGENT_LOOP
    if "function_agent_tool_call" in job_names:
        return FunctionCallingBenchmarkKind.AGENT_TOOL_CALL
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
    require_completion_style_remote_protocol(args, benchmark_name=f"function-calling/{benchmark_name}")
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
    _normalize_sample_worker_args(args)
    run = _resolve_run(args)
    _validate_sample_worker_benchmark(args, run)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.AGENT_LOOP:
        return _run_agent_loop(args, run, run_context=run_context)
    if run.benchmark_kind is FunctionCallingBenchmarkKind.AGENT_TOOL_CALL:
        return _run_simple_tool_call(
            args,
            run,
            default_job_name="function_agent_tool_call",
            run_context=run_context,
        )
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


def _normalize_sample_worker_args(args: argparse.Namespace) -> None:
    workers = int(getattr(args, "sample_workers", 1) or 1)
    if workers < 1:
        raise ValueError("--sample-workers must be >= 1")
    if workers == 1:
        args.sample_workers = 1
        return
    infer_base_url = str(getattr(args, "infer_base_url", "") or "").strip()
    if not infer_base_url:
        # 本地进程内 episode 并发尚未实现：非远端时回退为串行而非报错，
        # 以便 sample_workers 默认 >1 不破坏本地运行。
        print("⚠️ --sample-workers > 1 需要远端推理；当前为本地模式，回退为 1。")
        args.sample_workers = 1
        return
    args.sample_workers = workers
    args.infer_max_workers = max(workers, int(getattr(args, "infer_max_workers", 1) or 1))


def _validate_sample_worker_benchmark(args: argparse.Namespace, run: ResolvedFunctionCallingRun) -> None:
    if int(getattr(args, "sample_workers", 1) or 1) <= 1:
        return
    if run.benchmark_kind in _SAMPLE_WORKER_ENABLED_KINDS:
        return
    enabled = ", ".join(sorted(kind.value for kind in _SAMPLE_WORKER_ENABLED_KINDS))
    # 该 benchmark 尚未实现 episode 并发：回退为串行而非报错，
    # 让默认 >1 的 sample_workers 对未实现的 kind 安全降级。
    print(
        f"⚠️ --sample-workers > 1 目前仅实现于 {enabled}；"
        f"benchmark {run.benchmark_kind.value} 回退为 1。"
    )
    args.sample_workers = 1


__all__ = [
    "FunctionCallingBenchmarkKind",
    "ResolvedFunctionCallingRun",
    "main",
    "parse_args",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
