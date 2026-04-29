from __future__ import annotations

"""Run service-based performance benchmarks against OpenAI-compatible infer servers."""

import argparse
import sys
from pathlib import Path
from typing import Sequence

from src.eval.performance.config import load_perf_config_defaults
from src.eval.performance.layout import default_performance_result_path
from src.eval.performance.runner import ServiceBenchmarkConfig, run_service_benchmark, write_benchmark_result
from src.eval.performance.service_client import OpenAIChatServiceClient
from src.eval.performance.tokenizers import load_benchmark_tokenizer
from src.eval.performance.vllm_launcher import (
    DEFAULT_VLLM_COMMAND,
    VllmLaunchConfig,
    launch_vllm_server,
    parse_shell_args,
)
from src.eval.performance.workload import parse_int_csv


def _arg_default(defaults: dict[str, object], key: str, fallback: object) -> object:
    return defaults.get(key, fallback)


def _build_parser(defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    resolved_defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Run service-based inference performance benchmarks")
    parser.add_argument("--config", help="Perf config path (.toml) or config name under configs/perf/")
    parser.add_argument(
        "--base-url",
        default=_arg_default(resolved_defaults, "base_url", None),
        help="Base URL of the OpenAI-compatible infer service",
    )
    parser.add_argument(
        "--model",
        default=_arg_default(resolved_defaults, "model", None),
        help="Model name exposed by the infer service; defaults to first listed model",
    )
    parser.add_argument(
        "--api-key",
        default=_arg_default(resolved_defaults, "api_key", ""),
        help="Optional API key for the infer service",
    )
    parser.add_argument(
        "--protocol",
        choices=("openai-chat",),
        default=_arg_default(resolved_defaults, "protocol", "openai-chat"),
        help="Service protocol",
    )
    parser.add_argument(
        "--stack-name",
        default=_arg_default(resolved_defaults, "stack_name", None),
        help="Display name for the full inference stack",
    )
    parser.add_argument(
        "--engine-name",
        default=_arg_default(resolved_defaults, "engine_name", None),
        help="Engine label for reporting",
    )
    parser.add_argument(
        "--precision",
        default=_arg_default(resolved_defaults, "precision", "unknown"),
        help="Precision / quantization label for reporting",
    )
    parser.add_argument(
        "--tokenizer-type",
        choices=("rwkv", "hf"),
        default=_arg_default(resolved_defaults, "tokenizer_type", None),
        help="Tokenizer type",
    )
    parser.add_argument(
        "--tokenizer-ref",
        default=_arg_default(resolved_defaults, "tokenizer_ref", None),
        help="Tokenizer vocab path or HF model/tokenizer reference",
    )
    parser.add_argument(
        "--ctx-lens",
        default=_arg_default(resolved_defaults, "ctx_lens", "512,1024,2048,4096,8192"),
        help="Comma-separated input token lengths",
    )
    parser.add_argument(
        "--concurrency-levels",
        default=_arg_default(resolved_defaults, "concurrency_levels", "1,2,4,8,16,32"),
        help="Comma-separated concurrency levels (service-side batching pressure)",
    )
    parser.add_argument(
        "--batch-sizes",
        default=_arg_default(resolved_defaults, "batch_sizes", None),
        help="Comma-separated batch-size grid; defaults to concurrency levels",
    )
    parser.add_argument(
        "--skip-concurrency-matrix",
        action="store_true",
        default=bool(_arg_default(resolved_defaults, "skip_concurrency_matrix", False)),
        help="Skip the ctx_len x concurrency matrix",
    )
    parser.add_argument(
        "--skip-batch-size-matrix",
        action="store_true",
        default=bool(_arg_default(resolved_defaults, "skip_batch_size_matrix", False)),
        help="Skip the ctx_len x batch_size matrix",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=_arg_default(resolved_defaults, "output_tokens", 128),
        help="Fixed output tokens per request",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=_arg_default(resolved_defaults, "warmup_runs", 1),
        help="Warmup runs per point",
    )
    parser.add_argument(
        "--measure-runs",
        type=int,
        default=_arg_default(resolved_defaults, "measure_runs", 3),
        help="Measured runs per point",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=_arg_default(resolved_defaults, "temperature", 1.0),
        help="Generation temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=_arg_default(resolved_defaults, "top_p", 1.0),
        help="Generation top-p",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=_arg_default(resolved_defaults, "timeout_s", 600.0),
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_arg_default(resolved_defaults, "seed", None),
        help="Optional deterministic base seed",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=_arg_default(resolved_defaults, "gpu_index", None),
        help="Optional local GPU index for NVML VRAM sampling",
    )
    parser.add_argument(
        "--hardware-label",
        default=_arg_default(resolved_defaults, "hardware_label", None),
        help="Optional human-readable hardware label",
    )
    parser.add_argument(
        "--result-path",
        default=_arg_default(resolved_defaults, "result_path", None),
        help="Optional output JSON path",
    )
    parser.add_argument(
        "--launch-vllm",
        action="store_true",
        default=bool(_arg_default(resolved_defaults, "launch_vllm", False)),
        help="Launch a local vLLM OpenAI-compatible server for the benchmark",
    )
    parser.add_argument(
        "--vllm-command",
        default=_arg_default(resolved_defaults, "vllm_command", " ".join(DEFAULT_VLLM_COMMAND)),
        help="Command used to start the vLLM OpenAI server",
    )
    parser.add_argument(
        "--vllm-python",
        default=_arg_default(resolved_defaults, "vllm_python", None),
        help="Python executable used to launch vLLM; preferred over --vllm-command for cross-env usage",
    )
    parser.add_argument(
        "--vllm-host",
        default=_arg_default(resolved_defaults, "vllm_host", "127.0.0.1"),
        help="Host bound by the launched vLLM server",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=_arg_default(resolved_defaults, "vllm_port", 8000),
        help="Port bound by the launched vLLM server",
    )
    parser.add_argument(
        "--vllm-dtype",
        default=_arg_default(resolved_defaults, "vllm_dtype", None),
        help="Optional vLLM --dtype value",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=_arg_default(resolved_defaults, "vllm_tensor_parallel_size", None),
        help="Optional vLLM --tensor-parallel-size",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=_arg_default(resolved_defaults, "vllm_gpu_memory_utilization", None),
        help="Optional vLLM --gpu-memory-utilization value",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=_arg_default(resolved_defaults, "vllm_max_model_len", None),
        help="Optional vLLM --max-model-len",
    )
    parser.add_argument(
        "--vllm-max-num-seqs",
        type=int,
        default=_arg_default(resolved_defaults, "vllm_max_num_seqs", None),
        help="Optional vLLM --max-num-seqs",
    )
    parser.add_argument(
        "--vllm-max-num-batched-tokens",
        type=int,
        default=_arg_default(resolved_defaults, "vllm_max_num_batched_tokens", None),
        help="Optional vLLM --max-num-batched-tokens",
    )
    parser.add_argument(
        "--vllm-trust-remote-code",
        action="store_true",
        default=bool(_arg_default(resolved_defaults, "vllm_trust_remote_code", False)),
        help="Pass --trust-remote-code to vLLM",
    )
    parser.add_argument(
        "--vllm-extra-args",
        default=_arg_default(resolved_defaults, "vllm_extra_args", None),
        help="Additional vLLM CLI args, split with shell-style quoting",
    )
    parser.add_argument(
        "--vllm-startup-timeout-s",
        type=float,
        default=_arg_default(resolved_defaults, "vllm_startup_timeout_s", 600.0),
        help="Timeout for launched vLLM server readiness",
    )
    parser.add_argument(
        "--vllm-log-path",
        default=_arg_default(resolved_defaults, "vllm_log_path", None),
        help="Optional path for launched vLLM logs",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    defaults = load_perf_config_defaults(bootstrap_args.config) if bootstrap_args.config else {}
    parser = _build_parser(defaults)
    return parser.parse_args(argv)


def _resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return str(args.base_url)
    if args.launch_vllm:
        return f"http://{args.vllm_host}:{int(args.vllm_port)}"
    raise ValueError("必须提供 --base-url，或启用 --launch-vllm")


def _resolve_model(args: argparse.Namespace, *, base_url: str) -> str:
    if args.model:
        return str(args.model)
    client = OpenAIChatServiceClient(
        base_url=base_url,
        model="",
        api_key=str(args.api_key or ""),
        timeout_s=float(args.timeout_s),
    )
    names = client.list_models()
    if not names:
        raise ValueError("服务未返回可用模型，请显式传入 --model")
    return names[0]


def _resolve_tokenizer_args(
    args: argparse.Namespace,
    *,
    default_model: str | None = None,
) -> tuple[str, str | None]:
    tokenizer_type = str(args.tokenizer_type or "").strip()
    if not tokenizer_type:
        if args.launch_vllm:
            tokenizer_type = "hf"
        else:
            raise ValueError("必须提供 --tokenizer-type")

    tokenizer_ref = None if args.tokenizer_ref in (None, "") else str(args.tokenizer_ref)
    if tokenizer_type == "hf" and tokenizer_ref is None:
        model_name = str(args.model or default_model or "").strip()
        if not model_name:
            raise ValueError("HF tokenizer 模式需要提供 --tokenizer-ref，或至少显式传入 --model")
        tokenizer_ref = model_name
    return tokenizer_type, tokenizer_ref


def _resolve_level_grid(raw: str | None, *, fallback: str | None = None, enabled: bool = True) -> tuple[int, ...]:
    if not enabled:
        return ()
    text = str(raw or fallback or "").strip()
    return parse_int_csv(text)


def _build_vllm_launch_config(args: argparse.Namespace) -> VllmLaunchConfig:
    model_name = str(args.model or "").strip()
    if not model_name:
        raise ValueError("启用 --launch-vllm 时必须提供 --model")
    explicit_python = None if args.vllm_python in (None, "") else str(args.vllm_python)
    parsed_command = parse_shell_args(args.vllm_command)
    use_current_python = explicit_python is None and (
        not parsed_command or tuple(parsed_command) == DEFAULT_VLLM_COMMAND
    )
    return VllmLaunchConfig(
        model=model_name,
        host=str(args.vllm_host),
        port=int(args.vllm_port),
        api_key=str(args.api_key or ""),
        python_executable=explicit_python or (sys.executable if use_current_python else None),
        command=() if (explicit_python or use_current_python) else parsed_command,
        dtype=None if args.vllm_dtype in (None, "") else str(args.vllm_dtype),
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        trust_remote_code=bool(args.vllm_trust_remote_code),
        extra_args=parse_shell_args(args.vllm_extra_args),
        startup_timeout_s=float(args.vllm_startup_timeout_s),
        log_path=Path(args.vllm_log_path).expanduser() if args.vllm_log_path else None,
    )


def _run_benchmark(args: argparse.Namespace) -> int:
    if args.skip_concurrency_matrix and args.skip_batch_size_matrix:
        raise ValueError("至少需要保留一种测试矩阵")

    base_url = _resolve_base_url(args)
    model_name = _resolve_model(args, base_url=base_url)
    tokenizer_type, tokenizer_ref = _resolve_tokenizer_args(args, default_model=model_name)
    tokenizer = load_benchmark_tokenizer(
        tokenizer_type=tokenizer_type,
        tokenizer_ref=tokenizer_ref,
    )
    engine_name = args.engine_name or ("vllm" if args.launch_vllm else "openai-compatible-service")
    stack_name = args.stack_name or f"{model_name}+{engine_name}+{args.precision}"
    concurrency_levels = _resolve_level_grid(
        args.concurrency_levels,
        enabled=not bool(args.skip_concurrency_matrix),
    )
    batch_sizes = _resolve_level_grid(
        args.batch_sizes,
        fallback=args.concurrency_levels,
        enabled=not bool(args.skip_batch_size_matrix),
    )

    result = run_service_benchmark(
        ServiceBenchmarkConfig(
            base_url=base_url,
            model=model_name,
            api_key=str(args.api_key or ""),
            protocol=str(args.protocol),
            stack_name=stack_name,
            engine_name=str(engine_name),
            precision=str(args.precision),
            tokenizer_label=tokenizer.label,
            tokenizer=tokenizer,
            ctx_lens=parse_int_csv(str(args.ctx_lens)),
            concurrency_levels=concurrency_levels,
            batch_sizes=batch_sizes,
            output_tokens=int(args.output_tokens),
            warmup_runs=max(0, int(args.warmup_runs)),
            measure_runs=max(1, int(args.measure_runs)),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            timeout_s=float(args.timeout_s),
            base_seed=None if args.seed is None else int(args.seed),
            gpu_index=None if args.gpu_index is None else int(args.gpu_index),
            service_metadata={
                "base_url": base_url,
                "model": model_name,
                "protocol": str(args.protocol),
                "launch_vllm": bool(args.launch_vllm),
            },
            hardware_metadata={
                "hardware_label": args.hardware_label or "",
                "gpu_index": args.gpu_index,
            },
        )
    )

    result_path = (
        Path(args.result_path).expanduser()
        if args.result_path
        else default_performance_result_path(
            model_name=model_name,
            protocol=str(args.protocol),
            stack_name=stack_name,
        )
    )
    write_benchmark_result(result_path, result)
    print(f"性能 benchmark 结果已写入: {result_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.launch_vllm:
        return _run_benchmark(args)

    with launch_vllm_server(_build_vllm_launch_config(args)):
        return _run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
