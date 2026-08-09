from __future__ import annotations

"""Probe a remote inference endpoint without importing the full scheduler CLI."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.eval.scheduler.remote_profiler import DEFAULT_REMOTE_PROBE_PROMPT, probe_remote_inference, write_remote_probe_result
from src.infer.backend import REMOTE_INFERENCE_PROTOCOL_CHOICES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe remote inference concurrency")
    parser.add_argument("--infer-base-url", "--base-url", required=True, help="Remote inference base URL")
    parser.add_argument("--infer-model", "--model", required=True, help="Remote model name")
    parser.add_argument("--infer-api-key", "--api-key", default="", help="Remote inference bearer token")
    parser.add_argument("--infer-timeout-s", "--timeout-s", type=float, default=600.0, help="Request timeout")
    parser.add_argument(
        "--infer-protocol",
        "--protocol",
        choices=REMOTE_INFERENCE_PROTOCOL_CHOICES,
        default="vllm",
        help="Remote inference protocol",
    )
    parser.add_argument("--candidates", default="1,2,4,8,16,32,64", help="Comma-separated concurrency candidates")
    parser.add_argument("--prompt", default=DEFAULT_REMOTE_PROBE_PROMPT, help="Probe prompt")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max generated tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Sampling top-p")
    parser.add_argument("--top-k", type=int, default=50, help="Sampling top-k")
    parser.add_argument("--stop-suffix", help="Optional text stop suffix for every prompt")
    parser.add_argument("--gpu-index", type=int, help="Optional local GPU index to sample with NVML")
    parser.add_argument("--target-gpu-utilization", type=float, default=90.0, help="GPU utilization target")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Warmup requests excluded from the curve")
    parser.add_argument("--max-p95-latency-s", type=float, help="Reject candidate points above this p95 latency")
    parser.add_argument(
        "--min-throughput-gain",
        type=float,
        default=0.03,
        help="Smallest relative throughput gain worth keeping when selecting the concurrency knee",
    )
    parser.add_argument("--output-json", help="Optional JSON summary path")
    return parser.parse_args(argv)


def parse_int_csv(value: str) -> tuple[int, ...]:
    items: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed <= 0:
            raise ValueError("concurrency candidates must be positive")
        items.append(parsed)
    if not items:
        raise ValueError("at least one concurrency candidate is required")
    return tuple(items)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        candidates = parse_int_csv(str(args.candidates))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    result = probe_remote_inference(
        base_url=str(args.infer_base_url),
        model=str(args.infer_model),
        api_key=str(args.infer_api_key or ""),
        timeout_s=float(args.infer_timeout_s),
        protocol=str(args.infer_protocol),  # type: ignore[arg-type]
        candidates=candidates,
        prompt=str(args.prompt),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        stop_suffix=args.stop_suffix,
        gpu_index=args.gpu_index,
        target_gpu_utilization=float(args.target_gpu_utilization),
        warmup_requests=int(args.warmup_requests),
        max_p95_latency_s=args.max_p95_latency_s,
        min_throughput_gain=float(args.min_throughput_gain),
    )
    payload = result.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if args.output_json:
        write_remote_probe_result(Path(args.output_json).expanduser(), result)
    return 0


__all__ = ["main", "parse_args", "parse_int_csv"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
