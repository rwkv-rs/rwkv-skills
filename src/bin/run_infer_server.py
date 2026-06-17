from __future__ import annotations

"""Launch the standalone RWKV infer service."""

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence, cast

import uvicorn

from src.infer.auto_config import AutoConfigMode, GpuProfile, choose_infer_auto_config, detect_visible_gpu_profile
from src.infer.backend import InferenceBackend, LocalInferenceBackend
from src.infer.model import ModelLoadConfig
from src.infer.rwkv_lightning_server import (
    DEFAULT_RWKV_LIGHTNING_PATH,
    RWKVLightningServerConfig,
    build_rwkv_lightning_app,
)
from src.infer.server import create_app
from src.infer.service import InferenceService


_AUTO_CONFIG_MODES = ("off", "balanced", "throughput")
_ENGINE_MODE_CHOICES = ("rwkv-lightning", "lightning", "classic")
_AUTO_CONFIG_DEFAULTS = {
    "max_batch_size": 32,
    "batch_collect_ms": 5,
    "max_num_seqs": 512,
    "max_num_batched_tokens": 16384,
    "rwkv_prefill_token_budget": 2048,
    "rwkv_prefill_max_batch_size": 128,
    "max_state_slots": -1,
    "gpu_memory_utilization": 0.9,
}
_LOG = logging.getLogger(__name__)


def _default_auto_config_mode() -> str:
    return os.environ.get("RWKV_INFER_AUTO_CONFIG", "throughput").strip().lower()


def _default_engine_mode() -> str:
    value = os.environ.get("RWKV_INFER_ENGINE_MODE", "rwkv-lightning").strip().lower()
    if value in _ENGINE_MODE_CHOICES:
        return value
    return "rwkv-lightning"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone RWKV infer service")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compatibility flag; nano-vLLM selects CUDA devices from CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--engine-mode",
        choices=_ENGINE_MODE_CHOICES,
        default=_default_engine_mode(),
        help="Inference backend implementation; rwkv-lightning is the formal server backend",
    )
    parser.add_argument(
        "--state-db-path",
        help="Path to the local sqlite state cache database used by the lightning engine",
    )
    parser.add_argument(
        "--rwkv-lightning-path",
        default=os.environ.get("RWKV_LIGHTNING_PATH", str(DEFAULT_RWKV_LIGHTNING_PATH)),
        help="Path to the vendored RWKV-Lightning server source",
    )
    parser.add_argument("--model-name", help="Public model name exposed by the infer API")
    parser.add_argument(
        "--infer-auto-config",
        "--auto-config-mode",
        dest="infer_auto_config",
        choices=_AUTO_CONFIG_MODES,
        default=_default_auto_config_mode(),
        help=(
            "Startup parameter selector for omitted infer tuning flags. "
            "throughput uses the highest usable tier, balanced steps one tier down, "
            "off keeps legacy defaults. Explicit CLI values always win."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="nano-vLLM scheduler max active sequences",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="nano-vLLM scheduler max batched tokens",
    )
    parser.add_argument("--max-model-len", type=int, default=4096, help="nano-vLLM max model length")
    parser.add_argument(
        "--rwkv-prefill-token-budget",
        type=int,
        default=None,
        help="nano-vLLM RWKV prefill token budget",
    )
    parser.add_argument(
        "--rwkv-prefill-max-batch-size",
        type=int,
        default=None,
        help="nano-vLLM RWKV prefill max batch size",
    )
    parser.add_argument(
        "--rwkv-prefill-chunk-size",
        type=int,
        default=-1,
        help="nano-vLLM RWKV prefill chunk size; -1 lets the engine use its default",
    )
    parser.add_argument(
        "--rwkv-state-cache-enable",
        action="store_true",
        help="Enable nano-vLLM RWKV state cache",
    )
    parser.add_argument("--max-state-slots", type=int, default=None, help="nano-vLLM max RWKV state slots")
    parser.add_argument(
        "--rwkv-state-cache-safety-reserve-slots",
        type=int,
        default=0,
        help="nano-vLLM RWKV state cache reserve slots",
    )
    parser.add_argument(
        "--sampling-bucket-temperature-resolution",
        type=float,
        default=0.0,
        help="nano-vLLM sampler temperature bucket resolution",
    )
    parser.add_argument(
        "--sampling-bucket-top-p-resolution",
        type=float,
        default=0.0,
        help="nano-vLLM sampler top-p bucket resolution",
    )
    parser.add_argument("--rwkv-quant-int8", action="store_true", help="Enable nano-vLLM RWKV int8 quantization")
    parser.add_argument(
        "--rwkv-int8-fp16-lm-head",
        action="store_true",
        help="Use fp16 lm_head with nano-vLLM int8 quantization",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="nano-vLLM GPU memory utilization target",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="nano-vLLM tensor parallel size")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable nano-vLLM CUDA graph capture")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8081, help="Bind port")
    parser.add_argument("--api-key", default="", help="Optional bearer token required by the infer API")
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Max number of queued requests per infer batch",
    )
    parser.add_argument(
        "--batch-collect-ms",
        type=int,
        default=None,
        help="How long the worker waits to collect a batch",
    )
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    args = parser.parse_args(argv)
    return apply_startup_auto_config(args)


def apply_startup_auto_config(
    args: argparse.Namespace,
    *,
    gpu_profile: GpuProfile | None = None,
) -> argparse.Namespace:
    mode = str(args.infer_auto_config).strip().lower()
    if mode not in _AUTO_CONFIG_MODES:
        raise ValueError(f"invalid infer auto config mode: {args.infer_auto_config!r}")

    omitted_fields = [field for field in _AUTO_CONFIG_DEFAULTS if getattr(args, field) is None]
    if mode != "off" and omitted_fields:
        profile = gpu_profile if gpu_profile is not None else detect_visible_gpu_profile()
        config = choose_infer_auto_config(
            model_path=str(args.model_path),
            model_name=str(args.model_name) if args.model_name else None,
            gpu_profile=profile,
            mode=cast(AutoConfigMode, mode),
        )
        for field in omitted_fields:
            setattr(args, field, getattr(config, field))
        args.infer_auto_config_applied = True
        args.infer_auto_config_reason = config.reason
        return args

    for field, default in _AUTO_CONFIG_DEFAULTS.items():
        if getattr(args, field) is None:
            setattr(args, field, default)
    args.infer_auto_config_applied = False
    args.infer_auto_config_reason = "disabled" if mode == "off" else "all managed fields explicitly set"
    return args


def build_backend(args: argparse.Namespace) -> InferenceBackend:
    engine_mode = str(args.engine_mode or "lightning").strip().lower()
    if engine_mode == "rwkv-lightning":
        raise ValueError("rwkv-lightning is a standalone server mode; call main() to launch it")
    if engine_mode in {"lightning", "classic"}:
        return LocalInferenceBackend.from_model_config(
            ModelLoadConfig(
                weights_path=str(args.model_path),
                device=str(args.device or "cuda"),
            ),
            engine_mode=engine_mode,
            state_db_path=None if args.state_db_path in (None, "") else str(args.state_db_path),
        )
    raise ValueError(f"unsupported engine mode: {args.engine_mode!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(getattr(args, "infer_auto_config_applied", False)):
        logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
        _LOG.info("applied infer auto config: %s", args.infer_auto_config_reason)
    if str(args.engine_mode).strip().lower() == "rwkv-lightning":
        server = build_rwkv_lightning_app(
            RWKVLightningServerConfig(
                model_path=str(args.model_path),
                model_name=str(args.model_name) if args.model_name else None,
                password=str(args.api_key) if args.api_key else None,
                rwkv_lightning_path=str(args.rwkv_lightning_path),
            )
        )
        try:
            uvicorn.run(
                server.app,
                host=args.host,
                port=int(args.port),
                log_level=str(args.log_level),
                access_log=False,
            )
        finally:
            server.cleanup()
        return 0
    backend = build_backend(args)
    service = InferenceService(
        backend,
        max_batch_size=args.max_batch_size,
        batch_collect_ms=args.batch_collect_ms,
    )
    app = create_app(service, api_key=args.api_key)
    uvicorn.run(
        app,
        host=args.host,
        port=int(args.port),
        log_level=str(args.log_level),
        access_log=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
