from __future__ import annotations

"""Launch the RWKV vLLM server used by rwkv-skills evaluations."""

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


def _bootstrap_cuda_visible_devices(argv: Sequence[str]) -> None:
    for index, arg in enumerate(argv):
        if arg == "--cuda-visible-devices" and index + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[index + 1])
            return
        if arg.startswith("--cuda-visible-devices="):
            os.environ["CUDA_VISIBLE_DEVICES"] = arg.split("=", 1)[1]
            return


_bootstrap_cuda_visible_devices(sys.argv[1:])

from src.infer.auto_config import AutoConfigMode, GpuProfile, choose_infer_auto_config, detect_visible_gpu_profile


DEFAULT_VLLM_RWKV_PATH = Path.home() / "GitHub" / "vllm-rwkv"
_AUTO_CONFIG_MODES = ("off", "balanced", "throughput")
_AUTO_CONFIG_DEFAULTS = {
    "max_num_seqs": 512,
    "max_num_batched_tokens": 16384,
    "gpu_memory_utilization": 0.9,
}
_LOG = logging.getLogger(__name__)


def _default_auto_config_mode() -> str:
    return os.environ.get("RWKV_INFER_AUTO_CONFIG", "throughput").strip().lower()


def _default_vllm_rwkv_path() -> str:
    return os.environ.get("VLLM_RWKV_PATH", str(DEFAULT_VLLM_RWKV_PATH))


def _default_vllm_python() -> str:
    return os.environ.get("VLLM_PYTHON", sys.executable)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RWKV vLLM infer service")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument(
        "--cuda-visible-devices",
        help=(
            "Process-local CUDA_VISIBLE_DEVICES value applied before vLLM loads. "
            "Use this to bind a server to one physical GPU without changing shell or .env configuration."
        ),
    )
    parser.add_argument(
        "--vllm-rwkv-path",
        default=_default_vllm_rwkv_path(),
        help="Path to the vllm-rwkv source checkout",
    )
    parser.add_argument(
        "--vllm-python",
        default=_default_vllm_python(),
        help="Python executable used to launch vllm-rwkv",
    )
    parser.add_argument("--model-name", help="Public model name exposed by the OpenAI-compatible API")
    parser.add_argument(
        "--infer-auto-config",
        "--auto-config-mode",
        dest="infer_auto_config",
        choices=_AUTO_CONFIG_MODES,
        default=_default_auto_config_mode(),
        help=(
            "Startup parameter selector for omitted vLLM tuning flags. "
            "throughput uses the highest usable tier, balanced steps one tier down, "
            "off keeps static defaults. Explicit CLI values always win."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="vLLM scheduler max active sequences; values <= 0 omit the flag",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="vLLM scheduler max batched tokens",
    )
    parser.add_argument("--max-model-len", type=int, default=4096, help="vLLM max model length")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM GPU memory utilization target",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable vLLM CUDA graph capture")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8081, help="Bind port")
    parser.add_argument("--api-key", default="", help="Optional bearer token required by the infer API")
    parser.add_argument(
        "--rwkv-wkv-mode",
        default=os.environ.get("VLLM_RWKV7_WKV_MODE", "fp16"),
        help="VLLM_RWKV7_WKV_MODE passed to vllm-rwkv",
    )
    parser.add_argument(
        "--rwkv-emb-device",
        default=os.environ.get("VLLM_RWKV7_EMB_DEVICE", "gpu"),
        help="VLLM_RWKV7_EMB_DEVICE passed to vllm-rwkv",
    )
    parser.add_argument(
        "--rwkv-rkv-mode",
        default=os.environ.get("VLLM_RWKV7_RKV_MODE", "off"),
        help="VLLM_RWKV7_RKV_MODE passed to vllm-rwkv",
    )
    parser.add_argument(
        "--rwkv-cmix-sparse",
        default=os.environ.get("VLLM_RWKV7_CMIX_SPARSE", "no-fc"),
        help="VLLM_RWKV7_CMIX_SPARSE passed to vllm-rwkv",
    )
    parser.add_argument(
        "--rwkv-low-rank-weight",
        default=os.environ.get("VLLM_RWKV7_LOW_RANK_WEIGHT", "both"),
        help="VLLM_RWKV7_LOW_RANK_WEIGHT passed to vllm-rwkv",
    )
    parser.add_argument(
        "--extra-vllm-arg",
        action="append",
        default=[],
        help="Additional raw argument appended to `vllm serve`; repeat for multiple arguments",
    )
    parser.add_argument("--log-level", default="info", help="vLLM log level")
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
            mode=mode,  # type: ignore[arg-type]
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


def build_vllm_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.vllm_python),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model_path),
        "--host",
        str(args.host),
        "--port",
        str(int(args.port)),
        "--max-model-len",
        str(int(args.max_model_len)),
        "--gpu-memory-utilization",
        str(float(args.gpu_memory_utilization)),
        "--tensor-parallel-size",
        str(int(args.tensor_parallel_size)),
    ]
    if args.model_name:
        command.extend(["--served-model-name", str(args.model_name)])
    if args.api_key:
        command.extend(["--api-key", str(args.api_key)])
    if int(args.max_num_seqs) > 0:
        command.extend(["--max-num-seqs", str(int(args.max_num_seqs))])
    if int(args.max_num_batched_tokens) > 0:
        command.extend(["--max-num-batched-tokens", str(int(args.max_num_batched_tokens))])
    if bool(args.enforce_eager):
        command.append("--enforce-eager")
    if str(args.log_level or "").strip():
        command.extend(["--uvicorn-log-level", str(args.log_level)])
    command.extend(str(item) for item in (args.extra_vllm_arg or ()))
    return command


def build_vllm_env(args: argparse.Namespace, *, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    vllm_path = str(Path(str(args.vllm_rwkv_path)).expanduser().resolve())
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = vllm_path if not pythonpath else f"{vllm_path}{os.pathsep}{pythonpath}"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["VLLM_RWKV7_WKV_MODE"] = str(args.rwkv_wkv_mode)
    env["VLLM_RWKV7_EMB_DEVICE"] = str(args.rwkv_emb_device)
    env["VLLM_RWKV7_RKV_MODE"] = str(args.rwkv_rkv_mode)
    env["VLLM_RWKV7_CMIX_SPARSE"] = str(args.rwkv_cmix_sparse)
    env["VLLM_RWKV7_LOW_RANK_WEIGHT"] = str(args.rwkv_low_rank_weight)
    return env


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    if bool(getattr(args, "infer_auto_config_applied", False)):
        _LOG.info("applied infer auto config: %s", args.infer_auto_config_reason)
    command = build_vllm_command(args)
    env = build_vllm_env(args)
    _LOG.info("launching vllm-rwkv: %s", " ".join(command))
    return int(subprocess.run(command, env=env, check=False).returncode)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
