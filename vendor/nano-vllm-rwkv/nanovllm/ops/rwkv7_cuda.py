import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_FAST_EXTENSIONS_LOADED: set[str] = set()
_FAST_EXTENSION_SENTINELS = {
    "rwkv7_v3a_ops": "linear_f16_orig",
    "rwkv7_fast_ops_fp16": "cmix_mix",
    "rwkv7_wkv_fp16_v2": "wkv_seq",
    "rwkv7_wkv_fp32_v2": "forward",
}


def _cuda_flags() -> list[str]:
    return [
        "-O3",
        "--use_fast_math",
        "--extra-device-vectorization",
    ] + (["-Xptxas", "-O3"] if os.name != "nt" else [])


def _ensure_torch_cuda_arch_list() -> None:
    if os.getenv("NANOVLLM_KEEP_TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def _extension_op_registered(name: str) -> bool:
    sentinel = _FAST_EXTENSION_SENTINELS.get(name)
    if sentinel is None:
        return False
    return hasattr(getattr(torch.ops, name), sentinel)


def _load_extension_once(name: str, sources: list[str], extra_cuda_cflags: list[str] | None = None):
    if name in _FAST_EXTENSIONS_LOADED and _extension_op_registered(name):
        return
    if _extension_op_registered(name):
        _FAST_EXTENSIONS_LOADED.add(name)
        return
    _ensure_torch_cuda_arch_list()
    load(
        name=name,
        sources=sources,
        is_python_module=False,
        verbose=False,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags or _cuda_flags(),
    )
    _FAST_EXTENSIONS_LOADED.add(name)


def ensure_faster3a_loaded(wkv_mode: str | None = None):
    """Load the required Albatross faster3a CUDA op surface."""
    if wkv_mode is None:
        wkv_mode = os.getenv("NANOVLLM_RWKV7_WKV_MODE", "fp16")
    cur = Path(__file__).resolve().parent
    cuda_dir = cur / "cuda"
    _load_extension_once(
        "rwkv7_v3a_ops",
        [str(cuda_dir / "rwkv7_v3a_ops.cpp"), str(cuda_dir / "rwkv7_v3a_ops.cu")],
    )
    _load_extension_once(
        "rwkv7_fast_ops_fp16",
        [str(cuda_dir / "rwkv7_fast_ops_fp16.cpp"), str(cuda_dir / "rwkv7_fast_ops_fp16.cu")],
    )
    if wkv_mode == "fp16":
        _load_extension_once(
            "rwkv7_wkv_fp16_v2",
            [str(cuda_dir / "rwkv7_wkv_fp16_v2.cpp"), str(cuda_dir / "rwkv7_wkv_fp16_v2.cu")],
            extra_cuda_cflags=[
                "-O3",
                "-res-usage",
                "--extra-device-vectorization",
            ] + (["-Xptxas", "-O3"] if os.name != "nt" else []),
        )
    elif wkv_mode == "fp32io16":
        _load_extension_once(
            "rwkv7_wkv_fp32_v2",
            [str(cuda_dir / "rwkv7_wkv_fp32_v2.cpp"), str(cuda_dir / "rwkv7_wkv_fp32_v2.cu")],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-D_IO_FP16_",
            ] + (["-Xptxas", "-O3"] if os.name != "nt" else []),
        )
    else:
        raise ValueError(f"unknown NANOVLLM_RWKV7_WKV_MODE={wkv_mode!r}")
