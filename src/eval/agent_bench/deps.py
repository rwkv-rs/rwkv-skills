from __future__ import annotations

"""Dynamic dependency/path helpers for TAU official runtimes."""

import importlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
TAU_V2_VENDOR_ROOT = ROOT / "data" / "tau_v2"
TAU_V2_DATA_ROOT = TAU_V2_VENDOR_ROOT / "data"
TAU_V2_REFERENCE_ROOT = REPO_ROOT / "references" / "tau2-bench"

T = TypeVar("T")


def tau_v2_vendor_root() -> Path:
    override = (
        os.environ.get("RWKV_TAU3_BENCH_ROOT")
        or os.environ.get("TAU3_BENCH_ROOT")
        or os.environ.get("RWKV_TAU2_BENCH_ROOT")
        or os.environ.get("TAU2_BENCH_ROOT")
    )
    if override:
        root = Path(override).expanduser().resolve()
        src_root = root / "src"
        if (src_root / "tau2").exists():
            return src_root
        return root
    reference_src = TAU_V2_REFERENCE_ROOT / "src"
    if (reference_src / "tau2").exists():
        return reference_src
    return TAU_V2_VENDOR_ROOT


def tau_v2_data_root() -> Path:
    override = (
        os.environ.get("RWKV_TAU3_DATA_ROOT")
        or os.environ.get("TAU3_DATA_ROOT")
        or os.environ.get("RWKV_TAU2_DATA_ROOT")
        or os.environ.get("TAU2_DATA_DIR")
    )
    if override:
        return Path(override).expanduser().resolve()
    vendor_root = tau_v2_vendor_root()
    if vendor_root.name == "src":
        return vendor_root.parent / "data"
    reference_data = TAU_V2_REFERENCE_ROOT / "data"
    if (reference_data / "tau2").exists():
        return reference_data
    return TAU_V2_DATA_ROOT


def ensure_tau_v2_vendor_path() -> Path:
    vendor_root = tau_v2_vendor_root()
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    os.environ.setdefault("TAU2_DATA_DIR", str(tau_v2_data_root()))
    return vendor_root


def import_module_with_auto_install(module_name: str, *, context: str = ""):
    if module_name == "tau2" or module_name.startswith("tau2."):
        ensure_tau_v2_vendor_path()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        label = f" for {context}" if context else ""
        raise ModuleNotFoundError(
            f"Missing module {module_name!r}{label}. Set RWKV_TAU2_BENCH_ROOT/TAU2_BENCH_ROOT "
            "or RWKV_TAU3_BENCH_ROOT/TAU3_BENCH_ROOT to the official tau2/tau3-bench checkout."
        ) from exc


def run_with_auto_install(func: Callable[[], T], *, context: str = "") -> T:
    try:
        return func()
    except ModuleNotFoundError as exc:
        label = f" for {context}" if context else ""
        raise ModuleNotFoundError(f"Missing TAU runtime dependency{label}: {exc}") from exc


__all__ = [
    "ROOT",
    "REPO_ROOT",
    "TAU_V2_DATA_ROOT",
    "TAU_V2_REFERENCE_ROOT",
    "TAU_V2_VENDOR_ROOT",
    "ensure_tau_v2_vendor_path",
    "import_module_with_auto_install",
    "run_with_auto_install",
    "tau_v2_data_root",
    "tau_v2_vendor_root",
]
