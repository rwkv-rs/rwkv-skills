from __future__ import annotations

"""Loader for the vendored RWKV-Lightning FastAPI server."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RWKV_LIGHTNING_PATH = _REPO_ROOT / "vendor" / "rwkv-lightning"
_VENDOR_TOP_LEVEL_MODULES = ("API_servers", "infer", "model_load", "state_manager")


@dataclass(slots=True, frozen=True)
class RWKVLightningServerConfig:
    model_path: str
    model_name: str | None = None
    password: str | None = None
    rwkv_lightning_path: str | Path = DEFAULT_RWKV_LIGHTNING_PATH
    # Perf: "gpu" offloads the embedding to GPU (faster); "cpu" saves VRAM.
    # Read by the vendored model loader via the EMB_DEVICE env var, so it must be
    # set before the runtime import/model load below.
    emb_device: str = "gpu"
    high_throughput_enabled: bool = False
    high_throughput_max_active_states: int = 512
    high_throughput_prefill_batch_size: int = 8
    high_throughput_prefill_area: int = 4096
    high_throughput_prefill_cache_shape_limit: int = 64
    high_throughput_cuda_cache_budget_gb: float = 6.0
    high_throughput_clear_cuda_cache_each_request: bool = False


@dataclass(slots=True, frozen=True)
class RWKVLightningRuntime:
    root: Path
    create_app: Callable[..., object]
    inference_engine_cls: type
    load_model_and_tokenizer: Callable[[str], tuple[object, object, object, bool]]
    shutdown_state_manager: Callable[[], None]
    high_throughput_config_cls: type | None = None


@dataclass(slots=True, frozen=True)
class RWKVLightningApp:
    app: object
    cleanup: Callable[[], None]


def load_rwkv_lightning_runtime(
    rwkv_lightning_path: str | Path = DEFAULT_RWKV_LIGHTNING_PATH,
) -> RWKVLightningRuntime:
    root = Path(rwkv_lightning_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RWKV-Lightning vendor path not found: {root}")
    if not (root / "API_servers" / "fastapi_service.py").is_file():
        raise FileNotFoundError(f"RWKV-Lightning FastAPI service not found under: {root}")
    if not (root / "infer" / "inference.py").is_file():
        raise FileNotFoundError(f"RWKV-Lightning inference engine not found under: {root}")

    _prepend_current_python_bin_to_path()
    _purge_non_vendor_modules(root)
    root_text = str(root)
    sys.path[:] = [path for path in sys.path if path != root_text]
    sys.path.insert(0, root_text)

    removed_paths = _remove_conflicting_vendor_sys_paths(root)
    try:
        from API_servers.fastapi_service import create_app
        from infer.inference import InferenceEngine
        from model_load.model_loader import load_model_and_tokenizer
        from state_manager.state_pool import shutdown_state_manager
        try:
            from infer.high_throughput import HighThroughputConfig
        except (ImportError, ModuleNotFoundError):
            HighThroughputConfig = None  # type: ignore[assignment]
    finally:
        _restore_sys_paths(removed_paths)

    return RWKVLightningRuntime(
        root=root,
        create_app=create_app,
        inference_engine_cls=InferenceEngine,
        load_model_and_tokenizer=load_model_and_tokenizer,
        shutdown_state_manager=shutdown_state_manager,
        high_throughput_config_cls=HighThroughputConfig,
    )


def build_rwkv_lightning_app(config: RWKVLightningServerConfig) -> RWKVLightningApp:
    emb_device = str(config.emb_device or "gpu").strip().lower()
    if emb_device:
        # Set before the vendor import + model load so the loader reads it whether
        # it consults EMB_DEVICE at import time or at load time.
        os.environ.setdefault("EMB_DEVICE", emb_device)
    runtime = load_rwkv_lightning_runtime(config.rwkv_lightning_path)
    cwd = Path.cwd()
    try:
        os.chdir(runtime.root)
        model, tokenizer, model_args, rocm_flag = runtime.load_model_and_tokenizer(config.model_path)
    finally:
        os.chdir(cwd)
    if config.model_name:
        model_args.MODEL_NAME = str(config.model_name)
    engine = runtime.inference_engine_cls(
        model=model,
        tokenizer=tokenizer,
        args=model_args,
        rocm_flag=rocm_flag,
    )
    high_throughput_config = _build_high_throughput_config(config, runtime=runtime)
    if high_throughput_config is None:
        app = runtime.create_app(engine, password=config.password or None)
    else:
        try:
            app = runtime.create_app(
                engine,
                password=config.password or None,
                high_throughput_config=high_throughput_config,
            )
        except TypeError as exc:
            raise RuntimeError(
                "selected RWKV-Lightning create_app does not accept high_throughput_config; "
                f"use a newer --rwkv-lightning-path, current path: {runtime.root}"
            ) from exc

    def cleanup() -> None:
        runtime.shutdown_state_manager()
        engine.shutdown()

    return RWKVLightningApp(app=app, cleanup=cleanup)


def _build_high_throughput_config(
    config: RWKVLightningServerConfig,
    *,
    runtime: RWKVLightningRuntime,
) -> object | None:
    if not bool(config.high_throughput_enabled):
        return None
    cls = runtime.high_throughput_config_cls
    if cls is None:
        raise RuntimeError(
            "selected RWKV-Lightning path does not provide infer.high_throughput; "
            f"use a newer --rwkv-lightning-path, current path: {runtime.root}"
        )
    return cls(
        enabled=True,
        decode_max_batch_size=max(1, int(config.high_throughput_max_active_states)),
        prefill_area=max(1, int(config.high_throughput_prefill_area)),
        prefill_target_batch_size=max(1, int(config.high_throughput_prefill_batch_size)),
        prefill_cache_shape_limit=max(0, int(config.high_throughput_prefill_cache_shape_limit)),
        cuda_cache_budget_gb=max(0.0, float(config.high_throughput_cuda_cache_budget_gb)),
        clear_cuda_cache_each_request=bool(config.high_throughput_clear_cuda_cache_each_request),
    )


def _purge_non_vendor_modules(root: Path) -> None:
    root_text = str(root)
    for name in list(sys.modules):
        if not _is_vendor_top_level_module(name):
            continue
        module = sys.modules.get(name)
        if module is None or _module_is_from_root(module, root_text):
            continue
        del sys.modules[name]


def _remove_conflicting_vendor_sys_paths(root: Path) -> list[tuple[int, str]]:
    removed: list[tuple[int, str]] = []
    for index, raw_path in reversed(list(enumerate(sys.path))):
        if not _sys_path_has_conflicting_vendor_package(raw_path, root):
            continue
        removed.append((index, raw_path))
        del sys.path[index]
    return removed


def _restore_sys_paths(paths: list[tuple[int, str]]) -> None:
    for index, raw_path in sorted(paths):
        if raw_path in sys.path:
            continue
        sys.path.insert(min(index, len(sys.path)), raw_path)


def _sys_path_has_conflicting_vendor_package(raw_path: str, root: Path) -> bool:
    try:
        base = Path(raw_path or os.getcwd()).resolve()
    except OSError:
        return False
    root_text = str(root)
    if str(base).startswith(root_text):
        return False
    return any((base / name / "__init__.py").is_file() for name in _VENDOR_TOP_LEVEL_MODULES)


def _prepend_current_python_bin_to_path() -> None:
    candidates = [str(Path(sys.executable).parent), str(Path(sys.executable).resolve().parent)]
    try:
        import ninja  # type: ignore[import]
    except Exception:
        pass
    else:
        bin_dir = getattr(ninja, "BIN_DIR", None)
        if bin_dir:
            candidates.append(str(bin_dir))
    path_parts = [part for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    prepended = []
    for candidate in candidates:
        if candidate and candidate not in path_parts and candidate not in prepended:
            prepended.append(candidate)
    if prepended:
        os.environ["PATH"] = os.pathsep.join([*prepended, *path_parts])


def _is_vendor_top_level_module(name: str) -> bool:
    return any(name == module_name or name.startswith(f"{module_name}.") for module_name in _VENDOR_TOP_LEVEL_MODULES)


def _module_is_from_root(module: ModuleType, root_text: str) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        return str(Path(module_file).resolve()).startswith(root_text)
    except OSError:
        return False


__all__ = [
    "DEFAULT_RWKV_LIGHTNING_PATH",
    "RWKVLightningApp",
    "RWKVLightningRuntime",
    "RWKVLightningServerConfig",
    "build_rwkv_lightning_app",
    "load_rwkv_lightning_runtime",
]
