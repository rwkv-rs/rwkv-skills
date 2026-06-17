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


@dataclass(slots=True, frozen=True)
class RWKVLightningRuntime:
    root: Path
    create_app: Callable[..., object]
    inference_engine_cls: type
    load_model_and_tokenizer: Callable[[str], tuple[object, object, object, bool]]
    shutdown_state_manager: Callable[[], None]


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

    _purge_non_vendor_modules(root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    from API_servers.fastapi_service import create_app
    from infer.inference import InferenceEngine
    from model_load.model_loader import load_model_and_tokenizer
    from state_manager.state_pool import shutdown_state_manager

    return RWKVLightningRuntime(
        root=root,
        create_app=create_app,
        inference_engine_cls=InferenceEngine,
        load_model_and_tokenizer=load_model_and_tokenizer,
        shutdown_state_manager=shutdown_state_manager,
    )


def build_rwkv_lightning_app(config: RWKVLightningServerConfig) -> RWKVLightningApp:
    emb_device = str(config.emb_device or "gpu").strip().lower()
    if emb_device:
        # Set before the vendor import + model load so the loader reads it whether
        # it consults EMB_DEVICE at import time or at load time.
        os.environ.setdefault("EMB_DEVICE", emb_device)
    runtime = load_rwkv_lightning_runtime(config.rwkv_lightning_path)
    model, tokenizer, model_args, rocm_flag = runtime.load_model_and_tokenizer(config.model_path)
    if config.model_name:
        model_args.MODEL_NAME = str(config.model_name)
    engine = runtime.inference_engine_cls(
        model=model,
        tokenizer=tokenizer,
        args=model_args,
        rocm_flag=rocm_flag,
    )
    app = runtime.create_app(engine, password=config.password or None)

    def cleanup() -> None:
        runtime.shutdown_state_manager()
        engine.shutdown()

    return RWKVLightningApp(app=app, cleanup=cleanup)


def _purge_non_vendor_modules(root: Path) -> None:
    root_text = str(root)
    for name in list(sys.modules):
        if not _is_vendor_top_level_module(name):
            continue
        module = sys.modules.get(name)
        if module is None or _module_is_from_root(module, root_text):
            continue
        del sys.modules[name]


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
