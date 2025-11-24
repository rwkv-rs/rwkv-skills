from __future__ import annotations

"""High-level entrypoints for数据集准备流程。

对外暴露两个能力：
1. `available_*_datasets` —— 返回各个任务家族在本地已实现的准备器名。
2. `prepare_dataset` —— 通过注册表找到对应函数、写出 JSONL，并在必要时记录 perf log。

内部会懒加载每个 family 目录下的 preparer 模块，避免 import 时就执行昂贵操作。
"""

import importlib
import pkgutil
import time
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - avoid heavy deps in tests
    from src.server.perf_logging import perf_logger  # type: ignore
except Exception:  # pragma: no cover
    class _NoopPerfLogger:
        enabled = False

        def log(self, *_args, **_kwargs) -> None:
            pass

    perf_logger = _NoopPerfLogger()

from . import common
from .prepper_registry import (
    CODE_GENERATION_REGISTRY,
    DatasetPreparer,
    FREE_ANSWER_REGISTRY,
    INSTRUCTION_FOLLOWING_REGISTRY,
    MULTIPLE_CHOICE_REGISTRY,
)

_FAMILY_MODULES = (
    "multiple_choice",
    "free_answer",
    "instruction_following",
    "code_generation",
)
_CORE_PACKAGE = __name__.rsplit(".", 1)[0]
_PACKAGE_ROOT = Path(__file__).resolve().parent
_REGISTRIES_INITIALIZED = False


def _import_family_modules(family: str) -> None:
    package_path = _PACKAGE_ROOT / family
    if not package_path.exists():
        return
    prefix = f"{_CORE_PACKAGE}.{family}"
    for module_info in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"{prefix}.{module_info.name}")


def _ensure_registries() -> None:
    global _REGISTRIES_INITIALIZED
    if _REGISTRIES_INITIALIZED:
        return
    for family in _FAMILY_MODULES:
        _import_family_modules(family)
    _REGISTRIES_INITIALIZED = True


def available_multiple_choice_datasets() -> Iterable[str]:
    """列出当前实现的多选数据集别名（用于 CLI 补全 / 校验）。"""
    _ensure_registries()
    return MULTIPLE_CHOICE_REGISTRY.names()


def available_free_answer_datasets() -> Iterable[str]:
    """列出可准备的自由问答数据集（math/翻译等）。"""
    _ensure_registries()
    return FREE_ANSWER_REGISTRY.names()


def available_instruction_following_datasets() -> Iterable[str]:
    """列出 IFEval/ifbench 等指令遵循类数据集。"""
    _ensure_registries()
    return INSTRUCTION_FOLLOWING_REGISTRY.names()


def available_code_generation_datasets() -> Iterable[str]:
    """列出 EvalPlus 风格的代码生成数据集。"""
    _ensure_registries()
    return CODE_GENERATION_REGISTRY.names()


def prepare_dataset(name: str, output_root: Path, split: str = "test") -> list[Path]:
    """执行指定数据集的准备函数，确保输出路径在 repo/data 下规范化。

    会根据 registry 自动定位 preparer，并附带 perf logging。返回值是写出的
    所有 JSONL 文件路径（有些数据集一个名字可能生成多个 split）。
    """
    _ensure_registries()
    key = name.lower()
    preparer = (
        MULTIPLE_CHOICE_REGISTRY.get(key)
        or FREE_ANSWER_REGISTRY.get(key)
        or INSTRUCTION_FOLLOWING_REGISTRY.get(key)
        or CODE_GENERATION_REGISTRY.get(key)
    )
    if preparer is None:
        raise ValueError(f"未知的 NeMo benchmark 数据集: {name}")

    output_root = output_root.expanduser().resolve()
    start = time.perf_counter()
    try:
        paths = preparer(output_root, split)
    except Exception as exc:
        if perf_logger.enabled:
            perf_logger.log(
                "dataset_prepare",
                dataset=key,
                split=split,
                output_root=str(output_root),
                elapsed_s=time.perf_counter() - start,
                status="error",
                error=str(exc),
            )
        raise
    if perf_logger.enabled:
        perf_logger.log(
            "dataset_prepare",
            dataset=key,
            split=split,
            output_root=str(output_root),
            elapsed_s=time.perf_counter() - start,
            status="ok",
            outputs=[str(path) for path in paths],
        )
    return paths


__all__ = [
    "available_multiple_choice_datasets",
    "available_free_answer_datasets",
    "available_instruction_following_datasets",
    "available_code_generation_datasets",
    "prepare_dataset",
    "common",
]
