from __future__ import annotations

import os
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALL_REGISTRY
from src.eval.function_calling.agent.adapters.apibank import (
    load_apibank_level2_rows_from_source_dir,
)
from src.eval.function_calling.one_step.apibank import (
    DEFAULT_OFFICIAL_APIBANK_ROOT,
    load_api_bank_rows_from_source,
    load_apibank_level1_rows_from_source_dir,
)
from src.eval.scheduler.config import REPO_ROOT

from ..data_utils import write_jsonl


def apibank_source_root() -> Path:
    override = (
        os.environ.get("API_BANK_SOURCE_ROOT")
        or os.environ.get("RWKV_API_BANK_SOURCE_ROOT")
        or os.environ.get("RWKV_APIBANK_SOURCE_ROOT")
        or os.environ.get("APIBANK_SOURCE_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        REPO_ROOT / "references" / "API-Bank",
        REPO_ROOT.parent / "API-Bank",
        DEFAULT_OFFICIAL_APIBANK_ROOT,
    ):
        if candidate.exists():
            return candidate.expanduser().resolve()
    return DEFAULT_OFFICIAL_APIBANK_ROOT.expanduser().resolve()


def _prepare_apibank_l1(output_root: Path, dataset_name: str) -> list[Path]:
    root = apibank_source_root()
    source_dir = root / "lv1-lv2-samples" / "level-1-given-desc"
    rows = load_apibank_level1_rows_from_source_dir(
        source_dir,
        official_root=root,
        dataset_name=dataset_name,
    )
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


def _prepare_apibank_level(output_root: Path, dataset_name: str, *, level: int) -> list[Path]:
    root = apibank_source_root()
    source_dir = root / "lv1-lv2-samples" / "level-1-given-desc"
    rows = load_api_bank_rows_from_source(source_dir, dataset_name=dataset_name, level=level)
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


def _prepare_apibank_l2(output_root: Path, dataset_name: str) -> list[Path]:
    root = apibank_source_root()
    source_dir = root / "lv1-lv2-samples" / "level-2-toolsearcher"
    rows = load_apibank_level2_rows_from_source_dir(
        source_dir,
        official_root=root,
        dataset_name=dataset_name,
    )
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


@FUNCTION_CALL_REGISTRY.register("apibank_level1")
def prepare_apibank_level1(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("apibank_level1 only provides test split")
    return _prepare_apibank_level(output_root, "apibank_level1", level=1)


@FUNCTION_CALL_REGISTRY.register("apibank_level2")
def prepare_apibank_level2(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("apibank_level2 only provides test split")
    return _prepare_apibank_level(output_root, "apibank_level2", level=2)


@FUNCTION_CALL_REGISTRY.register("apibank_l1")
def prepare_apibank_l1(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("apibank_l1 only provides test split")
    return _prepare_apibank_l1(output_root, "apibank_l1")


@FUNCTION_CALL_REGISTRY.register("apibank_l2")
def prepare_apibank_l2(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("apibank_l2 only provides test split")
    return _prepare_apibank_l2(output_root, "apibank_l2")


__all__ = [
    "apibank_source_root",
    "prepare_apibank_level1",
    "prepare_apibank_level2",
    "prepare_apibank_l1",
    "prepare_apibank_l2",
]
