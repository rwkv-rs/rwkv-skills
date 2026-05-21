from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling.api_bank import load_api_bank_rows_from_source
from src.eval.scheduler.config import REPO_ROOT

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_tool_calls")


def api_bank_source_root() -> Path:
    override = os.environ.get("API_BANK_SOURCE_ROOT") or os.environ.get("RWKV_API_BANK_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    reference_root = REPO_ROOT / "references" / "API-Bank"
    if reference_root.exists():
        return reference_root.resolve()
    return (REPO_ROOT.parent / "API-Bank").resolve()


def api_bank_lv1_lv2_dir() -> Path:
    return api_bank_source_root() / "lv1-lv2-samples" / "level-1-given-desc"


def _prepare_api_bank_spec(dataset_name: str, output_root: Path, split: str, *, level: int) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} only provides test split")

    def _paths() -> tuple[Path, ...]:
        return (api_bank_lv1_lv2_dir(),)

    def _load() -> list[dict[str, Any]]:
        return load_api_bank_rows_from_source(api_bank_lv1_lv2_dir(), dataset_name=dataset_name, level=level)

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="local_official_api_bank",
        required_paths=_paths,
        load_local_records=_load,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("apibank_level1")
def prepare_apibank_level1_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_api_bank_spec("apibank_level1", output_root, split, level=1)


@FUNCTION_CALLING_REGISTRY.register_spec("apibank_level2")
def prepare_apibank_level2_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_api_bank_spec("apibank_level2", output_root, split, level=2)


__all__ = [
    "api_bank_lv1_lv2_dir",
    "api_bank_source_root",
    "prepare_apibank_level1_spec",
    "prepare_apibank_level2_spec",
]
