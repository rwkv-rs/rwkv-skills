from __future__ import annotations

import os
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALL_REGISTRY
from src.eval.function_calling.agent.adapters.browsecomp_plus import (
    DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT,
    load_browsecomp_plus_rows_from_decrypted_jsonl,
)
from src.eval.scheduler.config import REPO_ROOT

from ..data_utils import write_jsonl


def browsecomp_plus_source_root() -> Path:
    override = (
        os.environ.get("RWKV_BROWSECOMP_PLUS_SOURCE_ROOT")
        or os.environ.get("BROWSECOMP_PLUS_SOURCE_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        REPO_ROOT / "references" / "BrowseComp-Plus",
        REPO_ROOT.parent / "GitHub" / "rwkv-skills" / "references" / "BrowseComp-Plus",
        DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT,
        REPO_ROOT.parent / "BrowseComp-Plus",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / "references" / "BrowseComp-Plus").resolve()


def browsecomp_plus_dataset_path() -> Path:
    override = (
        os.environ.get("RWKV_BROWSECOMP_PLUS_DATASET")
        or os.environ.get("BROWSECOMP_PLUS_DATASET")
    )
    if override:
        return Path(override).expanduser().resolve()
    root = browsecomp_plus_source_root()
    for candidate in (
        root / "data" / "browsecomp_plus_decrypted.jsonl",
        root / "browsecomp_plus_decrypted.jsonl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (root / "data" / "browsecomp_plus_decrypted.jsonl").resolve()


@FUNCTION_CALL_REGISTRY.register("browsecomp_plus")
def prepare_browsecomp_plus(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("browsecomp_plus only provides test split")
    source_path = browsecomp_plus_dataset_path()
    if not source_path.exists():
        raise FileNotFoundError(
            f"BrowseComp-Plus decrypted dataset not found at {source_path}. "
            "Run the official decrypt script first or set RWKV_BROWSECOMP_PLUS_DATASET."
        )
    dataset_name = "browsecomp_plus"
    rows = load_browsecomp_plus_rows_from_decrypted_jsonl(
        source_path,
        official_root=browsecomp_plus_source_root(),
        dataset_name=dataset_name,
    )
    if not rows:
        raise ValueError(f"{source_path} did not yield any BrowseComp-Plus rows")
    target = output_root / dataset_name / "test.jsonl"
    write_jsonl(target, rows)
    return [target]


__all__ = [
    "browsecomp_plus_dataset_path",
    "browsecomp_plus_source_root",
    "prepare_browsecomp_plus",
]
