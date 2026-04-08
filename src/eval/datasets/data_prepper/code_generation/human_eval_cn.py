from __future__ import annotations

"""Prepare HumanEval-CN (CodeGeeX HumanEval-X python split, bundled as fallback)."""

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.runtime import UrlDownloadFile, UrlFilesJsonlDatasetSpec, read_jsonl_items

DATA_URL = "https://hf-mirror.com/datasets/zai-org/humaneval-x/resolve/main/data/python/data/humaneval.jsonl"


_DEF_RE = re.compile(r"def\s+(?P<name>[\w_]+)\s*\(")


def _extract_entry_point(payload: dict[str, Any]) -> str | None:
    """Try to recover entry_point from upstream fields when missing."""
    raw = payload.get("entry_point")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    for key in ("declaration", "prompt"):
        text = payload.get(key)
        if not isinstance(text, str):
            continue
        match = _DEF_RE.search(text)
        if match:
            return match.group("name")
    return None


def _map_record(row: dict[str, Any]) -> dict[str, Any]:
    entry_point = _extract_entry_point(row)
    return {
        "task_id": row.get("task_id"),
        "prompt": row.get("prompt"),
        "canonical_solution": row.get("canonical_solution"),
        "entry_point": entry_point,
        "test": row.get("test"),
        "example_test": row.get("example_test"),
        "text": row.get("text"),
        "declaration": row.get("declaration"),
    }


@CODE_GENERATION_REGISTRY.register_spec("human_eval_cn")
def prepare_human_eval_cn_spec(output_root: Path, split: str = "test") -> UrlFilesJsonlDatasetSpec:
    if split != "test":
        raise ValueError("human_eval_cn 仅提供 test split")

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return read_jsonl_items(source_root / "humaneval.jsonl", parse_item=_map_record)

    return UrlFilesJsonlDatasetSpec(
        "human_eval_cn",
        output_root,
        split,
        files=(UrlDownloadFile(Path("humaneval.jsonl"), DATA_URL),),
        load_downloaded_records=_load,
        required_fields=("task_id", "prompt"),
    )


__all__ = ["prepare_human_eval_cn_spec"]
