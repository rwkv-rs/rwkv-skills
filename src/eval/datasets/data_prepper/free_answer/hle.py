from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

from ..data_utils import iter_hf_dataset, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

HLE_CATEGORIES_MAP: dict[str, str] = {
    "Other": "other",
    "Humanities/Social Science": "human",
    "Math": "math",
    "Physics": "phy",
    "Computer Science/AI": "cs",
    "Biology/Medicine": "bio",
    "Chemistry": "chem",
    "Engineering": "eng",
}

HLE_REVERSE_MAP = {v: k for k, v in HLE_CATEGORIES_MAP.items()}
AVAILABLE_SPLITS = ("all", "text", *HLE_REVERSE_MAP.keys())


def _records() -> list[dict]:
    dataset = iter_hf_dataset("cais/hle", split="test")
    rows = []
    for entry in dataset:
        if entry.get("image"):
            continue
        rows.append(
            {
                "id": entry["id"],
                "problem": entry["question"],
                "expected_answer": entry["answer"],
                "answer_type": entry["answer_type"],
                "reference_solution": entry["rationale"],
                "raw_subject": entry["raw_subject"],
                "category": entry["category"],
                "author_name": entry["author_name"],
                "canary": entry["canary"],
            }
        )
    return rows


def _write_split(rows: Iterable[dict], target: Path, *, category: str | None) -> None:
    if category is None:
        write_jsonl(target, rows)
        return
    filtered = (row for row in rows if row["category"] == category)
    write_jsonl(target, filtered)


@FREE_ANSWER_REGISTRY.register("hle")
def prepare_hle(output_root: Path, split: str = "all") -> list[Path]:
    if split not in AVAILABLE_SPLITS:
        raise ValueError(f"未知的 HLE split: {split}")
    dataset_dir = output_root / "hle"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = _records()
    outputs: list[Path] = []
    if split == "all":
        # all split feeds every category plus the text-only view and needs its own slug.
        targets = ["all", "text", *HLE_REVERSE_MAP.keys()]
    else:
        targets = [split]

    for target_name in targets:
        if target_name in {"all", "text"}:
            category = None
        else:
            category = HLE_REVERSE_MAP.get(target_name)
            if category is None:
                raise ValueError(f"未知的 HLE split: {target_name}")
        path = dataset_dir / f"{target_name}.jsonl"
        _write_split(rows, path, category=category)
        outputs.append(path)

    return outputs
