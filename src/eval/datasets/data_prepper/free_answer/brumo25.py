from __future__ import annotations

from pathlib import Path
from typing import List
from collections.abc import Iterator

from ..data_utils import iter_hf_dataset, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY


def _records() -> Iterator[dict]:
    for row in iter_hf_dataset("MathArena/brumo_2025", split="train"):
        record = dict(row)
        record["expected_answer"] = record.pop("answer")
        yield record


@FREE_ANSWER_REGISTRY.register("brumo25")
def prepare_brumo25(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("brumo25 仅提供 test split")
    dataset_dir = output_root / "brumo25"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / "test.jsonl"
    write_jsonl(target, _records())
    return [target]
