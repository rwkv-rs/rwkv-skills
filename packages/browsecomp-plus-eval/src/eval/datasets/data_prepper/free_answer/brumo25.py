from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

from ..data_utils import iter_hf_dataset
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec


def _records() -> Iterator[dict]:
    for row in iter_hf_dataset("MathArena/brumo_2025", split="train"):
        record = dict(row)
        record["expected_answer"] = record.pop("answer")
        yield record


def _records_for_split(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("brumo25 仅提供 test split")
    return _records()


@FREE_ANSWER_REGISTRY.register_spec("brumo25")
def prepare_brumo25_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("brumo25", output_root, split, load_rows=_records_for_split, source_kind="hf_dataset")
