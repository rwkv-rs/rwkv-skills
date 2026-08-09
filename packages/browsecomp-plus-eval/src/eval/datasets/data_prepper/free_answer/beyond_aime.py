from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from ..data_utils import iter_hf_dataset
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec


def _records() -> Iterator[dict[str, object]]:
    for row in iter_hf_dataset("ByteDance-Seed/BeyondAIME", split="test"):
        record: dict[str, object] = dict(row)
        record["expected_answer"] = record.pop("answer")
        yield record


def _records_for_split(split: str) -> Iterable[dict[str, object]]:
    if split != "test":
        raise ValueError("beyond-aime 仅提供 test split")
    return _records()


@FREE_ANSWER_REGISTRY.register_spec("beyond_aime")
def prepare_beyond_aime_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("beyond-aime", output_root, split, load_rows=_records_for_split, source_kind="hf_dataset")
