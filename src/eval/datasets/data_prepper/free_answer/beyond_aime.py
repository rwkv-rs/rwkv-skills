from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast
from collections.abc import Callable

from ..data_utils import iter_hf_dataset, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY


def _records() -> Iterator[dict[str, object]]:
    for row in iter_hf_dataset("ByteDance-Seed/BeyondAIME", split="test"):
        record: dict[str, object] = dict(row)
        record["expected_answer"] = record.pop("answer")
        yield record


@FREE_ANSWER_REGISTRY.register("beyond-aime")
def prepare_beyond_aime(output_root: Path, split: str = "test") -> list[Path]:
    if split != "test":
        raise ValueError("beyond-aime 仅提供 test split")
    dataset_dir = output_root / "beyond-aime"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / "test.jsonl"
    writer = cast(Callable[[Path, Iterable[dict[str, object]]], Path], write_jsonl)
    written = writer(target, _records())
    return [written]
