from __future__ import annotations

import csv
from pathlib import Path
from typing import List
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

DATA_URL = "https://raw.githubusercontent.com/joyheyueya/declarative-math-word-problem/main/algebra222.csv"


def _records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("algebra222 仅提供 test split")
    cache_dir = dataset_cache_dir(Path("data"), "algebra222")
    source_path = cache_dir / "algebra222.csv"
    download_file(DATA_URL, source_path)

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            answer = float(row["final_answer"])
            if int(answer) == answer:
                answer = int(answer)
            yield {
                "problem": row["question"],
                "expected_answer": answer,
            }


@FREE_ANSWER_REGISTRY.register("algebra222")
def prepare_algebra222(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "algebra222"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _records(split))
    return [target]
