from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext

DATA_URL = "https://raw.githubusercontent.com/joyheyueya/declarative-math-word-problem/main/algebra222.csv"


def _records(split: str, context: DatasetPrepareContext) -> Iterator[dict]:
    if split != "test":
        raise ValueError("algebra222 仅提供 test split")
    cache_dir = dataset_cache_dir(context.data_root, "algebra222")
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


@FREE_ANSWER_REGISTRY.register_spec("algebra222")
def prepare_algebra222_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("algebra222", output_root, split, load_rows=_records, source_kind="url_download")
