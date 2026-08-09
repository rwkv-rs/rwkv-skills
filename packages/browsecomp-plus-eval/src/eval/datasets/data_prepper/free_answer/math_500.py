from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext

DATA_URL = "https://github.com/openai/prm800k/raw/main/prm800k/math_splits/test.jsonl"


def _records(split: str, context: DatasetPrepareContext) -> Iterator[dict]:
    if split != "test":
        raise ValueError("math-500 仅提供 test split")
    cache_dir = dataset_cache_dir(context.data_root, "math_500")
    source_path = cache_dir / "math_500_test.jsonl"
    download_file(DATA_URL, source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload["expected_answer"] = payload.pop("answer")
            payload["reference_solution"] = payload.pop("solution")
            yield payload


@FREE_ANSWER_REGISTRY.register_spec("math_500")
def prepare_math_500_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("math-500", output_root, split, load_rows=_records, source_kind="url_download")
