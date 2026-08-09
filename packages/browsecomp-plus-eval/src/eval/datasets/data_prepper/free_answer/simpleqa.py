from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable, Iterator

from ..data_utils import dataset_cache_dir, download_file, iter_hf_dataset
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext

SIMPLEQA_TEST_CSV = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"


def _format_verified(entry: dict, idx: int) -> dict:
    metadata = dict(entry)
    return {
        "id": metadata.get("original_index", f"simpleqa_{idx}"),
        "question": entry["problem"],
        "expected_answer": entry["answer"],
        "metadata": metadata,
    }


def _format_test(row: dict, idx: int) -> dict:
    metadata_raw = row.get("metadata")
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw.replace("'", '"'))
        except json.JSONDecodeError:
            metadata = {}
    else:
        metadata = metadata_raw or {}
    return {
        "id": metadata.get("id", f"simpleqa_{idx}"),
        "question": row["problem"],
        "expected_answer": row["answer"],
        "metadata": metadata,
    }


def _records(split: str, context: DatasetPrepareContext) -> Iterator[dict]:
    if split not in {"verified", "test"}:
        raise ValueError("simpleqa 仅支持 verified 或 test split")

    if split == "verified":
        rows = list(iter_hf_dataset("codelion/SimpleQA-Verified", split="train"))
        for idx, entry in enumerate(rows):
            yield _format_verified(dict(entry), idx)
        return

    cache_dir = dataset_cache_dir(context.data_root, "simpleqa")
    csv_path = cache_dir / "simpleqa_test.csv"
    download_file(SIMPLEQA_TEST_CSV, csv_path)
    with csv_path.open("r", encoding="utf-8") as handle:
        for idx, row in enumerate(handle):
            if idx == 0:
                headers = row.strip().split(",")
                continue
            values = row.strip().split(",")
            entry = dict(zip(headers, values))
            yield _format_test(entry, idx - 1)


@FREE_ANSWER_REGISTRY.register_spec("simpleqa")
def prepare_simpleqa_spec(output_root: Path, split: str = "verified") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("simpleqa", output_root, split, load_rows=_records, source_kind="mixed_remote")
