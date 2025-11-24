from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

from ..data_utils import dataset_cache_dir, download_file, iter_hf_dataset, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

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


@FREE_ANSWER_REGISTRY.register("simpleqa")
def prepare_simpleqa(output_root: Path, split: str = "verified") -> list[Path]:
    if split not in {"verified", "test"}:
        raise ValueError("simpleqa 仅支持 verified 或 test split")

    dataset_dir = output_root / "simpleqa"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"

    if split == "verified":
        rows = list(iter_hf_dataset("codelion/SimpleQA-Verified", split="train"))
        formatted = [_format_verified(dict(entry), idx) for idx, entry in enumerate(rows)]
        write_jsonl(target, formatted)
    else:
        cache_dir = dataset_cache_dir(Path("data"), "simpleqa")
        csv_path = cache_dir / "simpleqa_test.csv"
        download_file(SIMPLEQA_TEST_CSV, csv_path)
        formatted: list[dict] = []
        with csv_path.open("r", encoding="utf-8") as handle:
            for idx, row in enumerate(handle):
                if idx == 0:
                    headers = row.strip().split(",")
                    continue
                values = row.strip().split(",")
                entry = dict(zip(headers, values))
                formatted.append(_format_test(entry, idx - 1))
        write_jsonl(target, formatted)

    return [target]
