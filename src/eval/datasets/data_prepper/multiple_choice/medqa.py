from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_jsonl_items

_DATASET_ID = "GBaker/MedQA-USMLE-4-options"
_DATASET_REVISION = "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18"
_SOURCE_FILE = "phrases_no_exclude_test.jsonl"
_EXPECTED_ROWS = 1273


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    options = row.get("options")
    answer = str(row.get("answer_idx") or "").strip().upper()
    if not question:
        raise ValueError("MedQA contains an empty question")
    if not isinstance(options, dict):
        raise ValueError("MedQA options must be an object")
    if answer not in "ABCD" or len(answer) != 1:
        raise ValueError(f"MedQA contains invalid answer {answer!r}")

    choices: dict[str, str] = {}
    for label in "ABCD":
        choice = str(options.get(label) or "").strip()
        if not choice:
            raise ValueError(f"MedQA contains an empty {label} choice")
        choices[label] = choice

    answer_text = str(row.get("answer") or "").strip()
    if answer_text and choices[answer] != answer_text:
        raise ValueError(f"MedQA answer text does not match option {answer}")

    exam_step = str(row.get("meta_info") or "unknown").strip().lower() or "unknown"
    return {
        "question": question,
        "answer": answer,
        "subject": exam_step,
        "subset": "medicine",
        "source_answer": answer_text or choices[answer],
        **choices,
    }


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("MedQA benchmark only supports the test split")
    path = source_root / _SOURCE_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [_parse_row(dict(row)) for row in read_jsonl_items(path)]
    if len(records) != _EXPECTED_ROWS:
        raise ValueError(f"MedQA expected {_EXPECTED_ROWS} test rows, found {len(records)}")
    return records


@MULTIPLE_CHOICE_REGISTRY.register_spec("medqa")
def prepare_medqa_spec(output_root: Path, split: str = "test") -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "medqa",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B", "C", "D"),
        allow_patterns=[_SOURCE_FILE],
    )


__all__ = ["prepare_medqa_spec"]
