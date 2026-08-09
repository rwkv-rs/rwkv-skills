from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_parquet_items

_DATASET_ID = "microsoft/MMLU-CF"
_DATASET_REVISION = "c25b89a968a2062e422dd96ddf0fe3387507f0dd"
_CATEGORIES = (
    "Biology",
    "Business",
    "Chemistry",
    "Computer_Science",
    "Economics",
    "Engineering",
    "Health",
    "History",
    "Law",
    "Math",
    "Other",
    "Philosophy",
    "Physics",
    "Psychology",
)
_REQUIRED_FIELDS = ("question", "answer", "A", "B", "C", "D")


def _parse_row(row: dict[str, Any], category: str) -> dict[str, Any]:
    question = str(row.get("Question") or "").strip()
    answer = str(row.get("Answer") or "").strip().upper()
    if not question:
        raise ValueError(f"MMLU-CF {category} contains an empty question")
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"MMLU-CF {category} contains invalid answer {answer!r}")

    subject = category.lower()
    payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "subject": subject,
        "subset": subject,
        "source_category": category,
    }
    for label in "ABCD":
        choice = str(row.get(label) or "").strip()
        if not choice:
            raise ValueError(f"MMLU-CF {category} contains an empty {label} choice")
        payload[label] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "val":
        raise ValueError("MMLU-CF only exposes the public val split; the official test split is closed")

    records: list[dict[str, Any]] = []
    for category in _CATEGORIES:
        path = source_root / "val" / f"{category}_val.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        records.extend(
            _parse_row(dict(row), category)
            for row in read_parquet_items(path)
        )
    return records


@MULTIPLE_CHOICE_REGISTRY.register_spec("mmlu_cf")
def prepare_mmlu_cf_spec(output_root: Path, split: str = "val") -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "mmlu_cf",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=_REQUIRED_FIELDS,
        allow_patterns=["val/*_val.parquet"],
    )


__all__ = ["prepare_mmlu_cf_spec"]
