from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import (
    HfRepoJsonlDatasetSpec,
    collect_files_with_extension,
    read_csv_items,
)

_DATASET_ID = "HAERAE-HUB/KMMLU"
_DATASET_REVISION = "d61b3f19e552c576bf5960dd24289763edc36a88"
_EXPECTED_FILES = 45
_EXPECTED_ROWS = 35030


def _normalize_subject(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def _parse_row(row: dict[str, Any], source_subject: str) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    try:
        answer_index = int(str(row.get("answer") or "").strip()) - 1
    except ValueError as exc:
        raise ValueError(f"KMMLU {source_subject} contains invalid answer {row.get('answer')!r}") from exc
    if not question:
        raise ValueError(f"KMMLU {source_subject} contains an empty question")
    if answer_index not in range(4):
        raise ValueError(f"KMMLU {source_subject} answer index {answer_index} is outside four choices")

    category = str(row.get("Category") or source_subject).strip()
    payload: dict[str, Any] = {
        "question": question,
        "answer": chr(ord("A") + answer_index),
        "subject": _normalize_subject(category),
        "subset": "korean",
        "source_category": category,
    }
    human_accuracy = str(row.get("Human Accuracy") or "").strip()
    if human_accuracy:
        payload["human_accuracy"] = float(human_accuracy)
    for label in "ABCD":
        choice = str(row.get(label) or "").strip()
        if not choice:
            raise ValueError(f"KMMLU {source_subject} contains an empty {label} choice")
        payload[label] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("KMMLU benchmark only supports the test split")

    paths = [
        path
        for path in collect_files_with_extension(source_root / "data", "csv")
        if path.name.endswith("-test.csv")
    ]
    if len(paths) != _EXPECTED_FILES:
        raise ValueError(f"KMMLU expected {_EXPECTED_FILES} test files, found {len(paths)}")

    records: list[dict[str, Any]] = []
    for path in paths:
        source_subject = path.stem.removesuffix("-test")
        records.extend(_parse_row(dict(row), source_subject) for row in read_csv_items(path))
    if len(records) != _EXPECTED_ROWS:
        raise ValueError(f"KMMLU expected {_EXPECTED_ROWS} test rows, found {len(records)}")
    return records


@MULTIPLE_CHOICE_REGISTRY.register_spec("kmmlu")
def prepare_kmmlu_spec(output_root: Path, split: str = "test") -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "kmmlu",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B", "C", "D"),
        allow_patterns=["data/*-test.csv"],
    )


__all__ = ["prepare_kmmlu_spec"]
