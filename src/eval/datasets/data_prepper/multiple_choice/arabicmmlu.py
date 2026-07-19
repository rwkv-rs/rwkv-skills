from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_csv_items

_DATASET_ID = "MBZUAI/ArabicMMLU"
_DATASET_REVISION = "7aa530e2893ac420352b3f5c1a1310c010e9758b"
_EXPECTED_ROWS = 14455


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("Question") or "").strip()
    answer = str(row.get("Answer Key") or "").strip().upper()
    if not question:
        raise ValueError("ArabicMMLU contains an empty question")
    if answer not in "ABCDE" or len(answer) != 1:
        raise ValueError(f"ArabicMMLU contains invalid answer {answer!r}")

    choices = [str(row.get(f"Option {index}") or "").strip() for index in range(1, 6)]
    while choices and not choices[-1]:
        choices.pop()
    if not 2 <= len(choices) <= 5 or not all(choices):
        raise ValueError("ArabicMMLU choices must contain 2 to 5 contiguous options")
    if ord(answer) - ord("A") >= len(choices):
        raise ValueError(f"ArabicMMLU answer {answer!r} is outside {len(choices)} choices")

    source_subject = str(row.get("Subject") or "unknown").strip()
    level = str(row.get("Level") or "").strip()
    task = f"{source_subject} ({level})" if level else source_subject
    group = str(row.get("Group") or "unknown").strip()
    payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "subject": _normalize_label(task),
        "subset": _normalize_label(group),
        "source_subject": source_subject,
        "level": level or None,
        "country": str(row.get("Country") or "").strip() or None,
        "source_id": str(row.get("ID") or ""),
        "source": str(row.get("Source") or "").strip(),
    }
    context = str(row.get("Context") or "").strip()
    if context:
        payload["context"] = context
    for index, choice in enumerate(choices):
        payload[chr(ord("A") + index)] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("ArabicMMLU benchmark only supports the test split")
    path = source_root / "All" / "test.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [_parse_row(dict(row)) for row in read_csv_items(path)]
    if len(records) != _EXPECTED_ROWS:
        raise ValueError(f"ArabicMMLU expected {_EXPECTED_ROWS} test rows, found {len(records)}")
    return records


@MULTIPLE_CHOICE_REGISTRY.register_spec("arabicmmlu")
def prepare_arabicmmlu_spec(output_root: Path, split: str = "test") -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "arabicmmlu",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B"),
        allow_patterns=["All/test.csv"],
    )


__all__ = ["prepare_arabicmmlu_spec"]
