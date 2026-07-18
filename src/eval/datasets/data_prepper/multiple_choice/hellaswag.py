from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_parquet_items

_DATASET_ID = "Rowan/hellaswag"
_DATASET_REVISION = "218ec52e09a7e7462a5400043bb9a69a41d06b76"


def _normalize_subject(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    context = str(row.get("ctx") or "").strip()
    endings = row.get("endings")
    if not context:
        raise ValueError("HellaSwag contains an empty context")
    if not isinstance(endings, list) or not 2 <= len(endings) <= 26:
        raise ValueError("HellaSwag endings must contain between 2 and 26 choices")

    try:
        answer_index = int(str(row.get("label") or "").strip())
    except ValueError as exc:
        raise ValueError(f"HellaSwag contains invalid label {row.get('label')!r}") from exc
    if not 0 <= answer_index < len(endings):
        raise ValueError(f"HellaSwag answer index {answer_index} is outside {len(endings)} endings")

    activity = str(row.get("activity_label") or "unknown")
    payload: dict[str, Any] = {
        "question": context,
        "answer": chr(ord("A") + answer_index),
        "subject": _normalize_subject(activity),
        "subset": str(row.get("split_type") or "unknown"),
        "activity_label": activity,
        "source_id": str(row.get("source_id") or ""),
        "source_index": row.get("ind"),
    }
    for index, ending in enumerate(endings):
        choice = str(ending).strip()
        if not choice:
            raise ValueError(f"HellaSwag contains an empty ending at index {index}")
        payload[chr(ord("A") + index)] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "validation":
        raise ValueError("HellaSwag uses validation because the official test labels are private")
    path = source_root / "data" / "validation-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [_parse_row(dict(row)) for row in read_parquet_items(path)]


@MULTIPLE_CHOICE_REGISTRY.register_spec("hellaswag")
def prepare_hellaswag_spec(
    output_root: Path,
    split: str = "validation",
) -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "hellaswag",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B"),
        allow_patterns=["data/validation-*.parquet"],
    )


__all__ = ["prepare_hellaswag_spec"]
