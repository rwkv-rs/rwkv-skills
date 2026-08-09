from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_parquet_items

_DATASET_ID = "allenai/winogrande"
_DATASET_REVISION = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"
_CONFIG = "winogrande_xl"


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    sentence = str(row.get("sentence") or "").strip()
    choices = [str(row.get("option1") or "").strip(), str(row.get("option2") or "").strip()]
    if not sentence or "_" not in sentence:
        raise ValueError("WinoGrande sentence must contain a non-empty blank marker")
    if not all(choices):
        raise ValueError("WinoGrande contains an empty option")

    try:
        answer_index = int(str(row.get("answer") or "").strip()) - 1
    except ValueError as exc:
        raise ValueError(f"WinoGrande contains invalid answer {row.get('answer')!r}") from exc
    if answer_index not in (0, 1):
        raise ValueError(f"WinoGrande answer index {answer_index} is outside two options")

    return {
        "question": sentence,
        "answer": chr(ord("A") + answer_index),
        "subject": "winogrande",
        "subset": _CONFIG,
        "A": choices[0],
        "B": choices[1],
        "evaluation_protocol": "generated_choice_letter",
    }


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "validation":
        raise ValueError("WinoGrande uses validation because the official test labels are private")
    path = source_root / _CONFIG / "validation-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [_parse_row(dict(row)) for row in read_parquet_items(path)]


@MULTIPLE_CHOICE_REGISTRY.register_spec("winogrande")
def prepare_winogrande_spec(
    output_root: Path,
    split: str = "validation",
) -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "winogrande",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B"),
        allow_patterns=[f"{_CONFIG}/validation-*.parquet"],
    )


__all__ = ["prepare_winogrande_spec"]
