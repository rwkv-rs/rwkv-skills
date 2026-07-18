from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_parquet_items

_DATASET_ID = "openlifescienceai/medmcqa"
_DATASET_REVISION = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
_SOURCE_FILE = "data/validation-00000-of-00001.parquet"
_EXPECTED_ROWS = 4183


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("MedMCQA contains an empty question")

    raw_answer = row.get("cop")
    if isinstance(raw_answer, bool) or not isinstance(raw_answer, int) or not 0 <= raw_answer < 4:
        raise ValueError(f"MedMCQA contains invalid answer index {raw_answer!r}")
    answer = chr(ord("A") + raw_answer)

    choices: dict[str, str] = {}
    for label, source_key in zip("ABCD", ("opa", "opb", "opc", "opd"), strict=True):
        choice = str(row.get(source_key) or "").strip()
        if not choice:
            raise ValueError(f"MedMCQA contains an empty {label} choice")
        choices[label] = choice

    source_subject = str(row.get("subject_name") or "unknown").strip() or "unknown"
    payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "subject": _normalize_label(source_subject),
        "subset": "medicine",
        "source_subject": source_subject,
        "source_id": str(row.get("id") or ""),
        "choice_type": str(row.get("choice_type") or "").strip() or None,
        **choices,
    }
    topic = str(row.get("topic_name") or "").strip()
    if topic:
        payload["topic"] = topic
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "validation":
        raise ValueError("MedMCQA benchmark only supports the labeled validation split")
    path = source_root / _SOURCE_FILE
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [_parse_row(dict(row)) for row in read_parquet_items(path)]
    if len(records) != _EXPECTED_ROWS:
        raise ValueError(f"MedMCQA expected {_EXPECTED_ROWS} validation rows, found {len(records)}")
    return records


@MULTIPLE_CHOICE_REGISTRY.register_spec("medmcqa")
def prepare_medmcqa_spec(output_root: Path, split: str = "validation") -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "medmcqa",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B", "C", "D"),
        allow_patterns=[_SOURCE_FILE],
    )


__all__ = ["prepare_medmcqa_spec"]
