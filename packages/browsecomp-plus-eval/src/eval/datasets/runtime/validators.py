from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_non_empty_records(records: list[dict[str, Any]], dataset_name: str) -> None:
    if not records:
        raise ValueError(f"{dataset_name}: no records loaded")


def validate_required_fields(
    records: list[dict[str, Any]],
    required_fields: tuple[str, ...],
    dataset_name: str,
) -> None:
    if not required_fields:
        return
    for index, record in enumerate(records):
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise ValueError(f"{dataset_name}: record {index} missing required fields {missing}")


def validate_jsonl_file(path: str | Path, required_fields: tuple[str, ...] = ()) -> int:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)

    count = 0
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            missing = [field for field in required_fields if field not in record]
            if missing:
                raise ValueError(f"{target}: record {count} missing required fields {missing}")
            count += 1
    if count <= 0:
        raise ValueError(f"{target}: dataset is empty")
    return count


__all__ = [
    "validate_jsonl_file",
    "validate_non_empty_records",
    "validate_required_fields",
]
