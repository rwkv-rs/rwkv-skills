from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_parquet_items

_DATASET_ID = "truthfulqa/truthful_qa"
_DATASET_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"


def _normalize_category(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_") or "unknown"


def _parse_row(row: dict[str, Any], category: str, source: str) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    targets = row.get("mc1_targets")
    if not question:
        raise ValueError("TruthfulQA-MC1 contains an empty question")
    if not isinstance(targets, dict):
        raise ValueError("TruthfulQA-MC1 mc1_targets must be an object")

    choices = targets.get("choices")
    labels = targets.get("labels")
    if not isinstance(choices, list) or not isinstance(labels, list) or len(choices) != len(labels):
        raise ValueError("TruthfulQA-MC1 choices and labels must be equal-length lists")
    if not 2 <= len(choices) <= 26:
        raise ValueError(f"TruthfulQA-MC1 contains {len(choices)} choices")
    correct_indices = [index for index, label in enumerate(labels) if int(label) == 1]
    if len(correct_indices) != 1:
        raise ValueError(f"TruthfulQA-MC1 expected one correct choice, found {len(correct_indices)}")

    answer_index = correct_indices[0]
    payload: dict[str, Any] = {
        "question": question,
        "answer": chr(ord("A") + answer_index),
        "subject": _normalize_category(category),
        "subset": "truthfulness",
        "source_category": category,
        "source": source,
    }
    for index, text in enumerate(choices):
        choice = str(text).strip()
        if not choice:
            raise ValueError(f"TruthfulQA-MC1 contains an empty choice at index {index}")
        payload[chr(ord("A") + index)] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "validation":
        raise ValueError("TruthfulQA-MC1 only provides the validation split")

    generation_path = source_root / "generation" / "validation-00000-of-00001.parquet"
    multiple_choice_path = source_root / "multiple_choice" / "validation-00000-of-00001.parquet"
    if not generation_path.is_file():
        raise FileNotFoundError(generation_path)
    if not multiple_choice_path.is_file():
        raise FileNotFoundError(multiple_choice_path)

    metadata_by_question = {
        str(row["question"]).strip(): (str(row.get("category") or "unknown"), str(row.get("source") or ""))
        for row in read_parquet_items(generation_path)
    }
    records: list[dict[str, Any]] = []
    for raw_row in read_parquet_items(multiple_choice_path):
        row = dict(raw_row)
        question = str(row.get("question") or "").strip()
        try:
            category, source = metadata_by_question[question]
        except KeyError as exc:
            raise ValueError(f"TruthfulQA-MC1 category metadata missing for {question!r}") from exc
        records.append(_parse_row(row, category, source))
    return records


@MULTIPLE_CHOICE_REGISTRY.register_spec("truthfulqa_mc1")
def prepare_truthfulqa_mc1_spec(
    output_root: Path,
    split: str = "validation",
) -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "truthfulqa_mc1",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B"),
        allow_patterns=["generation/validation-*.parquet", "multiple_choice/validation-*.parquet"],
    )


__all__ = ["prepare_truthfulqa_mc1_spec"]
