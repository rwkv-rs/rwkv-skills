from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_parquet_items

_DATASET_ID = "allenai/openbookqa"
_DATASET_REVISION = "388097ea7776314e93a529163e0fea805b8a6454"
_CONFIG = "main"


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    question = str(row.get("question_stem") or "").strip()
    choices = row.get("choices")
    if not question:
        raise ValueError("OpenBookQA contains an empty question")
    if not isinstance(choices, dict):
        raise ValueError("OpenBookQA choices must be an object")

    texts = choices.get("text")
    labels = choices.get("label")
    if not isinstance(texts, list) or not isinstance(labels, list) or len(texts) != len(labels):
        raise ValueError("OpenBookQA choice texts and labels must be equal-length lists")
    if not 2 <= len(texts) <= 26:
        raise ValueError(f"OpenBookQA contains {len(texts)} choices")

    source_labels = [str(label).strip().upper() for label in labels]
    answer_key = str(row.get("answerKey") or "").strip().upper()
    try:
        answer_index = source_labels.index(answer_key)
    except ValueError as exc:
        raise ValueError(f"OpenBookQA answer {answer_key!r} is not in {source_labels!r}") from exc

    payload: dict[str, Any] = {
        "question": question,
        "answer": chr(ord("A") + answer_index),
        "subject": "openbookqa",
        "subset": "science",
        "source_id": str(row.get("id") or ""),
        "source_answer_label": answer_key,
    }
    for index, text in enumerate(texts):
        choice = str(text).strip()
        if not choice:
            raise ValueError(f"OpenBookQA contains an empty choice at index {index}")
        payload[chr(ord("A") + index)] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("OpenBookQA benchmark only supports the test split")
    path = source_root / _CONFIG / "test-00000-of-00001.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [_parse_row(dict(row)) for row in read_parquet_items(path)]


@MULTIPLE_CHOICE_REGISTRY.register_spec("openbookqa")
def prepare_openbookqa_spec(output_root: Path, split: str = "test") -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "openbookqa",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B"),
        allow_patterns=[f"{_CONFIG}/test-*.parquet"],
    )


__all__ = ["prepare_openbookqa_spec"]
