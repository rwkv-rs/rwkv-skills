from __future__ import annotations

import json
from pathlib import Path
from typing import List
from collections.abc import Sequence

from ..data_utils import write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

STATIC_DATASETS = {
    "aime24": "aime24_test.jsonl",
    "aime25": "aime25_test.jsonl",
    "comp-math-24-25": "comp-math-24-25_test.jsonl",
}


CANONICAL_QUESTION_KEYS: Sequence[str] = ("question", "problem", "prompt", "input")
CANONICAL_ANSWER_KEYS: Sequence[str] = (
    "expected_answer",
    "answer",
    "reference_solution",
    "solution",
    "output",
)


def _normalize_entry(entry: dict, dataset: str) -> dict:
    def _extract(keys: Sequence[str]) -> str | None:
        for key in keys:
            value = entry.get(key)
            if isinstance(value, str):
                return value
        return None

    question = _extract(CANONICAL_QUESTION_KEYS)
    answer = _extract(CANONICAL_ANSWER_KEYS)
    if question is None or answer is None:
        raise ValueError(f"{dataset}: 无法规范化题目，缺少 question/answer 字段: {entry}")

    subject = entry.get("subject") or entry.get("domain")
    if subject is not None and not isinstance(subject, str):
        subject = str(subject)

    normalized = {
        "question": question,
        "answer": answer,
    }
    if subject:
        normalized["subject"] = subject

    for key in ("id", "source", "expected_answer", "pass_ratio"):
        if key in entry:
            normalized[key] = entry[key]

    normalized["raw_record"] = entry
    normalized["dataset"] = dataset
    return normalized


def _load_static(filename: str, dataset: str) -> list[dict]:
    static_path = Path(__file__).parent / "static" / filename
    with static_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [_normalize_entry(row, dataset) for row in rows]


for dataset, filename in STATIC_DATASETS.items():
    rows = _load_static(filename, dataset)

    @FREE_ANSWER_REGISTRY.register(dataset)
    def _prepare(output_root: Path, split: str = "test", *, _rows=rows, _dataset=dataset) -> list[Path]:
        if split != "test":
            raise ValueError(f"{_dataset} 仅提供 test split")
        dataset_dir = output_root / _dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target = dataset_dir / f"{split}.jsonl"
        write_jsonl(target, _rows)
        return [target]
