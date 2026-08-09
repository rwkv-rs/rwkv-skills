from __future__ import annotations

"""Prepare the selected MMLU-SR Question+Answer representative set.

The upstream CSV files intentionally have no header.  Reading the generated
Hugging Face parquet configs would drop the first example from every subject,
so this prepper downloads and parses the raw per-subject files instead.
"""

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec

from .mmlu import _SUBCATEGORIES as _MMLU_SUBCATEGORIES


_DATASET_ID = "NiniCat/MMLU-SR"
_DATASET_REVISION = "2c0b9096737078969a0af2b548e0f7682271fff1"
_VARIANTS = ("question_and_answer",)
_CHOICE_LETTERS = ("A", "B", "C", "D")
_REQUIRED_FIELDS = ("question", "answer", *_CHOICE_LETTERS)


def _iter_subject_csv(path: Path, *, subject: str) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 6:
                raise ValueError(f"{path}:{row_number} 应包含 6 列，实际为 {len(row)} 列")

            question, *choices, raw_answer = (cell.strip() for cell in row)
            answer = raw_answer.upper()
            if not question:
                raise ValueError(f"{path}:{row_number} 的题干为空")
            if answer not in _CHOICE_LETTERS:
                raise ValueError(f"{path}:{row_number} 的答案不是 A-D: {raw_answer!r}")

            # Three upstream MMLU-SR rows serialize the literal MMLU choice
            # "None" as an empty CSV cell.  Restore that choice explicitly and
            # retain an audit marker in the materialized record.
            repaired_choices = [choice or "None" for choice in choices]
            repaired_letters = [
                letter
                for letter, original in zip(_CHOICE_LETTERS, choices, strict=True)
                if not original
            ]

            payload: dict[str, Any] = {
                "question": question,
                **dict(zip(_CHOICE_LETTERS, repaired_choices, strict=True)),
                "answer": answer,
                "subject": subject,
                "subset": _MMLU_SUBCATEGORIES[subject],
                "source_dataset": _DATASET_ID,
            }
            if repaired_letters:
                payload["source_repairs"] = [
                    f"empty_option_{letter}_to_None" for letter in repaired_letters
                ]
            yield payload


def _load_variant_records(source_root: Path, *, variant: str, split: str) -> list[dict[str, Any]]:
    if variant not in _VARIANTS:
        raise ValueError(f"未知 MMLU-SR 变体: {variant}")
    if split != "test":
        raise ValueError("MMLU-SR 目前仅提供 test split")

    source_dir = source_root / f"{variant}_test"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)

    expected_paths = {
        subject: source_dir / f"{variant}_{subject}_test.csv"
        for subject in _MMLU_SUBCATEGORIES
    }
    missing = [subject for subject, path in expected_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MMLU-SR {variant} 缺少学科文件: {', '.join(missing)}")

    unexpected = sorted(
        path.name
        for path in source_dir.glob("*.csv")
        if path not in set(expected_paths.values())
    )
    if unexpected:
        raise ValueError(f"MMLU-SR {variant} 出现未知 CSV 文件: {', '.join(unexpected)}")

    records: list[dict[str, Any]] = []
    for subject, path in expected_paths.items():
        records.extend(_iter_subject_csv(path, subject=subject))
    return records


def _prepare_variant_spec(
    variant: str,
    output_root: Path,
    split: str,
) -> HfRepoJsonlDatasetSpec:
    if split != "test":
        raise ValueError("MMLU-SR 目前仅提供 test split")

    dataset_name = f"mmlu_sr_{variant}"

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return _load_variant_records(source_root, variant=variant, split=split)

    return HfRepoJsonlDatasetSpec(
        dataset_name,
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        repo_type="dataset",
        allow_patterns=[f"{variant}_test/*.csv"],
        load_downloaded_records=_load,
        required_fields=_REQUIRED_FIELDS,
    )


@MULTIPLE_CHOICE_REGISTRY.register_spec("mmlu_sr_question_and_answer")
def prepare_mmlu_sr_question_and_answer_spec(
    output_root: Path,
    split: str = "test",
) -> HfRepoJsonlDatasetSpec:
    return _prepare_variant_spec("question_and_answer", output_root, split)


__all__ = [
    "prepare_mmlu_sr_question_and_answer_spec",
]
