from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, CoTMode, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.mmlu_cf import (
    _DATASET_REVISION,
    _load_records,
    _parse_row,
    prepare_mmlu_cf_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_mmlu_cf_registry_and_scheduler_use_public_validation_split() -> None:
    metadata = resolve_benchmark_metadata("mmlu_cf_val")
    slug = canonical_slug("mmlu_cf_val")

    assert metadata.name == "mmlu_cf"
    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.cot_modes == (CoTMode.NO_COT, CoTMode.COT)
    assert metadata.default_split == "val"
    assert metadata.scheduler_jobs == (
        "multi_choice_plain",
        "multi_choice_cot",
        "multi_choice_plain_naive",
        "multi_choice_cot_naive",
    )
    for job_name in metadata.scheduler_jobs:
        assert slug in JOB_CATALOGUE[job_name].dataset_slugs
    assert DATASET_PREP_SPECS[slug].dataset == "mmlu_cf"
    assert DATASET_PREP_SPECS[slug].split == "val"


def test_mmlu_cf_parser_preserves_choices_answer_and_category() -> None:
    record = _parse_row(
        {
            "Question": "Which option is correct? ",
            "A": " First ",
            "B": "Second",
            "C": "Third",
            "D": "Fourth",
            "Answer": " b ",
        },
        "Computer_Science",
    )

    assert record == {
        "question": "Which option is correct?",
        "answer": "B",
        "subject": "computer_science",
        "subset": "computer_science",
        "source_category": "Computer_Science",
        "A": "First",
        "B": "Second",
        "C": "Third",
        "D": "Fourth",
    }


def test_mmlu_cf_spec_is_pinned_to_public_validation_parquet(tmp_path: Path) -> None:
    spec = prepare_mmlu_cf_spec(tmp_path)

    assert spec.name == "mmlu_cf"
    assert spec.split == "val"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["val/*_val.parquet"]
    assert spec.artifact_path == tmp_path / "mmlu_cf" / "val.jsonl"

    with pytest.raises(ValueError, match="official test split is closed"):
        _load_records(tmp_path, "test")


@pytest.mark.parametrize("answer", ["", "E", "0"])
def test_mmlu_cf_parser_rejects_invalid_answers(answer: str) -> None:
    with pytest.raises(ValueError, match="invalid answer"):
        _parse_row(
            {
                "Question": "Question",
                "A": "One",
                "B": "Two",
                "C": "Three",
                "D": "Four",
                "Answer": answer,
            },
            "Math",
        )
