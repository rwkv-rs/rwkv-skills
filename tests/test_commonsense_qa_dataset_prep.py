from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.commonsense_qa import (
    _DATASET_REVISION,
    _load_records,
    _parse_row,
    prepare_commonsense_qa_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_commonsense_qa_registry_and_scheduler_use_validation_split() -> None:
    metadata = resolve_benchmark_metadata("commonsense_qa_validation")
    slug = canonical_slug("commonsense_qa_validation")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "validation"
    assert DATASET_PREP_SPECS[slug].dataset == "commonsense_qa"
    assert DATASET_PREP_SPECS[slug].split == "validation"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_commonsense_qa_parser_maps_five_choices_and_answer() -> None:
    record = _parse_row(
        {
            "id": "csqa-1",
            "question": "Which option is correct?",
            "question_concept": "option",
            "choices": {
                "text": ["First", "Second", "Third", "Fourth", "Fifth"],
                "label": ["A", "B", "C", "D", "E"],
            },
            "answerKey": "E",
        }
    )

    assert record["answer"] == "E"
    assert [record[label] for label in "ABCDE"] == ["First", "Second", "Third", "Fourth", "Fifth"]
    assert record["question_concept"] == "option"
    assert record["subset"] == "commonsense"


def test_commonsense_qa_spec_is_pinned_to_labeled_validation_parquet(tmp_path: Path) -> None:
    spec = prepare_commonsense_qa_spec(tmp_path)

    assert spec.name == "commonsense_qa"
    assert spec.split == "validation"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["data/validation-*.parquet"]
    assert spec.artifact_path == tmp_path / "commonsense_qa" / "validation.jsonl"

    with pytest.raises(ValueError, match="test labels are private"):
        _load_records(tmp_path, "test")
