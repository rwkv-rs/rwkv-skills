from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.medmcqa import (
    _DATASET_REVISION,
    _parse_row,
    prepare_medmcqa_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_medmcqa_registry_and_scheduler_use_validation_split() -> None:
    metadata = resolve_benchmark_metadata("medmcqa_validation")
    slug = canonical_slug("medmcqa_validation")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "validation"
    assert DATASET_PREP_SPECS[slug].dataset == "medmcqa"
    assert DATASET_PREP_SPECS[slug].split == "validation"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_medmcqa_parser_maps_zero_based_answer_and_subject() -> None:
    record = _parse_row(
        {
            "id": "medical-1",
            "question": "Which option is correct?",
            "opa": "First",
            "opb": "Second",
            "opc": "Third",
            "opd": "Fourth",
            "cop": 2,
            "choice_type": "single",
            "subject_name": "Forensic Medicine",
            "topic_name": "Injuries",
        }
    )

    assert record == {
        "question": "Which option is correct?",
        "answer": "C",
        "subject": "forensic_medicine",
        "subset": "medicine",
        "source_subject": "Forensic Medicine",
        "source_id": "medical-1",
        "choice_type": "single",
        "topic": "Injuries",
        "A": "First",
        "B": "Second",
        "C": "Third",
        "D": "Fourth",
    }


def test_medmcqa_spec_is_pinned_to_labeled_validation_parquet(tmp_path: Path) -> None:
    spec = prepare_medmcqa_spec(tmp_path)

    assert spec.name == "medmcqa"
    assert spec.split == "validation"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["data/validation-00000-of-00001.parquet"]
    assert spec.artifact_path == tmp_path / "medmcqa" / "validation.jsonl"


def test_medmcqa_parser_rejects_unlabeled_test_answer() -> None:
    with pytest.raises(ValueError, match="invalid answer index"):
        _parse_row(
            {
                "question": "Question",
                "opa": "One",
                "opb": "Two",
                "opc": "Three",
                "opd": "Four",
                "cop": -1,
            }
        )
