from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.arc_easy import (
    _DATASET_REVISION,
    _parse_row,
    prepare_arc_easy_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_arc_easy_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("arc_easy_test")
    slug = canonical_slug("arc_easy_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "arc_easy"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_arc_easy_parser_maps_source_labels_to_canonical_letters() -> None:
    record = _parse_row(
        {
            "id": "science-1",
            "question": "Which option is correct?",
            "choices": {
                "text": ["First", "Second", "Third", "Fourth"],
                "label": ["1", "2", "3", "4"],
            },
            "answerKey": "2",
        }
    )

    assert record == {
        "question": "Which option is correct?",
        "answer": "B",
        "subject": "arc_easy",
        "subset": "science",
        "source_id": "science-1",
        "source_answer_label": "2",
        "A": "First",
        "B": "Second",
        "C": "Third",
        "D": "Fourth",
    }


def test_arc_easy_spec_is_pinned_to_official_test_parquet(tmp_path: Path) -> None:
    spec = prepare_arc_easy_spec(tmp_path)

    assert spec.name == "arc_easy"
    assert spec.split == "test"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["ARC-Easy/test-*.parquet"]
    assert spec.artifact_path == tmp_path / "arc_easy" / "test.jsonl"


def test_arc_easy_parser_rejects_answer_outside_choice_labels() -> None:
    with pytest.raises(ValueError, match="is not in"):
        _parse_row(
            {
                "question": "Question",
                "choices": {"text": ["One", "Two"], "label": ["A", "B"]},
                "answerKey": "C",
            }
        )
