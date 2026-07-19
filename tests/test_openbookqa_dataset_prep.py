from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.openbookqa import (
    _DATASET_REVISION,
    _parse_row,
    prepare_openbookqa_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_openbookqa_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("openbookqa_test")
    slug = canonical_slug("openbookqa_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "openbookqa"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_openbookqa_parser_maps_choices_and_answer() -> None:
    record = _parse_row(
        {
            "id": "obqa-1",
            "question_stem": "Which option is correct?",
            "choices": {
                "text": ["First", "Second", "Third", "Fourth"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        }
    )

    assert record == {
        "question": "Which option is correct?",
        "answer": "D",
        "subject": "openbookqa",
        "subset": "science",
        "source_id": "obqa-1",
        "source_answer_label": "D",
        "A": "First",
        "B": "Second",
        "C": "Third",
        "D": "Fourth",
    }


def test_openbookqa_spec_is_pinned_to_official_main_test_parquet(tmp_path: Path) -> None:
    spec = prepare_openbookqa_spec(tmp_path)

    assert spec.name == "openbookqa"
    assert spec.split == "test"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["main/test-*.parquet"]
    assert spec.artifact_path == tmp_path / "openbookqa" / "test.jsonl"


def test_openbookqa_parser_rejects_answer_outside_choice_labels() -> None:
    with pytest.raises(ValueError, match="is not in"):
        _parse_row(
            {
                "question_stem": "Question",
                "choices": {"text": ["One", "Two"], "label": ["A", "B"]},
                "answerKey": "C",
            }
        )
