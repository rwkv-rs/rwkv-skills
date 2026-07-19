from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.medqa import (
    _DATASET_REVISION,
    _parse_row,
    prepare_medqa_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_medqa_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("medqa_test")
    slug = canonical_slug("medqa_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "medqa"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_medqa_parser_maps_four_options_and_exam_step() -> None:
    record = _parse_row(
        {
            "question": "Which treatment is appropriate?",
            "answer": "Treatment B",
            "options": {
                "A": "Treatment A",
                "B": "Treatment B",
                "C": "Treatment C",
                "D": "Treatment D",
            },
            "meta_info": "step2",
            "answer_idx": "B",
        }
    )

    assert record == {
        "question": "Which treatment is appropriate?",
        "answer": "B",
        "subject": "step2",
        "subset": "medicine",
        "source_answer": "Treatment B",
        "A": "Treatment A",
        "B": "Treatment B",
        "C": "Treatment C",
        "D": "Treatment D",
    }


def test_medqa_spec_is_pinned_to_public_usmle_four_option_test(tmp_path: Path) -> None:
    spec = prepare_medqa_spec(tmp_path)

    assert spec.name == "medqa"
    assert spec.split == "test"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["phrases_no_exclude_test.jsonl"]
    assert spec.artifact_path == tmp_path / "medqa" / "test.jsonl"


def test_medqa_parser_rejects_mismatched_answer_text() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _parse_row(
            {
                "question": "Question",
                "answer": "Different text",
                "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
                "answer_idx": "A",
            }
        )
