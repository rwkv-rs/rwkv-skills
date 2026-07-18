from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.truthfulqa_mc1 import (
    _DATASET_REVISION,
    _parse_row,
    prepare_truthfulqa_mc1_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_truthfulqa_mc1_registry_and_scheduler_use_validation_split() -> None:
    metadata = resolve_benchmark_metadata("truthfulqa_mc1_validation")
    slug = canonical_slug("truthfulqa_mc1_validation")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "validation"
    assert DATASET_PREP_SPECS[slug].dataset == "truthfulqa_mc1"
    assert DATASET_PREP_SPECS[slug].split == "validation"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_truthfulqa_mc1_parser_uses_single_positive_label() -> None:
    record = _parse_row(
        {
            "question": "Which option is truthful?",
            "mc1_targets": {
                "choices": ["False", "Also false", "Truthful"],
                "labels": [0, 0, 1],
            },
        },
        "Misconceptions and urban legends",
        "https://example.test/source",
    )

    assert record["answer"] == "C"
    assert [record[label] for label in "ABC"] == ["False", "Also false", "Truthful"]
    assert record["subject"] == "misconceptions_and_urban_legends"
    assert record["source_category"] == "Misconceptions and urban legends"


def test_truthfulqa_mc1_spec_is_pinned_to_both_validation_sources(tmp_path: Path) -> None:
    spec = prepare_truthfulqa_mc1_spec(tmp_path)

    assert spec.name == "truthfulqa_mc1"
    assert spec.split == "validation"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == [
        "generation/validation-*.parquet",
        "multiple_choice/validation-*.parquet",
    ]
    assert spec.artifact_path == tmp_path / "truthfulqa_mc1" / "validation.jsonl"


def test_truthfulqa_mc1_parser_rejects_multiple_positive_labels() -> None:
    with pytest.raises(ValueError, match="expected one correct choice"):
        _parse_row(
            {
                "question": "Question",
                "mc1_targets": {"choices": ["One", "Two"], "labels": [1, 1]},
            },
            "Category",
            "Source",
        )
