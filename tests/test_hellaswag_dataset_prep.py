from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.hellaswag import (
    _DATASET_REVISION,
    _load_records,
    _parse_row,
    prepare_hellaswag_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_hellaswag_registry_and_scheduler_use_validation_split() -> None:
    metadata = resolve_benchmark_metadata("hellaswag_validation")
    slug = canonical_slug("hellaswag_validation")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "validation"
    assert DATASET_PREP_SPECS[slug].dataset == "hellaswag"
    assert DATASET_PREP_SPECS[slug].split == "validation"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_hellaswag_parser_maps_numeric_label_and_activity() -> None:
    record = _parse_row(
        {
            "ind": 24,
            "activity_label": "Roof shingle removal",
            "ctx": "A man is sitting on a roof. he",
            "endings": ["falls", "waits", "waves", "starts pulling up roofing"],
            "source_id": "activitynet~video",
            "split_type": "indomain",
            "label": "3",
        }
    )

    assert record["answer"] == "D"
    assert [record[label] for label in "ABCD"] == [
        "falls",
        "waits",
        "waves",
        "starts pulling up roofing",
    ]
    assert record["subject"] == "roof_shingle_removal"
    assert record["subset"] == "indomain"


def test_hellaswag_spec_is_pinned_to_labeled_validation_parquet(tmp_path: Path) -> None:
    spec = prepare_hellaswag_spec(tmp_path)

    assert spec.name == "hellaswag"
    assert spec.split == "validation"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["data/validation-*.parquet"]
    assert spec.artifact_path == tmp_path / "hellaswag" / "validation.jsonl"

    with pytest.raises(ValueError, match="test labels are private"):
        _load_records(tmp_path, "test")
