from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.winogrande import (
    _DATASET_REVISION,
    _load_records,
    _parse_row,
    prepare_winogrande_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_winogrande_registry_and_scheduler_use_validation_split() -> None:
    metadata = resolve_benchmark_metadata("winogrande_validation")
    slug = canonical_slug("winogrande_validation")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "validation"
    assert DATASET_PREP_SPECS[slug].dataset == "winogrande"
    assert DATASET_PREP_SPECS[slug].split == "validation"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_winogrande_parser_maps_one_based_answer() -> None:
    record = _parse_row(
        {
            "sentence": "Sarah was better prepared than Maria, so _ got the harder case.",
            "option1": "Sarah",
            "option2": "Maria",
            "answer": "1",
        }
    )

    assert record["question"].endswith("_ got the harder case.")
    assert record["answer"] == "A"
    assert record["A"] == "Sarah"
    assert record["B"] == "Maria"
    assert record["evaluation_protocol"] == "generated_choice_letter"


def test_winogrande_spec_is_pinned_to_xl_validation_parquet(tmp_path: Path) -> None:
    spec = prepare_winogrande_spec(tmp_path)

    assert spec.name == "winogrande"
    assert spec.split == "validation"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["winogrande_xl/validation-*.parquet"]
    assert spec.artifact_path == tmp_path / "winogrande" / "validation.jsonl"

    with pytest.raises(ValueError, match="test labels are private"):
        _load_records(tmp_path, "test")
