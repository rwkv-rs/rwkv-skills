from __future__ import annotations

from pathlib import Path

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.kmmlu import (
    _DATASET_REVISION,
    _parse_row,
    prepare_kmmlu_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_kmmlu_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("kmmlu_test")
    slug = canonical_slug("kmmlu_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "kmmlu"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_kmmlu_parser_maps_one_based_answer_and_category() -> None:
    record = _parse_row(
        {
            "question": "정답은 무엇입니까?",
            "answer": "4",
            "A": "첫째",
            "B": "둘째",
            "C": "셋째",
            "D": "넷째",
            "Category": "Computer-Science",
            "Human Accuracy": "0.625",
        },
        "Computer-Science",
    )

    assert record["answer"] == "D"
    assert [record[label] for label in "ABCD"] == ["첫째", "둘째", "셋째", "넷째"]
    assert record["subject"] == "computer_science"
    assert record["human_accuracy"] == 0.625


def test_kmmlu_spec_is_pinned_to_all_test_csv_files(tmp_path: Path) -> None:
    spec = prepare_kmmlu_spec(tmp_path)

    assert spec.name == "kmmlu"
    assert spec.split == "test"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["data/*-test.csv"]
    assert spec.artifact_path == tmp_path / "kmmlu" / "test.jsonl"
