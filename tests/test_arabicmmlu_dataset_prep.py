from __future__ import annotations

from pathlib import Path

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.arabicmmlu import (
    _DATASET_REVISION,
    _parse_row,
    prepare_arabicmmlu_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_arabicmmlu_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("arabicmmlu_test")
    slug = canonical_slug("arabicmmlu_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "arabicmmlu"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_arabicmmlu_parser_preserves_task_metadata_and_variable_choices() -> None:
    record = _parse_row(
        {
            "ID": "42",
            "Source": "https://example.test/source",
            "Country": "Jordan",
            "Group": "STEM",
            "Subject": "Computer Science",
            "Level": "High School",
            "Question": "ما هي الإجابة الصحيحة؟",
            "Context": "اقرأ السياق.",
            "Answer Key": "E",
            "Option 1": "الأول",
            "Option 2": "الثاني",
            "Option 3": "الثالث",
            "Option 4": "الرابع",
            "Option 5": "الخامس",
        }
    )

    assert record["answer"] == "E"
    assert [record[label] for label in "ABCDE"] == ["الأول", "الثاني", "الثالث", "الرابع", "الخامس"]
    assert record["subject"] == "computer_science_high_school"
    assert record["subset"] == "stem"
    assert record["context"] == "اقرأ السياق."


def test_arabicmmlu_spec_is_pinned_to_nonduplicated_all_test_csv(tmp_path: Path) -> None:
    spec = prepare_arabicmmlu_spec(tmp_path)

    assert spec.name == "arabicmmlu"
    assert spec.split == "test"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == ["All/test.csv"]
    assert spec.artifact_path == tmp_path / "arabicmmlu" / "test.jsonl"
