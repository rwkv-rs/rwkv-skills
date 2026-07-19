from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.bbh_mcq import (
    _MCQ_TASKS,
    _NON_MCQ_TASKS,
    _REPO_REVISION,
    UnsupportedMalformedQuestionError,
    _parse_row,
    prepare_bbh_mcq_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_bbh_mcq_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("bbh_mcq_test")
    slug = canonical_slug("bbh_mcq_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "bbh_mcq"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_bbh_mcq_parser_extracts_embedded_options() -> None:
    record = _parse_row(
        {
            "input": "What is tomorrow's date?\nOptions:\n(A) Monday\n(B) Tuesday\n(C) Wednesday",
            "target": "(B)",
        },
        "date_understanding",
    )

    assert record["question"] == "What is tomorrow's date?"
    assert record["answer"] == "B"
    assert [record[label] for label in "ABC"] == ["Monday", "Tuesday", "Wednesday"]
    assert record["subject"] == "date_understanding"


def test_bbh_mcq_spec_pins_only_explicit_choice_tasks(tmp_path: Path) -> None:
    spec = prepare_bbh_mcq_spec(tmp_path)

    assert spec.name == "bbh_mcq"
    assert spec.split == "test"
    assert len(spec._files) == 17
    assert all(_REPO_REVISION in file.url for file in spec._files)
    assert len(_MCQ_TASKS) == 17
    assert len(_NON_MCQ_TASKS) == 10
    assert spec.manifest_extra()["excluded_malformed_rows"] == 4


@pytest.mark.parametrize(
    ("raw_input", "target"),
    [
        ("Question\nOptions:\n(A) only one", "(A)"),
        ("Question\nOptions:\n(A) one\n(B) two", "answer text"),
    ],
)
def test_bbh_mcq_parser_rejects_malformed_examples(raw_input: str, target: str) -> None:
    with pytest.raises(UnsupportedMalformedQuestionError):
        _parse_row({"input": raw_input, "target": target}, "task")
