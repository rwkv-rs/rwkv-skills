from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.agieval_mcq import (
    _REPO_REVISION,
    _SUBJECT_LANGUAGES,
    _parse_row,
    prepare_agieval_mcq_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_agieval_mcq_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("agieval_mcq_test")
    slug = canonical_slug("agieval_mcq_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "agieval_mcq"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_agieval_mcq_parser_handles_single_element_labels_and_passage() -> None:
    record = _parse_row(
        {
            "passage": "Read this passage.",
            "question": "Which option is correct?",
            "options": ["(A) First", "(B) Second", "(C) Third", "(D) Fourth"],
            "label": ["C"],
        },
        "jec-qa-ca",
    )

    assert record["answer"] == "C"
    assert [record[label] for label in "ABCD"] == ["First", "Second", "Third", "Fourth"]
    assert record["context"] == "Read this passage."
    assert record["subject"] == "jec_qa_ca"
    assert record["subset"] == "zh"


def test_agieval_mcq_spec_pins_all_official_mcq_subjects(tmp_path: Path) -> None:
    spec = prepare_agieval_mcq_spec(tmp_path)

    assert spec.name == "agieval_mcq"
    assert spec.split == "test"
    assert len(spec._files) == 18
    assert all(_REPO_REVISION in file.url for file in spec._files)
    filenames = {file.relative_path.name for file in spec._files}
    assert filenames == {f"{subject}.jsonl" for subject in _SUBJECT_LANGUAGES}
    assert "math.jsonl" not in filenames
    assert "gaokao-mathcloze.jsonl" not in filenames
    assert "sat-en-without-passage.jsonl" not in filenames
    assert spec.manifest_extra()["excluded_multi_answer_rows"] == 7
    assert spec.manifest_extra()["excluded_malformed_rows"] == 1


def test_agieval_mcq_parser_rejects_multi_answer_labels() -> None:
    with pytest.raises(ValueError, match="expected one answer"):
        _parse_row(
            {
                "question": "Question",
                "options": ["(A) One", "(B) Two"],
                "label": ["A", "B"],
            },
            "jec-qa-kd",
        )
