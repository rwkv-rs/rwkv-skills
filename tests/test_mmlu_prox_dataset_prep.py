from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.datasets.data_prepper.multiple_choice.mmlu_prox import (
    _DATASET_REVISION,
    _EXPECTED_ROWS,
    _LANGUAGES,
    _is_known_malformed,
    _parse_row,
    prepare_mmlu_prox_spec,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import DATASET_PREP_SPECS, JOB_CATALOGUE


def test_mmlu_prox_registry_and_scheduler_use_test_split() -> None:
    metadata = resolve_benchmark_metadata("mmlu_prox_test")
    slug = canonical_slug("mmlu_prox_test")

    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert DATASET_PREP_SPECS[slug].dataset == "mmlu_prox"
    assert DATASET_PREP_SPECS[slug].split == "test"
    assert all(slug in JOB_CATALOGUE[name].dataset_slugs for name in metadata.scheduler_jobs)


def test_mmlu_prox_parser_maps_variable_options_and_language() -> None:
    record = _parse_row(
        {
            "question_id": 42,
            "question": "Which option is correct?",
            "option_0": "First",
            "option_1": "Second",
            "option_2": "Third",
            "option_3": None,
            "answer": "C",
            "answer_index": 2,
            "category": "computer science",
            "src": "ori_mmlu-machine_learning",
            "question_id_src": 7,
        },
        "en",
    )

    assert record == {
        "question": "Which option is correct?",
        "answer": "C",
        "subject": "computer_science",
        "subset": "en",
        "language": "en",
        "category": "computer_science",
        "source": "ori_mmlu-machine_learning",
        "source_id": 42,
        "source_question_id": 7,
        "source_answer_index": 2,
        "answer_index_consistent": True,
        "A": "First",
        "B": "Second",
        "C": "Third",
    }


def test_mmlu_prox_answer_field_is_canonical_when_source_index_disagrees() -> None:
    record = _parse_row(
        {
            "question": "Question",
            "option_0": "One",
            "option_1": "Two",
            "option_2": "Three",
            "answer": "C",
            "answer_index": 1,
            "category": "chemistry",
        },
        "en",
    )

    assert record["answer"] == "C"
    assert record["source_answer_index"] == 1
    assert record["answer_index_consistent"] is False


def test_mmlu_prox_excludes_known_zh_row_with_missing_i_but_present_j() -> None:
    assert _is_known_malformed({"question_id": 3299}, "zh")
    assert not _is_known_malformed({"question_id": 3299}, "en")


def test_mmlu_prox_spec_covers_all_29_languages_at_pinned_revision(tmp_path: Path) -> None:
    spec = prepare_mmlu_prox_spec(tmp_path)

    assert len(_LANGUAGES) == 29
    assert _EXPECTED_ROWS == 341010
    assert spec.name == "mmlu_prox"
    assert spec.split == "test"
    assert spec.revision == _DATASET_REVISION
    assert spec._allow_patterns == [
        f"{language}/test-00000-of-00001.parquet" for language in _LANGUAGES
    ]
    assert spec.artifact_path == tmp_path / "mmlu_prox" / "test.jsonl"


def test_mmlu_prox_parser_rejects_non_contiguous_options() -> None:
    with pytest.raises(ValueError, match="non-contiguous"):
        _parse_row(
            {
                "question": "Question",
                "option_0": "One",
                "option_1": "Two",
                "option_2": None,
                "option_3": "Four",
                "answer": "A",
            },
            "en",
        )
