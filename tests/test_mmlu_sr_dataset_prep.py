from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.eval.datasets.data_prepper.multiple_choice.mmlu_sr import (
    _DATASET_REVISION,
    _iter_subject_csv,
    prepare_mmlu_sr_answer_only_spec,
    prepare_mmlu_sr_question_and_answer_spec,
    prepare_mmlu_sr_question_only_spec,
)


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerows(rows)


def test_mmlu_sr_raw_csv_parser_keeps_first_headerless_row_and_subject(tmp_path: Path) -> None:
    source = tmp_path / "question_only_abstract_algebra_test.csv"
    _write_csv(
        source,
        [
            [
                "Suppose 'Dragon' means degree. What is the Dragon?",
                "0",
                "4",
                "2",
                "6",
                "B",
            ],
            ["Second question", "one", "two", "three", "four", "d"],
        ],
    )

    records = list(_iter_subject_csv(source, subject="abstract_algebra"))

    assert len(records) == 2
    assert records[0] == {
        "question": "Suppose 'Dragon' means degree. What is the Dragon?",
        "A": "0",
        "B": "4",
        "C": "2",
        "D": "6",
        "answer": "B",
        "subject": "abstract_algebra",
        "subset": "math",
        "source_dataset": "NiniCat/MMLU-SR",
    }
    assert records[1]["answer"] == "D"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (["question", "A", "B", "C", "D"], "应包含 6 列"),
        (["question", "A", "B", "C", "D", "E"], "答案不是 A-D"),
        (["", "A", "B", "C", "D", "A"], "题干为空"),
    ],
)
def test_mmlu_sr_raw_csv_parser_rejects_malformed_rows(
    tmp_path: Path,
    row: list[str],
    message: str,
) -> None:
    source = tmp_path / "bad.csv"
    _write_csv(source, [row])

    with pytest.raises(ValueError, match=message):
        list(_iter_subject_csv(source, subject="abstract_algebra"))


def test_mmlu_sr_raw_csv_parser_restores_upstream_empty_none_choice(tmp_path: Path) -> None:
    source = tmp_path / "question_only_college_computer_science_test.csv"
    _write_csv(source, [["Question", "", "III only", "I and II only", "I, II, and III", "D"]])

    [record] = list(_iter_subject_csv(source, subject="college_computer_science"))

    assert record["A"] == "None"
    assert record["source_repairs"] == ["empty_option_A_to_None"]


def test_mmlu_sr_specs_pin_raw_variant_directories(tmp_path: Path) -> None:
    specs = (
        prepare_mmlu_sr_question_only_spec(tmp_path),
        prepare_mmlu_sr_answer_only_spec(tmp_path),
        prepare_mmlu_sr_question_and_answer_spec(tmp_path),
    )

    assert [spec.name for spec in specs] == [
        "mmlu_sr_question_only",
        "mmlu_sr_answer_only",
        "mmlu_sr_question_and_answer",
    ]
    assert all(spec.revision == _DATASET_REVISION for spec in specs)
    assert [spec._allow_patterns for spec in specs] == [
        ["question_only_test/*.csv"],
        ["answer_only_test/*.csv"],
        ["question_and_answer_test/*.csv"],
    ]
    assert all(spec.required_fields == ("question", "answer", "A", "B", "C", "D") for spec in specs)


def test_mmlu_sr_specs_reject_non_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="仅提供 test split"):
        prepare_mmlu_sr_question_only_spec(tmp_path, "train")
