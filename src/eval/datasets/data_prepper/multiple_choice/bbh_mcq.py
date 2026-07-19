from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import UrlDownloadFile, UrlFilesJsonlDatasetSpec

_REPO_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
_RAW_ROOT = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{_REPO_REVISION}/bbh"
_MCQ_TASKS = (
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
)
_NON_MCQ_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "dyck_languages",
    "formal_fallacies",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "sports_understanding",
    "web_of_lies",
    "word_sorting",
)
_EXPECTED_ROWS = 4070
_EXPECTED_EXCLUDED_MALFORMED = 4
_OPTION_PATTERN = re.compile(r"(?m)^\(([A-Z])\)\s*(.+)$")


class UnsupportedMalformedQuestionError(ValueError):
    pass


def _parse_row(row: dict[str, Any], task: str) -> dict[str, Any]:
    raw_input = str(row.get("input") or "").strip()
    target = str(row.get("target") or "").strip().upper()
    matches = list(_OPTION_PATTERN.finditer(raw_input))
    if len(matches) < 2:
        raise UnsupportedMalformedQuestionError(f"BBH {task} contains fewer than two choices")
    if not re.fullmatch(r"\([A-Z]\)", target):
        raise UnsupportedMalformedQuestionError(f"BBH {task} contains non-letter target {target!r}")

    source_labels = [match.group(1) for match in matches]
    answer_label = target[1]
    try:
        answer_index = source_labels.index(answer_label)
    except ValueError as exc:
        raise UnsupportedMalformedQuestionError(
            f"BBH {task} target {answer_label!r} is outside {source_labels!r}"
        ) from exc

    question = raw_input[: matches[0].start()].strip()
    question = re.sub(r"(?i)\n?options:\s*$", "", question).strip()
    if not question:
        raise UnsupportedMalformedQuestionError(f"BBH {task} contains an empty question")

    payload: dict[str, Any] = {
        "question": question,
        "answer": chr(ord("A") + answer_index),
        "subject": task,
        "subset": "reasoning",
        "source_answer_label": answer_label,
    }
    for index, match in enumerate(matches):
        choice = match.group(2).strip()
        if not choice:
            raise UnsupportedMalformedQuestionError(f"BBH {task} contains an empty choice")
        payload[chr(ord("A") + index)] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("BBH-MCQ only provides the test split")

    records: list[dict[str, Any]] = []
    excluded_malformed = 0
    for task in _MCQ_TASKS:
        path = source_root / "bbh" / f"{task}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise ValueError(f"BBH {task} examples must be a list")
        for row in examples:
            try:
                records.append(_parse_row(dict(row), task))
            except UnsupportedMalformedQuestionError:
                excluded_malformed += 1

    if excluded_malformed != _EXPECTED_EXCLUDED_MALFORMED:
        raise ValueError(
            f"BBH-MCQ expected {_EXPECTED_EXCLUDED_MALFORMED} malformed exclusions, found {excluded_malformed}"
        )
    if len(records) != _EXPECTED_ROWS:
        raise ValueError(f"BBH-MCQ expected {_EXPECTED_ROWS} rows, found {len(records)}")
    return records


class BbhMcqDatasetSpec(UrlFilesJsonlDatasetSpec):
    def manifest_extra(self) -> dict[str, Any]:
        return {
            **super().manifest_extra(),
            "repo_revision": _REPO_REVISION,
            "included_mcq_tasks": list(_MCQ_TASKS),
            "excluded_non_mcq_tasks": list(_NON_MCQ_TASKS),
            "excluded_malformed_rows": _EXPECTED_EXCLUDED_MALFORMED,
        }


@MULTIPLE_CHOICE_REGISTRY.register_spec("bbh_mcq")
def prepare_bbh_mcq_spec(output_root: Path, split: str = "test") -> BbhMcqDatasetSpec:
    files = [
        UrlDownloadFile(relative_path=Path("bbh") / f"{task}.json", url=f"{_RAW_ROOT}/{task}.json")
        for task in _MCQ_TASKS
    ]
    return BbhMcqDatasetSpec(
        "bbh_mcq",
        output_root,
        split,
        files=files,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B"),
        tasks=8,
    )


__all__ = ["prepare_bbh_mcq_spec"]
