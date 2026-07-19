from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import HfRepoJsonlDatasetSpec, read_parquet_items

_DATASET_ID = "li-lab/MMLU-ProX"
_DATASET_REVISION = "8e6106a6c6ce1c5027e66cc338143cf997b2aa09"
_LANGUAGES = (
    "af",
    "ar",
    "bn",
    "cs",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "mr",
    "ne",
    "pt",
    "ru",
    "sr",
    "sw",
    "te",
    "th",
    "uk",
    "ur",
    "vi",
    "wo",
    "yo",
    "zh",
    "zu",
)
_EXPECTED_ROWS_PER_LANGUAGE = 11759
# The official zh row 3299 has option_8 empty but option_9 populated. The
# pipeline's canonical MC format cannot preserve a missing I alongside J
# without relabeling choices, so exclude that one malformed source row.
_KNOWN_MALFORMED_ROWS = {("zh", 3299)}
_EXPECTED_ROWS = len(_LANGUAGES) * _EXPECTED_ROWS_PER_LANGUAGE - len(_KNOWN_MALFORMED_ROWS)


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def _parse_row(row: dict[str, Any], language: str) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip().upper()
    if not question:
        raise ValueError(f"MMLU-ProX {language} contains an empty question")
    if answer not in "ABCDEFGHIJ" or len(answer) != 1:
        raise ValueError(f"MMLU-ProX {language} contains invalid answer {answer!r}")

    choices: list[str] = []
    reached_gap = False
    for index in range(10):
        choice = str(row.get(f"option_{index}") or "").strip()
        if not choice:
            reached_gap = True
            continue
        if reached_gap:
            raise ValueError(f"MMLU-ProX {language} contains non-contiguous options")
        choices.append(choice)
    if not 3 <= len(choices) <= 10:
        raise ValueError(f"MMLU-ProX {language} contains {len(choices)} choices")
    answer_index = ord(answer) - ord("A")
    if answer_index >= len(choices):
        raise ValueError(f"MMLU-ProX {language} answer {answer!r} is outside {len(choices)} choices")

    category = _normalize_label(str(row.get("category") or "unknown"))
    source_answer_index = row.get("answer_index")
    payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "subject": category,
        "subset": language,
        "language": language,
        "category": category,
        "source": str(row.get("src") or "").strip(),
        "source_id": row.get("question_id"),
        "source_question_id": row.get("question_id_src"),
        "source_answer_index": source_answer_index,
        "answer_index_consistent": source_answer_index == answer_index,
    }
    for index, choice in enumerate(choices):
        payload[chr(ord("A") + index)] = choice
    return payload


def _is_known_malformed(row: dict[str, Any], language: str) -> bool:
    return (language, row.get("question_id")) in _KNOWN_MALFORMED_ROWS


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("MMLU-ProX benchmark only supports the test split")

    records: list[dict[str, Any]] = []
    for language in _LANGUAGES:
        path = source_root / language / "test-00000-of-00001.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        source_rows = [dict(row) for row in read_parquet_items(path)]
        language_records = [
            _parse_row(row, language)
            for row in source_rows
            if not _is_known_malformed(row, language)
        ]
        expected_language_rows = _EXPECTED_ROWS_PER_LANGUAGE - sum(
            known_language == language for known_language, _source_id in _KNOWN_MALFORMED_ROWS
        )
        if len(language_records) != expected_language_rows:
            raise ValueError(
                f"MMLU-ProX {language} expected {expected_language_rows} test rows, "
                f"found {len(language_records)}"
            )
        records.extend(language_records)
    if len(records) != _EXPECTED_ROWS:
        raise ValueError(f"MMLU-ProX expected {_EXPECTED_ROWS} test rows, found {len(records)}")
    return records


@MULTIPLE_CHOICE_REGISTRY.register_spec("mmlu_prox")
def prepare_mmlu_prox_spec(output_root: Path, split: str = "test") -> HfRepoJsonlDatasetSpec:
    return HfRepoJsonlDatasetSpec(
        "mmlu_prox",
        output_root,
        split,
        repo=_DATASET_ID,
        revision=_DATASET_REVISION,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B", "C"),
        allow_patterns=[f"{language}/test-00000-of-00001.parquet" for language in _LANGUAGES],
    )


__all__ = ["prepare_mmlu_prox_spec"]
