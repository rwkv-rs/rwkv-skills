from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import UrlDownloadFile, UrlFilesJsonlDatasetSpec, read_jsonl_items

_REPO_REVISION = "84ab72d94318290aad2e4ec820d535a95a1f7552"
_RAW_ROOT = f"https://raw.githubusercontent.com/ruixiangcui/AGIEval/{_REPO_REVISION}/data/v1_1"
_SUBJECT_LANGUAGES = {
    "aqua-rat": "en",
    "gaokao-biology": "zh",
    "gaokao-chemistry": "zh",
    "gaokao-chinese": "zh",
    "gaokao-english": "en",
    "gaokao-geography": "zh",
    "gaokao-history": "zh",
    "gaokao-mathqa": "zh",
    "gaokao-physics": "zh",
    "jec-qa-ca": "zh",
    "jec-qa-kd": "zh",
    "logiqa-en": "en",
    "logiqa-zh": "zh",
    "lsat-ar": "en",
    "lsat-lr": "en",
    "lsat-rc": "en",
    "sat-en": "en",
    "sat-math": "en",
}
_EXPECTED_ROWS = 5940
_EXPECTED_EXCLUDED_MULTI_ANSWER = 7
_EXPECTED_EXCLUDED_MALFORMED = 1


class UnsupportedMultiAnswerError(ValueError):
    pass


class UnsupportedMalformedQuestionError(ValueError):
    pass


def _answer_letter(raw_label: Any, subject: str) -> str:
    if isinstance(raw_label, list):
        if len(raw_label) != 1:
            raise UnsupportedMultiAnswerError(f"AGIEval {subject} expected one answer, found {raw_label!r}")
        raw_label = raw_label[0]
    answer = str(raw_label or "").strip().upper()
    compact_answer = re.sub(r"[^A-Z]", "", answer)
    if len(compact_answer) > 1:
        raise UnsupportedMultiAnswerError(f"AGIEval {subject} expected one answer, found {raw_label!r}")
    if len(answer) != 1 or answer not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        raise ValueError(f"AGIEval {subject} contains invalid answer {raw_label!r}")
    return answer


def _strip_choice_prefix(text: str, label: str) -> str:
    patterns = (
        rf"^\s*\({re.escape(label)}\)\s*",
        rf"^\s*{re.escape(label)}[\.\):：、]\s*",
    )
    for pattern in patterns:
        stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
        if stripped != text:
            return stripped.strip()
    return text.strip()


def _parse_row(row: dict[str, Any], subject: str) -> dict[str, Any]:
    question = str(row.get("question") or "").strip()
    options = row.get("options")
    if not question:
        raise ValueError(f"AGIEval {subject} contains an empty question")
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise ValueError(f"AGIEval {subject} options must contain between 2 and 26 choices")

    answer = _answer_letter(row.get("label"), subject)
    answer_index = ord(answer) - ord("A")
    if answer_index >= len(options):
        raise ValueError(f"AGIEval {subject} answer {answer!r} is outside {len(options)} choices")

    payload: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "subject": subject.replace("-", "_"),
        "subset": _SUBJECT_LANGUAGES[subject],
        "source_subject": subject,
    }
    passage = row.get("passage")
    if isinstance(passage, str) and passage.strip():
        payload["context"] = passage.strip()
    for index, option in enumerate(options):
        label = chr(ord("A") + index)
        choice = _strip_choice_prefix(str(option), label)
        if not choice:
            raise UnsupportedMalformedQuestionError(f"AGIEval {subject} contains an empty {label} choice")
        payload[label] = choice
    return payload


def _load_records(source_root: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("AGIEval-MCQ only provides the test split")

    records: list[dict[str, Any]] = []
    excluded_multi_answer = 0
    excluded_malformed = 0
    for subject in _SUBJECT_LANGUAGES:
        path = source_root / "data" / "v1_1" / f"{subject}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in read_jsonl_items(path):
            try:
                records.append(_parse_row(dict(row), subject))
            except UnsupportedMultiAnswerError:
                excluded_multi_answer += 1
            except UnsupportedMalformedQuestionError:
                excluded_malformed += 1
    if excluded_multi_answer != _EXPECTED_EXCLUDED_MULTI_ANSWER:
        raise ValueError(
            f"AGIEval-MCQ expected {_EXPECTED_EXCLUDED_MULTI_ANSWER} multi-answer exclusions, "
            f"found {excluded_multi_answer}"
        )
    if excluded_malformed != _EXPECTED_EXCLUDED_MALFORMED:
        raise ValueError(
            f"AGIEval-MCQ expected {_EXPECTED_EXCLUDED_MALFORMED} malformed exclusion, "
            f"found {excluded_malformed}"
        )
    if len(records) != _EXPECTED_ROWS:
        raise ValueError(f"AGIEval-MCQ expected {_EXPECTED_ROWS} rows, found {len(records)}")
    return records


class AgievalMcqDatasetSpec(UrlFilesJsonlDatasetSpec):
    def manifest_extra(self) -> dict[str, Any]:
        return {
            **super().manifest_extra(),
            "repo_revision": _REPO_REVISION,
            "included_subjects": list(_SUBJECT_LANGUAGES),
            "excluded_multi_answer_rows": _EXPECTED_EXCLUDED_MULTI_ANSWER,
            "excluded_malformed_rows": _EXPECTED_EXCLUDED_MALFORMED,
            "excluded_diagnostic_files": [
                "math.jsonl",
                "gaokao-mathcloze.jsonl",
                "sat-en-without-passage.jsonl",
            ],
        }


@MULTIPLE_CHOICE_REGISTRY.register_spec("agieval_mcq")
def prepare_agieval_mcq_spec(output_root: Path, split: str = "test") -> AgievalMcqDatasetSpec:
    files = [
        UrlDownloadFile(
            relative_path=Path("data") / "v1_1" / f"{subject}.jsonl",
            url=f"{_RAW_ROOT}/{subject}.jsonl",
        )
        for subject in _SUBJECT_LANGUAGES
    ]
    return AgievalMcqDatasetSpec(
        "agieval_mcq",
        output_root,
        split,
        files=files,
        load_downloaded_records=lambda source_root: _load_records(source_root, split),
        required_fields=("question", "answer", "A", "B"),
        tasks=8,
    )


__all__ = ["prepare_agieval_mcq_spec"]
