from __future__ import annotations

"""Multiple-choice evaluation over canonical `results/completions` JSONL."""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable

import orjson

from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.results.schema import make_eval_payload

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTER_RE = re.compile(r"[A-Z]")


@dataclass(slots=True)
class MultipleChoiceMetrics:
    accuracy: float
    accuracy_by_subject: dict[str | None, float]
    samples: int


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _max_stage_index(payload: dict) -> int:
    stage = 0
    for key in payload:
        if key.startswith("completion") and key.removeprefix("completion").isdigit():
            stage = max(stage, int(key.removeprefix("completion")))
    return stage


def _extract_choice_letter(token_text: str) -> str | None:
    match = _LETTER_RE.search(token_text or "")
    return match.group(0) if match else None


def evaluate_multiple_choice(
    completions_path: str | Path,
    *,
    dataset_path: str | Path,
    eval_output_path: str | Path,
) -> MultipleChoiceMetrics:
    """Evaluate multiple-choice completions and write canonical evaluator JSONL."""

    dataset = list(JsonlMultipleChoiceLoader(str(dataset_path)).load())
    eval_output_path = Path(eval_output_path)
    eval_output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    subject_totals: dict[str | None, tuple[int, int]] = {}

    with eval_output_path.open("wb") as out_f:
        for payload in _iter_jsonl(completions_path):
            sample_index = int(payload.get("sample_index", 0))
            last_stage = _max_stage_index(payload)
            token_text = str(payload.get(f"completion{last_stage}", ""))
            predicted = _extract_choice_letter(token_text)
            if sample_index < 0 or sample_index >= len(dataset):
                # Unknown sample index -> mark incorrect, but still emit an eval row.
                passed = False
                subject = None
                answer_letter = None
            else:
                record = dataset[sample_index]
                subject = record.subject
                answer_letter = ALPHABET[record.answer_index]
                passed = bool(predicted) and predicted == answer_letter

            total += 1
            if passed:
                correct += 1

            sub_total, sub_hits = subject_totals.get(subject, (0, 0))
            sub_total += 1
            if passed:
                sub_hits += 1
            subject_totals[subject] = (sub_total, sub_hits)

            out_f.write(
                orjson.dumps(
                    make_eval_payload(
                        payload,
                        is_passed=passed,
                        answer=predicted or "",
                        ref_answer=answer_letter or "",
                    ),
                    option=orjson.OPT_APPEND_NEWLINE,
                )
            )

    accuracy_by_subject = {
        subj: (hits / count if count else 0.0) for subj, (count, hits) in subject_totals.items()
    }
    accuracy = correct / total if total else 0.0
    return MultipleChoiceMetrics(accuracy=accuracy, accuracy_by_subject=accuracy_by_subject, samples=total)


__all__ = ["MultipleChoiceMetrics", "evaluate_multiple_choice"]
