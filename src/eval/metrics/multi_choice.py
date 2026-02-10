from __future__ import annotations

"""Multiple-choice evaluation over canonical `results/completions` JSONL."""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable


from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.results.schema import make_eval_payload, strict_nonneg_int

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LETTER_RE = re.compile(r"[A-Z]")


@dataclass(slots=True)
class MultipleChoiceMetrics:
    accuracy: float
    accuracy_by_subject: dict[str | None, float]
    samples: int
    payloads: list[dict]


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_completions(source: Iterable[dict] | str | Path) -> Iterable[dict]:
    if isinstance(source, (str, Path)):
        yield from _iter_jsonl(source)
        return
    yield from source


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
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
) -> MultipleChoiceMetrics:
    """Evaluate multiple-choice completions and return canonical eval payloads."""

    dataset = list(JsonlMultipleChoiceLoader(str(dataset_path)).load())
    total = 0
    correct = 0
    subject_totals: dict[str | None, tuple[int, int]] = {}
    eval_payloads: list[dict] = []

    for payload in _iter_completions(completions):
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
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

        eval_payloads.append(
            make_eval_payload(
                payload,
                is_passed=passed,
                answer=predicted or "",
                ref_answer=answer_letter or "",
            )
        )

    accuracy_by_subject = {
        subj: (hits / count if count else 0.0) for subj, (count, hits) in subject_totals.items()
    }
    accuracy = correct / total if total else 0.0
    return MultipleChoiceMetrics(
        accuracy=accuracy,
        accuracy_by_subject=accuracy_by_subject,
        samples=total,
        payloads=eval_payloads,
    )


__all__ = ["MultipleChoiceMetrics", "evaluate_multiple_choice"]
