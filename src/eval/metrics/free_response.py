from __future__ import annotations

"""Free-form QA metrics：支持 exact match 和 LLM judge。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable, Sequence
import unicodedata

import orjson

from openai import OpenAI


@dataclass(slots=True)
class FreeResponseSample:
    sample_index: int
    dataset: str
    question: str
    answer: str
    prediction: str
    subject: str | None
    cot: str | None
    problem_index: int | None = None
    sample_id: int | None = None


@dataclass(slots=True)
class FreeResponseSampleResult:
    sample: FreeResponseSample
    correct_exact: bool
    judge_correct: bool | None = None


@dataclass(slots=True)
class FreeResponseMetrics:
    exact_accuracy: float
    judge_accuracy: float | None
    samples: list[FreeResponseSampleResult]
    pass_at_k: dict[str, float] | None = None
    avg_at_k: dict[str, float] | None = None


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    """Collapse common formatting artefacts (LaTeX spacing, double spaces, NBSP)."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\\ ", " ").replace("\u00a0", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized.strip())
    return normalized


def load_samples(path: str | Path) -> list[FreeResponseSample]:
    path = Path(path)
    samples: list[FreeResponseSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            dataset = payload.get("dataset", path.stem)
            sample_index = int(payload.get("sample_index", len(samples)))
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            subject = payload.get("subject")
            cot = payload.get("output1")
            prediction = payload.get("prediction") or payload.get("output2") or ""
            samples.append(
                FreeResponseSample(
                    sample_index=sample_index,
                    dataset=dataset,
                    question=question,
                    answer=answer,
                    prediction=prediction.strip(),
                    subject=subject,
                    cot=cot,
                    problem_index=payload.get("problem_index"),
                    sample_id=payload.get("sample_id"),
                )
            )
    return samples


def evaluate_exact(samples: Iterable[FreeResponseSample]) -> FreeResponseMetrics:
    results: list[FreeResponseSampleResult] = []
    total = 0
    correct = 0
    for sample in samples:
        normalized_prediction = _normalize_text(sample.prediction)
        normalized_answer = _normalize_text(sample.answer)
        is_correct = normalized_prediction == normalized_answer
        results.append(FreeResponseSampleResult(sample, correct_exact=is_correct))
        total += 1
        if is_correct:
            correct += 1
    accuracy = correct / total if total else 0.0
    return FreeResponseMetrics(exact_accuracy=accuracy, judge_accuracy=None, samples=results)


def _estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    if total - correct < k:
        return 1.0
    product = 1.0
    for n in range(total - correct + 1, total + 1):
        product *= 1.0 - k / n
    return 1.0 - product


def compute_pass_at_k(
    samples: Iterable[FreeResponseSampleResult],
    ks: Sequence[int],
    *,
    use_judge: bool = False,
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for result in samples:
        sample = result.sample
        flag = result.correct_exact
        if use_judge and result.judge_correct is not None:
            flag = result.judge_correct
        if flag is None:
            continue
        flag_bool = bool(flag)
        if sample.problem_index is not None:
            key = f"{sample.dataset}:{sample.problem_index}"
        elif sample.question:
            key = f"{sample.dataset}:q:{sample.question.strip()}"
        else:
            key = f"{sample.dataset}:idx:{sample.sample_index}"
        grouped.setdefault(key, []).append(flag_bool)

    totals: list[int] = []
    corrects: list[int] = []
    for values in grouped.values():
        totals.append(len(values))
        corrects.append(sum(1 for flag in values if flag))

    metrics: dict[str, float] = {}
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        acc_values: list[float] = []
        for total, correct in zip(totals, corrects):
            if total < k:
                continue
            acc_values.append(_estimate_pass_at_k(total, correct, k))
        if acc_values:
            metrics[f"pass@{k}"] = sum(acc_values) / len(acc_values)
    return metrics


def compute_avg_at_k(
    samples: Iterable[FreeResponseSampleResult],
    ks: Sequence[int],
    *,
    use_judge: bool = False,
) -> dict[str, float]:
    """Compute average accuracy across the first k samples for each problem."""

    grouped: dict[str, list[tuple[int, bool]]] = {}
    for result in samples:
        sample = result.sample
        flag = result.correct_exact
        if use_judge and result.judge_correct is not None:
            flag = result.judge_correct
        if flag is None:
            continue
        sample_key = f"{sample.dataset}:idx:{sample.sample_index}"
        if sample.problem_index is not None:
            sample_key = f"{sample.dataset}:{sample.problem_index}"
        elif sample.question:
            sample_key = f"{sample.dataset}:q:{sample.question.strip()}"
        sample_id = sample.sample_id if sample.sample_id is not None else sample.sample_index
        grouped.setdefault(sample_key, []).append((int(sample_id), bool(flag)))

    metrics: dict[str, float] = {}
    for k in ks:
        k = int(k)
        if k <= 0:
            continue
        correct = 0
        total = 0
        for entries in grouped.values():
            ordered = sorted(entries, key=lambda pair: pair[0])
            if len(ordered) < k:
                continue
            selected = ordered[:k]
            correct += sum(1 for _, flag in selected if flag)
            total += k
        if total > 0:
            metrics[f"avg@{k}"] = correct / total
    return metrics


@dataclass(slots=True)
class LLMJudgeConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_workers: int = 32
    prompt_template: str = (
        "You are a rigorous AI judge. Your task is to evaluate whether a student's "
        "answer is semantically completely equivalent to the reference answer, based on "
        "the provided question and reference answer.\n\nInput:\nQuestion: <Q>\nReference Answer: <REF>\n"
        "Student's Answer: <A>\n\nOutput Format:\nStrictly adhere to the output format: Only output 'True' or 'False'."
    )


class LLMJudge:
    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def judge(self, samples: Iterable[FreeResponseSampleResult]) -> None:
        pending = [sample for sample in samples if sample.judge_correct is None]
        if not pending:
            return

        def worker(sample: FreeResponseSampleResult) -> bool:
            prompt = self.config.prompt_template
            prompt = prompt.replace("<Q>", sample.sample.question)
            prompt = prompt.replace("<REF>", sample.sample.answer)
            prompt = prompt.replace("<A>", sample.sample.prediction)
            response = self.client.chat.completions.create(
                model=self.config.model,
                stream=False,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (response.choices[0].message.content or "").strip()
            if content not in {"True", "False"}:
                raise ValueError(f"LLM judge 输出非法值: {content}")
            return content == "True"

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(worker, sample): sample for sample in pending
            }
            for future in as_completed(futures):
                sample = futures[future]
                sample.judge_correct = future.result()


def evaluate_with_judge(samples: Iterable[FreeResponseSample], judge: LLMJudge) -> FreeResponseMetrics:
    metrics = evaluate_exact(samples)
    judge.judge(metrics.samples)
    total = 0
    correct = 0
    for sample in metrics.samples:
        if sample.judge_correct is None:
            continue
        total += 1
        if sample.judge_correct:
            correct += 1
    judge_accuracy = correct / total if total else None
    metrics.judge_accuracy = judge_accuracy
    return metrics


# ---------------------------------------------------------------------------
# Eval JSONL helpers
# ---------------------------------------------------------------------------
def write_sample_results(
    results: Iterable[FreeResponseSampleResult],
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        for result in results:
            sample = result.sample
            payload = {
                "sample_index": sample.sample_index,
                "dataset": sample.dataset,
                "question": sample.question,
                "answer": sample.answer,
                "prediction": sample.prediction,
                "subject": sample.subject,
                "cot": sample.cot,
                "correct_exact": result.correct_exact,
                "judge_correct": result.judge_correct,
                "problem_index": sample.problem_index,
                "sample_id": sample.sample_id,
            }
            fh.write(orjson.dumps(payload, option=orjson.OPT_APPEND_NEWLINE))
    return target


__all__ = [
    "FreeResponseSample",
    "FreeResponseSampleResult",
    "FreeResponseMetrics",
    "load_samples",
    "evaluate_exact",
    "LLMJudge",
    "LLMJudgeConfig",
    "evaluate_with_judge",
    "write_sample_results",
    "compute_pass_at_k",
    "compute_avg_at_k",
]
