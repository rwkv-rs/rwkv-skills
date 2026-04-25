from __future__ import annotations

"""Free-form QA evaluation over canonical `results/completions` JSONL."""

import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import tqdm
from openai import OpenAI

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k
from src.eval.results.schema import make_eval_payload, strict_nonneg_int

_WHITESPACE_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_PREFERRED_ANSWER_KEYS = (
    "expected_answer",
    "reference_answer",
    "target",
    "final_answer",
)

DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE = (
    "You are a rigorous AI judge. Your task is to evaluate whether a student's "
    "answer is semantically completely equivalent to the reference answer, based on "
    "the provided question and reference answer.\\n\\nInput:\\nQuestion: <Q>\\nReference Answer: <REF>\\n"
    "Student's Answer: <A>\\n\\nOutput Format:\\nStrictly adhere to the output format: Only output 'True' or 'False'."
)


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\\ ", " ").replace("\u00a0", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized.strip())
    return normalized


def _normalize_answer_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        normalized = str(value)
        return normalized.strip() or None
    normalized = str(value).strip()
    return normalized or None


def _extract_number(text: str) -> str | None:
    if not text:
        return None
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    return value or None


def _format_answer_for_storage(prediction: str, reference: str) -> str:
    ref_num = _extract_number(reference)
    if ref_num is not None:
        pred_num = _extract_number(prediction)
        return pred_num or ""
    return _normalize_text(prediction)


def _is_exact_match(prediction: str, reference: str) -> bool:
    ref_num = _extract_number(reference)
    if ref_num is not None:
        pred_num = _extract_number(prediction)
        if pred_num is not None:
            return pred_num == ref_num
    return _normalize_text(prediction) == _normalize_text(reference)


def resolve_reference_answer(record: FreeAnswerRecord) -> str:
    metadata = record.metadata or {}
    for key in _PREFERRED_ANSWER_KEYS:
        normalized = _normalize_answer_value(metadata.get(key))
        if normalized:
            return normalized
    raw_record = metadata.get("raw_record")
    if isinstance(raw_record, dict):
        for key in _PREFERRED_ANSWER_KEYS:
            normalized = _normalize_answer_value(raw_record.get(key))
            if normalized:
                return normalized
    return record.answer


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


@dataclass(slots=True)
class FreeResponseEvaluation:
    exact_accuracy: float
    judge_accuracy: float | None
    samples: int
    rows: list[tuple[int, int, bool]]
    payloads: list[dict]


@dataclass(slots=True)
class LLMJudgeConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_workers: int = 4
    max_completion_tokens: int | None = None

    max_retries: int = 3
    backoff_base: float = 0.5

    prompt_template: str = DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE


@dataclass(slots=True)
class LLMJudgeStats:
    total: int = 0
    parsed_count: int = 0
    invalid_output_count: int = 0
    request_error_count: int = 0
    invalid_output_examples: list[str] = field(default_factory=list)
    request_error_examples: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return self.invalid_output_count + self.request_error_count

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "parsed_count": self.parsed_count,
            "invalid_output_count": self.invalid_output_count,
            "request_error_count": self.request_error_count,
            "error_count": self.error_count,
            "invalid_output_examples": self.invalid_output_examples,
            "request_error_examples": self.request_error_examples,
        }


class LLMJudge:
    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.last_run_stats: LLMJudgeStats | None = None

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]:
        """Return judge flags for (question, reference, prediction) items."""

        def worker(entry: tuple[str, str, str]) -> tuple[bool, str, str | None]:
            question, reference, prediction = entry
            prompt = self.config.prompt_template
            prompt = prompt.replace("<Q>", question)
            prompt = prompt.replace("<REF>", reference)
            prompt = prompt.replace("<A>", prediction)

            # Retry loop with exponential backoff
            last_error = ""
            last_error_kind = "request_error"
            for attempt in range(self.config.max_retries + 1):
                try:
                    request_kwargs: dict[str, Any] = {
                        "model": self.config.model,
                        "stream": False,
                        "temperature": 0.0,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if self.config.max_completion_tokens is not None:
                        request_kwargs["max_tokens"] = self.config.max_completion_tokens
                    response = self.client.chat.completions.create(
                        **request_kwargs,
                    )
                    content = (response.choices[0].message.content or "").strip()

                    if content not in {"True", "False"}:
                        last_error_kind = "invalid_output"
                        last_error = f"invalid output: {content!r}"
                        raise ValueError(f"LLM judge 输出非法值: {content!r}")

                    return content == "True", "parsed", None

                except Exception as exc:
                    if not last_error:
                        last_error = repr(exc)
                    if last_error_kind != "invalid_output":
                        last_error_kind = "request_error"
                    # Final attempt failed: don't crash overall eval, just return False
                    if attempt == self.config.max_retries:
                        return False, last_error_kind, last_error

                    backoff = self.config.backoff_base * (2**attempt)
                    time.sleep(backoff)

            return False, last_error_kind, last_error or None

        results: list[bool] = [False for _ in range(len(items))]
        stats = LLMJudgeStats(total=len(items))
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(worker, entry): idx for idx, entry in enumerate(items)
            }
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="LLM judging"
            ):
                idx = futures[future]
                passed, status, detail = future.result()
                results[idx] = passed
                if status == "parsed":
                    stats.parsed_count += 1
                elif status == "invalid_output":
                    stats.invalid_output_count += 1
                    if detail and len(stats.invalid_output_examples) < 5:
                        stats.invalid_output_examples.append(detail)
                else:
                    stats.request_error_count += 1
                    if detail and len(stats.request_error_examples) < 5:
                        stats.request_error_examples.append(detail)
        self.last_run_stats = stats
        return results


def evaluate_free_response(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    judge: LLMJudge | None = None,
) -> FreeResponseEvaluation:
    """Evaluate free-response completions and return canonical eval payloads."""

    dataset = list(JsonlFreeAnswerLoader(str(dataset_path)))
    payloads: list[dict] = []
    exact_flags: list[bool] = []
    answers: list[str] = []
    ref_answers: list[str] = []
    judge_inputs: list[tuple[str, str, str]] = []

    # First pass: compute exact + gather judge inputs
    for payload in _iter_completions(completions):
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        if sample_index < 0 or sample_index >= len(dataset):
            prediction = ""
            reference = ""
            question = ""
            exact = False
        else:
            record = dataset[sample_index]
            question = record.question
            reference = resolve_reference_answer(record)
            last_stage = _max_stage_index(payload)
            prediction = str(payload.get(f"completion{last_stage}", "")).strip()
            exact = _is_exact_match(prediction, reference)

        payloads.append(payload)
        exact_flags.append(bool(exact))
        answers.append(_format_answer_for_storage(prediction, reference))
        ref_answers.append(reference)
        if judge is not None:
            judge_inputs.append((question, reference, prediction))

    judge_flags: list[bool] | None = None
    if judge is not None:
        judge_flags = judge.judge(judge_inputs)

    # Second pass: write eval rows
    rows_for_at_k: list[tuple[int, int, bool]] = []
    total_exact = sum(1 for flag in exact_flags if flag)
    total_judge = sum(1 for flag in (judge_flags or []) if flag) if judge_flags is not None else 0

    eval_payloads: list[dict] = []
    for idx, payload in enumerate(payloads):
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        if judge_flags is not None:
            passed = bool(judge_flags[idx])
        else:
            passed = bool(exact_flags[idx])
        rows_for_at_k.append((sample_index, repeat_index, passed))
        eval_payloads.append(
            make_eval_payload(
                payload,
                is_passed=passed,
                answer=answers[idx],
                ref_answer=ref_answers[idx],
            )
        )

    samples = len(payloads)
    exact_accuracy = total_exact / samples if samples else 0.0
    judge_accuracy = (
        (total_judge / samples) if (judge_flags is not None and samples) else None
    )
    return FreeResponseEvaluation(
        exact_accuracy=exact_accuracy,
        judge_accuracy=judge_accuracy,
        samples=samples,
        rows=rows_for_at_k,
        payloads=eval_payloads,
    )


__all__ = [
    "LLMJudge",
    "LLMJudgeConfig",
    "LLMJudgeStats",
    "DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE",
    "FreeResponseEvaluation",
    "evaluate_free_response",
    "compute_avg_at_k",
    "compute_pass_at_k",
    "resolve_reference_answer",
]
