from __future__ import annotations

"""Free-form QA evaluation over canonical `results/completions` JSONL."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable
import unicodedata

import orjson

from openai import OpenAI

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k
from src.eval.results.schema import make_eval_payload

_WHITESPACE_RE = re.compile(r"\s+")
_PREFERRED_ANSWER_KEYS = (
    "expected_answer",
    "reference_answer",
    "target",
    "final_answer",
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


@dataclass(slots=True)
class LLMJudgeConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_workers: int = 32
    prompt_template: str = (
        "You are a rigorous AI judge. Your task is to evaluate whether a student's "
        "answer is semantically completely equivalent to the reference answer, based on "
        "the provided question and reference answer.\\n\\nInput:\\nQuestion: <Q>\\nReference Answer: <REF>\\n"
        "Student's Answer: <A>\\n\\nOutput Format:\\nStrictly adhere to the output format: Only output 'True' or 'False'."
    )


class LLMJudge:
    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]:
        """Return judge flags for (question, reference, prediction) items."""

        def worker(entry: tuple[str, str, str]) -> bool:
            question, reference, prediction = entry
            prompt = self.config.prompt_template
            prompt = prompt.replace("<Q>", question)
            prompt = prompt.replace("<REF>", reference)
            prompt = prompt.replace("<A>", prediction)
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

        results: list[bool] = [False for _ in range(len(items))]
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(worker, entry): idx for idx, entry in enumerate(items)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results


def evaluate_free_response(
    completions_path: str | Path,
    *,
    dataset_path: str | Path,
    eval_output_path: str | Path | None,
    judge: LLMJudge | None = None,
) -> FreeResponseEvaluation:
    """Evaluate free-response completions and write canonical evaluator JSONL."""

    dataset = list(JsonlFreeAnswerLoader(str(dataset_path)))
    eval_path = Path(eval_output_path) if eval_output_path is not None else None
    if eval_path is not None:
        eval_path.parent.mkdir(parents=True, exist_ok=True)

    payloads: list[dict] = []
    exact_flags: list[bool] = []
    answers: list[str] = []
    ref_answers: list[str] = []
    judge_inputs: list[tuple[str, str, str]] = []

    # First pass: compute exact + gather judge inputs
    for payload in _iter_jsonl(completions_path):
        sample_index = int(payload.get("sample_index", 0))
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
            exact = _normalize_text(prediction) == _normalize_text(reference)

        payloads.append(payload)
        exact_flags.append(bool(exact))
        answers.append(prediction)
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

    if eval_path is not None:
        with eval_path.open("wb") as out_f:
            for idx, payload in enumerate(payloads):
                sample_index = int(payload.get("sample_index", 0))
                repeat_index = int(payload.get("repeat_index", 0))
                if judge_flags is not None:
                    passed = bool(judge_flags[idx])
                else:
                    passed = bool(exact_flags[idx])
                rows_for_at_k.append((sample_index, repeat_index, passed))
                out_f.write(
                    orjson.dumps(
                        make_eval_payload(
                            payload,
                            is_passed=passed,
                            answer=answers[idx],
                            ref_answer=ref_answers[idx],
                        ),
                        option=orjson.OPT_APPEND_NEWLINE,
                    )
                )
    else:
        for idx, payload in enumerate(payloads):
            sample_index = int(payload.get("sample_index", 0))
            repeat_index = int(payload.get("repeat_index", 0))
            if judge_flags is not None:
                passed = bool(judge_flags[idx])
            else:
                passed = bool(exact_flags[idx])
            rows_for_at_k.append((sample_index, repeat_index, passed))

    samples = len(payloads)
    exact_accuracy = total_exact / samples if samples else 0.0
    judge_accuracy = (total_judge / samples) if (judge_flags is not None and samples) else None
    return FreeResponseEvaluation(
        exact_accuracy=exact_accuracy,
        judge_accuracy=judge_accuracy,
        samples=samples,
        rows=rows_for_at_k,
    )


__all__ = [
    "LLMJudge",
    "LLMJudgeConfig",
    "FreeResponseEvaluation",
    "evaluate_free_response",
    "compute_avg_at_k",
    "compute_pass_at_k",
    "resolve_reference_answer",
]
