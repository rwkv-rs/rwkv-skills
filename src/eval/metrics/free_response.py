"""Free-response evaluation using full completions and math_verify."""

from __future__ import annotations

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import tqdm
from openai import OpenAI

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.k_values import NumericK, filter_metrics_by_k
from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k
from src.eval.results.io import iter_jsonl
from src.eval.results.schema import make_eval_payload, strict_nonneg_int

USER_SENTINEL = "\nUser:"
REPAIR_FINAL_CUE = "Therefore, the final answer is "
STRATEGY_A = "strategy_a"
STRATEGY_B = "strategy_b"
STRATEGY_C = "strategy_c"
STRATEGY_GROUPS = (STRATEGY_A, STRATEGY_B, STRATEGY_C)
STRATEGY_LABELS = {
    STRATEGY_A: "strategy_a",
    STRATEGY_B: "strategy_b",
    STRATEGY_C: "strategy_c",
}

_WHITESPACE_RE = re.compile(r"\s+")
_JUDGEMENT_LABEL_RE = re.compile(r"\bjudg(?:e)?ment\s*:\s*(yes|no)\b", re.IGNORECASE)
_TRAILING_JUDGEMENT_LABEL_RE = re.compile(
    r"(?:\bjudg(?:e)?ment\s*:\s*)?\b(yes|no)\b\s*[.!。]?\s*$",
    re.IGNORECASE,
)
_PREFERRED_ANSWER_KEYS = (
    "expected_judgement",
    "expected_answer",
    "reference_answer",
    "target",
    "final_answer",
)

DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE = (
    "You are a rigorous AI judge. Your task is to evaluate whether a student's "
    "answer is mathematically equivalent to the reference answer, based on "
    "the provided question and reference answer. Accept different wording or formatting "
    "only when the mathematical value is unchanged and all required components are present.\\n\\n"
    "Input:\\nQuestion: <Q>\\nReference Answer: <REF>\\n"
    "Student's Answer: <A>\\n\\nOutput Format:\\nStrictly adhere to the output format: Only output 'True' or 'False'."
)


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\\ ", " ").replace("\u00a0", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized.strip())
    return normalized


def _is_exact_match(prediction: str, reference: str) -> bool:
    return bool(reference) and _normalize_text(prediction) == _normalize_text(reference)


def _extract_judgement_label(value: str) -> str | None:
    normalized = _normalize_text(value)
    if not normalized:
        return None
    matches = _JUDGEMENT_LABEL_RE.findall(normalized)
    if matches:
        return f"Judgement: {matches[-1].capitalize()}"
    trailing = _TRAILING_JUDGEMENT_LABEL_RE.search(normalized[-200:])
    if trailing:
        return f"Judgement: {trailing.group(1).capitalize()}"
    return None


def _is_judgement_reference(reference: str) -> bool:
    return _extract_judgement_label(reference) is not None


def _short_text(value: str, *, limit: int = 1200) -> str:
    normalized = _normalize_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


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


def _iter_completions(source: Iterable[dict] | str | Path) -> Iterable[dict]:
    if isinstance(source, (str, Path)):
        yield from iter_jsonl(source)
        return
    yield from source


@dataclass(slots=True)
class FreeResponseEvaluation:
    metrics_by_group: dict[str, dict[str, float]]
    rows_by_group: dict[str, list[tuple[int, int, bool]]]
    samples: int
    payloads: list[dict]
    payloads_by_group: dict[str, list[dict]] = field(default_factory=dict)
    judge_stats_by_group: dict[str, dict[str, object]] = field(default_factory=dict)
    primary_group: str = STRATEGY_A

    @property
    def exact_accuracy(self) -> float:
        return float(self.metrics_by_group.get(self.primary_group, {}).get("exact_accuracy", 0.0))

    @property
    def judge_accuracy(self) -> float | None:
        value = self.metrics_by_group.get(self.primary_group, {}).get("judge_accuracy")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def rows(self) -> list[tuple[int, int, bool]]:
        return list(self.rows_by_group.get(self.primary_group, []))


@dataclass(slots=True)
class LLMJudgeConfig:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_s: float = 60.0
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
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=max(1.0, float(config.timeout_s)),
            max_retries=0,
        )
        self.last_run_stats: LLMJudgeStats | None = None

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]:
        """Return judge flags for (question, reference, prediction) items."""

        def worker(entry: tuple[str, str, str]) -> tuple[bool, str, str | None]:
            question, reference, prediction = entry
            prompt = self.config.prompt_template
            prompt = prompt.replace("<Q>", question)
            prompt = prompt.replace("<REF>", reference)
            prompt = prompt.replace("<A>", prediction)

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
                    if "qwen3" in self.config.model.lower():
                        request_kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": False}
                        }
                    if self.config.max_completion_tokens is not None:
                        request_kwargs["max_tokens"] = self.config.max_completion_tokens
                    response = self.client.chat.completions.create(**request_kwargs)
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


@dataclass(slots=True)
class _MathVerifyResult:
    passed: bool
    answer: str
    fail_reason: str


@dataclass(slots=True)
class _ScoredCompletion:
    source_payload: dict[str, Any]
    sample_index: int
    repeat_index: int
    question: str
    reference: str
    scoring_text: str
    display_answer: str
    math_passed: bool
    final_passed: bool
    fail_reason: str


@lru_cache(maxsize=1)
def _load_math_verify() -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    return parse, verify


def _reference_expr(reference: str) -> str:
    if "\\boxed" in reference:
        return reference
    return f"$\\boxed{{{reference}}}$"


def _math_verify(reference: str, scoring_text: str) -> _MathVerifyResult:
    api = _load_math_verify()
    display_answer = _short_text(scoring_text)
    if _is_exact_match(scoring_text, reference):
        return _MathVerifyResult(
            passed=True,
            answer=display_answer,
            fail_reason="",
        )
    if api is None:
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason="math_verify_missing",
        )
    parse, verify = api
    try:
        gold = parse(_reference_expr(reference))
    except Exception as exc:  # noqa: BLE001
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason=f"reference_parse_error:{type(exc).__name__}",
        )
    try:
        pred = parse(scoring_text)
    except Exception as exc:  # noqa: BLE001
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason=f"prediction_parse_error:{type(exc).__name__}",
        )
    if pred:
        display_answer = _short_text(_parsed_answer_text(pred))
    try:
        passed = bool(pred and verify(gold, pred, strict=False))
    except Exception as exc:  # noqa: BLE001
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason=f"math_verify_error:{type(exc).__name__}",
        )
    return _MathVerifyResult(
        passed=passed,
        answer=display_answer,
        fail_reason="" if passed else "math_verify_false",
    )


def _judgement_verify(reference: str, scoring_text: str) -> _MathVerifyResult:
    expected = _extract_judgement_label(reference)
    actual = _extract_judgement_label(scoring_text)
    display_answer = actual or _short_text(scoring_text)
    passed = bool(expected and actual and expected == actual)
    if passed:
        fail_reason = ""
    elif actual is None:
        fail_reason = "judgement_label_missing"
    else:
        fail_reason = "judgement_label_mismatch"
    return _MathVerifyResult(
        passed=passed,
        answer=display_answer,
        fail_reason=fail_reason,
    )


def _parsed_answer_text(parsed: Any) -> str:
    if isinstance(parsed, (list, tuple)) and parsed:
        item = parsed[-1]
    else:
        item = parsed
    if isinstance(item, (list, tuple)) and item:
        item = item[-1]
    return str(item)


def _stage_text(payload: dict[str, Any], stage: int) -> str:
    text = str(payload.get(f"completion{stage}") or "")
    return text.split(USER_SENTINEL, 1)[0]


def _stage_prompt(payload: dict[str, Any], stage: int) -> str:
    return str(payload.get(f"prompt{stage}") or "")


def _stage_stop_reason(payload: dict[str, Any], stage: int) -> str:
    return str(payload.get(f"stop_reason{stage}") or "")


def _has_stage(payload: dict[str, Any], stage: int) -> bool:
    return f"completion{stage}" in payload or f"prompt{stage}" in payload


def _completion_text(payload: dict[str, Any]) -> str:
    return _stage_text(payload, 1)


def _completion_prompt(payload: dict[str, Any]) -> str:
    return _stage_prompt(payload, 1)


def _completion_stop_reason(payload: dict[str, Any]) -> str:
    return _stage_stop_reason(payload, 1)


def _stage_is_truncated(payload: dict[str, Any], stage: int) -> bool:
    if _stage_stop_reason(payload, stage) in {"max_tokens", "max_length"}:
        return True
    stats = payload.get("stats")
    if isinstance(stats, dict):
        stage_stats = stats.get(f"stage{stage}")
        if isinstance(stage_stats, dict):
            return bool(stage_stats.get("truncated"))
    return False


def _is_truncated(payload: dict[str, Any]) -> bool:
    if _has_stage(payload, 2):
        if _stage_is_truncated(payload, 2):
            return True
        stats = payload.get("stats")
        return isinstance(stats, dict) and bool(stats.get("truncated"))
    if _stage_is_truncated(payload, 1):
        return True
    stats = payload.get("stats")
    return isinstance(stats, dict) and bool(stats.get("truncated"))


def _think_state(prompt: str, text: str) -> tuple[bool, bool]:
    context = f"{prompt}{text}".lower()
    has_think = "<think" in context
    has_close = "</think>" in context
    return has_think, has_close


def _two_stage_scoring_text(payload: dict[str, Any]) -> str:
    prompt = _stage_prompt(payload, 2)
    text = _stage_text(payload, 2)
    return f"{prompt}{text}" if prompt else text


def _has_unclosed_boxed(text: str) -> bool:
    last_box = text.rfind("\\boxed{")
    if last_box < 0:
        return False
    tail = text[last_box + len("\\boxed{") :]
    depth = 1
    for char in tail:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return False
    return depth > 0


def _repair_two_stage_scoring_text(payload: dict[str, Any], text: str) -> str:
    repaired = text.rstrip()
    if _has_unclosed_boxed(repaired):
        repaired = f"{repaired}}}"
    if _stage_is_truncated(payload, 2) and "\\boxed" not in repaired:
        repaired = f"{repaired}\n{REPAIR_FINAL_CUE}"
    return repaired


def _strategy_scoring_text(group: str, payload: dict[str, Any]) -> str:
    if group == STRATEGY_A:
        return _completion_text(payload)
    if _has_stage(payload, 2):
        two_stage_text = _two_stage_scoring_text(payload)
        if group == STRATEGY_B:
            return two_stage_text
        if group == STRATEGY_C:
            return _repair_two_stage_scoring_text(payload, two_stage_text)

    text = _completion_text(payload)
    prompt = _completion_prompt(payload)
    has_think, has_close = _think_state(prompt, text)
    unclosed_think = has_think and not has_close
    truncated = _is_truncated(payload)

    if group == STRATEGY_B and unclosed_think:
        return f"{text.rstrip()}\n</think>"
    if group == STRATEGY_C and unclosed_think:
        return f"{text.rstrip()}\n</think>\n{REPAIR_FINAL_CUE}"
    if group == STRATEGY_C and truncated and (not has_think or has_close):
        return f"{text.rstrip()}\n{REPAIR_FINAL_CUE}"
    return text


def _stop_rate(payloads: list[dict[str, Any]]) -> float:
    if not payloads:
        return 0.0
    return sum(1 for payload in payloads if _is_truncated(payload)) / len(payloads)


def evaluate_free_response(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    judge: LLMJudge | None = None,
) -> FreeResponseEvaluation:
    """Evaluate full-generation free-response completions."""

    dataset = list(JsonlFreeAnswerLoader(str(dataset_path)))
    completion_payloads = list(_iter_completions(completions))
    references = [resolve_reference_answer(record) for record in dataset]
    judgement_reference_count = sum(1 for reference in references if _is_judgement_reference(reference))
    judgement_label_dataset = bool(dataset) and judgement_reference_count == len(dataset)
    if not judgement_label_dataset and _load_math_verify() is None:
        raise RuntimeError("free-response evaluation requires math-verify; run `uv sync` after updating uv.lock.")

    grouped: dict[str, list[_ScoredCompletion]] = {group: [] for group in STRATEGY_GROUPS}
    primary_group = STRATEGY_C if judgement_label_dataset else STRATEGY_A

    def apply_judge(group: str) -> None:
        if judge is None:
            return
        records = grouped[group]
        judge_inputs: list[tuple[str, str, str]] = []
        judge_indices: list[int] = []
        for idx, record in enumerate(records):
            if record.final_passed:
                continue
            judge_inputs.append((record.question, record.reference, record.display_answer))
            judge_indices.append(idx)
        if not judge_inputs:
            return
        judged_flags = judge.judge(judge_inputs)
        stats = judge.last_run_stats
        if stats is not None:
            judge_stats_by_group[group] = stats.as_dict()
        for idx, judged in zip(judge_indices, judged_flags, strict=True):
            record = records[idx]
            record.final_passed = bool(judged)
            if judged:
                record.fail_reason = ""
            else:
                record.fail_reason = (
                    f"{record.fail_reason};judge_false" if record.fail_reason else "judge_false"
                )

    def score_group(
        group: str,
        payload: dict[str, Any],
        *,
        sample_index: int,
        repeat_index: int,
        question: str,
        reference: str,
    ) -> _ScoredCompletion:
        scoring_text = _strategy_scoring_text(group, payload)
        if _is_judgement_reference(reference):
            verify_result = _judgement_verify(reference, scoring_text)
        else:
            verify_result = _math_verify(reference, scoring_text)
        return _ScoredCompletion(
            source_payload=payload,
            sample_index=sample_index,
            repeat_index=repeat_index,
            question=question,
            reference=reference,
            scoring_text=scoring_text,
            display_answer=verify_result.answer,
            math_passed=verify_result.passed,
            final_passed=verify_result.passed,
            fail_reason=verify_result.fail_reason,
        )

    def inherit_from_a(a_record: _ScoredCompletion) -> _ScoredCompletion:
        return _ScoredCompletion(
            source_payload=a_record.source_payload,
            sample_index=a_record.sample_index,
            repeat_index=a_record.repeat_index,
            question=a_record.question,
            reference=a_record.reference,
            scoring_text=a_record.scoring_text,
            display_answer=a_record.display_answer,
            math_passed=a_record.math_passed,
            final_passed=a_record.final_passed,
            fail_reason=a_record.fail_reason,
        )

    judge_stats_by_group: dict[str, dict[str, object]] = {}
    record_contexts: list[tuple[dict[str, Any], int, int, str, str]] = []

    for payload in completion_payloads:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        if sample_index < 0 or sample_index >= len(dataset):
            question = ""
            reference = ""
        else:
            record = dataset[sample_index]
            question = record.question
            reference = resolve_reference_answer(record)

        record_contexts.append((payload, sample_index, repeat_index, question, reference))
        grouped[STRATEGY_A].append(
            score_group(
                STRATEGY_A,
                payload,
                sample_index=sample_index,
                repeat_index=repeat_index,
                question=question,
                reference=reference,
            )
        )

    apply_judge(STRATEGY_A)

    for group in (STRATEGY_B, STRATEGY_C):
        for idx, (payload, sample_index, repeat_index, question, reference) in enumerate(record_contexts):
            a_record = grouped[STRATEGY_A][idx]
            if a_record.final_passed:
                grouped[group].append(inherit_from_a(a_record))
                continue
            grouped[group].append(
                score_group(
                    group,
                    payload,
                    sample_index=sample_index,
                    repeat_index=repeat_index,
                    question=question,
                    reference=reference,
                )
            )
        apply_judge(group)

    rows_by_group: dict[str, list[tuple[int, int, bool]]] = {}
    metrics_by_group: dict[str, dict[str, float]] = {}
    eval_payloads: list[dict] = []
    eval_payloads_by_group: dict[str, list[dict]] = {}
    samples = len(completion_payloads)
    stop_rate = _stop_rate(completion_payloads)

    for group in STRATEGY_GROUPS:
        records = grouped[group]
        group_payloads: list[dict] = []
        rows = [
            (record.sample_index, record.repeat_index, bool(record.final_passed))
            for record in records
        ]
        rows_by_group[group] = rows
        exact_accuracy = (
            sum(1 for record in records if record.math_passed) / samples if samples else 0.0
        )
        metrics: dict[str, float] = {
            "exact_accuracy": exact_accuracy,
            "stop_rate": stop_rate,
        }
        if judge is not None:
            metrics["judge_accuracy"] = (
                sum(1 for record in records if record.final_passed) / samples if samples else 0.0
            )
        metrics_by_group[group] = metrics
        for record in records:
            group_payloads.append(
                make_eval_payload(
                    record.source_payload,
                    is_passed=record.final_passed,
                    fail_reason=record.fail_reason,
                    answer=record.display_answer,
                    ref_answer=record.reference,
                )
            )
        eval_payloads_by_group[group] = group_payloads
        if group == primary_group:
            eval_payloads.extend(group_payloads)

    return FreeResponseEvaluation(
        metrics_by_group=metrics_by_group,
        rows_by_group=rows_by_group,
        samples=samples,
        payloads=eval_payloads,
        payloads_by_group=eval_payloads_by_group,
        judge_stats_by_group=judge_stats_by_group,
        primary_group=primary_group,
    )


def build_grouped_metrics_payload(
    evaluation: FreeResponseEvaluation,
    *,
    pass_k: tuple[int, ...],
    avg_k: tuple[NumericK, ...],
    report_pass_k: tuple[int, ...] = (),
    report_avg_k: tuple[NumericK, ...] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    group = evaluation.primary_group
    rows = evaluation.rows_by_group.get(group, [])
    metrics_payload: dict[str, object] = dict(evaluation.metrics_by_group.get(group, {}))
    strategy_metrics: dict[str, dict[str, float]] = {}
    pass_metrics_all = compute_pass_at_k(rows, pass_k)
    avg_metrics_all = compute_avg_at_k(rows, avg_k)

    pass_payload = filter_metrics_by_k(pass_metrics_all, report_pass_k, "pass@")
    if report_pass_k and not pass_payload:
        pass_payload = pass_metrics_all or {}
    if pass_payload:
        metrics_payload.update(pass_payload)

    avg_payload = filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@")
    if report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    if avg_payload:
        metrics_payload.update(avg_payload)

    for strategy in STRATEGY_GROUPS:
        strategy_rows = evaluation.rows_by_group.get(strategy, [])
        group_metrics = dict(evaluation.metrics_by_group.get(strategy, {}))
        group_pass_all = compute_pass_at_k(strategy_rows, pass_k)
        group_avg_all = compute_avg_at_k(strategy_rows, avg_k)

        group_pass_payload = filter_metrics_by_k(group_pass_all, report_pass_k, "pass@")
        if report_pass_k and not group_pass_payload:
            group_pass_payload = group_pass_all or {}
        if group_pass_payload:
            group_metrics.update(group_pass_payload)

        group_avg_payload = filter_metrics_by_k(group_avg_all, report_avg_k, "avg@")
        if report_avg_k and not group_avg_payload:
            group_avg_payload = group_avg_all or {}
        if group_avg_payload:
            group_metrics.update(group_avg_payload)

        strategy_metrics[strategy] = group_metrics
    metrics_payload["strategy_metrics"] = strategy_metrics
    metrics_payload["strategy_diagnostics"] = _build_strategy_diagnostics(evaluation)

    task_details: dict[str, object] = {}
    primary_judge_stats = evaluation.judge_stats_by_group.get(group)
    if primary_judge_stats:
        task_details["judge_stats"] = primary_judge_stats
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all
    return metrics_payload, task_details


def attach_strategy_task_ids(metrics_payload: dict[str, object], task_ids: dict[str, int | str]) -> dict[str, object]:
    metrics_payload["strategy_task_ids"] = {key: int(value) for key, value in task_ids.items()}
    return metrics_payload


def _build_strategy_diagnostics(evaluation: FreeResponseEvaluation) -> dict[str, dict[str, float]]:
    primary_records = _records_by_key(evaluation.payloads_by_group.get(evaluation.primary_group, []))
    diagnostics: dict[str, dict[str, float]] = {}
    for strategy in STRATEGY_GROUPS:
        if strategy == evaluation.primary_group:
            continue
        rows = evaluation.payloads_by_group.get(strategy, [])
        changed = 0
        rescued = 0
        harmed = 0
        compared = 0
        for payload in rows:
            key = (
                strict_nonneg_int(payload.get("sample_index"), "sample_index"),
                strict_nonneg_int(payload.get("repeat_index"), "repeat_index"),
            )
            primary = primary_records.get(key)
            if primary is None:
                continue
            compared += 1
            primary_answer = _normalize_text(str(primary.get("answer") or ""))
            strategy_answer = _normalize_text(str(payload.get("answer") or ""))
            if primary_answer != strategy_answer:
                changed += 1
            primary_passed = bool(primary.get("is_passed"))
            strategy_passed = bool(payload.get("is_passed"))
            if not primary_passed and strategy_passed:
                rescued += 1
            if primary_passed and not strategy_passed:
                harmed += 1
        denominator = compared or 1
        diagnostics[strategy] = {
            "changed_answer_rate": changed / denominator,
            "rescued_rate": rescued / denominator,
            "harmed_rate": harmed / denominator,
        }
    return diagnostics


def _records_by_key(payloads: list[dict]) -> dict[tuple[int, int], dict]:
    records: dict[tuple[int, int], dict] = {}
    for payload in payloads:
        key = (
            strict_nonneg_int(payload.get("sample_index"), "sample_index"),
            strict_nonneg_int(payload.get("repeat_index"), "repeat_index"),
        )
        records[key] = payload
    return records


__all__ = [
    "DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE",
    "LLMJudge",
    "LLMJudgeConfig",
    "LLMJudgeStats",
    "REPAIR_FINAL_CUE",
    "STRATEGY_A",
    "STRATEGY_B",
    "STRATEGY_C",
    "STRATEGY_GROUPS",
    "STRATEGY_LABELS",
    "FreeResponseEvaluation",
    "attach_strategy_task_ids",
    "build_grouped_metrics_payload",
    "compute_avg_at_k",
    "compute_pass_at_k",
    "evaluate_free_response",
    "resolve_reference_answer",
]
