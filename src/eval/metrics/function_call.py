from __future__ import annotations

"""Metrics for function-call benchmarks."""

from dataclasses import dataclass
from typing import Any, Iterable

from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.function_calling.common.payload import extract_function_call_text, extract_stats
from src.eval.function_calling.one_step.scorer import record_env_type, score_one_step_prediction
from src.eval.k_values import NumericK
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.results.schema import build_context_from_completions, make_eval_payload, strict_nonneg_int


@dataclass(slots=True)
class FunctionCallMetrics:
    success_rate: float
    avg_steps: float
    avg_tool_calls: float
    avg_at_k: dict[str, float] | None = None
    samples: int = 0
    payloads: list[dict[str, Any]] | None = None
    env_breakdown: dict[str, float] | None = None


def evaluate_function_call(
    completions: Iterable[dict[str, Any]],
    *,
    dataset_path: str,
    avg_k: tuple[NumericK, ...] = (),
) -> FunctionCallMetrics:
    dataset = list(JsonlFunctionCallTaskLoader(str(dataset_path)).load())
    rows_for_avg: list[tuple[int, int, bool]] = []
    eval_payloads: list[dict[str, Any]] = []
    env_totals: dict[str, int] = {}
    env_correct: dict[str, int] = {}
    total = 0
    correct = 0
    step_sum = 0.0
    tool_call_sum = 0.0

    for payload in completions:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        record = dataset[sample_index] if 0 <= sample_index < len(dataset) else None
        prediction = extract_function_call_text(payload).strip()
        score = score_one_step_prediction(record, prediction)
        passed = score.passed
        env_type = record_env_type(record)
        env_totals[env_type] = env_totals.get(env_type, 0) + 1
        if passed:
            env_correct[env_type] = env_correct.get(env_type, 0) + 1
            correct += 1
        total += 1
        rows_for_avg.append((sample_index, repeat_index, passed))
        stats = extract_stats(payload)
        step_sum += float(stats.get("steps") or 0)
        tool_call_sum += float(stats.get("tool_calls") or 0)
        eval_payloads.append(
            make_eval_payload(
                payload,
                is_passed=passed,
                fail_reason=score.fail_reason,
                answer=prediction,
                ref_answer=score.reference,
            )
        )
        if eval_payloads[-1].get("context") == "":
            eval_payloads[-1]["context"] = build_context_from_completions(payload)

    metrics = FunctionCallMetrics(
        success_rate=(correct / total) if total else 0.0,
        avg_steps=(step_sum / total) if total else 0.0,
        avg_tool_calls=(tool_call_sum / total) if total else 0.0,
        samples=total,
        payloads=eval_payloads,
        env_breakdown={
            env: env_correct.get(env, 0) / count if count else 0.0
            for env, count in env_totals.items()
        },
    )
    if avg_k:
        metrics.avg_at_k = compute_avg_at_k(rows_for_avg, avg_k)
    return metrics


__all__ = [
    "FunctionCallMetrics",
    "evaluate_function_call",
]
