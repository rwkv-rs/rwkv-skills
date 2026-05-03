from __future__ import annotations

"""Metrics for function-call benchmarks."""

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.eval.datasets.data_loader.function_call import JsonlFunctionCallTaskLoader
from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
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
        prediction = _extract_function_call_text(payload).strip()
        passed, fail_reason, reference = _score_prediction(record, prediction)
        env_type = _record_env_type(record)
        env_totals[env_type] = env_totals.get(env_type, 0) + 1
        if passed:
            env_correct[env_type] = env_correct.get(env_type, 0) + 1
            correct += 1
        total += 1
        rows_for_avg.append((sample_index, repeat_index, passed))
        stats = _extract_stats(payload)
        step_sum += float(stats.get("steps") or 0)
        tool_call_sum += float(stats.get("tool_calls") or 0)
        eval_payloads.append(
            make_eval_payload(
                payload,
                is_passed=passed,
                fail_reason=fail_reason,
                answer=prediction,
                ref_answer=reference,
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


def _extract_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    stats = payload.get("stats")
    if isinstance(stats, dict):
        return stats
    context = payload.get("context")
    if isinstance(context, dict):
        stats = context.get("stats")
        if isinstance(stats, dict):
            return stats
    return {}


def _extract_function_call_text(payload: Mapping[str, Any]) -> str:
    final_answer = payload.get("final_answer")
    if isinstance(final_answer, str):
        return final_answer
    context = payload.get("context")
    if isinstance(context, dict):
        context_answer = context.get("final_answer")
        if isinstance(context_answer, str):
            return context_answer
        events = context.get("events")
        if isinstance(events, list):
            extracted = _extract_function_call_from_events(events)
            if extracted:
                return extracted
    events = payload.get("events")
    if isinstance(events, list):
        extracted = _extract_function_call_from_events(events)
        if extracted:
            return extracted
    completion_keys = sorted(
        int(key.removeprefix("completion"))
        for key in payload
        if key.startswith("completion") and key.removeprefix("completion").isdigit()
    )
    if completion_keys:
        return str(payload.get(f"completion{completion_keys[-1]}", "") or "")
    return ""


def _extract_function_call_from_events(events: list[Any]) -> str:
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("type") or event.get("kind") or "")
        if event_type in {"function_call", "tool_call"}:
            return str(event.get("content") or event.get("text") or "")
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if str(event.get("role") or "") == "assistant":
            return str(event.get("content") or event.get("text") or "")
    return ""


def _score_prediction(record: FunctionCallTaskRecord | None, prediction: str) -> tuple[bool, str, str]:
    if record is None:
        return False, "missing_record", ""
    scorer = record.scorer or {}
    scorer_type = str(scorer.get("type") or "json_function_call_exact")
    expected, expected_error = _normalize_call(record.expected_call)
    reference = _call_to_json(expected) if expected is not None else ""
    if expected is None:
        return False, expected_error or "missing_reference_call", ""
    if scorer_type == "json_function_call_exact":
        actual, actual_error = _parse_function_call(prediction)
        if actual is None:
            return False, actual_error or "invalid_json_function_call", reference
        if actual["name"] != expected["name"]:
            return False, "name_mismatch", reference
        if actual["arguments"] != expected["arguments"]:
            return False, "arguments_mismatch", reference
        return True, "", reference
    return False, f"unsupported_scorer:{scorer_type}", reference


def _parse_function_call(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = raw.strip()
    if not text:
        return None, "empty_output"
    decoder = json.JSONDecoder()
    try:
        parsed, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None, "invalid_json"
    if text[end:].strip():
        return None, "invalid_json_trailing_text"
    if not isinstance(parsed, dict):
        return None, "json_not_object"
    extra_keys = set(parsed) - {"name", "arguments"}
    if extra_keys:
        return None, "unexpected_call_fields"
    return _normalize_call(parsed)


def _normalize_call(call: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    name = call.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, "missing_name"
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        return None, "arguments_not_object"
    return {"name": name.strip(), "arguments": dict(arguments)}, ""


def _call_to_json(call: Mapping[str, Any]) -> str:
    return json.dumps(call, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_env_type(record: FunctionCallTaskRecord | None) -> str:
    if record is None:
        return "unknown"
    env = record.env or {}
    return str(env.get("type") or "unknown")


AgentMetrics = FunctionCallMetrics
evaluate_agent = evaluate_function_call


__all__ = [
    "FunctionCallMetrics",
    "evaluate_function_call",
    "AgentMetrics",
    "evaluate_agent",
]
