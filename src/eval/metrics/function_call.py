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
    if not record.expected_tool_calls:
        return False, "missing_expected_tool_calls", ""
    if _looks_like_template_leak(prediction):
        return False, "decision stage leaked internal template/control tokens", ""
    from src.eval.function_calling.simple_tool_call import decode_simple_tool_call_response

    parse_error: str | None = None
    decoded_calls: list[dict[str, Any]] = []
    try:
        decoded_calls = decode_simple_tool_call_response(prediction)
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
    if _uses_bfcl_exec_scorer(record):
        from src.eval.function_calling.bfcl_exec import evaluate_bfcl_executable_calls

        result = evaluate_bfcl_executable_calls(record, decoded_calls, parse_error=parse_error)
    elif _uses_toolalpaca_official_scorer(record):
        from src.eval.function_calling.toolalpaca_official import evaluate_toolalpaca_official_calls

        result = evaluate_toolalpaca_official_calls(record, decoded_calls, parse_error=parse_error)
    else:
        from src.eval.function_calling.simple_tool_call import (
            evaluate_simple_tool_calls,
            normalize_simple_tool_call_manifest_row,
        )

        simple_record = normalize_simple_tool_call_manifest_row(
            {
                "task_id": record.task_id,
                "instruction": record.instruction,
                "tools": record.tools,
                "expected_tool_calls": record.expected_tool_calls,
                "metadata": record.metadata,
            },
            index=0,
        )
        result = evaluate_simple_tool_calls(simple_record, decoded_calls, parse_error=parse_error)
    reference = json.dumps(
        result.details.get("expected_tool_calls") or [],
        ensure_ascii=False,
        sort_keys=True,
    )
    return bool(result.is_passed), result.fail_reason, reference


def _uses_bfcl_exec_scorer(record: FunctionCallTaskRecord) -> bool:
    scorer_type = str((record.scorer or {}).get("type") or "")
    if scorer_type == "bfcl_exec":
        return True
    category = str((record.metadata or {}).get("category") or "")
    if category in {"exec_simple", "exec_multiple"}:
        return True
    return str(record.task_id or "").startswith(("exec_simple_", "exec_multiple_"))


def _uses_toolalpaca_official_scorer(record: FunctionCallTaskRecord) -> bool:
    scorer_type = str((record.scorer or {}).get("type") or "")
    if scorer_type == "toolalpaca_official":
        return True
    source_format = str((record.metadata or {}).get("source_format") or "")
    return source_format == "official_toolalpaca" and str(record.task_id or "").startswith("toolalpaca_")


def _record_env_type(record: FunctionCallTaskRecord | None) -> str:
    if record is None:
        return "unknown"
    env = record.env or {}
    return str(env.get("type") or "simple_tool_call")


_TEMPLATE_LEAK_MARKERS = (
    "<system message>",
    "</system message>",
    "<assistant>",
    "</assistant>",
    "<user_input>",
    "</user_input>",
)


def _looks_like_template_leak(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if "<system message>" in lowered and "you are a helpful assistant" in lowered:
        return True
    marker_hits = sum(lowered.count(marker) for marker in _TEMPLATE_LEAK_MARKERS)
    return marker_hits >= 3


__all__ = [
    "FunctionCallMetrics",
    "evaluate_function_call",
]
