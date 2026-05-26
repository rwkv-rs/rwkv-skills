from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.eval.function_calling.common.payload import extract_function_call_text, extract_stats
from src.eval.results.schema import build_context_from_completions, make_eval_payload


@dataclass(frozen=True, slots=True)
class AgentScore:
    success: bool
    official_score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FunctionCallAgentMetrics:
    success_rate: float
    official_score: float
    avg_steps: float
    invalid_action_rate: float
    timeout_rate: float
    parse_error_rate: float
    samples: int = 0
    payloads: list[dict[str, Any]] | None = None


def evaluate_function_call_agent(completions: Iterable[dict[str, Any]]) -> FunctionCallAgentMetrics:
    total = 0
    success_count = 0
    score_sum = 0.0
    step_sum = 0.0
    invalid_actions = 0
    parse_errors = 0
    timeouts = 0
    eval_payloads: list[dict[str, Any]] = []

    for payload in completions:
        details = _agent_details(payload)
        stats = extract_stats(payload)
        steps = _numeric(stats.get("steps"), _numeric(details.get("steps"), 0.0))
        invalid_count = int(_numeric(details.get("invalid_action_count"), 0.0))
        parse_count = int(_numeric(details.get("parse_error_count"), 0.0))
        timeout = bool(details.get("timeout"))
        score = _numeric(payload.get("official_score"), None)
        if score is None:
            score = _numeric(details.get("score"), None)
        score_unavailable = bool(details.get("official_score_unavailable"))
        final_env_details = details.get("final_env_details")
        if isinstance(final_env_details, dict):
            score_unavailable = score_unavailable or bool(final_env_details.get("official_score_unavailable"))
        success = bool(payload.get("success"))
        if not success and score is not None:
            success = score > 0.0
        if score is None and score_unavailable:
            score = 0.0
        if score is None:
            score = 1.0 if success else 0.0

        total += 1
        success_count += 1 if success else 0
        score_sum += float(score)
        step_sum += steps
        invalid_actions += invalid_count
        parse_errors += parse_count
        timeouts += 1 if timeout else 0

        eval_payload = make_eval_payload(
            payload,
            is_passed=success,
            fail_reason="" if success else str(details.get("finish_reason") or "agent_failed"),
            answer=extract_function_call_text(payload),
            ref_answer=str(payload.get("official_score") if payload.get("official_score") is not None else ""),
        )
        if eval_payload.get("context") == "":
            eval_payload["context"] = build_context_from_completions(payload)
        eval_payloads.append(eval_payload)

    denominator_steps = step_sum if step_sum > 0 else float(total or 1)
    return FunctionCallAgentMetrics(
        success_rate=success_count / total if total else 0.0,
        official_score=score_sum / total if total else 0.0,
        avg_steps=step_sum / total if total else 0.0,
        invalid_action_rate=invalid_actions / denominator_steps if total else 0.0,
        timeout_rate=timeouts / total if total else 0.0,
        parse_error_rate=parse_errors / denominator_steps if total else 0.0,
        samples=total,
        payloads=eval_payloads,
    )


def _agent_details(payload: dict[str, Any]) -> dict[str, Any]:
    details = payload.get("agent_details")
    if isinstance(details, dict):
        return dict(details)
    context = payload.get("context")
    if isinstance(context, dict):
        details = context.get("agent_details")
        if isinstance(details, dict):
            return dict(details)
    return {}


def _numeric(value: Any, default: float | None) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


__all__ = [
    "AgentScore",
    "FunctionCallAgentMetrics",
    "evaluate_function_call_agent",
]
