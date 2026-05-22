from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .events import FunctionCallEvent, event_payloads


@dataclass(slots=True)
class FunctionCallRunStats:
    steps: int = 0
    tool_calls: int = 0
    summaries: int = 0
    truncated_observations: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0


def build_one_step_completion_payload(
    *,
    benchmark_name: str,
    dataset_split: str,
    sample_index: int,
    repeat_index: int,
    sampling_config: list[dict[str, Any]],
    prompt: str,
    completion: str,
    stop_reason: str,
    final_answer: str,
    events: list[FunctionCallEvent],
    stats: FunctionCallRunStats,
    env_type: str,
    scorer_type: str,
) -> dict[str, Any]:
    return {
        "benchmark_name": benchmark_name,
        "dataset_split": dataset_split,
        "sample_index": sample_index,
        "repeat_index": repeat_index,
        "sampling_config": sampling_config,
        "prompt1": prompt,
        "completion1": completion,
        "stop_reason1": stop_reason,
        "final_answer": final_answer,
        "events": event_payloads(events),
        "stats": asdict(stats),
        "function_call_env_type": env_type,
        "function_call_scorer_type": scorer_type,
    }


def build_agent_completion_payload(
    *,
    benchmark_name: str,
    dataset_split: str,
    sample_index: int,
    repeat_index: int,
    sampling_config: list[dict[str, Any]],
    prompts: list[str],
    completions: list[str],
    events: list[dict[str, Any]],
    stats: FunctionCallRunStats,
    env_type: str,
    scorer_type: str,
    success: bool,
    official_score: float | None,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    final_answer = ""
    for event in reversed(events):
        if isinstance(event, dict) and event.get("type") == "model_output":
            final_answer = str(event.get("content") or "")
            break
    return {
        "benchmark_name": benchmark_name,
        "dataset_split": dataset_split,
        "sample_index": sample_index,
        "repeat_index": repeat_index,
        "sampling_config": sampling_config,
        "prompt1": prompts[-1] if prompts else "",
        "completion1": completions[-1] if completions else "",
        "prompts": prompts,
        "completions": completions,
        "stop_reason1": str(details.get("finish_reason") or ""),
        "final_answer": final_answer,
        "events": events,
        "stats": asdict(stats),
        "function_call_subtype": "agent",
        "function_call_env_type": env_type,
        "function_call_scorer_type": scorer_type,
        "success": success,
        "official_score": official_score,
        "agent_details": dict(details),
    }


def extract_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    stats = payload.get("stats")
    if isinstance(stats, dict):
        return stats
    context = payload.get("context")
    if isinstance(context, dict):
        stats = context.get("stats")
        if isinstance(stats, dict):
            return stats
    return {}


def extract_function_call_text(payload: Mapping[str, Any]) -> str:
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
