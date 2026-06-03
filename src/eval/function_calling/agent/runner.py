from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Sequence

from src.eval.function_calling.agent.env import AgentObservation, FunctionCallingEnv
from src.eval.function_calling.common.action import ToolAction
from src.eval.function_calling.common.events import FunctionCallEvent, event_payloads
from src.eval.function_calling.common.parser import ToolCallParseResult, parse_json_tool_calls


@dataclass(frozen=True, slots=True)
class AgentRunConfig:
    max_steps: int = 8
    timeout_s: float | None = None
    stop_on_parse_error: bool = True


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    score: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


AgentActionGenerator = Callable[[Sequence[Mapping[str, Any]], AgentObservation, int], str]
AgentActionParser = Callable[[str], ToolCallParseResult]


def run_function_calling_agent(
    env: FunctionCallingEnv,
    generate_action: AgentActionGenerator,
    *,
    parse_action: AgentActionParser | None = None,
    config: AgentRunConfig | None = None,
) -> AgentRunResult:
    cfg = config or AgentRunConfig()
    parser = parse_action or parse_json_tool_calls
    started = time.monotonic()
    events: list[FunctionCallEvent] = []
    invalid_action_count = 0
    parse_error_count = 0
    timeout = False
    finish_reason = "max_steps"
    final_score: float | None = None
    final_success: bool | None = None
    final_env_details: dict[str, Any] = {}

    observation = env.reset()
    events.append(_observation_event(observation, step=0))

    for step in range(max(0, cfg.max_steps)):
        if cfg.timeout_s is not None and time.monotonic() - started > cfg.timeout_s:
            timeout = True
            finish_reason = "timeout"
            break

        model_output = generate_action(event_payloads(events), observation, step)
        events.append(
            FunctionCallEvent(
                type="model_output",
                role="assistant",
                content=str(model_output or ""),
                metadata={"step": step},
            )
        )

        parsed = parser(str(model_output or ""))
        if parsed.parse_error is not None:
            parse_error_count += 1
            invalid_action_count += 1
            events.append(
                FunctionCallEvent(
                    type="error",
                    role="parser",
                    content=parsed.parse_error.message,
                    metadata={"step": step, **parsed.parse_error.as_dict()},
                )
            )
            if cfg.stop_on_parse_error:
                finish_reason = "parse_error"
                break
            continue
        if not parsed.calls:
            invalid_action_count += 1
            events.append(
                FunctionCallEvent(
                    type="error",
                    role="parser",
                    content="model output did not contain a tool action",
                    metadata={"step": step, "kind": "empty_action"},
                )
            )
            if cfg.stop_on_parse_error:
                finish_reason = "empty_action"
                break
            continue

        actions = [ToolAction.from_tool_call(call) for call in parsed.calls]
        for call_index, action in enumerate(actions):
            events.append(
                FunctionCallEvent(
                    type="action",
                    role="assistant",
                    name=action.name,
                    arguments=dict(action.arguments),
                    metadata={"step": step, "call_index": call_index},
                )
            )

        step_many = getattr(env, "step_many", None)
        if callable(step_many):
            result = step_many(actions)
            observation = result.observation
            final_score = result.score if result.score is not None else final_score
            final_success = result.success if result.success is not None else final_success
            if result.details:
                final_env_details = dict(result.details)
            events.append(
                FunctionCallEvent(
                    type="env_result",
                    role="environment",
                    content=observation.content,
                    metadata={
                        "step": step,
                        "call_count": len(actions),
                        "done": result.done,
                        "score": result.score,
                        "success": result.success,
                        "details": dict(result.details),
                        "observation_metadata": dict(observation.metadata),
                    },
                )
            )
            if result.done:
                finish_reason = "done"
                break
            continue

        for call_index, action in enumerate(actions):
            result = env.step(action)
            observation = result.observation
            final_score = result.score if result.score is not None else final_score
            final_success = result.success if result.success is not None else final_success
            if result.details:
                final_env_details = dict(result.details)
            events.append(
                FunctionCallEvent(
                    type="env_result",
                    role="environment",
                    content=observation.content,
                    metadata={
                        "step": step,
                        "call_index": call_index,
                        "done": result.done,
                        "score": result.score,
                        "success": result.success,
                        "details": dict(result.details),
                        "observation_metadata": dict(observation.metadata),
                    },
                )
            )
            if result.done:
                finish_reason = "done"
                break
        if finish_reason == "done":
            break
    else:
        finish_reason = "max_steps"

    success = _coerce_success(final_success, final_score, finish_reason)
    details = {
        "finish_reason": finish_reason,
        "steps": sum(1 for event in events if event.type == "model_output"),
        "invalid_action_count": invalid_action_count,
        "parse_error_count": parse_error_count,
        "timeout": timeout,
    }
    if final_env_details:
        details["final_env_details"] = final_env_details
    events.append(
        FunctionCallEvent(
            type="final_score",
            role="scorer",
            metadata={
                "success": success,
                "score": final_score,
                **details,
            },
        )
    )
    return AgentRunResult(
        events=event_payloads(events),
        success=success,
        score=final_score,
        details=details,
    )


def _observation_event(observation: AgentObservation, *, step: int) -> FunctionCallEvent:
    return FunctionCallEvent(
        type="observation",
        role="environment",
        content=observation.content,
        metadata={"step": step, **dict(observation.metadata)},
    )


def _coerce_success(success: bool | None, score: float | None, finish_reason: str) -> bool:
    if success is not None:
        return bool(success)
    if finish_reason != "done":
        return False
    if score is None:
        return True
    return float(score) > 0.0


__all__ = [
    "AgentActionGenerator",
    "AgentActionParser",
    "AgentRunConfig",
    "AgentRunResult",
    "run_function_calling_agent",
]
