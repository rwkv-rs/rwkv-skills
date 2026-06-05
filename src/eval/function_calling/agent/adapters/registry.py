from __future__ import annotations

"""Function-calling agent adapter registry."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.agent.adapters.apibank import create_apibank_level2_env
from src.eval.function_calling.agent.adapters.browsecomp_plus import (
    browsecomp_plus_run_from_agent_details,
    create_browsecomp_plus_env,
)
from src.eval.function_calling.agent.adapters.complexfuncbench import (
    create_complexfuncbench_official_env,
)

PromptRenderer = Callable[
    [FunctionCallTaskRecord, Sequence[Mapping[str, Any]], Any, int, Sequence[Any]],
    str,
]
PayloadAugmentor = Callable[[dict[str, Any], Mapping[str, Any]], None]
EnvFactory = Callable[[FunctionCallTaskRecord], Any]
DEDICATED_AGENT_ENV_TYPES = frozenset({"tau_official"})


@dataclass(frozen=True, slots=True)
class FunctionCallAgentAdapter:
    env_type: str
    create_env: EnvFactory | None = None
    render_prompt: PromptRenderer = None  # type: ignore[assignment]
    control_tool_names: tuple[str, ...] = ()
    augment_payload: PayloadAugmentor | None = None

    def __post_init__(self) -> None:
        if self.render_prompt is None:
            object.__setattr__(self, "render_prompt", render_default_agent_prompt)


def render_agent_trajectory(events: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for event in events:
        event_type = str(event.get("type") or "")
        content = str(event.get("content") or "")
        if event_type == "observation":
            rendered.append(f"Environment: {content}")
        elif event_type == "model_output":
            rendered.append(f"Assistant action: {content}")
        elif event_type == "action":
            rendered.append(
                "Tool call: "
                + json.dumps(
                    {"name": event.get("name"), "arguments": event.get("arguments") or {}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        elif event_type == "env_result":
            rendered.append(f"Environment: {content}")
        elif event_type == "error":
            rendered.append(f"Error: {content}")
    return "\n".join(part for part in rendered if part.strip()).strip()


def render_default_agent_prompt(
    record: FunctionCallTaskRecord,
    events: Sequence[Mapping[str, Any]],
    observation: Any,
    step: int,
    tools: Sequence[Any],
) -> str:
    _ = record, step
    tools_json = json.dumps(list(tools), ensure_ascii=False, indent=2)
    trajectory = render_agent_trajectory(events)
    current = str(getattr(observation, "content", "") or "")
    return (
        "You are controlling tools in a function-calling environment.\n"
        "Respond with exactly one JSON tool call and no extra text.\n"
        'Use this shape: {"name":"ToolName","arguments":{"arg":"value"}}\n'
        "Available tools:\n"
        f"{tools_json}\n\n"
        f"Trajectory:\n{trajectory}\n\n"
        f"Current observation:\n{current}\n\n"
        "Assistant: <think>\n</think>\n```json\n"
    )


def render_complexfuncbench_prompt(
    record: FunctionCallTaskRecord,
    events: Sequence[Mapping[str, Any]],
    observation: Any,
    step: int,
    tools: Sequence[Any],
) -> str:
    _ = record
    tools_json = json.dumps(list(tools), ensure_ascii=False, indent=2)
    trajectory = render_agent_trajectory(events)
    current = str(getattr(observation, "content", "") or "")
    return (
        "You are running the official ComplexFuncBench function-calling task.\n"
        "Return only JSON and no extra text.\n"
        'For one call, use {"name":"ToolName","arguments":{"arg":"value"}}.\n'
        "For multiple calls in the same assistant turn, return a JSON array of those objects.\n"
        "After all official sandbox observations indicate the required calls are complete, "
        'call {"name":"final_answer","arguments":{"answer":"..."}}.\n'
        "Available tools:\n"
        f"{tools_json}\n\n"
        f"Trajectory:\n{trajectory}\n\n"
        f"Current observation:\n{current}\n\n"
        f"Step: {step}\n\n"
        "Assistant: <think>\n</think>\n```json\n"
    )


def augment_complexfuncbench_payload(payload: dict[str, Any], details: Mapping[str, Any]) -> None:
    final_env_details = details.get("final_env_details")
    if not isinstance(final_env_details, Mapping):
        return
    final_response = final_env_details.get("final_response")
    if isinstance(final_response, str):
        payload["final_answer"] = final_response
    payload["complexfuncbench_official_result"] = {
        key: final_env_details.get(key)
        for key in ("message", "count_dict", "resp_eval", "call_accuracy")
        if key in final_env_details
    }


def augment_browsecomp_plus_payload(payload: dict[str, Any], details: Mapping[str, Any]) -> None:
    browsecomp_plus_run = browsecomp_plus_run_from_agent_details(details)
    if browsecomp_plus_run is not None:
        payload["browsecomp_plus_run"] = browsecomp_plus_run


DEFAULT_AGENT_ADAPTER = FunctionCallAgentAdapter(env_type="")
AGENT_ENV_ADAPTERS: dict[str, FunctionCallAgentAdapter] = {
    "apibank_level2": FunctionCallAgentAdapter(
        env_type="apibank_level2",
        create_env=create_apibank_level2_env,
    ),
    "browsecomp_plus": FunctionCallAgentAdapter(
        env_type="browsecomp_plus",
        create_env=create_browsecomp_plus_env,
        augment_payload=augment_browsecomp_plus_payload,
    ),
    "complexfuncbench_official": FunctionCallAgentAdapter(
        env_type="complexfuncbench_official",
        create_env=create_complexfuncbench_official_env,
        render_prompt=render_complexfuncbench_prompt,
        control_tool_names=("final_answer",),
        augment_payload=augment_complexfuncbench_payload,
    ),
}


def prompt_adapter_for_env(env_type: str) -> FunctionCallAgentAdapter:
    return AGENT_ENV_ADAPTERS.get(str(env_type or ""), DEFAULT_AGENT_ADAPTER)


def agent_env_adapter(env_type: str) -> FunctionCallAgentAdapter:
    env_key = str(env_type or "")
    if env_key in DEDICATED_AGENT_ENV_TYPES:
        raise NotImplementedError(f"{env_key} records must run through TauOfficialAgentPipeline")
    adapter = AGENT_ENV_ADAPTERS.get(env_key)
    if adapter is None or adapter.create_env is None:
        raise NotImplementedError(f"Unsupported function-calling agent env.type: {env_type}")
    return adapter


def create_agent_env(record: FunctionCallTaskRecord) -> Any:
    env_type = str(record.env.get("type") or "")
    adapter = agent_env_adapter(env_type)
    return adapter.create_env(record)


__all__ = [
    "AGENT_ENV_ADAPTERS",
    "DEDICATED_AGENT_ENV_TYPES",
    "DEFAULT_AGENT_ADAPTER",
    "FunctionCallAgentAdapter",
    "agent_env_adapter",
    "create_agent_env",
    "prompt_adapter_for_env",
    "render_agent_trajectory",
    "render_default_agent_prompt",
]
