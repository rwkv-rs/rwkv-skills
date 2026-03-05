from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.eval.agent_bench.chat_bridge import RWKVChatBridge
from src.eval.agent_bench.deps import import_module_with_auto_install
from src.eval.agent_bench.envs.tau_v1 import TauV1Env
from src.eval.agent_bench.envs.tau_v2 import TauV2Env
from src.eval.env_config import OpenAIModelConfig
from src.eval.agent_bench.user_sim import HumanUser
from src.infer.sampling import SamplingConfig


@dataclass(slots=True)
class StageRecord:
    prompt: str
    completion: str
    stop_reason: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeResult:
    task_id: str
    domain: str
    reward: float
    stages: list[StageRecord]
    num_turns: int
    cost: float
    error: str | None = None
    is_passed: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)


def run_tau_v1_episode(
    *,
    bridge: RWKVChatBridge,
    env: TauV1Env,
    task_index: int,
    max_steps: int,
    sampling: SamplingConfig | None = None,
) -> EpisodeResult:
    reset = env.reset(task_index=task_index)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": env.system_prompt},
        {"role": "user", "content": reset.observation},
    ]
    stages: list[StageRecord] = []
    info = dict(reset.info)
    reward = 0.0

    for step_idx in range(max(1, int(max_steps))):
        chat = bridge.chat(
            messages,
            env.tools_schema,
            sampling=sampling,
            tool_choice="auto",
        )
        stages.append(
            StageRecord(
                prompt=chat.prompt,
                completion=chat.raw_text,
                stop_reason=chat.finish_reason,
                meta={
                    "parse_error": chat.parse_error,
                    "step": step_idx,
                },
            )
        )

        if chat.tool_calls:
            tool_call = chat.tool_calls[0]
            step = env.step_tool(name=tool_call.name, arguments=tool_call.arguments)
            assistant_message = {
                "role": "assistant",
                "content": chat.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": step.observation,
            }
            messages.extend([assistant_message, tool_message])
        else:
            response_text = (chat.content or chat.raw_text or "").strip()
            step = env.step_response(content=response_text)
            assistant_message = {"role": "assistant", "content": response_text}
            user_message = {"role": "user", "content": step.observation}
            messages.extend([assistant_message, user_message])

        reward = float(step.reward)
        info.update(step.info)
        if step.done:
            break

    cost = _safe_float(info.get("user_cost"), default=0.0)
    return EpisodeResult(
        task_id=str(task_index),
        domain=env.domain,
        reward=reward,
        stages=stages,
        num_turns=len(stages),
        cost=cost,
        is_passed=reward >= (1.0 - 1e-6),
        info=info,
        trace=messages,
    )


def run_tau_v2_episode(
    *,
    bridge: RWKVChatBridge,
    runtime_env: TauV2Env,
    task_payload: dict[str, Any],
    user_strategy: str,
    user_model: OpenAIModelConfig,
    max_steps: int,
    max_errors: int = 10,
    user_temperature: float = 0.0,
    sampling: SamplingConfig | None = None,
) -> EpisodeResult:
    task = runtime_env.load_task(task_payload)
    environment = runtime_env.create_environment()
    trajectory = runtime_env.apply_initial_state(environment=environment, task=task)
    stages: list[StageRecord] = []
    tool_errors = 0

    user_state: Any | None = None
    user: Any
    strategy = user_strategy.lower().strip()
    if strategy == "human":
        user = HumanUser()
    elif strategy == "llm":
        user_module = import_module_with_auto_install(
            "tau2.user.user_simulator",
            context="tau2 user simulator import",
        )
        base_module = import_module_with_auto_install("tau2.user.base", context="tau2 user base import")
        UserSimulator = getattr(user_module, "UserSimulator")
        is_valid_user_history_message = getattr(base_module, "is_valid_user_history_message")

        user = UserSimulator(
            instructions=getattr(task.user_scenario, "instructions", ""),
            llm=user_model.model_name,
            llm_args={
                "temperature": float(user_temperature),
                "api_key": user_model.api_key,
                "api_base": user_model.base_url,
            },
        )
        user_history = [item for item in trajectory if is_valid_user_history_message(item)]
        user_state = user.get_init_state(message_history=user_history)
    else:
        raise ValueError(f"Unsupported user_strategy for tau2: {user_strategy}")

    if not trajectory:
        greeting = runtime_env.default_assistant_greeting()
        trajectory.append(greeting)
        if strategy == "human":
            instruction = _tau2_instruction_text(task)
            user_message = runtime_env.build_user_message(user.reset(instruction))
        else:
            user_message, user_state = user.generate_next_message(greeting, user_state)
        trajectory.append(user_message)
    else:
        user_state = _bootstrap_tau2_trajectory(
            runtime_env=runtime_env,
            environment=environment,
            trajectory=trajectory,
            user=user,
            user_state=user_state,
            strategy=strategy,
        )

    termination_reason: str | None = None
    for step_idx in range(max(1, int(max_steps))):
        messages = runtime_env.to_chat_messages(
            system_prompt=runtime_env.system_prompt(environment),
            trajectory=trajectory,
        )
        chat = bridge.chat(
            messages,
            runtime_env.tools_schema(environment),
            sampling=sampling,
            tool_choice="auto",
        )
        stages.append(
            StageRecord(
                prompt=chat.prompt,
                completion=chat.raw_text,
                stop_reason=chat.finish_reason,
                meta={"parse_error": chat.parse_error, "step": step_idx},
            )
        )

        if chat.tool_calls:
            tool_calls = [
                runtime_env.build_tool_call(
                    tool_call_id=item.id,
                    name=item.name,
                    arguments=item.arguments,
                )
                for item in chat.tool_calls
            ]
            assistant = runtime_env.build_assistant_message(content=None, tool_calls=tool_calls)
            trajectory.append(assistant)
            for tool_call in tool_calls:
                tool_message = runtime_env.call_tool(environment=environment, tool_call=tool_call)
                trajectory.append(tool_message)
                if bool(getattr(tool_message, "error", False)):
                    tool_errors += 1
            if tool_errors >= max(1, int(max_errors)):
                termination_reason = "too_many_errors"
                break
            continue

        response_text = (chat.content or chat.raw_text or "").strip()
        assistant = runtime_env.build_assistant_message(content=response_text)
        trajectory.append(assistant)
        if runtime_env.assistant_stop_signal(assistant):
            termination_reason = "agent_stop"
            break

        if strategy == "human":
            user_message = runtime_env.build_user_message(user.step(response_text))
        else:
            user_message, user_state = user.generate_next_message(assistant, user_state)
        trajectory.append(user_message)
        if runtime_env.user_stop_signal(user_message):
            termination_reason = "user_stop"
            break

    if termination_reason is None:
        termination_reason = "max_steps"

    evaluation = runtime_env.evaluate(
        task=task,
        trajectory=trajectory,
        termination_reason=termination_reason,
        solo_mode=False,
    )
    info = dict(evaluation.details)
    info["termination_reason"] = evaluation.termination_reason
    info["tool_errors"] = tool_errors

    return EpisodeResult(
        task_id=str(getattr(task, "id", "")),
        domain=runtime_env.domain,
        reward=float(evaluation.reward),
        stages=stages,
        num_turns=len(stages),
        cost=_sum_message_costs(trajectory),
        is_passed=bool(evaluation.is_passed),
        info=info,
        trace=_trajectory_dump(trajectory),
    )


def _bootstrap_tau2_trajectory(
    *,
    runtime_env: TauV2Env,
    environment: Any,
    trajectory: list[Any],
    user: Any,
    user_state: Any,
    strategy: str,
) -> Any:
    if not trajectory:
        return user_state
    last = trajectory[-1]
    role = getattr(last, "role", None)

    if role == "assistant":
        tool_calls = getattr(last, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                tool_message = runtime_env.call_tool(environment=environment, tool_call=tool_call)
                trajectory.append(tool_message)
            return user_state
        if strategy == "human":
            trajectory.append(runtime_env.build_user_message(user.step(getattr(last, "content", ""))))
        else:
            user_message, user_state = user.generate_next_message(last, user_state)
            trajectory.append(user_message)
    return user_state


def _sum_message_costs(messages: list[Any]) -> float:
    total = 0.0
    for message in messages:
        total += _safe_float(getattr(message, "cost", None), default=0.0)
    return total


def _tau2_instruction_text(task: Any) -> str:
    user_scenario = getattr(task, "user_scenario", None)
    if user_scenario is None:
        return str(getattr(task, "id", ""))
    instructions = getattr(user_scenario, "instructions", None)
    if instructions is None:
        return str(user_scenario)
    if isinstance(instructions, str):
        return instructions
    if hasattr(instructions, "model_dump_json"):
        return instructions.model_dump_json(indent=2)
    return str(instructions)


def _trajectory_dump(trajectory: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in trajectory:
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                rows.append(dumped)
                continue
        rows.append(
            {
                "role": str(getattr(message, "role", "unknown")),
                "content": str(getattr(message, "content", "")),
            }
        )
    return rows


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


__all__ = [
    "StageRecord",
    "EpisodeResult",
    "run_tau_v1_episode",
    "run_tau_v2_episode",
]
