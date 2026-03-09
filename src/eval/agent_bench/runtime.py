from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from src.eval.agent_bench.chat_bridge import PromptProfile, RWKVChatBridge
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


@dataclass(slots=True)
class _TauV1ActiveEpisode:
    request_index: int
    task_id: str
    env: TauV1Env
    messages: list[dict[str, Any]]
    stages: list[StageRecord]
    info: dict[str, Any]
    reward: float = 0.0
    step_count: int = 0


@dataclass(slots=True)
class _TauV2ActiveEpisode:
    request_index: int
    task: Any
    environment: Any
    trajectory: list[Any]
    user: Any
    user_state: Any
    strategy: str
    stages: list[StageRecord]
    tool_errors: int = 0
    step_count: int = 0


def _run_parallel_ordered(
    operations: Sequence[Callable[[], Any]],
    *,
    max_workers: int,
) -> list[Any]:
    if not operations:
        return []
    worker_count = max(1, min(int(max_workers), len(operations)))
    if worker_count <= 1:
        return [operation() for operation in operations]
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(operation) for operation in operations]
        return [future.result() for future in futures]


def run_tau_v1_episode(
    *,
    bridge: RWKVChatBridge,
    env: TauV1Env,
    task_index: int,
    max_steps: int,
    sampling: SamplingConfig | None = None,
    prompt_profile: PromptProfile = "tau_v1",
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
            prompt_profile=prompt_profile,
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


def run_tau_v1_episodes(
    *,
    bridge: RWKVChatBridge,
    env_factory: Callable[[], TauV1Env],
    task_indices: Sequence[int],
    max_steps: int,
    max_concurrency: int = 1,
    sampling: SamplingConfig | None = None,
    prompt_profile: PromptProfile = "tau_v1",
    on_complete: Callable[[int, EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    if not task_indices:
        return []

    max_steps = max(1, int(max_steps))
    max_concurrency = max(1, min(int(max_concurrency), len(task_indices)))
    pending = deque((request_index, int(task_index)) for request_index, task_index in enumerate(task_indices))
    active: list[_TauV1ActiveEpisode] = []
    results: list[EpisodeResult | None] = [None] * len(task_indices)

    while pending or active:
        if pending and len(active) < max_concurrency:
            starters: list[Callable[[], Any]] = []
            while pending and len(active) + len(starters) < max_concurrency:
                request_index, task_index = pending.popleft()
                starters.append(
                    lambda request_index=request_index, task_index=task_index: _start_tau_v1_episode(
                        request_index=request_index,
                        env=env_factory(),
                        task_index=task_index,
                    )
                )
            active.extend(_run_parallel_ordered(starters, max_workers=max_concurrency))

        chats = bridge.chat_many(
            [state.messages for state in active],
            [state.env.tools_schema for state in active],
            sampling=sampling,
            tool_choice="auto",
            prompt_profile=prompt_profile,
        )

        advance_results = _run_parallel_ordered(
            [
                lambda state=state, chat=chat: _advance_tau_v1_episode(
                    state,
                    chat=chat,
                    max_steps=max_steps,
                )
                for state, chat in zip(active, chats)
            ],
            max_workers=max_concurrency,
        )
        completed_slots: list[int] = []
        for slot_index, (state, finished) in enumerate(zip(active, advance_results)):
            if not finished:
                continue
            result = _finish_tau_v1_episode(state)
            results[state.request_index] = result
            if on_complete is not None:
                on_complete(state.request_index, result)
            completed_slots.append(slot_index)

        for slot_index in reversed(completed_slots):
            active.pop(slot_index)

    if any(result is None for result in results):
        raise RuntimeError("tau_v1 batched rollout completed with missing episode results")
    return [result for result in results if result is not None]


def _start_tau_v1_episode(*, request_index: int, env: TauV1Env, task_index: int) -> _TauV1ActiveEpisode:
    reset = env.reset(task_index=task_index)
    return _TauV1ActiveEpisode(
        request_index=request_index,
        task_id=str(task_index),
        env=env,
        messages=[
            {"role": "system", "content": env.system_prompt},
            {"role": "user", "content": reset.observation},
        ],
        stages=[],
        info=dict(reset.info),
    )


def _advance_tau_v1_episode(
    state: _TauV1ActiveEpisode,
    *,
    chat: Any,
    max_steps: int,
) -> bool:
    state.stages.append(
        StageRecord(
            prompt=chat.prompt,
            completion=chat.raw_text,
            stop_reason=chat.finish_reason,
            meta={"parse_error": chat.parse_error, "step": state.step_count},
        )
    )

    if chat.tool_calls:
        tool_call = chat.tool_calls[0]
        step = state.env.step_tool(name=tool_call.name, arguments=tool_call.arguments)
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
        state.messages.extend([assistant_message, tool_message])
    else:
        response_text = (chat.content or chat.raw_text or "").strip()
        step = state.env.step_response(content=response_text)
        assistant_message = {"role": "assistant", "content": response_text}
        user_message = {"role": "user", "content": step.observation}
        state.messages.extend([assistant_message, user_message])

    state.reward = float(step.reward)
    state.info.update(step.info)
    state.step_count += 1
    return bool(step.done) or state.step_count >= max_steps


def _finish_tau_v1_episode(state: _TauV1ActiveEpisode) -> EpisodeResult:
    cost = _safe_float(state.info.get("user_cost"), default=0.0)
    return EpisodeResult(
        task_id=state.task_id,
        domain=state.env.domain,
        reward=state.reward,
        stages=state.stages,
        num_turns=len(state.stages),
        cost=cost,
        is_passed=state.reward >= (1.0 - 1e-6),
        info=state.info,
        trace=state.messages,
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
    prompt_profile: PromptProfile = "tau_v2",
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
            prompt_profile=prompt_profile,
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


def run_tau_v2_episodes(
    *,
    bridge: RWKVChatBridge,
    runtime_env: TauV2Env,
    task_payloads: Sequence[dict[str, Any]],
    user_strategy: str,
    user_model: OpenAIModelConfig,
    max_steps: int,
    max_errors: int = 10,
    max_concurrency: int = 1,
    user_temperature: float = 0.0,
    sampling: SamplingConfig | None = None,
    prompt_profile: PromptProfile = "tau_v2",
    on_complete: Callable[[int, EpisodeResult], None] | None = None,
) -> list[EpisodeResult]:
    if not task_payloads:
        return []

    strategy = user_strategy.lower().strip()
    if strategy == "human" and int(max_concurrency) > 1:
        raise ValueError("human user_strategy 不支持 batch_size > 1")

    max_steps = max(1, int(max_steps))
    max_errors = max(1, int(max_errors))
    max_concurrency = max(1, min(int(max_concurrency), len(task_payloads)))
    pending = deque((request_index, payload) for request_index, payload in enumerate(task_payloads))
    active: list[_TauV2ActiveEpisode] = []
    results: list[EpisodeResult | None] = [None] * len(task_payloads)

    while pending or active:
        if pending and len(active) < max_concurrency:
            starters: list[Callable[[], Any]] = []
            while pending and len(active) + len(starters) < max_concurrency:
                request_index, task_payload = pending.popleft()
                starters.append(
                    lambda request_index=request_index, task_payload=task_payload: _start_tau_v2_episode(
                        request_index=request_index,
                        runtime_env=runtime_env,
                        task_payload=task_payload,
                        strategy=strategy,
                        user_model=user_model,
                        user_temperature=user_temperature,
                    )
                )
            active.extend(_run_parallel_ordered(starters, max_workers=max_concurrency))

        chats = bridge.chat_many(
            [
                runtime_env.to_chat_messages(
                    system_prompt=runtime_env.system_prompt(state.environment),
                    trajectory=state.trajectory,
                )
                for state in active
            ],
            [runtime_env.tools_schema(state.environment) for state in active],
            sampling=sampling,
            tool_choice="auto",
            prompt_profile=prompt_profile,
        )

        termination_reasons = _run_parallel_ordered(
            [
                lambda state=state, chat=chat: _advance_tau_v2_episode(
                    state,
                    runtime_env=runtime_env,
                    chat=chat,
                    max_steps=max_steps,
                    max_errors=max_errors,
                )
                for state, chat in zip(active, chats)
            ],
            max_workers=max_concurrency,
        )
        completed_entries: list[tuple[int, _TauV2ActiveEpisode, str]] = []
        for slot_index, (state, termination_reason) in enumerate(zip(active, termination_reasons)):
            if termination_reason is None:
                continue
            completed_entries.append((slot_index, state, termination_reason))

        finalized = _run_parallel_ordered(
            [
                lambda state=state, termination_reason=termination_reason: _finish_tau_v2_episode(
                    state,
                    runtime_env=runtime_env,
                    termination_reason=termination_reason,
                )
                for _, state, termination_reason in completed_entries
            ],
            max_workers=max_concurrency,
        )
        completed_slots: list[int] = []
        for (slot_index, state, _termination_reason), result in zip(completed_entries, finalized):
            results[state.request_index] = result
            if on_complete is not None:
                on_complete(state.request_index, result)
            completed_slots.append(slot_index)

        for slot_index in reversed(completed_slots):
            active.pop(slot_index)

    if any(result is None for result in results):
        raise RuntimeError("tau_v2 batched rollout completed with missing episode results")
    return [result for result in results if result is not None]


def _start_tau_v2_episode(
    *,
    request_index: int,
    runtime_env: TauV2Env,
    task_payload: dict[str, Any],
    strategy: str,
    user_model: OpenAIModelConfig,
    user_temperature: float,
) -> _TauV2ActiveEpisode:
    task = runtime_env.load_task(task_payload)
    environment = runtime_env.create_environment()
    trajectory = runtime_env.apply_initial_state(environment=environment, task=task)
    user_state: Any | None = None
    user: Any

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
        raise ValueError(f"Unsupported user_strategy for tau2: {strategy}")

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

    return _TauV2ActiveEpisode(
        request_index=request_index,
        task=task,
        environment=environment,
        trajectory=trajectory,
        user=user,
        user_state=user_state,
        strategy=strategy,
        stages=[],
    )


def _advance_tau_v2_episode(
    state: _TauV2ActiveEpisode,
    *,
    runtime_env: TauV2Env,
    chat: Any,
    max_steps: int,
    max_errors: int,
) -> str | None:
    state.stages.append(
        StageRecord(
            prompt=chat.prompt,
            completion=chat.raw_text,
            stop_reason=chat.finish_reason,
            meta={"parse_error": chat.parse_error, "step": state.step_count},
        )
    )

    termination_reason: str | None = None
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
        state.trajectory.append(assistant)
        for tool_call in tool_calls:
            tool_message = runtime_env.call_tool(environment=state.environment, tool_call=tool_call)
            state.trajectory.append(tool_message)
            if bool(getattr(tool_message, "error", False)):
                state.tool_errors += 1
        if state.tool_errors >= max_errors:
            termination_reason = "too_many_errors"
    else:
        response_text = (chat.content or chat.raw_text or "").strip()
        assistant = runtime_env.build_assistant_message(content=response_text)
        state.trajectory.append(assistant)
        if runtime_env.assistant_stop_signal(assistant):
            termination_reason = "agent_stop"
        else:
            if state.strategy == "human":
                user_message = runtime_env.build_user_message(state.user.step(response_text))
            else:
                user_message, state.user_state = state.user.generate_next_message(assistant, state.user_state)
            state.trajectory.append(user_message)
            if runtime_env.user_stop_signal(user_message):
                termination_reason = "user_stop"

    state.step_count += 1
    if termination_reason is None and state.step_count >= max_steps:
        termination_reason = "max_steps"
    return termination_reason


def _finish_tau_v2_episode(
    state: _TauV2ActiveEpisode,
    *,
    runtime_env: TauV2Env,
    termination_reason: str,
) -> EpisodeResult:
    evaluation = runtime_env.evaluate(
        task=state.task,
        trajectory=state.trajectory,
        termination_reason=termination_reason,
        solo_mode=False,
    )
    info = dict(evaluation.details)
    info["termination_reason"] = evaluation.termination_reason
    info["tool_errors"] = state.tool_errors
    return EpisodeResult(
        task_id=str(getattr(state.task, "id", "")),
        domain=runtime_env.domain,
        reward=float(evaluation.reward),
        stages=state.stages,
        num_turns=len(state.stages),
        cost=_sum_message_costs(state.trajectory),
        is_passed=bool(evaluation.is_passed),
        info=info,
        trace=_trajectory_dump(state.trajectory),
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
    "run_tau_v1_episodes",
    "run_tau_v2_episode",
    "run_tau_v2_episodes",
]
