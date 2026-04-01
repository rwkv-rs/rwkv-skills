from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.eval.agent_bench.deps import (
    ensure_tau_v2_runtime_dependencies,
    import_module_with_auto_install,
)
from src.eval.agent_bench.judge import NLAssertionJudge
from src.eval.agent_bench.tasks import ensure_tau_v2_vendor_path


@dataclass(slots=True)
class TauV2Evaluation:
    reward: float
    is_passed: bool
    termination_reason: str
    details: dict[str, Any]


class TauV2Env:
    """Minimal runtime wrapper for vendored tau2 environments + reward logic."""

    def __init__(self, *, domain: str, judge: NLAssertionJudge | None = None) -> None:
        ensure_tau_v2_vendor_path()
        ensure_tau_v2_runtime_dependencies()
        self.domain = domain
        self.judge = judge

        if domain == "retail":
            self._environment_constructor = _tau2_attr(
                "tau2.domains.retail.environment",
                "get_environment",
                context="tau2 retail env import",
            )
        elif domain == "airline":
            self._environment_constructor = _tau2_attr(
                "tau2.domains.airline.environment",
                "get_environment",
                context="tau2 airline env import",
            )
        elif domain == "telecom":
            self._environment_constructor = _tau2_attr(
                "tau2.domains.telecom.environment",
                "get_environment_manual_policy",
                context="tau2 telecom env import",
            )
        else:
            raise ValueError(f"Unsupported tau2 domain: {domain}")

    def load_task(self, payload: dict[str, Any]) -> Any:
        Task = _tau2_attr("tau2.data_model.tasks", "Task", context="tau2 Task model import")
        return Task.model_validate(payload)

    def create_environment(self, *, solo_mode: bool = False) -> Any:
        return self._environment_constructor(solo_mode=solo_mode)

    def apply_initial_state(self, *, environment: Any, task: Any) -> list[Any]:
        initial_state = getattr(task, "initial_state", None)
        initialization_data = (
            getattr(initial_state, "initialization_data", None) if initial_state is not None else None
        )
        initialization_actions = (
            getattr(initial_state, "initialization_actions", None) if initial_state is not None else None
        )
        message_history = (
            list(getattr(initial_state, "message_history", []) or [])
            if initial_state is not None
            else []
        )
        environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )
        return message_history

    @staticmethod
    def default_assistant_greeting() -> Any:
        AssistantMessage = _tau2_attr(
            "tau2.data_model.message",
            "AssistantMessage",
            context="tau2 AssistantMessage import",
        )
        return AssistantMessage(role="assistant", content="Hi! How can I help you today?")

    @staticmethod
    def assistant_stop_signal(message: Any) -> bool:
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            return False
        return "###STOP###" in content

    @staticmethod
    def user_stop_signal(message: Any) -> bool:
        try:
            UserSimulator = _tau2_attr(
                "tau2.user.user_simulator",
                "UserSimulator",
                context="tau2 UserSimulator import",
            )
            return bool(UserSimulator.is_stop(message))
        except Exception:
            content = getattr(message, "content", "")
            if not isinstance(content, str):
                return False
            return (
                "###STOP###" in content
                or "###TRANSFER###" in content
                or "###OUT-OF-SCOPE###" in content
            )

    @staticmethod
    def build_user_message(content: str) -> Any:
        UserMessage = _tau2_attr("tau2.data_model.message", "UserMessage", context="tau2 UserMessage import")
        return UserMessage(role="user", content=content)

    @staticmethod
    def build_assistant_message(*, content: str | None, tool_calls: list[Any] | None = None) -> Any:
        AssistantMessage = _tau2_attr(
            "tau2.data_model.message",
            "AssistantMessage",
            context="tau2 AssistantMessage import",
        )
        return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls)

    @staticmethod
    def build_tool_call(
        *,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
        requestor: str = "assistant",
    ) -> Any:
        ToolCall = _tau2_attr("tau2.data_model.message", "ToolCall", context="tau2 ToolCall import")
        return ToolCall(
            id=tool_call_id,
            name=name,
            arguments=arguments,
            requestor=requestor,
        )

    def call_tool(self, *, environment: Any, tool_call: Any) -> Any:
        response = environment.get_response(tool_call)
        environment.sync_tools()
        return response

    @staticmethod
    def to_chat_messages(*, system_prompt: str, trajectory: list[Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for item in trajectory:
            role = getattr(item, "role", None)
            if role == "user":
                messages.append({"role": "user", "content": getattr(item, "content", "")})
            elif role == "assistant":
                tool_calls = getattr(item, "tool_calls", None)
                if tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": getattr(item, "content", None),
                            "tool_calls": [
                                {
                                    "id": getattr(call, "id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": getattr(call, "name", ""),
                                        "arguments": _json_dumps(getattr(call, "arguments", {})),
                                    },
                                }
                                for call in tool_calls
                            ],
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": getattr(item, "content", "")})
            elif role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(item, "id", ""),
                        "content": getattr(item, "content", ""),
                    }
                )
            elif role == "system":
                messages.append({"role": "system", "content": getattr(item, "content", "")})
        return messages

    @staticmethod
    def tools_schema(environment: Any) -> list[dict[str, Any]]:
        return [tool.openai_schema for tool in environment.get_tools()]

    @staticmethod
    def user_tools_schema(environment: Any) -> list[dict[str, Any]]:
        try:
            tools = environment.get_user_tools()
        except Exception:
            return []
        return [tool.openai_schema for tool in tools]

    @staticmethod
    def system_prompt(environment: Any) -> str:
        return str(environment.get_policy())

    def evaluate(
        self,
        *,
        task: Any,
        trajectory: list[Any],
        termination_reason: str,
        solo_mode: bool = False,
    ) -> TauV2Evaluation:
        if termination_reason not in {"agent_stop", "user_stop"}:
            details = {
                "termination_reason": termination_reason,
                "note": "simulation terminated before normal completion",
            }
            return TauV2Evaluation(
                reward=0.0,
                is_passed=False,
                termination_reason=termination_reason,
                details=details,
            )

        criteria = getattr(task, "evaluation_criteria", None)
        if criteria is None:
            return TauV2Evaluation(
                reward=1.0,
                is_passed=True,
                termination_reason=termination_reason,
                details={"termination_reason": termination_reason, "note": "no evaluation criteria"},
            )

        RewardType = _tau2_attr("tau2.data_model.tasks", "RewardType", context="tau2 RewardType import")
        ActionEvaluator = _tau2_attr(
            "tau2.evaluator.evaluator_action",
            "ActionEvaluator",
            context="tau2 ActionEvaluator import",
        )
        CommunicateEvaluator = _tau2_attr(
            "tau2.evaluator.evaluator_communicate",
            "CommunicateEvaluator",
            context="tau2 CommunicateEvaluator import",
        )
        EnvironmentEvaluator = _tau2_attr(
            "tau2.evaluator.evaluator_env",
            "EnvironmentEvaluator",
            context="tau2 EnvironmentEvaluator import",
        )

        env_reward = EnvironmentEvaluator.calculate_reward(
            environment_constructor=self._environment_constructor,
            task=task,
            full_trajectory=trajectory,
            solo_mode=solo_mode,
        )
        action_reward = ActionEvaluator.calculate_reward(task=task, full_trajectory=trajectory)
        communicate_reward = CommunicateEvaluator.calculate_reward(task=task, full_trajectory=trajectory)
        nl_reward = self._evaluate_nl_assertions(task=task, trajectory=trajectory)

        reward = 1.0
        reward_breakdown: dict[str, float] = {}

        task_reward_basis = set(getattr(criteria, "reward_basis", []) or [])
        env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
        action_bases = {RewardType.ACTION}
        nl_bases = {RewardType.NL_ASSERTION}
        comm_bases = {RewardType.COMMUNICATE}

        if task_reward_basis & env_bases:
            reward *= float(getattr(env_reward, "reward", 0.0))
            reward_breakdown.update(_reward_breakdown_dict(env_reward))
        if task_reward_basis & action_bases:
            reward *= float(getattr(action_reward, "reward", 0.0))
            reward_breakdown.update(_reward_breakdown_dict(action_reward))
        if task_reward_basis & nl_bases:
            reward *= float(getattr(nl_reward, "reward", 0.0))
            reward_breakdown.update(_reward_breakdown_dict(nl_reward))
        if task_reward_basis & comm_bases:
            reward *= float(getattr(communicate_reward, "reward", 0.0))
            reward_breakdown.update(_reward_breakdown_dict(communicate_reward))

        details = {
            "termination_reason": termination_reason,
            "env": _model_dump_safe(env_reward),
            "action": _model_dump_safe(action_reward),
            "communicate": _model_dump_safe(communicate_reward),
            "nl": _model_dump_safe(nl_reward),
            "reward_breakdown": reward_breakdown,
        }
        return TauV2Evaluation(
            reward=reward,
            is_passed=reward >= (1.0 - 1e-6),
            termination_reason=termination_reason,
            details=details,
        )

    def _evaluate_nl_assertions(self, *, task: Any, trajectory: list[Any]) -> Any:
        NLAssertionCheck = _tau2_attr(
            "tau2.data_model.simulation",
            "NLAssertionCheck",
            context="tau2 NLAssertionCheck import",
        )
        RewardInfo = _tau2_attr("tau2.data_model.simulation", "RewardInfo", context="tau2 RewardInfo import")
        RewardType = _tau2_attr("tau2.data_model.tasks", "RewardType", context="tau2 RewardType import")

        criteria = getattr(task, "evaluation_criteria", None)
        nl_assertions = getattr(criteria, "nl_assertions", None) if criteria is not None else None
        if not nl_assertions:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "no nl assertions"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        if self.judge is None:
            NLAssertionsEvaluator = _tau2_attr(
                "tau2.evaluator.evaluator_nl_assertions",
                "NLAssertionsEvaluator",
                context="tau2 NLAssertionsEvaluator import",
            )
            return NLAssertionsEvaluator.calculate_reward(task=task, full_trajectory=trajectory)

        judged = self.judge.evaluate(list(nl_assertions), trajectory)
        checks = [
            NLAssertionCheck(
                nl_assertion=item.assertion,
                met=bool(item.met),
                justification=item.justification,
            )
            for item in judged
        ]
        reward = 1.0 if all(item.met for item in judged) else 0.0
        return RewardInfo(
            reward=reward,
            nl_assertions=checks,
            reward_breakdown={RewardType.NL_ASSERTION: reward},
            info={"judge_model": self.judge.model_name},
        )


def _json_dumps(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return "{}"


def _model_dump_safe(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    if isinstance(value, dict):
        return value
    return {"value": str(value)}


def _reward_breakdown_dict(reward_info: Any) -> dict[str, float]:
    breakdown = getattr(reward_info, "reward_breakdown", None)
    if not isinstance(breakdown, dict):
        return {}
    output: dict[str, float] = {}
    for key, value in breakdown.items():
        output[str(key)] = float(value)
    return output


def _tau2_attr(module_name: str, attr_name: str, *, context: str) -> Any:
    module = import_module_with_auto_install(module_name, context=context)
    return getattr(module, attr_name)


__all__ = [
    "TauV2Evaluation",
    "TauV2Env",
]
