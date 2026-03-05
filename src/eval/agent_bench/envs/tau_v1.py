from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.eval.agent_bench.tasks import ensure_tau_v1_vendor_path


@dataclass(slots=True)
class TauV1Reset:
    observation: str
    info: dict[str, Any]


@dataclass(slots=True)
class TauV1Step:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any]


class TauV1Env:
    """Thin wrapper around vendored tau-bench v1 environments."""

    def __init__(
        self,
        *,
        domain: str,
        user_strategy: str,
        user_model: str,
        user_provider: str | None,
        task_split: str,
    ) -> None:
        ensure_tau_v1_vendor_path()
        from tau_bench.envs import get_env  # type: ignore

        self.domain = domain
        self._env = get_env(
            domain,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_split=task_split,
        )

        from tau_bench.types import RESPOND_ACTION_NAME  # type: ignore

        self.respond_action_name = RESPOND_ACTION_NAME

    @property
    def tools_schema(self) -> list[dict[str, Any]]:
        return list(self._env.tools_info)

    @property
    def system_prompt(self) -> str:
        return str(self._env.wiki)

    @property
    def task_count(self) -> int:
        return len(self._env.tasks)

    def reset(self, *, task_index: int) -> TauV1Reset:
        response = self._env.reset(task_index=task_index)
        info = response.info.model_dump() if hasattr(response.info, "model_dump") else dict(response.info)
        return TauV1Reset(observation=str(response.observation), info=info)

    def step_tool(self, *, name: str, arguments: dict[str, Any]) -> TauV1Step:
        return self._step(name=name, kwargs=arguments)

    def step_response(self, *, content: str) -> TauV1Step:
        return self._step(name=self.respond_action_name, kwargs={"content": content})

    def _step(self, *, name: str, kwargs: dict[str, Any]) -> TauV1Step:
        from tau_bench.types import Action  # type: ignore

        action = Action(name=name, kwargs=kwargs)
        response = self._env.step(action)
        info = response.info.model_dump() if hasattr(response.info, "model_dump") else dict(response.info)
        return TauV1Step(
            observation=str(response.observation),
            reward=float(response.reward),
            done=bool(response.done),
            info=info,
        )


__all__ = [
    "TauV1Reset",
    "TauV1Step",
    "TauV1Env",
]
