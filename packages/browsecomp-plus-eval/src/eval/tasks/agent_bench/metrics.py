from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from src.eval.metrics.at_k import compute_pass_at_k
from src.eval.results.schema import strict_nonneg_int


def compute_agent_metrics(
    eval_payloads: Sequence[dict[str, Any]],
    *,
    pass_k: Sequence[int],
    expected_count: int | None = None,
) -> dict[str, Any]:
    rewards: list[float] = []
    steps: list[int] = []
    costs: list[float] = []
    rows: list[tuple[int, int, bool]] = []

    for payload in eval_payloads:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        answer = _parse_answer_payload(payload.get("answer"))
        reward = float(answer.get("reward", 1.0 if payload.get("is_passed") else 0.0))
        num_turns = int(answer.get("num_turns", 0))
        cost = float(answer.get("cost", 0.0))
        passed = bool(payload.get("is_passed", reward >= (1.0 - 1e-6)))

        rewards.append(reward)
        steps.append(num_turns)
        costs.append(cost)
        rows.append((sample_index, repeat_index, passed))

    total = len(eval_payloads)
    success = sum(1 for row in rows if row[2])
    metrics: dict[str, Any] = {
        "avg_reward": (sum(rewards) / total) if total else 0.0,
        "success_rate": (success / total) if total else 0.0,
        "avg_steps": (sum(steps) / total) if total else 0.0,
        "avg_cost": (sum(costs) / total) if total else 0.0,
        "samples": total,
    }

    if expected_count is not None:
        metrics["expected_count"] = int(expected_count)

    pass_metrics = compute_pass_at_k(rows, pass_k)
    metrics.update(pass_metrics)
    for key, value in pass_metrics.items():
        if key.startswith("pass@"):
            metrics[f"pass_hat_{key.removeprefix('pass@')}"] = value
    return metrics


def _parse_answer_payload(answer: Any) -> dict[str, Any]:
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str):
        try:
            loaded = json.loads(answer)
        except json.JSONDecodeError:
            return {}
        if isinstance(loaded, dict):
            return loaded
    return {}


__all__ = ["compute_agent_metrics"]
