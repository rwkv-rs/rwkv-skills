from __future__ import annotations

import json
from typing import Any, Mapping

from src.eval.agent_bench.runtime import EpisodeResult
from src.eval.evaluators.common import SampleRecord, StageRecord
from src.eval.results.schema import make_eval_payload, strict_nonneg_int


def episode_to_completion_payload(
    episode: EpisodeResult,
    *,
    benchmark_name: str,
    dataset_split: str,
    sample_index: int,
    repeat_index: int,
    sampling_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stages = [
        StageRecord(
            prompt=stage.prompt,
            completion=stage.completion,
            stop_reason=stage.stop_reason,
        )
        for stage in episode.stages
    ]
    payload = SampleRecord(
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        sample_index=sample_index,
        repeat_index=repeat_index,
        stages=stages,
        sampling_config=dict(sampling_config or {}),
    ).as_payload()

    payload["agent_result"] = {
        "task_id": episode.task_id,
        "domain": episode.domain,
        "reward": float(episode.reward),
        "num_turns": int(episode.num_turns),
        "cost": float(episode.cost),
        "is_passed": bool(episode.is_passed),
        "error": episode.error,
    }
    if episode.info:
        payload["agent_info"] = episode.info
    if episode.trace:
        payload["agent_trace"] = episode.trace
    return payload


def completion_to_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = _extract_agent_result(payload)
    reward = float(result.get("reward", 0.0))
    passed = bool(result.get("is_passed", reward >= (1.0 - 1e-6)))
    answer_payload = {
        "reward": reward,
        "num_turns": int(result.get("num_turns", 0)),
        "cost": float(result.get("cost", 0.0)),
        "error": result.get("error"),
    }
    fail_reason = "" if passed else "reward_below_threshold"
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=fail_reason,
        answer=json.dumps(answer_payload, ensure_ascii=False),
        ref_answer="1.0",
    )


def _extract_agent_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("agent_result")
    if isinstance(result, dict):
        return result

    answer = payload.get("answer")
    if isinstance(answer, str):
        try:
            loaded = json.loads(answer)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            return loaded

    return {
        "reward": 1.0 if bool(payload.get("is_passed", False)) else 0.0,
        "num_turns": 0,
        "cost": 0.0,
        "is_passed": bool(payload.get("is_passed", False)),
        "error": None,
    }


def completion_rows_for_pass_k(payloads: list[dict[str, Any]]) -> list[tuple[int, int, bool]]:
    rows: list[tuple[int, int, bool]] = []
    for payload in payloads:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        result = _extract_agent_result(payload)
        reward = float(result.get("reward", 0.0))
        passed = bool(result.get("is_passed", reward >= (1.0 - 1e-6)))
        rows.append((sample_index, repeat_index, passed))
    return rows


__all__ = [
    "episode_to_completion_payload",
    "completion_to_eval_payload",
    "completion_rows_for_pass_k",
]
