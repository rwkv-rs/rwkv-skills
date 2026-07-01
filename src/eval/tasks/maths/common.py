from __future__ import annotations

"""Maths-domain shared helpers aligned with rwkv-rs maths modules."""
from enum import Enum
from pathlib import Path
from typing import Iterable

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.env_config import (
    resolve_judge_max_tokens,
    resolve_judge_max_workers,
    resolve_judge_model_config,
    resolve_judge_timeout_s,
)
from src.eval.k_values import NumericK, filter_metrics_by_k


class JudgeMode(str, Enum):
    EXACT = "exact"
    LLM = "llm"


def default_job_name(judge_mode: JudgeMode) -> str:
    if judge_mode is JudgeMode.LLM:
        return "free_response_judge"
    return "free_response"


def default_db_write_queue(judge_mode: JudgeMode) -> int:
    if judge_mode is JudgeMode.LLM:
        return 16
    return 8


def default_db_drain_every(judge_mode: JudgeMode) -> int:
    if judge_mode is JudgeMode.LLM:
        return 0
    return 8


def count_free_answer_records(path: str | Path, limit: int | None) -> int:
    loader = JsonlFreeAnswerLoader(str(path))
    count = 0
    for _ in loader:
        count += 1
        if limit and count >= limit:
            break
    return count


def resolve_sampling_pair(
    slug: str,
    model_name: str,
    *,
    cot_max_tokens: int | None = None,
    final_max_tokens: int | None = None,
):
    cot_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    final_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="final",
        fallback_templates="free_response_final_default",
    )
    if cot_sampling is None or final_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")
    return cot_sampling.clamp(cot_max_tokens), final_sampling.clamp(final_max_tokens)


def resolve_generation_sampling(
    slug: str,
    model_name: str,
    *,
    max_tokens: int | None = None,
):
    generation_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    if generation_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")
    return generation_sampling.clamp(max_tokens)


def filter_avg_metrics(metric_map: dict[str, float] | None, ks: Iterable[NumericK]) -> dict[str, float]:
    return filter_metrics_by_k(metric_map, tuple(ks), "avg@")


def filter_pass_metrics(metric_map: dict[str, float] | None, ks: Iterable[int]) -> dict[str, float]:
    if not metric_map:
        return {}
    allowed = {f"pass@{int(k)}" for k in ks if int(k) > 0}
    return {
        key: value
        for key, value in metric_map.items()
        if key.startswith("pass@") and key in allowed
    }


def build_llm_judge(
    *,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
    judge_max_workers: int | None = None,
    judge_max_tokens: int | None = None,
    required: bool = True,
):
    from src.eval.metrics.free_response import LLMJudge, LLMJudgeConfig

    try:
        config = resolve_judge_model_config(
            model_name=judge_model,
            api_key=judge_api_key,
            base_url=judge_base_url,
        )
    except ValueError:
        if required:
            raise
        return None
    if config is None:
        if required:
            raise ValueError(
                "free_response_judge 需要有效的 judge 配置："
                "请提供 --judge-model/--judge-api-key，或设置 JUDGE_MODEL + JUDGE_API_KEY。"
            )
        return None
    resolved_max_workers = resolve_judge_max_workers(judge_max_workers, default=16)
    resolved_max_tokens = resolve_judge_max_tokens(judge_max_tokens)
    resolved_timeout_s = resolve_judge_timeout_s(default=60.0)

    return LLMJudge(
        LLMJudgeConfig(
            api_key=config.api_key,
            model=config.model_name,
            base_url=config.base_url,
            timeout_s=resolved_timeout_s,
            max_workers=resolved_max_workers,
            max_completion_tokens=resolved_max_tokens,
        )
    )


__all__ = [
    "JudgeMode",
    "build_llm_judge",
    "count_free_answer_records",
    "default_db_drain_every",
    "default_db_write_queue",
    "default_job_name",
    "filter_avg_metrics",
    "filter_pass_metrics",
    "resolve_generation_sampling",
    "resolve_sampling_pair",
]
