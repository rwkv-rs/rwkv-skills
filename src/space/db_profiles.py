"""Read-only DB profile helpers for the Space dashboard."""

from __future__ import annotations

from typing import Any

from src.db.eval_db_service import EvalDbService


DEFAULT_PROFILE = "default"
MATH_PROFILE = "math"
MATH_DOMAIN = "math reasoning系列"


def math_db_enabled() -> bool:
    return False


def db_profile_for_domain(domain: str) -> str:
    _ = domain
    return DEFAULT_PROFILE


def fetch_latest_scores_for_profile(*, profile: str, include_param_search: bool = False) -> list[dict[str, Any]]:
    _ = profile
    return EvalDbService().list_latest_scores_for_space(include_param_search=include_param_search)


def list_eval_records_for_profile(
    *,
    profile: str,
    task_id: str,
    only_wrong: bool,
    limit: int | None = None,
    offset: int = 0,
    include_context: bool = True,
) -> list[dict[str, Any]]:
    _ = profile
    return EvalDbService().list_eval_records_for_space(
        task_id=task_id,
        only_wrong=only_wrong,
        limit=limit,
        offset=offset,
        include_context=include_context,
    )


def get_eval_context_for_profile(
    *,
    profile: str,
    task_id: str,
    sample_index: int,
    repeat_index: int,
) -> Any | None:
    _ = profile
    return EvalDbService().get_eval_context_for_space(
        task_id=task_id,
        sample_index=sample_index,
        repeat_index=repeat_index,
    )
