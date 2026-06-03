"""Read-only DB profile helpers for the Space dashboard.

The default profile keeps using the process-wide PG_* config.  The optional
math profile uses SPACE_MATH_PG_* and is intentionally isolated from the ORM
global engine so the other dashboard domains are not affected.
"""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any, Mapping

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, URL
from sqlalchemy.orm import Session

from src.db.eval_db_repo import EvalDbRepository
from src.db.eval_db_service import EvalDbService
from src.eval.scheduler.config import DBConfig, DEFAULT_DB_CONFIG


DEFAULT_PROFILE = "default"
MATH_PROFILE = "math"
MATH_DOMAIN = "math reasoning系列"


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return fallback
    return value


def _int_env(name: str, fallback: int) -> int:
    raw = _env(name)
    if raw is None:
        return fallback
    return int(raw)


def _math_db_config() -> DBConfig | None:
    dbname = _env("SPACE_MATH_PG_DBNAME")
    if not dbname:
        return None
    return DBConfig(
        host=_env("SPACE_MATH_PG_HOST", _env("PG_HOST", DEFAULT_DB_CONFIG.host)) or DEFAULT_DB_CONFIG.host,
        port=_int_env("SPACE_MATH_PG_PORT", int(_env("PG_PORT", str(DEFAULT_DB_CONFIG.port)) or DEFAULT_DB_CONFIG.port)),
        user=_env("SPACE_MATH_PG_USER", _env("PG_USER", DEFAULT_DB_CONFIG.user)) or DEFAULT_DB_CONFIG.user,
        password=_env("SPACE_MATH_PG_PASSWORD", _env("PG_PASSWORD", DEFAULT_DB_CONFIG.password))
        or DEFAULT_DB_CONFIG.password,
        dbname=dbname,
    )


def math_db_enabled() -> bool:
    cfg = _math_db_config()
    return bool(cfg and cfg.dbname != DEFAULT_DB_CONFIG.dbname)


def db_profile_for_domain(domain: str) -> str:
    if domain == MATH_DOMAIN and math_db_enabled():
        return MATH_PROFILE
    return DEFAULT_PROFILE


def _config_key(config: DBConfig) -> tuple[str, int, str, str, str]:
    return (config.host, int(config.port), config.user, config.password, config.dbname)


@lru_cache(maxsize=4)
def _engine_for_config(host: str, port: int, user: str, password: str, dbname: str) -> Engine:
    url = URL.create(
        "postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=port,
        database=dbname,
    )
    return create_engine(url, pool_pre_ping=True, future=True)


def _math_engine() -> Engine:
    cfg = _math_db_config()
    if cfg is None:
        raise RuntimeError("SPACE_MATH_PG_DBNAME is not configured")
    return _engine_for_config(*_config_key(cfg))


def fetch_latest_scores_for_profile(*, profile: str, include_param_search: bool = False) -> list[dict[str, Any]]:
    if profile != MATH_PROFILE:
        return EvalDbService().list_latest_scores_for_space(include_param_search=include_param_search)

    repo = EvalDbRepository()
    with Session(_math_engine()) as session:
        rows = repo.fetch_latest_scores_for_space(session, include_param_search=include_param_search)
    return [dict(row) for row in rows]


def _eval_record_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sample_index": int(row.get("sample_index", 0)),
        "repeat_index": int(row.get("repeat_index", 0)),
        "eval_group": str(row.get("eval_group") or "strategy_a"),
        "is_passed": bool(row.get("is_passed", False)),
        "answer": str(row.get("answer") or ""),
        "ref_answer": str(row.get("ref_answer") or ""),
        "fail_reason": str(row.get("fail_reason") or ""),
        "context_preview": str(row.get("context_preview") or ""),
    }
    if "context" in row:
        payload["context"] = row.get("context")
    return payload


def list_eval_records_for_profile(
    *,
    profile: str,
    task_id: str,
    only_wrong: bool,
    limit: int | None = None,
    offset: int = 0,
    include_context: bool = True,
) -> list[dict[str, Any]]:
    if profile != MATH_PROFILE:
        return EvalDbService().list_eval_records_for_space(
            task_id=task_id,
            only_wrong=only_wrong,
            limit=limit,
            offset=offset,
            include_context=include_context,
        )

    safe_limit = int(limit) if isinstance(limit, int) or (isinstance(limit, str) and limit.isdigit()) else None
    if safe_limit is not None and safe_limit <= 0:
        safe_limit = None
    safe_offset = max(0, int(offset))

    repo = EvalDbRepository()
    with Session(_math_engine()) as session:
        rows = repo.fetch_eval_with_completions_by_task(
            session,
            task_id=int(task_id),
            only_wrong=bool(only_wrong),
            limit=safe_limit,
            offset=safe_offset,
            include_context=bool(include_context),
        )
    return [_eval_record_payload(dict(row)) for row in rows if isinstance(row, Mapping)]


def get_eval_context_for_profile(
    *,
    profile: str,
    task_id: str,
    sample_index: int,
    repeat_index: int,
) -> Any | None:
    if profile != MATH_PROFILE:
        return EvalDbService().get_eval_context_for_space(
            task_id=task_id,
            sample_index=sample_index,
            repeat_index=repeat_index,
        )

    repo = EvalDbRepository()
    with Session(_math_engine()) as session:
        return repo.fetch_eval_context_by_task_sample_repeat(
            session,
            task_id=int(task_id),
            sample_index=int(sample_index),
            repeat_index=int(repeat_index),
        )
