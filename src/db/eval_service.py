from __future__ import annotations

"""Persistence backend selection for evaluation runs."""

import os
from typing import Any

from src.eval.scheduler.config import DBConfig


def use_json_eval_store(env: dict[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    raw = (
        source.get("RWKV_EVAL_STORE")
        or source.get("RWKV_EVAL_PERSISTENCE")
        or source.get("RWKV_EVAL_BACKEND")
        or "db"
    )
    normalized = raw.strip().lower()
    if normalized in {"db", "postgres", "postgresql", "sql"}:
        return False
    return normalized in {"json", "jsonl", "file", "files", "local-json"}


def init_eval_store(config: DBConfig | None = None) -> None:
    if use_json_eval_store():
        return
    from src.db.database import init_db

    init_db(config)


def create_eval_service() -> Any:
    if use_json_eval_store():
        from src.db.json_service import EvalJsonService

        return EvalJsonService()
    from src.db.eval_db_service import EvalDbService

    return EvalDbService()


__all__ = ["create_eval_service", "init_eval_store", "use_json_eval_store"]
