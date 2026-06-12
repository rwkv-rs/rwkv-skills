from __future__ import annotations

"""Evaluation persistence service factory."""

from typing import Any

from src.eval.scheduler.config import DBConfig


def init_eval_store(config: DBConfig | None = None) -> None:
    from src.db.database import init_db

    init_db(config)


def create_eval_service() -> Any:
    from src.db.eval_db_service import EvalDbService

    return EvalDbService()


__all__ = ["create_eval_service", "init_eval_store"]
