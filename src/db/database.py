"""Database initialization wrapper around orm.init_orm."""

from __future__ import annotations

from src.eval.scheduler.config import DBConfig
from .orm import init_orm, is_initialized

__all__ = ["init_db", "is_initialized"]


def init_db(config: DBConfig | None = None) -> None:
    """Initialize the database connection.

    Database and schema must already exist.
    Delegates to orm.init_orm().
    """
    init_orm(config)
