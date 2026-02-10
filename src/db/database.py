"""Database initialization - thin wrapper around orm.init_orm for backwards compatibility."""

from __future__ import annotations

from src.eval.scheduler.config import DBConfig
from .orm import init_orm, is_initialized

__all__ = ["init_db", "is_initialized"]


def init_db(config: DBConfig | None = None) -> None:
    """Initialize the database connection.

    This is the preferred entry point for database initialization.
    Delegates to orm.init_orm().
    """
    init_orm(config)
