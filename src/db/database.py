"""Database initialization wrapper around pooled PostgreSQL access."""

from __future__ import annotations

from src.eval.scheduler.config import DBConfig
from . import pool as db_pool
from .pool import get_db, init_db_pool

__all__ = ["get_db", "init_db", "is_initialized"]


def init_db(config: DBConfig | None = None) -> None:
    """Initialize the database connection.

    Database and schema must already exist.
    Initializes the shared PostgreSQL pool.
    """
    init_db_pool(config)


def is_initialized() -> bool:
    return db_pool._DB is not None
