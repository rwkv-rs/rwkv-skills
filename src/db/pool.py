from __future__ import annotations

"""Minimal PostgreSQL pooled access aligned with rwkv-rs `Db { pool }` shape."""

from dataclasses import dataclass

from psycopg_pool import ConnectionPool

from src.eval.scheduler.config import DBConfig, DEFAULT_DB_CONFIG

_DB: "Db | None" = None


@dataclass(slots=True)
class Db:
    pool: ConnectionPool


def _build_conninfo(config: DBConfig) -> str:
    parts = [
        f"host={config.host}",
        f"port={int(config.port)}",
        f"user={config.user}",
        f"dbname={config.dbname}",
    ]
    if config.password:
        parts.append(f"password={config.password}")
    sslmode = str(getattr(config, "sslmode", "") or "").strip()
    if sslmode:
        parts.append(f"sslmode={sslmode}")
    return " ".join(parts)


def init_db_pool(
    config: DBConfig | None = None,
    *,
    min_size: int = 1,
    max_size: int = 16,
) -> Db:
    global _DB
    if _DB is not None:
        return _DB
    resolved = config or DEFAULT_DB_CONFIG
    pool = ConnectionPool(
        conninfo=_build_conninfo(resolved),
        min_size=max(1, int(min_size)),
        max_size=max(1, int(max_size)),
        open=False,
        kwargs={"autocommit": False},
    )
    pool.open(wait=True)
    _DB = Db(pool=pool)
    return _DB


def get_db() -> Db:
    if _DB is None:
        return init_db_pool()
    return _DB


def close_db_pool() -> None:
    global _DB
    if _DB is None:
        return
    _DB.pool.close()
    _DB = None


__all__ = ["Db", "close_db_pool", "get_db", "init_db_pool"]
