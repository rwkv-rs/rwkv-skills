from __future__ import annotations

"""Minimal PostgreSQL pooled access aligned with rwkv-rs `Db { pool }` shape."""

from dataclasses import dataclass

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool

from src.eval.scheduler.config import DBConfig, DEFAULT_DB_CONFIG

_DB: "Db | None" = None


@dataclass(slots=True)
class Db:
    pool: ConnectionPool


def _build_conninfo(config: DBConfig, *, dbname: str | None = None) -> str:
    parts = [
        f"host={config.host}",
        f"port={int(config.port)}",
        f"user={config.user}",
        f"dbname={dbname or config.dbname}",
    ]
    if config.password:
        parts.append(f"password={config.password}")
    sslmode = str(getattr(config, "sslmode", "") or "").strip()
    if sslmode:
        parts.append(f"sslmode={sslmode}")
    return " ".join(parts)


def _ensure_database_exists(config: DBConfig) -> None:
    target_db = str(config.dbname or "").strip()
    if not target_db:
        raise ValueError("database name must not be empty")
    maintenance_dbs = ("template1",) if target_db == "postgres" else ("postgres", "template1")
    last_error: Exception | None = None
    for maintenance_db in maintenance_dbs:
        conninfo = _build_conninfo(config, dbname=maintenance_db)
        try:
            with psycopg.connect(conninfo, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target_db,))
                    if cur.fetchone() is not None:
                        return
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
                    return
        except psycopg.OperationalError as exc:
            last_error = exc
            if maintenance_db != maintenance_dbs[-1]:
                continue
            raise
    if last_error is not None:
        raise last_error


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
    _ensure_database_exists(resolved)
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
