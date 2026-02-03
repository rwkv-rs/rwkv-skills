from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from ..eval.scheduler.config import DBConfig


class DatabaseManager:
    _instance = None
    _pool = None
    _config: DBConfig | None = None

    def __init__(self):
        raise RuntimeError("Call instance() instead")

    @classmethod
    def instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def initialize(self, config: DBConfig) -> None:
        if not config.enabled:
            return
        if self._pool is not None:
            return
        self._config = config

        # 1. Connect to default 'postgres' db to check/create target db
        try:
            admin_conn_str = (
                f"host={config.host} port={config.port} "
                f"dbname=postgres user={config.user} password={config.password}"
            )
            # autocommit=True needed for CREATE DATABASE
            with psycopg.connect(admin_conn_str, autocommit=True) as conn:
                # Check if db exists
                res = conn.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (config.dbname,),
                ).fetchone()
                if not res:
                    print(f"📦 Database '{config.dbname}' does not exist. Creating...")
                    conn.execute(f'CREATE DATABASE "{config.dbname}"')
        except Exception as e:
            # If we can't connect to postgres db or create db, warn but try proceeding
            print(f"⚠️  Database creation check failed (proceeding anyway): {e}")

        # 2. Connect to target db
        conn_str = (
            f"host={config.host} port={config.port} "
            f"dbname={config.dbname} user={config.user} password={config.password}"
        )

        self._pool = ConnectionPool(conn_str, min_size=1, max_size=10)
        self._pool.wait()
        self._init_schema()

    def _init_schema(self) -> None:
        if self._pool is None:
            return

        with self.get_connection() as conn:
            schema_path = Path(__file__).resolve().parents[2] / "scripts" / "schema.sql"
            if not schema_path.exists():
                raise RuntimeError(f"Schema file not found: {schema_path}")
            sql = schema_path.read_text(encoding="utf-8")
            for statement in sql.split(";"):
                stmt = statement.strip()
                if stmt:
                    try:
                        conn.execute(stmt)
                    except Exception:
                        if "create extension" in stmt.lower() and "pgcrypto" in stmt.lower():
                            conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
                        else:
                            raise

    @contextmanager
    def get_connection(self) -> Iterator[psycopg.Connection]:
        if self._pool is None:
            raise RuntimeError("Database not initialized")

        with self._pool.connection() as conn:
            conn.row_factory = dict_row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        if self._pool:
            self._pool.close()
            self._pool = None

    @property
    def config(self) -> DBConfig | None:
        return self._config


db_manager = DatabaseManager.instance()
