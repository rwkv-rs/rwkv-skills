from __future__ import annotations

from src.db import pool as db_pool
from src.eval.scheduler.config import DBConfig


def test_ensure_database_exists_creates_missing_database(monkeypatch) -> None:
    events: list[tuple[object, object | None]] = []

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def execute(self, query, params=None) -> None:
            events.append((query, params))

        def fetchone(self):
            return None

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def cursor(self):
            return _Cursor()

    def _connect(conninfo: str, *, autocommit: bool):
        events.append(("connect", (conninfo, autocommit)))
        return _Connection()

    monkeypatch.setattr(db_pool.psycopg, "connect", _connect)

    db_pool._ensure_database_exists(
        DBConfig(
            host="127.0.0.1",
            port=5433,
            user="postgres",
            password="secret",
            dbname="rwkv-eval",
            sslmode="disable",
        )
    )

    assert events[0] == (
        "connect",
        (
            "host=127.0.0.1 port=5433 user=postgres dbname=postgres password=secret sslmode=disable",
            True,
        ),
    )
    assert events[1][1] == ("rwkv-eval",)
    assert len(events) == 3
