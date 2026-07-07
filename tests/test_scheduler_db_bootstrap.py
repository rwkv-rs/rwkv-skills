from __future__ import annotations

from pathlib import Path

from src.eval.scheduler import db_bootstrap
from src.eval.scheduler.config import DBConfig


class _Cursor:
    def __init__(self, rows_by_kind: dict[str, list[tuple[str]]]) -> None:
        self._rows_by_kind = rows_by_kind
        self._kind = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def execute(self, _query, params=None) -> None:
        self._kind = str((params or ("",))[0])

    def fetchall(self):
        return self._rows_by_kind.get(self._kind, [])


class _Connection:
    def __init__(self, rows_by_kind: dict[str, list[tuple[str]]] | None = None) -> None:
        self.rows_by_kind = rows_by_kind or {}
        self.executed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def cursor(self):
        return _Cursor(self.rows_by_kind)

    def execute(self, query) -> None:
        self.executed.append(str(query))


def _config() -> DBConfig:
    return DBConfig(
        host="127.0.0.1",
        port=5432,
        user="postgres",
        dbname="rwkv-eval",
        sslmode="disable",
    )


def test_check_db_schema_reports_missing_relations(monkeypatch) -> None:
    monkeypatch.setattr(db_bootstrap.db_pool, "_ensure_database_exists", lambda _config: None)
    monkeypatch.setattr(db_bootstrap.db_pool, "_build_conninfo", lambda _config: "conninfo")

    rows = {
        "BASE TABLE": [("benchmark",), ("model",)],
        "VIEW": [],
    }
    monkeypatch.setattr(db_bootstrap.psycopg, "connect", lambda _conninfo: _Connection(rows))

    report = db_bootstrap.check_db_schema(_config(), schema_path=Path("schema.sql"))

    assert report.database_ok is True
    assert report.schema_ok is False
    assert "task" in report.missing_tables
    assert report.missing_views == ("view_model_version",)
    assert report.ok is False


def test_bootstrap_db_schema_executes_schema_then_rechecks(monkeypatch, tmp_path: Path) -> None:
    schema = tmp_path / "schema.sql"
    schema.write_text("CREATE TABLE IF NOT EXISTS benchmark(id int);", encoding="utf-8")
    connection = _Connection()
    monkeypatch.setattr(db_bootstrap.db_pool, "_ensure_database_exists", lambda _config: None)
    monkeypatch.setattr(db_bootstrap.db_pool, "_build_conninfo", lambda _config: "conninfo")
    monkeypatch.setattr(
        db_bootstrap.psycopg,
        "connect",
        lambda _conninfo, autocommit=False: connection,
    )
    monkeypatch.setattr(
        db_bootstrap,
        "check_db_schema",
        lambda _config, *, schema_path: db_bootstrap.DbSchemaReport(
            host="127.0.0.1",
            port=5432,
            user="postgres",
            dbname="rwkv-eval",
            sslmode="disable",
            schema_path=str(schema_path),
            database_ok=True,
            schema_ok=True,
        ),
    )

    report = db_bootstrap.bootstrap_db_schema(_config(), schema_path=schema)

    assert report.ok is True
    assert connection.executed == ["CREATE TABLE IF NOT EXISTS benchmark(id int);"]
