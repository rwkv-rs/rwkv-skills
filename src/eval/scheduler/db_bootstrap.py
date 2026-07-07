from __future__ import annotations

"""PostgreSQL schema checks and idempotent bootstrap for scheduler launches."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import psycopg

from src.db import pool as db_pool

from .config import DBConfig, DEFAULT_DB_CONFIG, REPO_ROOT


DEFAULT_SCHEMA_PATH = REPO_ROOT / "scripts" / "schema.sql"
REQUIRED_TABLES = (
    "benchmark",
    "model",
    "task",
    "completions",
    "eval",
    "checker",
    "scores",
)
REQUIRED_VIEWS = ("view_model_version",)


@dataclass(frozen=True, slots=True)
class DbSchemaReport:
    host: str
    port: int
    user: str
    dbname: str
    sslmode: str
    schema_path: str
    database_ok: bool
    schema_ok: bool
    missing_tables: tuple[str, ...] = ()
    missing_views: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.database_ok and self.schema_ok and not self.error)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def check_db_schema(
    config: DBConfig | None = None,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> DbSchemaReport:
    resolved = config or DEFAULT_DB_CONFIG
    try:
        db_pool._ensure_database_exists(resolved)
        with psycopg.connect(db_pool._build_conninfo(resolved)) as conn:
            tables = _existing_relations(conn, kind="BASE TABLE", names=REQUIRED_TABLES)
            views = _existing_relations(conn, kind="VIEW", names=REQUIRED_VIEWS)
    except Exception as exc:  # noqa: BLE001 - caller needs a compact diagnostic
        return _report(
            resolved,
            schema_path=schema_path,
            database_ok=False,
            schema_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    missing_tables = tuple(name for name in REQUIRED_TABLES if name not in tables)
    missing_views = tuple(name for name in REQUIRED_VIEWS if name not in views)
    return _report(
        resolved,
        schema_path=schema_path,
        database_ok=True,
        schema_ok=not missing_tables and not missing_views,
        missing_tables=missing_tables,
        missing_views=missing_views,
    )


def bootstrap_db_schema(
    config: DBConfig | None = None,
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> DbSchemaReport:
    resolved = config or DEFAULT_DB_CONFIG
    if not schema_path.exists():
        return _report(
            resolved,
            schema_path=schema_path,
            database_ok=False,
            schema_ok=False,
            error=f"schema file not found: {schema_path}",
        )
    try:
        db_pool._ensure_database_exists(resolved)
        ddl = schema_path.read_text(encoding="utf-8")
        with psycopg.connect(db_pool._build_conninfo(resolved), autocommit=True) as conn:
            conn.execute(ddl)
    except Exception as exc:  # noqa: BLE001
        return _report(
            resolved,
            schema_path=schema_path,
            database_ok=False,
            schema_ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return check_db_schema(resolved, schema_path=schema_path)


def render_db_schema_report(report: DbSchemaReport) -> str:
    lines = [
        f"database={report.user}@{report.host}:{report.port}/{report.dbname}",
        f"schema_path={report.schema_path}",
        f"database_ok={str(report.database_ok).lower()}",
        f"schema_ok={str(report.schema_ok).lower()}",
    ]
    if report.missing_tables:
        lines.append("missing_tables=" + ",".join(report.missing_tables))
    if report.missing_views:
        lines.append("missing_views=" + ",".join(report.missing_views))
    if report.error:
        lines.append(f"error={report.error}")
    if not report.ok:
        lines.append("next_step=uv run rwkv-skills-scheduler bootstrap-db")
    return "\n".join(lines)


def _existing_relations(conn: psycopg.Connection[Any], *, kind: str, names: Sequence[str]) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = %s
              AND table_name = ANY(%s)
            """,
            (kind, list(names)),
        )
        return {str(row[0]) for row in cur.fetchall()}


def _report(
    config: DBConfig,
    *,
    schema_path: Path,
    database_ok: bool,
    schema_ok: bool,
    missing_tables: Sequence[str] = (),
    missing_views: Sequence[str] = (),
    error: str | None = None,
) -> DbSchemaReport:
    return DbSchemaReport(
        host=config.host,
        port=int(config.port),
        user=config.user,
        dbname=config.dbname,
        sslmode=config.sslmode,
        schema_path=str(schema_path),
        database_ok=database_ok,
        schema_ok=schema_ok,
        missing_tables=tuple(missing_tables),
        missing_views=tuple(missing_views),
        error=error,
    )


__all__ = [
    "DEFAULT_SCHEMA_PATH",
    "REQUIRED_TABLES",
    "REQUIRED_VIEWS",
    "DbSchemaReport",
    "bootstrap_db_schema",
    "check_db_schema",
    "render_db_schema_report",
]
