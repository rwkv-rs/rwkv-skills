from __future__ import annotations

"""Sync completed eval DB score bundles from a source PostgreSQL DB to a target.

The script is intentionally conservative: it defaults to dry-run, preserves
primary keys, and only copies score rows plus the task/completion/eval/checker
rows they depend on.
"""

import argparse
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


TABLE_PK = {
    "benchmark": "benchmark_id",
    "model": "model_id",
    "task": "task_id",
    "completions": "completions_id",
    "eval": "eval_id",
    "checker": "checker_id",
    "scores": "score_id",
}

JSON_COLUMNS = {
    ("task", "sampling_config"),
    ("completions", "context"),
    ("scores", "metrics"),
}


@dataclass(slots=True)
class DbArgs:
    host: str
    port: int
    user: str
    password: str
    dbname: str
    sslmode: str

    def conninfo(self) -> str:
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"user={self.user}",
            f"dbname={self.dbname}",
            f"sslmode={self.sslmode}",
        ]
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync rwkv-skills eval DB score bundles")
    parser.add_argument("--score-id-min", type=int, help="Sync scores with score_id >= this value")
    parser.add_argument("--score-id", type=int, action="append", default=[], help="Specific score_id to sync")
    parser.add_argument("--task-id", type=int, action="append", default=[], help="Specific task_id to sync")
    parser.add_argument("--write", action="store_true", help="Actually write target DB; default is dry-run")

    parser.add_argument("--source-host", default=_env("PG_HOST", "127.0.0.1"))
    parser.add_argument("--source-port", type=int, default=int(_env("PG_PORT", "5432")))
    parser.add_argument("--source-user", default=_env("PG_USER", "postgres"))
    parser.add_argument("--source-password", default=_env("PG_PASSWORD", ""))
    parser.add_argument("--source-dbname", default=_env("PG_DBNAME", "rwkv-eval"))
    parser.add_argument("--source-sslmode", default=_env("PG_SSLMODE", "prefer"))

    parser.add_argument("--target-host", default=_env("TARGET_PG_HOST", "127.0.0.1"))
    parser.add_argument("--target-port", type=int, default=int(_env("TARGET_PG_PORT", "15432")))
    parser.add_argument("--target-user", default=_env("TARGET_PG_USER", "postgres"))
    parser.add_argument("--target-password", default=_env("TARGET_PG_PASSWORD", ""))
    parser.add_argument("--target-dbname", default=_env("TARGET_PG_DBNAME", "chase_rwkv_skills"))
    parser.add_argument("--target-sslmode", default=_env("TARGET_PG_SSLMODE", "prefer"))
    return parser.parse_args()


def _db_args(args: argparse.Namespace, prefix: str) -> DbArgs:
    return DbArgs(
        host=getattr(args, f"{prefix}_host"),
        port=int(getattr(args, f"{prefix}_port")),
        user=getattr(args, f"{prefix}_user"),
        password=getattr(args, f"{prefix}_password"),
        dbname=getattr(args, f"{prefix}_dbname"),
        sslmode=getattr(args, f"{prefix}_sslmode"),
    )


def _where_in(column: str, values: Sequence[int]) -> tuple[sql.SQL, list[Any]]:
    if not values:
        return sql.SQL("FALSE"), []
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in values)
    return sql.SQL("{} IN ({})").format(sql.Identifier(column), placeholders), list(values)


def _fetch_rows(
    conn: psycopg.Connection[dict[str, Any]],
    table: str,
    where: sql.SQL,
    params: Sequence[Any],
) -> list[dict[str, Any]]:
    pk = TABLE_PK[table]
    query = sql.SQL("SELECT * FROM {} WHERE {} ORDER BY {}").format(
        sql.Identifier(table),
        where,
        sql.Identifier(pk),
    )
    with conn.cursor() as cur:
        cur.execute(query, list(params))
        return [dict(row) for row in cur.fetchall()]


def _ids(rows: Iterable[Mapping[str, Any]], key: str) -> list[int]:
    return sorted({int(row[key]) for row in rows if row.get(key) is not None})


def _adapt_value(table: str, column: str, value: Any) -> Any:
    if (table, column) in JSON_COLUMNS and value is not None:
        return Jsonb(value)
    return value


def _upsert_rows(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    table: str,
    rows: Sequence[Mapping[str, Any]],
    dry_run: bool,
) -> int:
    if not rows:
        return 0
    if dry_run:
        return len(rows)

    pk = TABLE_PK[table]
    columns = list(rows[0].keys())
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
    assignments = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
        for col in columns
        if col != pk
    )
    if assignments.as_string(conn) == "":
        conflict = sql.SQL("DO NOTHING")
    else:
        conflict = sql.SQL("DO UPDATE SET {}").format(assignments)
    query = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) {}").format(
        sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(col) for col in columns),
        placeholders,
        sql.Identifier(pk),
        conflict,
    )
    values = [
        tuple(_adapt_value(table, col, row.get(col)) for col in columns)
        for row in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(query, values)
    return len(rows)


def _reset_sequence(conn: psycopg.Connection[dict[str, Any]], *, table: str) -> None:
    pk = TABLE_PK[table]
    query = sql.SQL(
        "SELECT setval(pg_get_serial_sequence(%s, %s), COALESCE((SELECT MAX({}) FROM {}), 1), true)"
    ).format(sql.Identifier(pk), sql.Identifier(table))
    with conn.cursor() as cur:
        cur.execute(query, (table, pk))


def _collect_bundle(
    conn: psycopg.Connection[dict[str, Any]],
    *,
    score_ids: Sequence[int],
    score_id_min: int | None,
    task_ids: Sequence[int],
) -> dict[str, list[dict[str, Any]]]:
    score_clauses: list[sql.SQL] = []
    score_params: list[Any] = []
    if score_ids:
        where, params = _where_in("score_id", sorted(set(score_ids)))
        score_clauses.append(where)
        score_params.extend(params)
    if score_id_min is not None:
        score_clauses.append(sql.SQL("score_id >= %s"))
        score_params.append(int(score_id_min))

    score_rows: list[dict[str, Any]] = []
    if score_clauses:
        score_where = sql.SQL(" OR ").join(score_clauses)
        score_rows = _fetch_rows(conn, "scores", score_where, score_params)

    bundle_task_ids = sorted(set(task_ids) | set(_ids(score_rows, "task_id")))
    task_rows = _fetch_rows(conn, "task", *_where_in("task_id", bundle_task_ids))
    if not task_rows and not score_rows:
        return {table: [] for table in TABLE_PK}

    task_ids = _ids(task_rows, "task_id")
    completion_rows = _fetch_rows(conn, "completions", *_where_in("task_id", task_ids))
    completion_ids = _ids(completion_rows, "completions_id")
    eval_rows = _fetch_rows(conn, "eval", *_where_in("completions_id", completion_ids))
    checker_rows = _fetch_rows(conn, "checker", *_where_in("completions_id", completion_ids))

    benchmark_rows = _fetch_rows(conn, "benchmark", *_where_in("benchmark_id", _ids(task_rows, "benchmark_id")))
    model_rows = _fetch_rows(conn, "model", *_where_in("model_id", _ids(task_rows, "model_id")))
    if not score_rows and task_ids:
        score_rows = _fetch_rows(conn, "scores", *_where_in("task_id", task_ids))

    return {
        "benchmark": benchmark_rows,
        "model": model_rows,
        "task": task_rows,
        "completions": completion_rows,
        "eval": eval_rows,
        "checker": checker_rows,
        "scores": score_rows,
    }


def main() -> int:
    args = parse_args()
    if not args.score_id and args.score_id_min is None and not args.task_id:
        raise SystemExit("Provide --score-id-min, --score-id, or --task-id.")

    source = _db_args(args, "source")
    target = _db_args(args, "target")
    dry_run = not bool(args.write)

    with psycopg.connect(source.conninfo(), row_factory=dict_row) as src:
        bundle = _collect_bundle(
            src,
            score_ids=args.score_id,
            score_id_min=args.score_id_min,
            task_ids=args.task_id,
        )

    print("mode", "dry-run" if dry_run else "write")
    for table in TABLE_PK:
        print(table, len(bundle[table]))
    if dry_run:
        return 0

    with psycopg.connect(target.conninfo(), row_factory=dict_row) as dst:
        with dst.transaction():
            for table in ("benchmark", "model", "task", "completions", "eval", "checker", "scores"):
                _upsert_rows(dst, table=table, rows=bundle[table], dry_run=False)
            for table in ("benchmark", "model", "task", "completions", "eval", "checker", "scores"):
                _reset_sequence(dst, table=table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
