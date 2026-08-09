#!/usr/bin/env python3
"""Safely close explicitly audited stale G1i task rows.

This helper is intentionally narrow: it targets the strict-46 database, requires
every requested row to still be Running, refuses rows that already have a score,
and uses :class:`EvalDbService` for the status transition.  Completions and eval
rows are preserved unchanged for auditability.
"""

from __future__ import annotations

import argparse
import os

import psycopg
from psycopg.rows import dict_row


TARGET_DATABASE = "chase_rwkv_skills_frontend46_20260804"
TARGET_MODEL_PREFIX = "rwkv7-g1i-"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_ids", metavar="TASK_ID", type=int, nargs="+")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the status transitions; otherwise only validate and print",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    task_ids = tuple(dict.fromkeys(args.task_ids))
    os.environ["RWKV_EVAL_SPACE_DB_DATABASE_NAME"] = TARGET_DATABASE

    # Imports happen after the database override so the shared config is bound
    # to the intended audit database rather than the repository default.
    from src.db.eval_db_service import EvalDbService
    from src.db.pool import _build_conninfo
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    query = """
        SELECT t.task_id, t.status, m.model_name,
               count(DISTINCT c.completions_id) AS completions,
               count(DISTINCT e.eval_id) AS evals,
               count(DISTINCT s.score_id) AS scores
        FROM task t
        JOIN model m ON m.model_id = t.model_id
        LEFT JOIN completions c ON c.task_id = t.task_id
        LEFT JOIN eval e ON e.completions_id = c.completions_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        WHERE t.task_id = ANY(%s)
        GROUP BY t.task_id, t.status, m.model_name
        ORDER BY t.task_id
    """
    conninfo = _build_conninfo(DEFAULT_DB_CONFIG)
    with psycopg.connect(conninfo, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (list(task_ids),))
            before = list(cursor.fetchall())

    found = {int(row["task_id"]) for row in before}
    missing = sorted(set(task_ids) - found)
    if missing:
        raise SystemExit(f"refusing: task ids not found in {TARGET_DATABASE}: {missing}")
    for row in before:
        if str(row["status"]).lower() != "running":
            raise SystemExit(
                f"refusing: task {row['task_id']} status is {row['status']!r}, not Running"
            )
        if not str(row["model_name"]).startswith(TARGET_MODEL_PREFIX):
            raise SystemExit(
                f"refusing: task {row['task_id']} model is {row['model_name']!r}"
            )
        if int(row["scores"]) != 0:
            raise SystemExit(f"refusing: task {row['task_id']} already has a score")

    for row in before:
        print(
            "validated",
            row["task_id"],
            row["model_name"],
            row["status"],
            f"completions={row['completions']}",
            f"evals={row['evals']}",
        )
    if not args.apply:
        print("dry-run only; pass --apply after process-level verification")
        return 0

    service = EvalDbService()
    for task_id in task_ids:
        service.update_task_status(task_id=str(task_id), status="failed")

    with psycopg.connect(conninfo, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (list(task_ids),))
            after = list(cursor.fetchall())
    bad = [row for row in after if str(row["status"]).lower() != "failed"]
    if bad:
        raise SystemExit(f"postcondition failed: {bad}")
    print(f"marked failed: {len(after)} task(s); completions/evals preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
