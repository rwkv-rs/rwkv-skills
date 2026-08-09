#!/usr/bin/env python3
"""Mark an empty, unscored probe task failed after validating it is harmless."""

from __future__ import annotations

import argparse
import os

import psycopg
from psycopg.rows import dict_row


DB_NAME = "chase_rwkv_skills_frontend46_20260804"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    args = parser.parse_args()

    os.environ["RWKV_EVAL_SPACE_DB_DATABASE_NAME"] = DB_NAME
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService
    from src.db.pool import _build_conninfo
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    init_db(DEFAULT_DB_CONFIG)
    query = """
        SELECT t.status,
               COUNT(DISTINCT c.completions_id) AS completion_count,
               COUNT(DISTINCT s.score_id) AS score_count
        FROM task t
        LEFT JOIN completions c ON c.task_id = t.task_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        WHERE t.task_id = %s
        GROUP BY t.task_id, t.status
    """
    with psycopg.connect(
        _build_conninfo(DEFAULT_DB_CONFIG), row_factory=dict_row
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (args.task_id,))
            row = cursor.fetchone()
    if row is None:
        raise SystemExit(f"task {args.task_id} does not exist")
    if str(row["status"]).lower() != "running":
        raise SystemExit(f"task {args.task_id} status is {row['status']!r}, not Running")
    if int(row["completion_count"]) or int(row["score_count"]):
        raise SystemExit(
            f"task {args.task_id} is not empty: "
            f"completions={row['completion_count']} scores={row['score_count']}"
        )

    EvalDbService().update_task_status(task_id=str(args.task_id), status="failed")
    print(f"task {args.task_id} marked failed (validated empty and unscored)")


if __name__ == "__main__":
    main()
