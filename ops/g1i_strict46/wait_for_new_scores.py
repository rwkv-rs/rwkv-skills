#!/usr/bin/env python3
"""Wait read-only for a new strict-46 score before running the heavy audit."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import time

import psycopg
from psycopg.rows import dict_row


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
MODELS = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", required=True, type=datetime.fromisoformat)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--interval-s", type=float, default=20.0)
    args = parser.parse_args()

    os.environ["RWKV_EVAL_SPACE_DB_DATABASE_NAME"] = DB_NAME
    from src.db.pool import _build_conninfo
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    query = """
        SELECT s.created_at AS score_created_at, t.task_id, m.model_name,
               b.benchmark_name, b.benchmark_split, t.status, s.metrics
        FROM scores s
        JOIN task t ON t.task_id = s.task_id
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        WHERE m.model_name = ANY(%s) AND s.created_at > %s
        ORDER BY s.created_at, t.task_id
    """
    deadline = time.monotonic() + args.timeout_s
    while True:
        with psycopg.connect(
            _build_conninfo(DEFAULT_DB_CONFIG), row_factory=dict_row
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (list(MODELS), args.after))
                rows = list(cursor.fetchall())
        if rows:
            print(json.dumps(rows, ensure_ascii=False, default=str))
            return
        if time.monotonic() >= deadline:
            raise SystemExit(3)
        time.sleep(min(args.interval_s, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
