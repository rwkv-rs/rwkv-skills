#!/usr/bin/env python3
"""Read-only listing of scored model/task history for one benchmark."""

from __future__ import annotations

import argparse
import json
import os

import psycopg
from psycopg.rows import dict_row


DB_NAME = "chase_rwkv_skills_frontend46_20260804"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("--split", default="test")
    parser.add_argument("--database", default=DB_NAME)
    args = parser.parse_args()
    os.environ["RWKV_EVAL_SPACE_DB_DATABASE_NAME"] = args.database

    from src.db.pool import _build_conninfo
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    query = """
        SELECT m.model_name, t.task_id, t.evaluator, t.status,
               t.sampling_config, s.metrics, s.cot_mode,
               s.created_at AS score_created_at
        FROM scores s
        JOIN task t ON t.task_id = s.task_id
        JOIN model m ON m.model_id = t.model_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        WHERE b.benchmark_name = %s AND b.benchmark_split = %s
        ORDER BY m.model_name, t.task_id DESC
    """
    with psycopg.connect(
        _build_conninfo(DEFAULT_DB_CONFIG), row_factory=dict_row
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (args.benchmark, args.split))
            for row in cursor.fetchall():
                print(json.dumps(row, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
