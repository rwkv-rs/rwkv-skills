#!/usr/bin/env python3
"""Read-only, inexpensive progress snapshot for active G1i target tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


MODELS = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
)

QUERY = """
SELECT
    t.task_id,
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    t.evaluator,
    t.status,
    t.created_at AS task_created_at,
    COALESCE((t.sampling_config->>'effective_sample_count')::integer, 0) AS expected,
    COUNT(c.completions_id) AS completions,
    COUNT(c.completions_id) FILTER (
        WHERE c.created_at >= CURRENT_TIMESTAMP - interval '10 minutes'
    ) AS completions_10m,
    COUNT(c.completions_id) FILTER (
        WHERE c.created_at >= CURRENT_TIMESTAMP - interval '60 minutes'
    ) AS completions_60m,
    MAX(c.created_at) AS latest_completion_at,
    COUNT(e.eval_id) AS evals,
    s.created_at AS score_created_at
FROM task t
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN completions c ON c.task_id = t.task_id
LEFT JOIN eval e ON e.completions_id = c.completions_id
LEFT JOIN scores s ON s.task_id = t.task_id
WHERE m.model_name = ANY(%s)
  AND t.task_id >= %s
  AND (t.status IN ('Running', 'running') OR s.score_id IS NOT NULL)
GROUP BY
    t.task_id, m.model_name, b.benchmark_name, b.benchmark_split,
    t.evaluator, t.status, t.created_at, t.sampling_config, s.created_at
ORDER BY t.task_id
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default="chase_rwkv_skills_frontend46_20260804")
    parser.add_argument("--since-task-id", type=int, default=28527)
    args = parser.parse_args()

    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as connection:
        rows = list(connection.execute(QUERY, (list(MODELS), args.since_task_id)))
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
