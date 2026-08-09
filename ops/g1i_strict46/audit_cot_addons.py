#!/usr/bin/env python3
"""Read-only inventory for G1h/G1i Knowledge CoT and NoCoT cells."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from ops.g1i_strict46.audit_current import KNOWLEDGE


DATABASES = (
    "chase_rwkv_skills_frontend46_20260804",
    "chase_rwkv_skills",
    "rwkv-g1h-fallback-20260720",
)
MODELS = (
    "rwkv7-g1h-1.5b-20260710-ctx10240",
    "rwkv7-g1h-2.9b-20260710-ctx10240",
    "rwkv7-g1h-7.2b-20260710-ctx10240",
    "rwkv7-g1h-13.3b-20260710-ctx10240",
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
)
TARGETS = KNOWLEDGE
MODES = ("no_cot", "cot")

META_QUERY = """
SELECT
    %s AS database,
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    t.task_id,
    t.status,
    t.evaluator,
    t.created_at AS task_created_at,
    t.sampling_config,
    s.cot_mode,
    s.metrics,
    s.created_at AS score_created_at
FROM model m
JOIN task t ON t.model_id = m.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN scores s ON s.task_id = t.task_id
WHERE m.model_name = ANY(%s)
  AND b.benchmark_name = ANY(%s)
ORDER BY t.task_id
"""

STATS_QUERY = """
SELECT
    c.task_id,
    COUNT(c.completions_id) AS completion_count,
    COUNT(DISTINCT (c.sample_index, c.avg_repeat_index, c.pass_index))
        AS distinct_coordinates,
    COUNT(e.eval_id) AS eval_count,
    COUNT(*) FILTER (WHERE e.fail_reason = 'missing_prediction')
        AS missing_prediction_count,
    COUNT(*) FILTER (
        WHERE c.completions_id IS NOT NULL
          AND COALESCE(
              NULLIF(c.context->>'direct_raw_completion', ''),
              NULLIF(c.context #>> '{strategy_a,completion}', ''),
              NULLIF(c.context #>> '{stages,0,completion}', ''),
              ''
          ) = ''
    ) AS blank_primary_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
           OR COALESCE(c.context->>'direct_raw_finish_reason', '') IN
              ('length', 'max_length', 'max_tokens')
           OR COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN
              ('length', 'max_length', 'max_tokens')
    ) AS truncation_count
FROM completions c
LEFT JOIN eval e ON e.completions_id = c.completions_id
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
"""


def _json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    return str(value)


def _is_cot(value: Any) -> bool:
    return str(value or "").lower().replace("-", "_") in {"cot", "true", "1"}


def _mode(value: Any) -> str:
    return "cot" if _is_cot(value) else "no_cot"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    connections: dict[str, Any] = {}
    for database in DATABASES:
        config = replace(DEFAULT_DB_CONFIG, dbname=database)
        try:
            connection = psycopg.connect(_build_conninfo(config), row_factory=dict_row)
            connections[database] = connection
            with connection.cursor() as cursor:
                cursor.execute(
                    META_QUERY,
                    (
                        database,
                        list(MODELS),
                        sorted({name for name, _split in TARGETS}),
                    ),
                )
                rows.extend({key: _json(value) for key, value in row.items()} for row in cursor)
        except psycopg.Error as exc:
            rows.append({"database": database, "query_error": str(exc)})

    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        if "query_error" in row:
            continue
        benchmark_key = (str(row["benchmark_name"]), str(row["benchmark_split"]))
        if benchmark_key not in TARGETS:
            continue
        mode = _mode(row.get("cot_mode") or (row.get("sampling_config") or {}).get("cot_mode"))
        key = (
            str(row["model_name"]),
            str(row["benchmark_name"]),
            str(row["benchmark_split"]),
            mode,
        )
        current = selected.get(key)
        ordering = (str(row.get("score_created_at") or ""), int(row["task_id"]))
        current_ordering = (
            str(current.get("score_created_at") or ""),
            int(current["task_id"]),
        ) if current else None
        if current is None or ordering > current_ordering:
            selected[key] = row

    selected_by_database: dict[str, list[int]] = {}
    for row in selected.values():
        selected_by_database.setdefault(str(row["database"]), []).append(int(row["task_id"]))
    for database, task_ids in selected_by_database.items():
        connection = connections.get(database)
        if connection is None or not task_ids:
            continue
        try:
            with connection.cursor() as cursor:
                cursor.execute(STATS_QUERY, (task_ids,))
                stats = {int(row["task_id"]): dict(row) for row in cursor}
        except psycopg.Error as exc:
            rows.append({"database": database, "query_error": str(exc)})
            continue
        for row in selected.values():
            if row["database"] != database:
                continue
            row.update({key: _json(value) for key, value in stats.get(int(row["task_id"]), {}).items()})

    for connection in connections.values():
        connection.close()

    cells = []
    for model in MODELS:
        for benchmark, split in TARGETS:
            for mode in MODES:
                row = selected.get((model, benchmark, split, mode))
                cells.append(
                    {
                        "model_name": model,
                        "benchmark": f"{benchmark}__{split}",
                        "mode": mode,
                        "present": row is not None,
                        "latest": row,
                    }
                )
    payload = {
        "target_cells": len(cells),
        "present_cells": sum(cell["present"] for cell in cells),
        "missing_cells": sum(not cell["present"] for cell in cells),
        "cells": cells,
        "database_errors": [row for row in rows if "query_error" in row],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary:
        by_arch_mode: dict[str, dict[str, int]] = {}
        for cell in cells:
            architecture = "G1h" if "g1h" in cell["model_name"] else "G1i"
            key = f"{architecture}_{cell['mode']}"
            bucket = by_arch_mode.setdefault(key, {"present": 0, "missing": 0})
            bucket["present" if cell["present"] else "missing"] += 1
        print(
            json.dumps(
                {
                    "target_cells": payload["target_cells"],
                    "present_cells": payload["present_cells"],
                    "missing_cells": payload["missing_cells"],
                    "by_architecture_mode": by_arch_mode,
                    "database_errors": payload["database_errors"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
