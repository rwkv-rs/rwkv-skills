#!/usr/bin/env python3
"""Read-only replay of persisted HumanEval-family completions.

This verifies that evaluator infrastructure changes preserve historical pass/
fail decisions before the new implementation is used for subsequent tasks.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.metrics.code_generation.evaluate import (
    evaluate_mbpp_dataset,
    extract_code_completion,
)
from src.eval.metrics.code_generation.human_eval.data import read_problems
from src.eval.metrics.code_generation.human_eval.execution import check_correctness
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DEFAULT_DB_NAME = "chase_rwkv_skills_frontend46_20260804"

COMPLETION_QUERY = """
SELECT
    c.completions_id,
    c.sample_index,
    c.avg_repeat_index,
    c.pass_index,
    COALESCE(
        NULLIF(c.context #>> '{stages,1,completion}', ''),
        NULLIF(c.context #>> '{stages,0,completion}', ''),
        NULLIF(c.context->>'direct_raw_completion', ''),
        ''
    ) AS completion,
    e.is_passed
FROM completions c
JOIN eval e ON e.completions_id = c.completions_id
WHERE c.task_id = %s
ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index
"""


def _evenly_spaced(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    return [rows[(index * len(rows)) // limit] for index in range(limit)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--family", choices=("human_eval", "mbpp"), default="human_eval")
    parser.add_argument("--dbname", default=DEFAULT_DB_NAME)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args()

    db_config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    with psycopg.connect(_build_conninfo(db_config), row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(COMPLETION_QUERY, (args.task_id,))
            all_rows = list(cursor.fetchall())

    selected = _evenly_spaced(all_rows, args.limit)
    problems = list(read_problems(str(args.dataset)).values())

    def replay(row: dict[str, Any]) -> dict[str, Any]:
        sample_index = int(row["sample_index"])
        result = check_correctness(
            problems[sample_index],
            extract_code_completion(str(row["completion"] or "")),
            args.timeout,
            completion_id=int(row["completions_id"]),
        )
        return {
            "completions_id": int(row["completions_id"]),
            "sample_index": sample_index,
            "avg_repeat_index": int(row["avg_repeat_index"]),
            "pass_index": int(row["pass_index"]),
            "stored_is_passed": bool(row["is_passed"]),
            "replayed_is_passed": bool(result["passed"]),
            "replayed_result": str(result["result"]),
        }

    if args.family == "human_eval":
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            replayed = list(executor.map(replay, selected))
    else:
        completion_payloads = [
            {
                "benchmark_name": "mbpp_plus",
                "dataset_split": "test",
                "sample_index": int(row["sample_index"]),
                "repeat_index": int(row["avg_repeat_index"]),
                "completion1": str(row["completion"] or ""),
            }
            for row in selected
        ]
        _metrics, eval_payloads = evaluate_mbpp_dataset(
            completion_payloads,
            dataset_path=args.dataset,
            pass_k=(1,),
            n_workers=max(1, args.workers),
            timeout=args.timeout,
        )
        eval_by_key = {
            (int(payload["sample_index"]), int(payload["repeat_index"])): payload
            for payload in eval_payloads
        }
        replayed = []
        for row in selected:
            key = (int(row["sample_index"]), int(row["avg_repeat_index"]))
            payload = eval_by_key[key]
            replayed.append(
                {
                    "completions_id": int(row["completions_id"]),
                    "sample_index": key[0],
                    "avg_repeat_index": key[1],
                    "pass_index": int(row["pass_index"]),
                    "stored_is_passed": bool(row["is_passed"]),
                    "replayed_is_passed": bool(payload["is_passed"]),
                    "replayed_result": str(payload.get("fail_reason") or "passed"),
                }
            )

    mismatches = [
        row for row in replayed if row["stored_is_passed"] != row["replayed_is_passed"]
    ]
    payload = {
        "task_id": args.task_id,
        "family": args.family,
        "dataset": str(args.dataset),
        "stored_rows": len(all_rows),
        "replayed_rows": len(replayed),
        "mismatch_count": len(mismatches),
        "active_children_after_replay": len(multiprocessing.active_children()),
        "mismatch_examples": mismatches[:10],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if mismatches or payload["active_children_after_replay"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
