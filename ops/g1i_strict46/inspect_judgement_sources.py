#!/usr/bin/env python3
"""Inspect which generated lane supplied stored judgement-label eval rows."""

from __future__ import annotations

import argparse
import json
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import _completion_payload_from_context
from src.db.pool import _build_conninfo
from src.eval.metrics.free_response import (
    STRATEGY_A,
    STRATEGY_C,
    _extract_judgement_label,
    _strategy_judgement_text,
)
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


def _label(group: str, payload: dict[str, Any]) -> str | None:
    return _extract_judgement_label(_strategy_judgement_text(group, payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id", type=int)
    parser.add_argument("--database", default="chase_rwkv_skills_frontend46_20260804")
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    load_dotenv()
    query = """
        SELECT c.completions_id, c.sample_index, c.avg_repeat_index,
               c.pass_index, c.context, e.answer, e.ref_answer,
               e.is_passed, e.fail_reason
        FROM completions c
        JOIN eval e ON e.completions_id = c.completions_id
        WHERE c.task_id = %s
        ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index
    """
    with psycopg.connect(
        _build_conninfo(DEFAULT_DB_CONFIG, dbname=args.database),
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (args.task_id,))
            rows = list(cursor.fetchall())

    counters = {
        "rows": len(rows),
        "stored_differs_from_strategy_c": 0,
        "stored_differs_from_evaluator_source": 0,
        "strategy_a_inherited": 0,
    }
    examples: list[dict[str, Any]] = []
    for row in rows:
        payload = _completion_payload_from_context(row.get("context"))
        reference = _extract_judgement_label(str(row.get("ref_answer") or ""))
        strategy_a = _label(STRATEGY_A, payload)
        strategy_c = _label(STRATEGY_C, payload)
        stored = _extract_judgement_label(str(row.get("answer") or ""))
        # evaluate_free_response inherits A into B/C when A already passes.
        inherited_a = bool(reference and strategy_a == reference)
        expected = strategy_a if inherited_a else strategy_c
        counters["strategy_a_inherited"] += int(inherited_a)
        counters["stored_differs_from_strategy_c"] += int(stored != strategy_c)
        counters["stored_differs_from_evaluator_source"] += int(stored != expected)
        if (stored != strategy_c or stored != expected) and len(examples) < args.examples:
            examples.append(
                {
                    "completions_id": row["completions_id"],
                    "sample_index": row["sample_index"],
                    "avg_repeat_index": row["avg_repeat_index"],
                    "pass_index": row["pass_index"],
                    "reference": reference,
                    "strategy_a": strategy_a,
                    "strategy_c": strategy_c,
                    "stored": stored,
                    "evaluator_source": expected,
                    "strategy_a_inherited": inherited_a,
                    "is_passed": row["is_passed"],
                    "fail_reason": row["fail_reason"],
                }
            )

    print(json.dumps({"task_id": args.task_id, **counters, "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
