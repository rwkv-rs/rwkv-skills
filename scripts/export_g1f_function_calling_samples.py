#!/usr/bin/env python3
"""Export selected g1f-13.3B function-calling completions from the eval DB."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORE_INDEX = REPO_ROOT / "results" / "space" / "score_index.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "exports" / "g1f_13_3b_function_calling_samples.json"
TARGET_MODEL = "rwkv7-g1f-13.3b-20260415-ctx8192"
TARGET_DATASETS = (
    "toolalpaca_eval_real_test",
    "bfcl_exec_multiple_test",
    "bfcl_multiple_test",
    "bfcl_exec_simple_test",
    "bfcl_simple_python_test",
)


def _load_task_ids(score_index: Path, model: str, datasets: tuple[str, ...]) -> dict[str, int]:
    selected: dict[str, tuple[int, int]] = {}
    with score_index.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            dataset = str(item.get("dataset") or "")
            if dataset not in datasets or str(item.get("model") or "") != model:
                continue
            task_id = item.get("task_id")
            if task_id is None:
                continue
            selected[dataset] = (line_no, int(task_id))
    return {dataset: task_id for dataset, (_line_no, task_id) in selected.items()}


def _json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _completion_from_context(context: Any) -> str:
    if not isinstance(context, dict):
        return ""
    stages = context.get("stages")
    if not isinstance(stages, list):
        return ""
    for stage in reversed(stages):
        if isinstance(stage, dict) and isinstance(stage.get("completion"), str):
            return stage["completion"]
    return ""


def _simplify_row(dataset: str, task_id: int, row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("context")
    if not isinstance(context, dict):
        context = {}
    return {
        "dataset": dataset,
        "task_id": task_id,
        "sample_index": int(row["sample_index"]),
        "repeat_index": int(row["repeat_index"]),
        "pass_index": int(row["pass_index"]),
        "is_passed": bool(row["is_passed"]),
        "completion": _completion_from_context(context),
        "answer": _json_or_text(str(row.get("answer") or "")),
        "ref_answer": _json_or_text(str(row.get("ref_answer") or "")),
        "fail_reason": str(row.get("fail_reason") or ""),
        "instruction": str(context.get("instruction") or ""),
        "case_id": str(context.get("task_id") or ""),
        "agent_info": context.get("agent_info") if isinstance(context.get("agent_info"), dict) else {},
        "agent_trace": context.get("agent_trace") if isinstance(context.get("agent_trace"), list) else [],
        "stages": context.get("stages") if isinstance(context.get("stages"), list) else [],
    }


def _fetch_samples(conn: psycopg.Connection[Any], task_id: int, *, is_passed: bool, limit: int) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                c.sample_index AS sample_index,
                c.avg_repeat_index AS repeat_index,
                c.pass_index AS pass_index,
                e.is_passed AS is_passed,
                e.answer AS answer,
                e.ref_answer AS ref_answer,
                e.fail_reason AS fail_reason,
                c.context AS context
            FROM completions c
            JOIN eval e ON e.completions_id = c.completions_id
            WHERE c.task_id = %s
              AND e.is_passed = %s
            ORDER BY c.sample_index ASC, c.avg_repeat_index ASC, c.pass_index ASC, e.eval_id ASC
            LIMIT %s
            """,
            (task_id, is_passed, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def _connect(args: argparse.Namespace) -> psycopg.Connection[Any]:
    return psycopg.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname,
    )


def parse_args() -> argparse.Namespace:
    load_dotenv(REPO_ROOT / ".env", override=False, encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-index", type=Path, default=DEFAULT_SCORE_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--host", default=os.environ.get("PG_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PG_PORT", "5433")))
    parser.add_argument("--user", default=os.environ.get("PG_USER", "rwkv"))
    parser.add_argument("--password", default=os.environ.get("PG_PASSWORD", ""))
    parser.add_argument("--dbname", default=os.environ.get("PG_DBNAME", "rwkv-eval"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = TARGET_DATASETS
    task_ids = _load_task_ids(args.score_index, args.model, datasets)
    missing = [dataset for dataset in datasets if dataset not in task_ids]
    if missing:
        raise SystemExit(f"missing task_id in {args.score_index}: {', '.join(missing)}")

    payload: dict[str, Any] = {
        "model": args.model,
        "source_score_index": str(args.score_index),
        "exported_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "limit_per_outcome": int(args.limit),
        "task_ids": task_ids,
        "datasets": {},
    }
    with _connect(args) as conn:
        for dataset in datasets:
            task_id = task_ids[dataset]
            correct_rows = _fetch_samples(conn, task_id, is_passed=True, limit=args.limit)
            wrong_rows = _fetch_samples(conn, task_id, is_passed=False, limit=args.limit)
            payload["datasets"][dataset] = {
                "task_id": task_id,
                "correct": [_simplify_row(dataset, task_id, row) for row in correct_rows],
                "wrong": [_simplify_row(dataset, task_id, row) for row in wrong_rows],
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
