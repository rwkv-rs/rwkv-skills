#!/usr/bin/env python3
from __future__ import annotations

"""Export BrowseComp-Plus DB completions to the official evaluator input shape."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, type=int, help="DB task_id to export.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for <query_id>.json files. Defaults to /tmp/browsecomp_plus_official_runs_task<TASK_ID>.",
    )
    parser.add_argument("--expected-count", type=int, default=830, help="Expected exported row count.")
    parser.add_argument("--force", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args(argv)


def connect_from_env():
    from src.eval.env_config import load_env_file
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    load_env_file()
    return psycopg.connect(
        host=DEFAULT_DB_CONFIG.host,
        port=DEFAULT_DB_CONFIG.port,
        user=DEFAULT_DB_CONFIG.user,
        password=DEFAULT_DB_CONFIG.password,
        dbname=DEFAULT_DB_CONFIG.dbname,
        row_factory=dict_row,
    )


def load_task_and_completions(task_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with connect_from_env() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    t.task_id,
                    t.status,
                    t.evaluator,
                    m.model_name,
                    b.benchmark_name,
                    b.benchmark_split
                FROM task t
                JOIN model m ON m.model_id = t.model_id
                JOIN benchmark b ON b.benchmark_id = t.benchmark_id
                WHERE t.task_id = %s
                """,
                (int(task_id),),
            )
            task = cur.fetchone()
            if not task:
                raise SystemExit(f"task_id {task_id} not found")
            cur.execute(
                """
                SELECT completions_id, sample_index, avg_repeat_index, pass_index, context
                FROM completions
                WHERE task_id = %s
                ORDER BY sample_index ASC, avg_repeat_index ASC, pass_index ASC, completions_id ASC
                """,
                (int(task_id),),
            )
            rows = cur.fetchall()
    return dict(task), [dict(row) for row in rows]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any, *, field: str, query_id: str) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, Mapping) and "__truncated_items__" in item:
            raise ValueError(f"query_id={query_id} has truncated {field}; rerun with fixed DB context persistence")
        out.append(str(item))
    return out


def _result_list(value: Any, *, query_id: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        item_type = str(item.get("type") or "")
        output = str(item.get("output") or "")
        if item_type and output:
            result.append({"type": item_type, "output": output})
    if len(result) > 1:
        raise ValueError(f"query_id={query_id} has multiple output records; expected at most one")
    return result


def completion_to_official_run(task: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    context = _mapping(row.get("context"))
    run = _mapping(context.get("browsecomp_plus_run"))
    metadata = _mapping(context.get("metadata"))
    if not run:
        raise ValueError(
            f"completions_id={row.get('completions_id')} is missing context.browsecomp_plus_run; "
            "rerun after the DB persistence fix"
        )
    query_id = str(run.get("query_id") or metadata.get("query_id") or "").strip()
    if not query_id:
        raise ValueError(f"completions_id={row.get('completions_id')} is missing query_id")
    result = _result_list(run.get("result"), query_id=query_id)
    status = str(run.get("status") or ("completed" if result else "incomplete"))
    if status == "completed" and not result:
        status = "incomplete"
    return {
        "query_id": query_id,
        "tool_call_counts": dict(_mapping(run.get("tool_call_counts"))),
        "status": status,
        "retrieved_docids": _string_list(run.get("retrieved_docids"), field="retrieved_docids", query_id=query_id),
        "result": result if status == "completed" else [],
        "metadata": {
            "model": str(task.get("model_name") or ""),
            "task_id": int(task["task_id"]),
            "completions_id": int(row["completions_id"]),
            "sample_index": int(row["sample_index"]),
            "avg_repeat_index": int(row["avg_repeat_index"]),
            "pass_index": int(row["pass_index"]),
            "benchmark_name": str(task.get("benchmark_name") or ""),
            "benchmark_split": str(task.get("benchmark_split") or ""),
        },
    }


def prepare_output_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise SystemExit(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()) and not force:
            raise SystemExit(f"output directory is not empty; pass --force to reuse: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    opts = parse_args(argv)
    output_dir = opts.output_dir or Path(f"/tmp/browsecomp_plus_official_runs_task{opts.task_id}")
    task, rows = load_task_and_completions(int(opts.task_id))
    if task.get("benchmark_name") != "browsecomp_plus":
        raise SystemExit(f"task_id {opts.task_id} is {task.get('benchmark_name')!r}, not browsecomp_plus")
    prepare_output_dir(output_dir, force=bool(opts.force))
    seen: set[str] = set()
    completed = 0
    for row in rows:
        payload = completion_to_official_run(task, row)
        query_id = str(payload["query_id"])
        if query_id in seen:
            raise SystemExit(f"duplicate query_id in task {opts.task_id}: {query_id}")
        seen.add(query_id)
        if payload["status"] == "completed":
            completed += 1
        with (output_dir / f"{query_id}.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    if opts.expected_count and len(seen) != int(opts.expected_count):
        raise SystemExit(f"exported {len(seen)} rows, expected {opts.expected_count}")
    print(
        json.dumps(
            {
                "task_id": int(opts.task_id),
                "output_dir": str(output_dir),
                "exported": len(seen),
                "completed": completed,
                "incomplete": len(seen) - completed,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
