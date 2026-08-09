#!/usr/bin/env python3
"""Restore BrowseComp-Plus audit summaries lost by a judge round-trip.

The repair is deliberately limited to completion.context audit fields. It does
not change model answers, judge decisions, eval rows, task state, or scores.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.db.eval_db_service import EvalDbService
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.tasks.function_calling.browsecomp_plus import (
    _browsecomp_plus_confidence_percent,
    _browsecomp_plus_official_final_output,
    load_browsecomp_plus_manifest_records,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=830)
    parser.add_argument("--execute", action="store_true", help="Apply the repair; default is dry-run.")
    return parser.parse_args(argv)


def _connect() -> psycopg.Connection[Any]:
    cfg = DEFAULT_DB_CONFIG
    return psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        password=cfg.password,
        dbname=cfg.dbname,
        sslmode=cfg.sslmode,
        row_factory=dict_row,
    )


def _contains_compaction_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "__truncated_items__" in value:
            return True
        return any(_contains_compaction_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_compaction_marker(item) for item in value)
    return False


def _last_cumulative_run(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, list):
        return {}
    for entry in reversed(trace):
        if not isinstance(entry, Mapping):
            continue
        details = entry.get("details")
        if not isinstance(details, Mapping):
            continue
        run = details.get("browsecomp_plus_run")
        if isinstance(run, Mapping):
            return deepcopy(dict(run))
    return {}


def _restore_run(context: Mapping[str, Any], *, query_id: str) -> tuple[dict[str, Any], bool]:
    trace = context.get("agent_trace")
    run = _last_cumulative_run(trace)
    partial = not run or _contains_compaction_marker(trace) or _contains_compaction_marker(run)
    info = context.get("agent_info")
    info = dict(info) if isinstance(info, Mapping) else {}
    final_answer = str(info.get("final_answer") or "").strip()
    decoded = info.get("decoded_final_answer_call")
    decoded = dict(decoded) if isinstance(decoded, Mapping) else {}
    arguments = decoded.get("arguments")
    arguments = dict(arguments) if isinstance(arguments, Mapping) else {}

    run["query_id"] = str(run.get("query_id") or query_id)
    run["status"] = "completed" if final_answer else "incomplete"
    if final_answer:
        explanation = str(arguments.get("explanation") or run.get("final_explanation") or "")
        confidence = _browsecomp_plus_confidence_percent(
            arguments.get("confidence", run.get("final_confidence")),
            fallback=50.0,
        )
        run["final_explanation"] = explanation
        run["final_confidence"] = confidence
        run["result"] = [
            {
                "type": "output_text",
                "output": _browsecomp_plus_official_final_output(
                    final_answer,
                    explanation=explanation,
                    confidence=confidence,
                ),
            }
        ]
    else:
        run.setdefault("final_explanation", "")
        run.setdefault("final_confidence", 0.0)
        run["result"] = []
    run["audit_reconstruction"] = {
        "reconstructed_after_judge": True,
        "source": "agent_trace.details.browsecomp_plus_run",
        "partial_due_to_compaction": partial,
    }
    compacted = EvalDbService._compact_browsecomp_plus_run(run)
    if not isinstance(compacted, dict):
        raise TypeError("compacted BrowseComp-Plus run must be an object")
    return compacted, partial


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_browsecomp_plus_manifest_records(args.dataset)
    if len(records) != args.expected_samples:
        raise RuntimeError(f"dataset has {len(records)} samples; expected {args.expected_samples}")

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.task_id, t.evaluator, t.status, b.benchmark_name,
                       m.model_name, s.metrics
                FROM public.task t
                JOIN public.benchmark b ON b.benchmark_id = t.benchmark_id
                JOIN public.model m ON m.model_id = t.model_id
                LEFT JOIN public.scores s ON s.task_id = t.task_id
                WHERE t.task_id = %s
                """,
                (args.task_id,),
            )
            task = cur.fetchone()
            if not task:
                raise RuntimeError(f"task {args.task_id} does not exist")
            if task["evaluator"] != "function_browsecomp_plus" or task["benchmark_name"] != "browsecomp_plus":
                raise RuntimeError(f"task {args.task_id} is not BrowseComp-Plus: {task}")
            score_before = deepcopy(task.get("metrics"))
            cur.execute(
                """
                SELECT completions_id, sample_index, context
                FROM public.completions
                WHERE task_id = %s AND status = 'Completed'
                ORDER BY sample_index, avg_repeat_index, pass_index
                """,
                (args.task_id,),
            )
            rows = list(cur.fetchall())

        if len(rows) != args.expected_samples:
            raise RuntimeError(f"task has {len(rows)} completions; expected {args.expected_samples}")

        updates: list[tuple[Jsonb, Jsonb, int, int]] = []
        partial_count = 0
        already_present = 0
        for row in rows:
            sample_index = int(row["sample_index"])
            context = row["context"]
            if not isinstance(context, Mapping):
                raise TypeError(f"completion {row['completions_id']} context is not an object")
            if isinstance(context.get("browsecomp_plus_run"), Mapping) and isinstance(context.get("metadata"), Mapping):
                already_present += 1
                continue
            record = records[sample_index]
            run, partial = _restore_run(context, query_id=record.query_id)
            metadata = EvalDbService._compact_completion_extra(record.metadata)
            if not isinstance(metadata, dict):
                raise TypeError(f"sample {sample_index} metadata did not compact to an object")
            partial_count += int(partial)
            updates.append((Jsonb(run), Jsonb(metadata), int(row["completions_id"]), args.task_id))

        print(
            json.dumps(
                {
                    "task_id": args.task_id,
                    "completion_rows": len(rows),
                    "repair_rows": len(updates),
                    "already_present": already_present,
                    "partial_reconstructions": partial_count,
                    "execute": bool(args.execute),
                },
                sort_keys=True,
            )
        )
        if not args.execute:
            conn.rollback()
            return 0

        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE public.completions
                SET context = jsonb_set(
                    jsonb_set(context, '{browsecomp_plus_run}', %s, true),
                    '{metadata}', %s, true
                )
                WHERE completions_id = %s
                  AND task_id = %s
                  AND NOT (context ? 'browsecomp_plus_run')
                """,
                updates,
            )
            cur.execute(
                """
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE context ? 'browsecomp_plus_run') AS with_run,
                       count(*) FILTER (WHERE context ? 'metadata') AS with_metadata,
                       count(*) FILTER (
                           WHERE context->'agent_info'->>'judge_pending' = 'true'
                       ) AS judge_pending,
                       count(*) FILTER (
                           WHERE context->'agent_result'->>'is_passed' = 'true'
                       ) AS passed
                FROM public.completions
                WHERE task_id = %s AND status = 'Completed'
                """,
                (args.task_id,),
            )
            verification = dict(cur.fetchone() or {})
            cur.execute("SELECT metrics FROM public.scores WHERE task_id = %s", (args.task_id,))
            score_after_row = cur.fetchone()
            score_after = score_after_row.get("metrics") if score_after_row else None
        if score_after != score_before:
            raise RuntimeError("score changed during audit-field restoration")
        if verification != {
            "total": args.expected_samples,
            "with_run": args.expected_samples,
            "with_metadata": args.expected_samples,
            "judge_pending": 0,
            "passed": 18,
        }:
            raise RuntimeError(f"post-repair verification failed: {verification}")
        conn.commit()
        print(json.dumps({"verified": verification, "score_unchanged": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
