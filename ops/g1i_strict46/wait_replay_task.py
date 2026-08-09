#!/usr/bin/env python3
"""Wait for scored source tasks, then replay them with the current adapter.

This is an operational safety net for tasks whose runner imported an older
evaluator before a global repair was deployed.  It never mutates the source
task.  A fresh replay task is written by ``recompute_math_from_completions``
and the audit tag makes the operation idempotent across service restarts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DEFAULT_DB_NAME = "chase_rwkv_skills_frontend46_20260804"


def _append_event(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _source_state(
    connection: psycopg.Connection[dict_row], task_id: int
) -> dict[str, object] | None:
    row = connection.execute(
        """
        SELECT t.task_id, t.status, s.score_id, s.created_at AS score_created_at
        FROM task t
        LEFT JOIN scores s ON s.task_id = t.task_id
        WHERE t.task_id = %s
        ORDER BY s.score_id DESC NULLS LAST
        LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    return dict(row) if row is not None else None


def _existing_replay(
    connection: psycopg.Connection[dict_row], task_id: int, reason_tag: str
) -> dict[str, object] | None:
    marker = f"%replay_source_task_id={task_id};{reason_tag}%"
    row = connection.execute(
        """
        SELECT t.task_id, t.status, s.score_id, s.created_at AS score_created_at
        FROM task t
        JOIN scores s ON s.task_id = t.task_id
        WHERE t.desc LIKE %s AND t.status = 'Completed'
        ORDER BY s.score_id DESC
        LIMIT 1
        """,
        (marker,),
    ).fetchone()
    return dict(row) if row is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="+", type=int)
    parser.add_argument("--dbname", default=DEFAULT_DB_NAME)
    parser.add_argument("--reason-tag", required=True)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--judge-mode", choices=("auto", "exact", "llm"), default="auto")
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/audits/g1i_waited_replays"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/g1i_waited_replay_events.jsonl"),
    )
    args = parser.parse_args()
    if args.interval_s < 1:
        parser.error("--interval-s must be at least 1")
    if args.judge_max_workers < 1:
        parser.error("--judge-max-workers must be at least 1")

    repo = Path(__file__).resolve().parents[2]
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    pending = set(args.task_ids)

    while pending:
        ready: list[int] = []
        with psycopg.connect(
            _build_conninfo(config), row_factory=dict_row
        ) as connection:
            for task_id in sorted(pending):
                replay = _existing_replay(connection, task_id, args.reason_tag)
                if replay is not None:
                    _append_event(
                        args.events,
                        {
                            "event": "replay_already_completed",
                            "observed_at": datetime.now().astimezone(),
                            "source_task_id": task_id,
                            "replay": replay,
                        },
                    )
                    pending.remove(task_id)
                    continue
                state = _source_state(connection, task_id)
                if state is None:
                    raise RuntimeError(f"source task {task_id} does not exist")
                status = str(state.get("status") or "")
                if status.lower() == "failed":
                    raise RuntimeError(f"source task {task_id} failed before replay")
                if status.lower() == "completed" and state.get("score_id") is not None:
                    ready.append(task_id)

        for task_id in ready:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output = args.output_dir / f"source_{task_id}_{args.reason_tag}.json"
            command = [
                str(repo / ".venv/bin/python"),
                str(replay_script),
                str(task_id),
                "--dbname",
                args.dbname,
                "--judge-mode",
                args.judge_mode,
                "--judge-max-workers",
                str(args.judge_max_workers),
                "--reason-tag",
                args.reason_tag,
                "--commit",
                "--summary",
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=repo,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            event = {
                "event": "waited_replay_completed" if completed.returncode == 0 else "waited_replay_failed",
                "observed_at": datetime.now().astimezone(),
                "source_task_id": task_id,
                "reason_tag": args.reason_tag,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "output": str(output),
            }
            _append_event(args.events, event)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"replay for source task {task_id} failed: {completed.stderr[-1000:]}"
                )
            pending.remove(task_id)

        if pending:
            time.sleep(args.interval_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
