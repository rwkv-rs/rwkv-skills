#!/usr/bin/env python3
"""Reconcile strict-46 Math scores after the global answer extractor deploys.

The monitor is append-only and fail-closed.  It selects one canonical root
source for each G1i model x Math benchmark cell, waits for its immutable
completion grid to settle, and replays that grid with the globally deployed
extractor.  A post-deployment valid root or an exact source/reason marker
resolves the cell without another replay.

``--deployed-at`` is intentionally required.  It must be the first database
timestamp after the candidate extractor was deployed *and every evaluator
process was restarted*.  A process that merely creates a task after a file
copy but still has the old Python module loaded is not post-deployment proof.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import MATH, MODELS, STRICT_CONFIG_ROOT
from ops.g1i_strict46.monitor_blank_recovery_replays import (
    _filter_source_candidates,
)
from ops.g1i_strict46.monitor_judge_determinism_replays import (
    TASK_QUERY,
    _append_event,
    _build_replay_command as _generic_build_replay_command,
    _eligible_source,
    _marker_replays_by_source,
    _once_exit_code,
    _parse_datetime,
    _plan_ids,
    _plan_replays,
    _replay_artifact,
    _select_latest_valid_sources,
    _split_post_candidates,
    _terminal_action,
)
from ops.g1i_strict46.replay_lock import (
    held_replay_advisory_locks,
    replay_advisory_lock_keys,
)
from src.db.pool import _build_conninfo
from src.eval.env_config import load_env_file
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
CURRENT_WAVE_STARTED_AT = datetime(2026, 8, 6, 12, 54, 0)
REASON_TAG_PREFIX = "global_complete_answer_extractor"
SOURCE_QUERY = TASK_QUERY.replace(
    "ORDER BY t.task_id",
    """
  AND t.evaluator IN ('free_response_naive', 'free_response_judge_naive')
  AND LOWER(COALESCE(t.sampling_config->>'prompt_profile', '')) = 'naive'
  AND REGEXP_REPLACE(
        LOWER(COALESCE(t.sampling_config->>'cot_mode', '')),
        '[^a-z]', '', 'g'
      ) = 'cot'
ORDER BY t.task_id
""",
)


def _reason_tag_for_sha256(extractor_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", extractor_sha256):
        raise ValueError("extractor SHA-256 must be 64 lowercase hex characters")
    return f"{REASON_TAG_PREFIX}_{extractor_sha256[:8]}_20260808"


def _split_candidates(
    rows: list[dict[str, Any]],
    *,
    deployed_at: datetime,
    now: datetime | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    """Partition canonical roots around the explicit process-restart cutoff."""

    candidates = _filter_source_candidates(rows)
    pre = [
        row
        for row in candidates
        if isinstance(row.get("task_created_at"), datetime)
        and row["task_created_at"] < deployed_at
    ]
    post = [
        row
        for row in candidates
        if isinstance(row.get("task_created_at"), datetime)
        and row["task_created_at"] >= deployed_at
    ]
    sources, invalid_sources = _select_latest_valid_sources(pre, now=now)
    resolved, pending = _split_post_candidates(post, now=now)
    return sources, resolved, pending, invalid_sources


def _fetch_rows(
    connection: psycopg.Connection[Any],
    *,
    wave_started_at: datetime,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            SOURCE_QUERY,
            (
                list(MODELS),
                wave_started_at,
                sorted({name for name, _split in MATH}),
            ),
        ).fetchall()
    ]


def _scan(
    connection: psycopg.Connection[Any],
    *,
    wave_started_at: datetime,
    deployed_at: datetime,
    reason_tag: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = _fetch_rows(connection, wave_started_at=wave_started_at)
    sources, resolved, pending, invalid = _split_candidates(
        rows,
        deployed_at=deployed_at,
    )
    exact_marker_replays = _marker_replays_by_source(rows, sources, reason_tag)
    plan = _plan_replays(
        sources,
        resolved,
        pending,
        exact_marker_replays,
        invalid,
        deployed_at=deployed_at,
    )
    return rows, plan


def _build_replay_command(
    *,
    repo: Path,
    replay_script: Path,
    source_task_id: int,
    dbname: str,
    reason_tag: str,
    output: Path,
    judge_max_workers: int,
) -> list[str]:
    return _generic_build_replay_command(
        repo=repo,
        replay_script=replay_script,
        source_task_id=source_task_id,
        dbname=dbname,
        reason_tag=reason_tag,
        output=output,
        judge_max_workers=judge_max_workers,
    )


def _subprocess_result(
    *,
    completed: subprocess.CompletedProcess[str],
    output: Path,
    source_task_id: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Preserve a fail-closed replay artifact even on a non-zero exit."""

    replay, artifact_error = _replay_artifact(output, source_task_id)
    if completed.returncode != 0:
        detail = artifact_error or "replay_subprocess_reported_failure"
        return replay, f"subprocess_returncode:{completed.returncode};{detail}"
    return replay, artifact_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument("--wave-started-at", type=_parse_datetime, default=CURRENT_WAVE_STARTED_AT)
    parser.add_argument(
        "--deployed-at",
        type=_parse_datetime,
        required=True,
        help="first DB time after extractor deployment and evaluator-process restart",
    )
    parser.add_argument(
        "--extractor-sha256",
        required=True,
        help="SHA-256 of the exact deployed candidate free_response.py",
    )
    parser.add_argument(
        "--reason-tag",
        help="defaults to a marker containing the deployed extractor SHA prefix",
    )
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/g1i_complete_answer_extractor_replay_events.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/audits/g1i_complete_answer_extractor_replays"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("logs/audits/g1i_complete_answer_extractor_replay_monitor.lock"),
    )
    args = parser.parse_args()
    try:
        default_reason_tag = _reason_tag_for_sha256(args.extractor_sha256)
    except ValueError as error:
        parser.error(str(error))
    if args.reason_tag is None:
        args.reason_tag = default_reason_tag
    if args.interval_s < 1:
        parser.error("--interval-s must be at least 1")
    if args.judge_max_workers < 1:
        parser.error("--judge-max-workers must be at least 1")
    if args.wave_started_at >= args.deployed_at:
        parser.error("--wave-started-at must be earlier than --deployed-at")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.reason_tag):
        parser.error(
            "--reason-tag may contain only letters, digits, dot, underscore, and dash"
        )
    if args.extractor_sha256[:8] not in args.reason_tag:
        parser.error(
            "--reason-tag must contain the deployed extractor SHA-256 prefix "
            f"{args.extractor_sha256[:8]}"
        )

    repo = Path(__file__).resolve().parents[2]
    load_env_file(repo / ".env")
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    conninfo = _build_conninfo(replace(DEFAULT_DB_CONFIG, dbname=args.dbname))
    replay_env = dict(os.environ)
    replay_env["RWKV_BENCHMARK_CONFIG_ROOT"] = str(STRICT_CONFIG_ROOT)
    last_state: dict[str, list[int]] | None = None
    locally_blocked: dict[int, str] = {}

    with args.lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _append_event(
            args.events,
            {
                "event": "complete_answer_extractor_monitor_started",
                "observed_at": datetime.now().astimezone(),
                "database": args.dbname,
                "wave_started_at": args.wave_started_at,
                "deployed_at": args.deployed_at,
                "extractor_sha256": args.extractor_sha256,
                "reason_tag": args.reason_tag,
            },
        )
        while True:
            with psycopg.connect(conninfo, row_factory=dict_row) as connection:
                rows, plan = _scan(
                    connection,
                    wave_started_at=args.wave_started_at,
                    deployed_at=args.deployed_at,
                    reason_tag=args.reason_tag,
                )

            state = _plan_ids(plan)
            if state != last_state:
                _append_event(
                    args.events,
                    {
                        "event": "complete_answer_extractor_monitor_state",
                        "observed_at": datetime.now().astimezone(),
                        "candidate_task_count": len(rows),
                        "task_ids_by_state": state,
                    },
                )
                last_state = state

            eligible_ids = {
                int(row["task_id"]) for row in plan["eligible_to_replay"]
            }
            locally_blocked = {
                task_id: reason
                for task_id, reason in locally_blocked.items()
                if task_id in eligible_ids
            }
            replay_failed = False
            replay_succeeded = False
            replay_lock_busy = False
            state_changed_under_lock = False
            for source in plan["eligible_to_replay"]:
                source_task_id = int(source["task_id"])
                if source_task_id in locally_blocked:
                    continue
                lock_keys = replay_advisory_lock_keys(
                    dbname=args.dbname,
                    source_task_id=source_task_id,
                    model_name=str(source["model_name"]),
                    benchmark_name=str(source["benchmark_name"]),
                    benchmark_split=str(source["benchmark_split"]),
                )
                with psycopg.connect(
                    conninfo,
                    row_factory=dict_row,
                    autocommit=True,
                ) as lock_connection:
                    with held_replay_advisory_locks(
                        lock_connection, lock_keys
                    ) as acquired:
                        if not acquired:
                            replay_lock_busy = True
                            _append_event(
                                args.events,
                                {
                                    "event": "complete_answer_extractor_replay_lock_busy",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "lock_keys": lock_keys,
                                },
                            )
                            continue

                        # The source and logical-cell locks stay held through the
                        # subprocess.  Re-scan only after both locks are owned.
                        _fresh_rows, fresh_plan = _scan(
                            lock_connection,
                            wave_started_at=args.wave_started_at,
                            deployed_at=args.deployed_at,
                            reason_tag=args.reason_tag,
                        )
                        fresh_source = _eligible_source(fresh_plan, source_task_id)
                        if fresh_source is None:
                            state_changed_under_lock = True
                            _append_event(
                                args.events,
                                {
                                    "event": "complete_answer_extractor_replay_cancelled",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "reason": "database_state_reconciled_under_lock",
                                    "task_ids_by_state": _plan_ids(fresh_plan),
                                },
                            )
                            continue

                        attempt_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                        output = args.output_dir / (
                            f"source_{source_task_id}_{args.reason_tag}_{attempt_id}.json"
                        )
                        command = _build_replay_command(
                            repo=repo,
                            replay_script=replay_script,
                            source_task_id=source_task_id,
                            dbname=args.dbname,
                            reason_tag=args.reason_tag,
                            output=output,
                            judge_max_workers=args.judge_max_workers,
                        )
                        started_at = datetime.now().astimezone()
                        try:
                            completed = subprocess.run(
                                command,
                                cwd=repo,
                                env=replay_env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False,
                            )
                        except OSError as error:
                            returncode: int | None = None
                            stdout_tail = ""
                            stderr_tail = repr(error)
                            replay = None
                            failure_reason = (
                                "subprocess_launch_error:"
                                f"{type(error).__name__}:{error}"
                            )
                        else:
                            returncode = completed.returncode
                            stdout_tail = completed.stdout[-4000:]
                            stderr_tail = completed.stderr[-4000:]
                            replay, failure_reason = _subprocess_result(
                                completed=completed,
                                output=output,
                                source_task_id=source_task_id,
                            )
                        event: dict[str, Any] = {
                            "event": (
                                "complete_answer_extractor_replay_completed"
                                if failure_reason is None
                                else "complete_answer_extractor_replay_failed_blocked"
                            ),
                            "observed_at": datetime.now().astimezone(),
                            "started_at": started_at,
                            "source": fresh_source,
                            "returncode": returncode,
                            "failure_reason": failure_reason,
                            "command": command,
                            "output": str(output),
                            "stdout_tail": stdout_tail,
                            "stderr_tail": stderr_tail,
                        }
                        if replay is not None:
                            event["replay"] = replay
                        _append_event(args.events, event)
                        if failure_reason is None:
                            replay_succeeded = True
                        else:
                            replay_failed = True
                            locally_blocked[source_task_id] = failure_reason

            if args.once:
                return _once_exit_code(replay_failed=replay_failed, plan=plan)
            if replay_succeeded or state_changed_under_lock:
                continue
            if replay_lock_busy:
                time.sleep(args.interval_s)
                continue
            action = _terminal_action(plan, set(locally_blocked))
            if action in {"blocked", "complete"}:
                _append_event(
                    args.events,
                    {
                        "event": f"complete_answer_extractor_monitor_{action}",
                        "observed_at": datetime.now().astimezone(),
                        "task_ids_by_state": state,
                        "locally_blocked": locally_blocked,
                    },
                )
                return 2 if action == "blocked" else 0
            time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
