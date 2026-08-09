#!/usr/bin/env python3
"""Replay pre-fix G1i Math tasks that contain a blank recovery stage.

The monitor is intentionally narrow and idempotent:

* only strict-46 G1i Math/CoT Naive source tasks from the current wave and
  created before the fail-closed evaluator deployment are considered;
* a source is replayed only after it is ``Completed`` and has a persisted
  score, and only when the exact raw-completion aggregate used by
  :mod:`ops.g1i_strict46.audit_current` reports a blank/sentinel stage 2;
* replay tasks are append-only and are deduplicated by the exact
  ``replay_source_task_id=<id>;<reason>`` provenance marker.

The source task, completions, eval rows, and score are never changed.  The
normal monitor exits once every pre-cutoff source is settled and every
eligible source has a scored replay (or a pre-existing replay row that needs
operator attention).  ``--once`` performs one reconciliation pass.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import (
    BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT,
    BLANK_RECOVERY_STAGE_SQL_PREDICATE,
    MATH,
    MODELS,
    STRICT_CONFIG_ROOT,
    canonical_target_benchmark,
)
from ops.g1i_strict46.monitor_judge_determinism_replays import (
    TASK_QUERY,
    _cell,
    _classify_existing_replays,
    _marker_replays_by_source,
    _replay_artifact,
    _select_latest_valid_sources,
    _split_post_candidates,
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
REASON_TAG = "blank_recovery_fail_closed_20260807"
TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "canceled", "stopped"})

BLANK_RECOVERY_COUNTS_QUERY = f"""
SELECT
    c.task_id,
    COUNT(*) FILTER (
        WHERE {BLANK_RECOVERY_STAGE_SQL_PREDICATE}
    ) AS blank_recovery_stage_count
FROM completions c
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
ORDER BY c.task_id
"""

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


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _provenance_marker(source_task_id: int, reason_tag: str) -> str:
    return f"replay_source_task_id={source_task_id};{reason_tag}"


def _filter_source_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the strict-46 Math/CoT Naive source lane."""

    math_targets = set(MATH)
    candidates: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if str(row.get("evaluator") or "") not in {
            "free_response_naive",
            "free_response_judge_naive",
        }:
            continue
        benchmark = canonical_target_benchmark(
            str(row.get("benchmark_name") or ""),
            str(row.get("benchmark_split") or ""),
        )
        if benchmark not in math_targets:
            continue
        sampling_config = _json_object(row.get("sampling_config"))
        if str(sampling_config.get("prompt_profile") or "").lower() != "naive":
            continue
        configured_mode = re.sub(
            r"[^a-z]", "", str(sampling_config.get("cot_mode") or "").lower()
        )
        if configured_mode != "cot":
            continue
        # A running source has no score mode yet.  Once scored, only the
        # strict-46 Math CoT lane is eligible for replay.
        cot_mode = row.get("cot_mode")
        if cot_mode is not None and str(cot_mode).lower() != "cot":
            continue
        row["benchmark_name"], row["benchmark_split"] = benchmark
        candidates.append(row)
    return candidates


def _source_candidates(
    connection: psycopg.Connection[Any],
    *,
    wave_started_at: datetime,
    deployed_at: datetime,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        SOURCE_QUERY,
        (
            list(MODELS),
            wave_started_at,
            sorted({name for name, _split in MATH}),
        ),
    ).fetchall()
    candidates = _filter_source_candidates(
        [
            dict(row)
            for row in rows
            if isinstance(row.get("task_created_at"), datetime)
            and row["task_created_at"] < deployed_at
        ]
    )
    selected, _invalid = _select_latest_valid_sources(candidates)
    return list(selected.values())


def _blank_recovery_counts(
    connection: psycopg.Connection[Any], task_ids: list[int]
) -> dict[int, int]:
    """Use the auditor's exact raw-content aggregate and sentinel rules."""

    if not task_ids:
        return {}
    rows = connection.execute(
        BLANK_RECOVERY_COUNTS_QUERY,
        (task_ids,),
    ).fetchall()
    return {
        int(row["task_id"]): int(row["blank_recovery_stage_count"] or 0) for row in rows
    }


def _source_is_settled(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").lower()
    if status in TERMINAL_FAILURE_STATUSES:
        return True
    # Wait through the small persistence interval between task completion and
    # score insertion; otherwise a one-shot terminal scan could miss a replay.
    return status == "completed" and row.get("score_id") is not None


def _replay_is_scored(row: dict[str, Any]) -> bool:
    return (
        str(row.get("replay_status") or "").lower() == "completed"
        and row.get("replay_score_id") is not None
    )


def _replay_is_pending(row: dict[str, Any]) -> bool:
    status = str(row.get("replay_status") or "").lower()
    return status not in TERMINAL_FAILURE_STATUSES and status != "completed"


def _plan_replays(
    sources: list[dict[str, Any]],
    blank_counts: dict[int, int],
    existing_replays: dict[int, list[dict[str, Any]]],
    resolved_by_cell: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    pending_post_by_cell: dict[tuple[str, str, str], dict[str, Any]] | None = None,
    invalid_sources: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Pure reconciliation plan used by the monitor and unit tests."""

    plan: dict[str, list[dict[str, Any]]] = {
        "pending_sources": [],
        "terminal_failed_sources": [],
        "eligible_to_replay": [],
        "already_replayed": [],
        "pending_existing_replay": [],
        "blocked_existing_replay": [],
        "not_affected": [],
        "ignored_invalid_sources": list(invalid_sources or []),
        "blocked_invalid_source_cells": [],
    }
    selected_cells = {_cell(row) for row in sources}
    unresolved_invalid_cells: set[tuple[str, str, str]] = set()
    for invalid in invalid_sources or []:
        invalid_cell = _cell(invalid)
        if invalid_cell in selected_cells or invalid_cell in unresolved_invalid_cells:
            continue
        unresolved_invalid_cells.add(invalid_cell)
        plan["blocked_invalid_source_cells"].append(invalid)
    resolved_by_cell = resolved_by_cell or {}
    pending_post_by_cell = pending_post_by_cell or {}
    for source in sources:
        row = dict(source)
        task_id = int(row["task_id"])
        row["blank_recovery_stage_count"] = int(blank_counts.get(task_id, 0))
        status = str(row.get("status") or "").lower()
        if not _source_is_settled(row):
            plan["pending_sources"].append(row)
            continue
        if status in TERMINAL_FAILURE_STATUSES:
            plan["terminal_failed_sources"].append(row)
            continue
        if not row["blank_recovery_stage_count"]:
            plan["not_affected"].append(row)
            continue

        resolved = resolved_by_cell.get(_cell(row))
        if resolved is not None:
            row["post_cutoff_task_id"] = int(resolved["task_id"])
            plan["already_replayed"].append(row)
            continue
        pending_post = pending_post_by_cell.get(_cell(row))
        if pending_post is not None:
            row["post_cutoff_task_id"] = int(pending_post["task_id"])
            plan["pending_existing_replay"].append(row)
            continue

        replay_state, replay = _classify_existing_replays(
            existing_replays.get(task_id, []),
            deployed_at=BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT,
        )
        if replay_state == "missing":
            plan["eligible_to_replay"].append(row)
            continue
        annotated = {**row, "existing_replay": replay}
        if replay_state == "valid":
            plan["already_replayed"].append(annotated)
        elif replay_state == "pending":
            # Another process already owns this exact replay.  Wait for its
            # terminal state instead of creating a duplicate or failing early.
            plan["pending_existing_replay"].append(annotated)
        else:
            # Never create a second marker-identical replay.  A Failed replay
            # or Completed replay without a score needs operator attention.
            plan["blocked_existing_replay"].append(annotated)
    return plan


def _blocking_plan_entries(
    plan: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return terminal states that cannot become valid by merely waiting."""

    return (
        plan.get("terminal_failed_sources", [])
        + plan.get("blocked_existing_replay", [])
        + plan.get("blocked_invalid_source_cells", [])
    )


def _once_exit_code(
    *,
    replay_failed: bool,
    replay_lock_busy: bool,
    plan: dict[str, list[dict[str, Any]]],
) -> int:
    if replay_failed:
        return 1
    return 2 if replay_lock_busy or _blocking_plan_entries(plan) else 0


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
    return [
        str(repo / ".venv/bin/python"),
        str(replay_script),
        str(source_task_id),
        "--dbname",
        dbname,
        "--judge-mode",
        "auto",
        "--judge-max-workers",
        str(judge_max_workers),
        "--reason-tag",
        reason_tag,
        "--commit",
        "--summary",
        "--output",
        str(output),
    ]


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _plan_summary(plan: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {key: len(value) for key, value in plan.items()}


def _plan_task_ids(
    plan: dict[str, list[dict[str, Any]]],
) -> dict[str, list[int]]:
    return {key: [int(row["task_id"]) for row in rows] for key, rows in plan.items()}


def _eligible_source(
    plan: dict[str, list[dict[str, Any]]], source_task_id: int
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in plan["eligible_to_replay"]
            if int(row["task_id"]) == int(source_task_id)
        ),
        None,
    )


def _scan(
    connection: psycopg.Connection[Any],
    *,
    wave_started_at: datetime,
    deployed_at: datetime,
    reason_tag: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_rows = [
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
    candidates = _filter_source_candidates(all_rows)
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
    sources_by_cell, invalid_sources = _select_latest_valid_sources(pre)
    sources = list(sources_by_cell.values())
    resolved_by_cell, pending_post_by_cell = _split_post_candidates(post)
    # Only a completed source with a persisted score can be replayed.  Do not
    # make the monitor repeatedly JSON-scan large in-flight completion sets;
    # terminal failures and running sources are still tracked by the source
    # plan, and the exact raw predicate is applied as soon as a score exists.
    task_ids = [
        int(row["task_id"])
        for row in sources
        if str(row.get("status") or "").lower() == "completed"
        and row.get("score_id") is not None
    ]
    blank_counts = _blank_recovery_counts(connection, task_ids)
    existing_replays = _marker_replays_by_source(
        all_rows, sources_by_cell, reason_tag
    )
    return sources, _plan_replays(
        sources,
        blank_counts,
        existing_replays,
        resolved_by_cell,
        pending_post_by_cell,
        invalid_sources,
    )


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if parsed.tzinfo is not None:
        raise argparse.ArgumentTypeError(
            "timestamps must be timezone-naive database time"
        )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument(
        "--wave-started-at",
        type=_parse_datetime,
        default=CURRENT_WAVE_STARTED_AT,
    )
    parser.add_argument(
        "--deployed-at",
        type=_parse_datetime,
        default=BLANK_RECOVERY_PROTOCOL_DEPLOYED_AT,
    )
    parser.add_argument("--reason-tag", default=REASON_TAG)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/g1i_blank_recovery_replay_events.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/audits/g1i_blank_recovery_replays"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("logs/audits/g1i_blank_recovery_replay_monitor.lock"),
    )
    args = parser.parse_args()
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

    repo = Path(__file__).resolve().parents[2]
    load_env_file(repo / ".env")
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    conninfo = _build_conninfo(config)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    last_state: dict[str, list[int]] | None = None
    replay_env = dict(os.environ)
    replay_env["RWKV_BENCHMARK_CONFIG_ROOT"] = str(STRICT_CONFIG_ROOT)

    with args.lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _append_event(
            args.events,
            {
                "event": "blank_recovery_monitor_started",
                "observed_at": datetime.now().astimezone(),
                "database": args.dbname,
                "wave_started_at": args.wave_started_at,
                "deployed_at": args.deployed_at,
                "reason_tag": args.reason_tag,
            },
        )
        while True:
            with psycopg.connect(
                conninfo, row_factory=dict_row
            ) as connection:
                sources, plan = _scan(
                    connection,
                    wave_started_at=args.wave_started_at,
                    deployed_at=args.deployed_at,
                    reason_tag=args.reason_tag,
                )

            summary = _plan_summary(plan)
            task_ids_by_state = _plan_task_ids(plan)
            if task_ids_by_state != last_state:
                _append_event(
                    args.events,
                    {
                        "event": "blank_recovery_monitor_state",
                        "observed_at": datetime.now().astimezone(),
                        "source_count": len(sources),
                        "source_task_ids_by_state": task_ids_by_state,
                        **summary,
                    },
                )
                last_state = task_ids_by_state

            replay_failed = False
            replay_lock_busy = False
            for source in plan["eligible_to_replay"]:
                source_task_id = int(source["task_id"])
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
                                    "event": "blank_recovery_replay_lock_busy",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "lock_keys": lock_keys,
                                },
                            )
                            continue
                        _fresh_sources, fresh_plan = _scan(
                            lock_connection,
                            wave_started_at=args.wave_started_at,
                            deployed_at=args.deployed_at,
                            reason_tag=args.reason_tag,
                        )
                        fresh_source = _eligible_source(
                            fresh_plan, source_task_id
                        )
                        if fresh_source is None:
                            _append_event(
                                args.events,
                                {
                                    "event": "blank_recovery_replay_cancelled",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "reason": "database_state_reconciled_under_lock",
                                },
                            )
                            continue
                        args.output_dir.mkdir(parents=True, exist_ok=True)
                        attempt_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                        output = args.output_dir / (
                            f"source_{source_task_id}_{args.reason_tag}_"
                            f"{attempt_id}.json"
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
                            replay, artifact_error = (
                                _replay_artifact(output, source_task_id)
                                if completed.returncode == 0
                                else (None, None)
                            )
                            failure_reason = (
                                f"subprocess_returncode:{completed.returncode}"
                                if completed.returncode != 0
                                else artifact_error
                            )
                        event: dict[str, Any] = {
                            "event": (
                                "blank_recovery_replay_completed"
                                if failure_reason is None
                                else "blank_recovery_replay_failed"
                            ),
                            "observed_at": datetime.now().astimezone(),
                            "started_at": started_at,
                            "source": fresh_source,
                            "reason_tag": args.reason_tag,
                            "returncode": returncode,
                            "failure_reason": failure_reason,
                            "stdout_tail": stdout_tail,
                            "stderr_tail": stderr_tail,
                            "output": str(output),
                        }
                        if replay is not None:
                            event["replay"] = replay
                        _append_event(args.events, event)
                        replay_failed = replay_failed or failure_reason is not None

            if args.once:
                return _once_exit_code(
                    replay_failed=replay_failed,
                    replay_lock_busy=replay_lock_busy,
                    plan=plan,
                )

            if replay_lock_busy:
                time.sleep(args.interval_s)
                continue

            if not plan["pending_sources"] and not plan["pending_existing_replay"]:
                # Re-read marker rows after subprocesses finish.  This is the
                # idempotency/termination gate, not trust in process exit code.
                with psycopg.connect(
                    conninfo, row_factory=dict_row
                ) as connection:
                    final_sources, final_plan = _scan(
                        connection,
                        wave_started_at=args.wave_started_at,
                        deployed_at=args.deployed_at,
                        reason_tag=args.reason_tag,
                    )
                still_waiting = bool(
                    final_plan["pending_sources"]
                    or final_plan["pending_existing_replay"]
                )
                if still_waiting:
                    time.sleep(args.interval_s)
                    continue
                incomplete = bool(
                    final_plan["eligible_to_replay"]
                    or _blocking_plan_entries(final_plan)
                )
                _append_event(
                    args.events,
                    {
                        "event": (
                            "blank_recovery_monitor_incomplete"
                            if incomplete
                            else "blank_recovery_monitor_complete"
                        ),
                        "observed_at": datetime.now().astimezone(),
                        "source_count": len(final_sources),
                        "source_task_ids_by_state": _plan_task_ids(final_plan),
                        **_plan_summary(final_plan),
                    },
                )
                return 2 if incomplete else 0

            time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
