#!/usr/bin/env python3
"""Monitor G1g/G1i Core20 task evidence and refresh the full matrix audit.

The monitor is read-only with respect to evaluation tables.  It records a
local snapshot and append-only issue transitions so a recovery controller can
rerun only cells whose persisted completions, eval rows, or score are invalid.
Natural output truncation is intentionally not an error here; the Core20 audit
keeps it as a warning and applies the Math final-stage policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any
from zoneinfo import ZoneInfo

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_core20_dual import G1G_MODELS, G1I_MODELS, TARGETS
from ops.g1i_strict46.monitor_new_scores import _append_event, _write_json_atomic
from src.db.pool import _build_conninfo
from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
MODELS = G1G_MODELS + G1I_MODELS
DB_NAIVE_TIMESTAMP_TZ = ZoneInfo("Asia/Shanghai")

TASK_QUERY = """
SELECT
    t.task_id,
    t.status,
    t.evaluator,
    t.created_at AS task_created_at,
    t.sampling_config,
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    COALESCE((t.sampling_config->>'effective_sample_count')::bigint, 0)
        AS expected,
    COUNT(c.completions_id)::bigint AS completion_count,
    COUNT(c.completions_id) FILTER (WHERE c.status = 'Completed')::bigint
        AS completed_completion_count,
    COUNT(e.eval_id)::bigint AS eval_count,
    MAX(c.created_at) AS latest_completion_at,
    MAX(e.created_at) AS latest_eval_at,
    MAX(s.score_id) AS score_id,
    MAX(s.created_at) AS score_created_at,
    COUNT(c.completions_id) FILTER (
        WHERE COALESCE(
            NULLIF(BTRIM(c.context->>'direct_raw_completion'), ''),
            NULLIF(BTRIM(c.context #>> '{strategy_a,completion}'), ''),
            NULLIF(BTRIM(c.context #>> '{stages,0,completion}'), ''),
            ''
        ) = ''
    )::bigint AS blank_primary_count,
    COUNT(e.eval_id) FILTER (WHERE e.fail_reason = 'missing_prediction')::bigint
        AS missing_prediction_count,
    COUNT(c.completions_id) FILTER (WHERE c.status <> 'Completed')::bigint
        AS noncompleted_completion_count,
    COUNT(e.eval_id) FILTER (
        WHERE e.eval_id IS NOT NULL AND NULLIF(BTRIM(e.answer), '') IS NULL
    )::bigint AS blank_eval_answer_count,
    COUNT(c.completions_id) FILTER (
        WHERE COALESCE(
            NULLIF(c.context->>'direct_raw_completion', ''),
            NULLIF(c.context #>> '{strategy_a,completion}', ''),
            NULLIF(c.context #>> '{stages,0,completion}', ''),
            ''
        ) ~ '^[[:space:]]*>?</think>'
    )::bigint AS leading_orphan_close_count
FROM public.task t
JOIN public.model m ON m.model_id = t.model_id
JOIN public.benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN public.completions c ON c.task_id = t.task_id
LEFT JOIN public.eval e ON e.completions_id = c.completions_id
LEFT JOIN public.scores s ON s.task_id = t.task_id
WHERE t.is_tmp = FALSE
  AND t.is_param_search = FALSE
  AND t.task_id >= %s
  AND m.model_name = ANY(%s)
  AND (b.benchmark_name, b.benchmark_split) IN (
      SELECT * FROM unnest(%s::text[], %s::text[])
  )
GROUP BY
    t.task_id, t.status, t.evaluator, t.created_at, t.sampling_config,
    m.model_name, b.benchmark_name, b.benchmark_split
ORDER BY t.task_id
"""


def _as_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        # The scoreboard database stores created_at as timestamp without time
        # zone, but the server writes it in its local +08:00 timezone. Treating
        # it as UTC makes fresh rows appear to be in the future and suppresses
        # all generation/evaluation stall alerts.
        return value.replace(tzinfo=DB_NAIVE_TIMESTAMP_TZ)
    return value


def _seconds_since(now: datetime, value: datetime | None) -> float | None:
    aware = _as_aware(value)
    if aware is None:
        return None
    return max(0.0, (now - aware).total_seconds())


def _task_issues(
    rows: list[dict[str, Any]],
    *,
    now: datetime,
    generation_stall: timedelta,
    evaluation_stall: timedelta,
    score_stall: timedelta,
) -> dict[str, dict[str, Any]]:
    """Classify evidence failures without treating ordinary truncation as bad."""

    issues: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = int(row["task_id"])
        status = str(row.get("status") or "").lower()
        expected = int(row.get("expected") or 0)
        completions = int(row.get("completion_count") or 0)
        completed = int(row.get("completed_completion_count") or 0)
        evals = int(row.get("eval_count") or 0)
        score_id = row.get("score_id")
        common = {
            "task_id": task_id,
            "model_name": row.get("model_name"),
            "benchmark": (
                f"{row.get('benchmark_name')}__{row.get('benchmark_split')}"
            ),
            "evaluator": row.get("evaluator"),
            "expected": expected,
            "completions": completions,
            "completed_completions": completed,
            "evals": evals,
            "score_id": score_id,
        }

        if status == "failed":
            issues[f"failed_task:{task_id}"] = {"kind": "failed_task", **common}
            continue

        if status == "running":
            if expected > completions:
                activity = (
                    row.get("latest_completion_at")
                    or row.get("runtime_activity_at")
                    or row.get("task_created_at")
                )
                idle = _seconds_since(now, activity)
                if idle is not None and idle > generation_stall.total_seconds():
                    issues[f"generation_stalled:{task_id}"] = {
                        "kind": "generation_stalled",
                        "idle_seconds": idle,
                        "last_activity_at": activity,
                        **common,
                    }
            elif expected > 0 and evals < completions:
                activity = row.get("latest_eval_at") or row.get(
                    "latest_completion_at"
                )
                idle = _seconds_since(now, activity)
                if idle is not None and idle > evaluation_stall.total_seconds():
                    issues[f"evaluation_stalled:{task_id}"] = {
                        "kind": "evaluation_stalled",
                        "idle_seconds": idle,
                        "last_activity_at": activity,
                        **common,
                    }
            elif expected > 0 and evals == completions and score_id is None:
                activity = row.get("latest_eval_at") or row.get(
                    "latest_completion_at"
                )
                idle = _seconds_since(now, activity)
                if idle is not None and idle > score_stall.total_seconds():
                    issues[f"score_stalled:{task_id}"] = {
                        "kind": "score_stalled",
                        "idle_seconds": idle,
                        "last_activity_at": activity,
                        **common,
                    }

        if status == "completed" and score_id is None:
            issues[f"completed_without_score:{task_id}"] = {
                "kind": "completed_without_score",
                **common,
            }

        if score_id is None:
            continue

        evidence_reasons: list[str] = []
        if expected <= 0:
            evidence_reasons.append("missing_expected_count")
        else:
            if completions != expected:
                evidence_reasons.append(f"completions:{completions}!={expected}")
            if completed != expected:
                evidence_reasons.append(
                    f"completed_completions:{completed}!={expected}"
                )
        if evals != completions:
            evidence_reasons.append(f"evals:{evals}!=completions:{completions}")
        blank = int(row.get("blank_primary_count") or 0)
        missing = int(row.get("missing_prediction_count") or 0)
        noncompleted = int(row.get("noncompleted_completion_count") or 0)
        blank_eval_answer = int(row.get("blank_eval_answer_count") or 0)
        leading_orphan_close = int(row.get("leading_orphan_close_count") or 0)
        is_multiple_choice = str(row.get("evaluator") or "").startswith(
            "multi_choice"
        )
        if blank:
            evidence_reasons.append(f"blank_primary:{blank}")
        if is_multiple_choice and missing:
            evidence_reasons.append(f"missing_prediction:{missing}")
        if noncompleted:
            evidence_reasons.append(f"noncompleted_completions:{noncompleted}")
        if is_multiple_choice and blank_eval_answer:
            evidence_reasons.append(f"blank_eval_answers:{blank_eval_answer}")
        if leading_orphan_close:
            evidence_reasons.append(
                f"leading_orphan_close:{leading_orphan_close}"
            )
        if evidence_reasons:
            issues[f"invalid_score_evidence:{task_id}"] = {
                "kind": "invalid_score_evidence",
                "reasons": evidence_reasons,
                **common,
            }
    return issues


def _attach_runtime_activity(
    rows: list[dict[str, Any]], *, run_log_root: Path
) -> None:
    """Attach newest matching runner-log activity to uncommitted tasks.

    Long generation phases buffer results before committing completions. The
    run log is therefore the only authoritative liveness signal before the
    first database batch appears.
    """

    if not run_log_root.is_dir():
        return
    for row in rows:
        config = row.get("sampling_config") or {}
        mode = str(config.get("cot_mode") or "").strip().lower().replace("-", "_")
        suffix = "__cot" if mode in {"cot", "true", "1"} else ""
        dataset_slug = canonical_slug(
            f"{row.get('benchmark_name')}_{row.get('benchmark_split')}"
        )
        model_slug = safe_slug(str(row.get("model_name") or ""))
        filename = f"{dataset_slug}{suffix}.log"
        latest_mtime: float | None = None
        for candidate in run_log_root.glob(f"**/{model_slug}/{filename}"):
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            latest_mtime = mtime if latest_mtime is None else max(latest_mtime, mtime)
        if latest_mtime is not None:
            row["runtime_activity_at"] = datetime.fromtimestamp(
                latest_mtime, tz=DB_NAIVE_TIMESTAMP_TZ
            )


def _transition_events(
    previous: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for key in sorted(current.keys() - previous.keys()):
        events.append({"event": "core20_issue_started", "issue_key": key, **current[key]})
    for key in sorted(previous.keys() - current.keys()):
        events.append({"event": "core20_issue_resolved", "issue_key": key, **previous[key]})
    return events


def _new_score_rows(
    rows: list[dict[str, Any]], previous_score_ids: set[int]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("score_id") is not None
        and int(row["score_id"]) not in previous_score_ids
    ]


def _row_json(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in row.items()
    }


def _snapshot(
    connection: psycopg.Connection[Any],
    *,
    since_task_id: int,
    run_log_root: Path,
    generation_stall_minutes: float,
    evaluation_stall_minutes: float,
    score_stall_minutes: float,
) -> dict[str, Any]:
    names = [name for name, _split, _domain in TARGETS]
    splits = [split for _name, split, _domain in TARGETS]
    rows = [
        dict(row)
        for row in connection.execute(
            TASK_QUERY,
            (since_task_id, list(MODELS), names, splits),
        )
    ]
    _attach_runtime_activity(rows, run_log_root=run_log_root)
    now = datetime.now().astimezone()
    issues = _task_issues(
        rows,
        now=now,
        generation_stall=timedelta(minutes=generation_stall_minutes),
        evaluation_stall=timedelta(minutes=evaluation_stall_minutes),
        score_stall=timedelta(minutes=score_stall_minutes),
    )
    by_model: dict[str, dict[str, int]] = {}
    for model in MODELS:
        model_rows = [row for row in rows if row["model_name"] == model]
        statuses = Counter(str(row["status"]) for row in model_rows)
        by_model[model] = {
            "tasks": len(model_rows),
            "running": statuses.get("Running", 0),
            "completed": statuses.get("Completed", 0),
            "failed": statuses.get("Failed", 0),
            "completions": sum(int(row.get("completion_count") or 0) for row in model_rows),
            "evals": sum(int(row.get("eval_count") or 0) for row in model_rows),
            "scores": sum(row.get("score_id") is not None for row in model_rows),
        }
    return {
        "observed_at": now,
        "since_task_id": since_task_id,
        "rows": [_row_json(row) for row in rows],
        "by_model": by_model,
        "score_ids": sorted(
            int(row["score_id"]) for row in rows if row.get("score_id") is not None
        ),
        "issues": issues,
    }


def _run_matrix_audit(repo: Path, output: Path, dbname: str) -> dict[str, Any]:
    temporary = output.with_suffix(output.suffix + ".tmp")
    completed = subprocess.run(
        [
            str(repo / ".venv/bin/python"),
            str(repo / "ops/g1i_strict46/audit_core20_dual.py"),
            "--dbname",
            dbname,
            "--family",
            "all",
            "--output",
            str(temporary),
        ],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
        }
    temporary.replace(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    return {
        "returncode": 0,
        **{
            key: payload[key]
            for key in (
                "target_cells",
                "valid_cells",
                "invalid_scored_cells",
                "warning_scored_cells",
                "running_cells",
                "missing_cells",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument("--since-task-id", type=int, default=28890)
    parser.add_argument("--run-log-root", type=Path, default=Path("logs/runs"))
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--generation-stall-minutes", type=float, default=90.0)
    parser.add_argument("--evaluation-stall-minutes", type=float, default=90.0)
    parser.add_argument("--score-stall-minutes", type=float, default=30.0)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("logs/audits/core20_campaign_state.json"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/core20_campaign_events.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("logs/audits/g1g_g1i_core20_dual_current.json"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    run_log_root = (
        args.run_log_root
        if args.run_log_root.is_absolute()
        else repo / args.run_log_root
    )
    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    previous = (
        json.loads(args.state.read_text(encoding="utf-8"))
        if args.state.exists()
        else None
    )
    while True:
        with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as connection:
            current = _snapshot(
                connection,
                since_task_id=args.since_task_id,
                run_log_root=run_log_root,
                generation_stall_minutes=args.generation_stall_minutes,
                evaluation_stall_minutes=args.evaluation_stall_minutes,
                score_stall_minutes=args.score_stall_minutes,
            )
        current["matrix"] = _run_matrix_audit(repo, args.audit_output, args.dbname)

        if previous is None:
            _append_event(
                args.events,
                {
                    "event": "core20_monitor_initialized",
                    "observed_at": current["observed_at"],
                    "matrix": current["matrix"],
                    "issues": current["issues"],
                },
            )
        else:
            for event in _transition_events(
                previous.get("issues", {}), current["issues"]
            ):
                _append_event(
                    args.events,
                    {"observed_at": current["observed_at"], **event},
                )
            prior_scores = {int(value) for value in previous.get("score_ids", [])}
            new_scores = _new_score_rows(
                [dict(row) for row in current["rows"]], prior_scores
            )
            if new_scores:
                _append_event(
                    args.events,
                    {
                        "event": "core20_scores_observed",
                        "observed_at": current["observed_at"],
                        "scores": new_scores,
                        "matrix": current["matrix"],
                    },
                )

        _write_json_atomic(args.state, current)
        previous = current
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
