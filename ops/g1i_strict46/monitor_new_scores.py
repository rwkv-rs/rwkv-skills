#!/usr/bin/env python3
"""Refresh the read-only strict-46 audit whenever a new G1i score appears.

The monitor never changes task, completion, eval, or score rows. It persists
only a local cursor and an append-only event log under ``logs/audits``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
MODELS = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
)

LATEST_QUERY = """
SELECT s.score_id, s.created_at AS score_created_at, t.task_id
FROM scores s
JOIN task t ON t.task_id = s.task_id
JOIN model m ON m.model_id = t.model_id
WHERE m.model_name = ANY(%s)
ORDER BY s.score_id DESC
LIMIT 1
"""

NEW_QUERY = """
SELECT s.score_id, s.created_at AS score_created_at, t.task_id, m.model_name,
       b.benchmark_name, b.benchmark_split, t.status, s.metrics
FROM scores s
JOIN task t ON t.task_id = s.task_id
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
WHERE m.model_name = ANY(%s)
  AND s.score_id > %s
ORDER BY s.score_id
"""


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_cursor(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    score_id = payload.get("score_id")
    return int(score_id) if score_id is not None else None


def _save_cursor(path: Path, row: dict[str, Any]) -> None:
    _write_json_atomic(
        path,
        {
            "score_id": int(row["score_id"]),
            "score_created_at": row["score_created_at"],
            "task_id": int(row["task_id"]),
        },
    )


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n"
        )


def _row_mentions_task_id(value: Any, task_id: int) -> bool:
    """Return whether a nested audit row references ``task_id``.

    The aggregate audit uses different task-id keys for parameter-curve and
    architecture-reference comparisons.  Walking only fields whose name ends
    in ``task_id`` keeps this helper generic without matching unrelated numeric
    values such as sample counts or score ids.
    """

    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("task_id") and item == task_id:
                return True
            if isinstance(item, (dict, list)) and _row_mentions_task_id(
                item, task_id
            ):
                return True
    elif isinstance(value, list):
        return any(_row_mentions_task_id(item, task_id) for item in value)
    return False


def _build_per_score_audit(
    aggregate: dict[str, Any], score_row: dict[str, Any]
) -> dict[str, Any]:
    """Build a traceable strict-protocol snapshot for one observed score."""

    task_id = int(score_row["task_id"])
    valid = next(
        (
            row
            for row in aggregate.get("valid_task_rows", [])
            if int(row.get("task_id", -1)) == task_id
        ),
        None,
    )
    invalid = next(
        (
            row
            for row in aggregate.get("invalid_scored_tasks", [])
            if int(row.get("task_id", -1)) == task_id
        ),
        None,
    )

    if valid is not None:
        classification = "accepted"
        task_audit = valid
    elif invalid is not None:
        classification = "invalid"
        task_audit = invalid
    else:
        classification = "score_not_in_strict_audit"
        task_audit = None

    curve_rows = [
        row
        for row in aggregate.get("curve_comparisons", [])
        if _row_mentions_task_id(row, task_id)
    ]
    reference_rows = [
        row
        for row in aggregate.get("reference_comparisons", [])
        if _row_mentions_task_id(row, task_id)
    ]
    choice_bias_rows = [
        row
        for row in aggregate.get("choice_bias_signals", [])
        if _row_mentions_task_id(row, task_id)
    ]
    truncation_examples_by_task = aggregate.get("truncation_examples_by_task", {})
    if isinstance(truncation_examples_by_task, dict):
        truncation_examples = truncation_examples_by_task.get(
            str(task_id), truncation_examples_by_task.get(task_id, [])
        )
    else:
        truncation_examples = []
    if not isinstance(truncation_examples, list):
        truncation_examples = []

    investigation_signals: list[str] = []
    if classification == "invalid":
        investigation_signals.extend(
            f"invalid:{reason}" for reason in invalid.get("invalid_reasons", [])
        )
    elif classification == "score_not_in_strict_audit":
        investigation_signals.append("score_not_in_strict_audit")

    if valid is not None:
        for field in (
            "blank_primary_generation_count",
            "leading_orphan_close_count",
            "missing_prediction_count",
        ):
            count = int(valid.get(field) or 0)
            if count:
                investigation_signals.append(f"{field}:{count}")
        truncation_field = (
            "final_stage_truncation_count"
            if valid.get("domain") == "math"
            else "overall_truncation_count"
        )
        truncation = int(valid.get(truncation_field) or 0)
        if truncation:
            investigation_signals.append(f"{truncation_field}:{truncation}")
    else:
        truncation_field = None
        truncation = None

    if any(bool(row.get("investigate")) for row in curve_rows):
        investigation_signals.append("parameter_curve_investigation")
    if any(bool(row.get("investigate")) for row in reference_rows):
        investigation_signals.append("architecture_reference_investigation")
    if choice_bias_rows:
        investigation_signals.append("choice_bias_investigation")

    return {
        "generated_at": datetime.now().astimezone(),
        "database": aggregate.get("database"),
        "score": score_row,
        "strict_protocol_status": classification,
        "accepted_by_strict_audit": classification == "accepted",
        "truncation_policy": (
            {
                "scope": (
                    "math_final_stage_only"
                    if valid.get("domain") == "math"
                    else "evaluator_facing_final_output"
                ),
                "field": truncation_field,
                "count": truncation,
            }
            if valid is not None
            else None
        ),
        "task_audit": task_audit,
        "curve_comparisons": curve_rows,
        "reference_comparisons": reference_rows,
        "choice_bias_signals": choice_bias_rows,
        "truncation_examples": truncation_examples,
        "investigation_signals": investigation_signals,
    }


def _write_per_score_audits(
    aggregate_path: Path,
    output_dir: Path,
    score_rows: list[dict[str, Any]],
) -> list[Path]:
    """Persist one append-safe audit snapshot per newly observed score."""

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    written: list[Path] = []
    for row in score_rows:
        task_id = int(row["task_id"])
        path = output_dir / f"task_{task_id}_strict_audit.json"
        _write_json_atomic(path, _build_per_score_audit(aggregate, row))
        written.append(path)
    return written


def _latest_score(connection: psycopg.Connection[Any]) -> dict[str, Any]:
    row = connection.execute(LATEST_QUERY, (list(MODELS),)).fetchone()
    if row is None:
        return {"score_id": 0, "score_created_at": datetime.min, "task_id": 0}
    return dict(row)


def _new_scores(
    connection: psycopg.Connection[Any], score_id: int
) -> list[dict[str, Any]]:
    return list(
        connection.execute(
            NEW_QUERY,
            (list(MODELS), score_id),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("logs/audits/g1i_score_monitor_state.json"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/g1i_score_monitor_events.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("logs/audits/g1i_strict46_current.json"),
    )
    parser.add_argument(
        "--per-score-dir",
        type=Path,
        default=Path("logs/audits/g1i_new_scores"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    os.environ["RWKV_EVAL_SPACE_DB_DATABASE_NAME"] = args.dbname
    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    repo = Path(__file__).resolve().parents[2]
    audit_script = repo / "ops/g1i_strict46/audit_current.py"

    score_id = _load_cursor(args.state)
    with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as connection:
        if score_id is None:
            latest = _latest_score(connection)
            score_id = int(latest["score_id"])
            _save_cursor(args.state, latest)
            _append_event(
                args.events,
                {
                    "event": "initialized",
                    "observed_at": datetime.now().astimezone(),
                    "cursor": latest,
                },
            )
            if args.once:
                return 0

    while True:
        with psycopg.connect(
            _build_conninfo(config), row_factory=dict_row
        ) as connection:
            rows = _new_scores(connection, score_id)

        if rows:
            completed = subprocess.run(
                [
                    str(repo / ".venv/bin/python"),
                    str(audit_script),
                    "--output",
                    str(args.audit_output),
                ],
                cwd=repo,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            per_score_outputs: list[str] = []
            if completed.returncode == 0:
                per_score_outputs = [
                    str(path)
                    for path in _write_per_score_audits(
                        args.audit_output,
                        args.per_score_dir,
                        rows,
                    )
                ]
            _append_event(
                args.events,
                {
                    "event": "scores_observed",
                    "observed_at": datetime.now().astimezone(),
                    "scores": rows,
                    "audit_returncode": completed.returncode,
                    "audit_stderr": completed.stderr[-4000:],
                    "audit_output": str(args.audit_output),
                    "per_score_outputs": per_score_outputs,
                },
            )
            if completed.returncode == 0:
                last = rows[-1]
                score_id = int(last["score_id"])
                _save_cursor(args.state, last)

        if args.once:
            return 0
        time.sleep(max(1.0, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
