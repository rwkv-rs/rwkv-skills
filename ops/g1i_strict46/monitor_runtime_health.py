#!/usr/bin/env python3
"""Read-only runtime health monitor for the G1i strict-46 campaign.

The monitor records endpoint mismatches, newly failed target tasks, and
running tasks whose committed completion count has not advanced within a
conservative timeout.  It never changes database rows or restarts services;
the existing systemd restart and handoff units remain the recovery authority.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import TARGETS
from ops.g1i_strict46.monitor_new_scores import (
    DB_NAME,
    MODELS,
    _append_event,
    _write_json_atomic,
)
from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


MODEL_ENDPOINTS = {
    "rwkv7-g1i-1.5b-20260805-ctx16384": "http://127.0.0.1:19439/v1",
    "rwkv7-g1i-2.9b-20260805-ctx16384": "http://127.0.0.1:19439/v1",
    "rwkv7-g1i-7.2b-20260805-ctx16384": "http://127.0.0.1:29574/v1",
    "rwkv7-g1i-13.3b-20260805-ctx16384": "http://127.0.0.1:29574/v1",
}


def _local_strict46_schedulers(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    """Return local strict-46 scheduler processes and their kernel state.

    A scheduler stopped with SIGSTOP remains present and keeps its systemd unit
    ``active``.  Endpoint and database probes therefore cannot distinguish it
    from a healthy dispatcher.  Reading procfs gives the authoritative Linux
    process state without mutating the service.
    """

    processes: list[dict[str, Any]] = []
    for process_dir in proc_root.glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
            command = (process_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            stat = (process_dir / "stat").read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            continue
        if "src.eval.scheduler.cli dispatch" not in command or "g1i_strict46" not in command:
            continue
        closing_paren = stat.rfind(")")
        fields = stat[closing_paren + 2 :].split() if closing_paren >= 0 else []
        state = fields[0] if fields else "?"
        processes.append({"pid": pid, "state": state, "command": command})
    return sorted(processes, key=lambda row: int(row["pid"]))


def _scheduler_process_issues(
    processes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    issues: dict[str, dict[str, Any]] = {}
    for process in processes:
        state = str(process.get("state") or "?")
        if not state.lower().startswith("t"):
            continue
        pid = int(process["pid"])
        issues[f"stopped_scheduler:{pid}"] = {
            "kind": "stopped_scheduler",
            "pid": pid,
            "process_state": state,
            "command": process.get("command"),
        }
    return issues

TASK_QUERY = """
SELECT
    t.task_id,
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    t.status,
    t.created_at AS task_created_at,
    COALESCE((t.sampling_config->>'effective_sample_count')::integer, 0)
        AS expected,
    COUNT(c.completions_id) AS completions,
    MAX(c.created_at) AS latest_completion_at
FROM task t
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN completions c ON c.task_id = t.task_id
WHERE m.model_name = ANY(%s)
  AND b.benchmark_name = ANY(%s)
  AND t.task_id >= %s
  AND lower(t.status) IN ('running', 'failed')
GROUP BY
    t.task_id, m.model_name, b.benchmark_name, b.benchmark_split,
    t.status, t.created_at, t.sampling_config
ORDER BY t.task_id
"""


def _as_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _probe_model(endpoint: str, model: str, timeout_s: float) -> dict[str, Any]:
    request = Request(
        endpoint.rstrip("/") + "/models",
        headers={"Authorization": "Bearer rwkv-skills"},
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    served = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    return {
        "ok": model in served,
        "served_models": sorted(served),
        "error": None if model in served else "requested model is not served",
    }


def _health_issues(
    rows: list[dict[str, Any]],
    probes: dict[str, dict[str, Any]],
    *,
    now: datetime,
    stall_after: timedelta,
) -> dict[str, dict[str, Any]]:
    issues: dict[str, dict[str, Any]] = {}
    running_models: set[str] = set()
    for row in rows:
        task_id = int(row["task_id"])
        status = str(row.get("status") or "").lower()
        if status == "failed":
            issues[f"failed_task:{task_id}"] = {
                "kind": "failed_task",
                "task_id": task_id,
                "model_name": row["model_name"],
                "benchmark": f"{row['benchmark_name']}__{row['benchmark_split']}",
            }
            continue
        if status != "running":
            continue
        model = str(row["model_name"])
        running_models.add(model)
        expected = int(row.get("expected") or 0)
        completions = int(row.get("completions") or 0)
        last = _as_aware(row.get("latest_completion_at"))
        created = _as_aware(row.get("task_created_at"))
        activity_at = last or created
        if (
            expected > completions
            and activity_at is not None
            and now - activity_at > stall_after
        ):
            issues[f"stalled_task:{task_id}"] = {
                "kind": "stalled_task",
                "task_id": task_id,
                "model_name": model,
                "benchmark": f"{row['benchmark_name']}__{row['benchmark_split']}",
                "completions": completions,
                "expected": expected,
                "last_activity_at": activity_at,
                "idle_seconds": (now - activity_at).total_seconds(),
            }
    for model in sorted(running_models):
        probe = probes.get(model, {"ok": False, "error": "probe missing"})
        if not probe.get("ok"):
            issues[f"endpoint:{model}"] = {
                "kind": "endpoint",
                "model_name": model,
                "endpoint": MODEL_ENDPOINTS[model],
                **probe,
            }
    return issues


def _transition_events(
    previous: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for key in sorted(current.keys() - previous.keys()):
        events.append({"event": "runtime_issue_started", "issue_key": key, **current[key]})
    for key in sorted(previous.keys() - current.keys()):
        events.append({"event": "runtime_issue_resolved", "issue_key": key, **previous[key]})
    return events


def _without_baseline_failures(
    issues: dict[str, dict[str, Any]], baseline: set[str]
) -> dict[str, dict[str, Any]]:
    return {key: value for key, value in issues.items() if key not in baseline}


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(
    connection: psycopg.Connection[Any],
    *,
    since_task_id: int,
    stall_minutes: float,
    probe: Callable[[str, str, float], dict[str, Any]] = _probe_model,
) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in connection.execute(
            TASK_QUERY,
            (list(MODELS), sorted({name for name, _split in TARGETS}), since_task_id),
        )
    ]
    running_models = {
        str(row["model_name"])
        for row in rows
        if str(row.get("status") or "").lower() == "running"
    }
    probes = {
        model: probe(MODEL_ENDPOINTS[model], model, 5.0)
        for model in sorted(running_models)
    }
    now = datetime.now().astimezone()
    issues = _health_issues(
        rows,
        probes,
        now=now,
        stall_after=timedelta(minutes=stall_minutes),
    )
    scheduler_processes = _local_strict46_schedulers()
    issues.update(_scheduler_process_issues(scheduler_processes))
    return {
        "observed_at": now,
        "rows": rows,
        "probes": probes,
        "scheduler_processes": scheduler_processes,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument("--since-task-id", type=int, default=28527)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--stall-minutes", type=float, default=75.0)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("logs/audits/g1i_runtime_health_state.json"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/g1i_runtime_health_events.jsonl"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    previous_state = _load_state(args.state)
    while True:
        with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as connection:
            current = _snapshot(
                connection,
                since_task_id=args.since_task_id,
                stall_minutes=args.stall_minutes,
            )
        previous_issues = (previous_state or {}).get("issues", {})
        baseline_failed_issue_keys = set(
            (previous_state or {}).get("baseline_failed_issue_keys", [])
        )
        # Migrate the first version of the state file, which persisted all
        # pre-monitor failures in ``issues`` but did not yet name the baseline.
        if previous_state is not None and not baseline_failed_issue_keys:
            baseline_failed_issue_keys = {
                key for key in previous_issues if key.startswith("failed_task:")
            }
        if previous_state is None:
            baseline_failed_issue_keys = {
                key for key in current["issues"] if key.startswith("failed_task:")
            }
            _append_event(
                args.events,
                {
                    "event": "runtime_monitor_initialized",
                    "observed_at": current["observed_at"],
                    "baseline_failed_issue_keys": sorted(baseline_failed_issue_keys),
                },
            )
        else:
            previous_issues = _without_baseline_failures(
                previous_issues, baseline_failed_issue_keys
            )
            current["issues"] = _without_baseline_failures(
                current["issues"], baseline_failed_issue_keys
            )
            for event in _transition_events(previous_issues, current["issues"]):
                _append_event(
                    args.events,
                    {"observed_at": current["observed_at"], **event},
                )
        current["issues"] = _without_baseline_failures(
            current["issues"], baseline_failed_issue_keys
        )
        current["baseline_failed_issue_keys"] = sorted(baseline_failed_issue_keys)
        _write_json_atomic(args.state, current)
        previous_state = current
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
