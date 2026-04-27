#!/usr/bin/env python3
"""Export latest wrong eval records for a model matrix.

This script is meant to run from a local machine against an SSH-forwarded
PostgreSQL port, for example:

    ssh -i "$HOME\\.ssh\\id_server_new" -p 2333 -N -L 15432:127.0.0.1:5432 caizus@47.115.88.183

It exports one JSON file per latest formal benchmark task:

    <output>/<data_version>/<param>/<benchmark_name>.json

Each JSON file is a list of records with only:
    context, answer, ref-answer, fail-reason
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover - local dependency guard
    raise SystemExit(
        "Missing dependency: psycopg. Install it with `python -m pip install \"psycopg[binary]\"`."
    ) from exc


DEFAULT_DATA_VERSIONS = ("g1e", "g1f")
DEFAULT_PARAMS = ("1.5b", "2.9b", "7.2b", "13.3b")


LATEST_TASKS_SQL = """
WITH latest_scores AS (
    SELECT
        s.task_id,
        s.score_id,
        s.created_at AS score_created_at,
        s.is_cot,
        m.model_name,
        lower(m.arch_version) AS arch_version,
        lower(m.data_version) AS data_version,
        lower(m.num_params) AS num_params_raw,
        regexp_replace(lower(m.num_params), '[^a-z0-9]', '', 'g') AS param_norm,
        CASE
            WHEN b.benchmark_split <> ''
            THEN b.benchmark_name || '_' || b.benchmark_split
            ELSE b.benchmark_name
        END AS dataset_slug,
        regexp_replace(
            lower(
                CASE
                    WHEN b.benchmark_split <> ''
                    THEN b.benchmark_name || '_' || b.benchmark_split
                    ELSE b.benchmark_name
                END
            ),
            '_(test|eval|val)$',
            ''
        ) AS benchmark_base,
        CASE WHEN s.is_cot THEN 'cot' ELSE 'nocot' END AS eval_method,
        row_number() OVER (
            PARTITION BY
                lower(m.arch_version),
                lower(m.data_version),
                regexp_replace(lower(m.num_params), '[^a-z0-9]', '', 'g'),
                t.benchmark_id,
                s.is_cot
            ORDER BY s.created_at DESC, s.score_id DESC
        ) AS rn
    FROM scores s
    JOIN task t ON t.task_id = s.task_id
    JOIN model m ON m.model_id = t.model_id
    JOIN benchmark b ON b.benchmark_id = t.benchmark_id
    WHERE t.is_param_search IS false
      AND t.status = 'completed'
      AND lower(m.arch_version) = %(arch)s
      AND lower(m.data_version) = ANY(%(data_versions)s::text[])
      AND regexp_replace(lower(m.num_params), '[^a-z0-9]', '', 'g') = ANY(%(param_norms)s::text[])
)
SELECT
    task_id,
    score_id,
    score_created_at,
    model_name,
    arch_version,
    data_version,
    num_params_raw,
    param_norm,
    dataset_slug,
    benchmark_base,
    eval_method,
    benchmark_base || '_' || eval_method AS benchmark_display_name
FROM latest_scores
WHERE rn = 1
ORDER BY data_version, param_norm, benchmark_display_name;
"""


WRONG_ROWS_SQL = """
WITH latest_eval AS (
    SELECT
        e.*,
        row_number() OVER (
            PARTITION BY e.completions_id
            ORDER BY e.created_at DESC, e.eval_id DESC
        ) AS rn
    FROM eval e
    JOIN completions c ON c.completions_id = e.completions_id
    WHERE c.task_id = ANY(%(task_ids)s::int[])
)
SELECT
    c.task_id,
    c.sample_index,
    c.repeat_index,
    c.context,
    e.answer,
    e.ref_answer,
    e.fail_reason
FROM completions c
JOIN latest_eval e ON e.completions_id = c.completions_id AND e.rn = 1
WHERE c.task_id = ANY(%(task_ids)s::int[])
  AND c.status = 'answer'
  AND e.is_passed IS false
ORDER BY c.task_id, c.sample_index, c.repeat_index;
"""


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._-") or "unknown"


def _param_label_map(params: Sequence[str]) -> dict[str, str]:
    return {_norm_token(item): item.lower().replace("_", ".") for item in params}


def _parse_benchmark_filter(raw_items: Sequence[str]) -> set[str]:
    return {item.strip().lower() for item in raw_items if item.strip()}


def _connect(config: DbConfig) -> psycopg.Connection[Any]:
    return psycopg.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        dbname=config.dbname,
        row_factory=dict_row,
    )


def _task_matches_benchmark_filter(task: dict[str, Any], filters: set[str]) -> bool:
    if not filters:
        return True
    candidates = {
        str(task.get("benchmark_display_name") or "").lower(),
        str(task.get("benchmark_base") or "").lower(),
        str(task.get("dataset_slug") or "").lower(),
    }
    return bool(candidates & filters)


def _record_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "context": row.get("context"),
        "answer": str(row.get("answer") or ""),
        "ref-answer": str(row.get("ref_answer") or ""),
        "fail-reason": str(row.get("fail_reason") or ""),
    }


def export_wrong_records(
    *,
    db: DbConfig,
    output_dir: Path,
    arch: str,
    data_versions: Sequence[str],
    params: Sequence[str],
    benchmarks: Sequence[str],
    write_empty: bool,
) -> dict[str, Any]:
    param_labels = _param_label_map(params)
    param_norms = sorted(param_labels)
    benchmark_filters = _parse_benchmark_filter(benchmarks)

    with _connect(db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                LATEST_TASKS_SQL,
                {
                    "arch": arch.lower(),
                    "data_versions": [item.lower() for item in data_versions],
                    "param_norms": param_norms,
                },
            )
            tasks = [
                dict(row)
                for row in cur.fetchall()
                if _task_matches_benchmark_filter(dict(row), benchmark_filters)
            ]

            task_ids = [int(task["task_id"]) for task in tasks]
            wrong_by_task: dict[int, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
            if task_ids:
                cur.execute(WRONG_ROWS_SQL, {"task_ids": task_ids})
                for row in cur.fetchall():
                    item = dict(row)
                    wrong_by_task.setdefault(int(item["task_id"]), []).append(_record_payload(item))

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    skipped_empty = 0
    for task in tasks:
        task_id = int(task["task_id"])
        records = wrong_by_task.get(task_id, [])
        if not records and not write_empty:
            skipped_empty += 1
            continue

        data_version = str(task.get("data_version") or "unknown").lower()
        param_norm = str(task.get("param_norm") or "")
        param_label = param_labels.get(param_norm, str(task.get("num_params_raw") or "unknown"))
        benchmark_name = str(task.get("benchmark_display_name") or task.get("benchmark_base") or "unknown")
        output_path = output_dir / _safe_name(data_version) / _safe_name(param_label) / f"{_safe_name(benchmark_name)}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        exported.append(
            {
                "path": str(output_path),
                "task_id": task_id,
                "model_name": task.get("model_name"),
                "benchmark": benchmark_name,
                "wrong_count": len(records),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db": {
            "host": db.host,
            "port": db.port,
            "user": db.user,
            "dbname": db.dbname,
        },
        "arch": arch,
        "data_versions": list(data_versions),
        "params": list(params),
        "benchmarks_filter": list(benchmarks),
        "latest_tasks_found": len(tasks),
        "files_written": len(exported),
        "empty_tasks_skipped": skipped_empty,
        "exports": exported,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export latest wrong eval records from PostgreSQL.")
    parser.add_argument("--env-file", default=".env", help="Optional .env file with PG_* values.")
    parser.add_argument("--pg-host", default=None, help="PostgreSQL host. Default: PG_HOST or 127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=None, help="PostgreSQL port. Default: PG_PORT or 15432")
    parser.add_argument("--pg-user", default=None, help="PostgreSQL user. Default: PG_USER or postgres")
    parser.add_argument("--pg-password", default=None, help="PostgreSQL password. Default: PG_PASSWORD")
    parser.add_argument("--pg-dbname", default=None, help="PostgreSQL database. Default: PG_DBNAME or rwkv-eval")
    parser.add_argument("--arch", default="rwkv7", help="Exact arch_version to export.")
    parser.add_argument("--data-version", action="append", dest="data_versions", help="Data version, repeatable.")
    parser.add_argument("--param", action="append", dest="params", help="Parameter size, repeatable.")
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Optional frontend benchmark name filter, e.g. math_500 or math_500_cot. Repeatable.",
    )
    parser.add_argument("--output-dir", default="wrong_records_export", help="Output directory.")
    parser.add_argument("--write-empty", action="store_true", help="Also write [] for benchmarks with zero wrong rows.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _load_env_file(Path(args.env_file).expanduser())

    db = DbConfig(
        host=args.pg_host or os.environ.get("PG_HOST", "127.0.0.1"),
        port=args.pg_port or int(os.environ.get("PG_PORT", "15432")),
        user=args.pg_user or os.environ.get("PG_USER", "postgres"),
        password=args.pg_password if args.pg_password is not None else os.environ.get("PG_PASSWORD", ""),
        dbname=args.pg_dbname or os.environ.get("PG_DBNAME", "rwkv-eval"),
    )
    data_versions = tuple(args.data_versions or DEFAULT_DATA_VERSIONS)
    params = tuple(args.params or DEFAULT_PARAMS)

    try:
        manifest = export_wrong_records(
            db=db,
            output_dir=Path(args.output_dir).expanduser(),
            arch=args.arch,
            data_versions=data_versions,
            params=params,
            benchmarks=tuple(args.benchmark),
            write_empty=bool(args.write_empty),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"export failed: {exc}", file=sys.stderr)
        return 1

    print(
        "export done: "
        f"latest_tasks={manifest['latest_tasks_found']} "
        f"files={manifest['files_written']} "
        f"output={Path(args.output_dir).expanduser()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
