#!/usr/bin/env python3
"""Export wrong MMLU-Pro rows for one model through a forwarded PostgreSQL port."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: psycopg. Run through this repo's uv environment.") from exc


LATEST_TASK_SQL = """
WITH latest_scores AS (
    SELECT
        s.score_id,
        s.task_id,
        s.is_cot,
        s.created_at AS score_created_at,
        t.evaluator,
        m.model_name,
        m.arch_version,
        m.data_version,
        m.num_params,
        row_number() OVER (
            PARTITION BY m.model_id, s.is_cot
            ORDER BY s.created_at DESC, s.score_id DESC
        ) AS rn
    FROM scores s
    JOIN task t ON t.task_id = s.task_id
    JOIN benchmark b ON b.benchmark_id = t.benchmark_id
    JOIN model m ON m.model_id = t.model_id
    WHERE b.benchmark_name = %(benchmark_name)s
      AND b.benchmark_split = %(benchmark_split)s
      AND t.status = 'completed'
      AND t.is_param_search IS false
      AND lower(m.data_version) = %(data_version)s
      AND regexp_replace(lower(m.num_params), '[^a-z0-9]', '', 'g') = %(param_norm)s
      AND s.is_cot = %(is_cot)s
)
SELECT *
FROM latest_scores
WHERE rn = 1
ORDER BY score_created_at DESC, score_id DESC
LIMIT 1;
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
    WHERE c.task_id = %(task_id)s
)
SELECT
    c.completions_id,
    c.task_id,
    c.sample_index,
    c.repeat_index,
    c.context,
    e.answer,
    e.ref_answer,
    e.fail_reason,
    e.created_at AS eval_created_at
FROM completions c
JOIN latest_eval e ON e.completions_id = c.completions_id AND e.rn = 1
WHERE c.task_id = %(task_id)s
  AND c.status = 'answer'
  AND e.is_passed IS false
ORDER BY c.sample_index, c.repeat_index, c.completions_id;
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _norm_param(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _context_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=4)


def _connect(config: DbConfig) -> psycopg.Connection[Any]:
    return psycopg.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        dbname=config.dbname,
        row_factory=dict_row,
    )


def export_wrong_rows(
    *,
    db: DbConfig,
    output_path: Path,
    export_name: str,
    benchmark_name: str,
    benchmark_split: str,
    data_version: str,
    param: str,
    method: str,
) -> dict[str, Any]:
    is_cot = method == "cot"
    with _connect(db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                LATEST_TASK_SQL,
                {
                    "benchmark_name": benchmark_name,
                    "benchmark_split": benchmark_split,
                    "data_version": data_version.lower(),
                    "param_norm": _norm_param(param),
                    "is_cot": is_cot,
                },
            )
            task = cur.fetchone()
            if task is None:
                raise RuntimeError(
                    f"no latest completed score found for {benchmark_name}_{benchmark_split} "
                    f"{data_version}-{param} method={method}"
                )

            task_id = int(task["task_id"])
            cur.execute(WRONG_ROWS_SQL, {"task_id": task_id})
            rows = [dict(row) for row in cur.fetchall()]

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "name": export_name,
                "source": export_name,
                "dataset": benchmark_name,
                "task_id": int(row["task_id"]),
                "completions_id": int(row["completions_id"]),
                "sample_index": int(row["sample_index"]),
                "repeat_index": int(row["repeat_index"]),
                "pass_index": 0,
                "answer": str(row.get("answer") or ""),
                "ref_answer": str(row.get("ref_answer") or ""),
                "fail_reason": str(row.get("fail_reason") or ""),
                "context": _context_to_text(row.get("context")),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name": export_name,
        "output_path": str(output_path),
        "wrong_count": len(records),
        "resolved": {
            "benchmark": f"{benchmark_name}_{benchmark_split}",
            "method": method,
            "task_id": int(task["task_id"]),
            "score_id": int(task["score_id"]),
            "score_created_at": task["score_created_at"].isoformat(),
            "evaluator": task["evaluator"],
            "model_name": task["model_name"],
            "data_version": task["data_version"],
            "num_params": task["num_params"],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--pg-host", default=None)
    parser.add_argument("--pg-port", type=int, default=None)
    parser.add_argument("--pg-user", default=None)
    parser.add_argument("--pg-password", default=None)
    parser.add_argument("--pg-dbname", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="g1f-7.2B-mmlu-pro")
    parser.add_argument("--benchmark-name", default="mmlu_pro")
    parser.add_argument("--benchmark-split", default="test")
    parser.add_argument("--data-version", default="g1f")
    parser.add_argument("--param", default="7.2b")
    parser.add_argument("--method", choices=("cot", "nocot"), default="cot")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_env_file(Path(args.env_file).expanduser())
    db = DbConfig(
        host=args.pg_host or os.environ.get("PG_HOST", "127.0.0.1"),
        port=args.pg_port or int(os.environ.get("PG_PORT", "15432")),
        user=args.pg_user or os.environ.get("PG_USER", "postgres"),
        password=args.pg_password if args.pg_password is not None else os.environ.get("PG_PASSWORD", ""),
        dbname=args.pg_dbname or os.environ.get("PG_DBNAME", "rwkv-eval"),
    )
    try:
        manifest = export_wrong_rows(
            db=db,
            output_path=Path(args.output).expanduser(),
            export_name=args.name,
            benchmark_name=args.benchmark_name,
            benchmark_split=args.benchmark_split,
            data_version=args.data_version,
            param=args.param,
            method=args.method,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"export failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
