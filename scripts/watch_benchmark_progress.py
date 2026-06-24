from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import tuple_row


ERROR_PATTERN = (
    "Traceback|RemoteHTTPError|timeout|Timeout|AttributeError|RuntimeError|ERROR|"
    "CUDA out of memory|HTTP 5|Connection refused|BrokenPipe|ReadTimeout|ConnectTimeout"
)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def env_value(env: dict[str, str], key: str, default: str) -> str:
    return os.environ.get(key) or env.get(key) or default


def conninfo(env: dict[str, str]) -> str:
    parts = [
        f"host={env_value(env, 'PG_HOST', '127.0.0.1')}",
        f"port={env_value(env, 'PG_PORT', '5432')}",
        f"user={env_value(env, 'PG_USER', 'postgres')}",
        f"dbname={env_value(env, 'PG_DBNAME', 'rwkv-eval')}",
    ]
    password = env_value(env, "PG_PASSWORD", "")
    if password:
        parts.append(f"password={password}")
    sslmode = env_value(env, "PG_SSLMODE", "prefer")
    if sslmode:
        parts.append(f"sslmode={sslmode}")
    return " ".join(parts)


def run_command(
    args: list[str],
    *,
    cwd: Path,
    timeout: float = 15.0,
    extra_env: dict[str, str] | None = None,
) -> str:
    child_env = os.environ.copy()
    if extra_env:
        child_env.update(extra_env)
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001 - monitor must keep running.
        return f"[command-error] {args!r}: {exc}\n"
    return completed.stdout


def rows_text(rows: list[tuple[Any, ...]]) -> str:
    return "".join("|".join("" if value is None else str(value) for value in row) + "\n" for row in rows)


def query_text(conn: psycopg.Connection[Any], sql: str) -> str:
    try:
        with conn.cursor(row_factory=tuple_row) as cur:
            cur.execute(sql)
            return rows_text(cur.fetchall())
    except Exception as exc:  # noqa: BLE001 - monitor must keep running.
        return f"[sql-error] {exc}\n"


def collect_once(*, root: Path, run_tag: str, score_id_min: int, env: dict[str, str]) -> str:
    scheduler_log_dir = root / "logs" / "scheduler" / run_tag
    run_log_dir = root / "results" / "logs" / run_tag
    pid_dir = scheduler_log_dir / "pids"
    dispatcher_pid_file = scheduler_log_dir / "dispatcher.pid"
    lines: list[str] = []

    def section(name: str) -> None:
        lines.append(f"\n===== {time.strftime('%F %T %Z')} {name} =====\n")

    section("process")
    dispatcher_pid = dispatcher_pid_file.read_text(encoding="utf-8").strip() if dispatcher_pid_file.exists() else ""
    lines.append(f"dispatcher_pid={dispatcher_pid or 'missing'}\n")
    if dispatcher_pid:
        lines.append(run_command(["ps", "-p", dispatcher_pid, "-o", "pid,stat,etime,cmd", "--no-headers"], cwd=root))
    runner_count = len(list(pid_dir.glob("*.pid"))) if pid_dir.exists() else 0
    lines.append(f"runner_pid_files={runner_count}\n")

    section("db")
    with psycopg.connect(conninfo(env)) as conn:
        lines.append(query_text(conn, "select max(score_id), count(*) from scores;"))
        lines.append(query_text(conn, "select status, count(*) from task group by status order by status;"))
        lines.append(
            query_text(
                conn,
                f"""
                select s.score_id, s.task_id, t.evaluator, b.benchmark_name, m.model_name,
                       s.cot_mode, s.created_at, s.metrics
                from scores s
                join task t on t.task_id=s.task_id
                join benchmark b on b.benchmark_id=t.benchmark_id
                join model m on m.model_id=t.model_id
                where s.score_id >= {int(score_id_min)}
                order by s.score_id desc
                limit 20;
                """,
            )
        )

        section("recent-running")
        lines.append(
            query_text(
                conn,
                """
                select t.task_id, t.evaluator, b.benchmark_name, m.model_name, t.status,
                       coalesce((select count(*) from completions c where c.task_id=t.task_id),0) comps,
                       coalesce((select count(*) from eval e join completions c on c.completions_id=e.completions_id where c.task_id=t.task_id),0) evals
                from task t
                join benchmark b on b.benchmark_id=t.benchmark_id
                join model m on m.model_id=t.model_id
                where t.status='Running'
                order by t.created_at desc
                limit 30;
                """,
            )
        )

    section("backpressure")
    lines.append(
        run_command(
            [
                "bash",
                "-lc",
                "curl -fsS http://127.0.0.1:19083/v1/backpressure | "
                "jq -r '.models | to_entries[] | [.key, (.value.status // \"\"), "
                "(.value.pending_queue // 0), (.value.service_queue // 0), "
                "(.value.engine_inbox // 0), (.value.active_records // 0), "
                "(.value.scheduler_waiting // 0), (.value.scheduler_running // 0)] | @tsv'",
            ],
            cwd=root,
            timeout=10.0,
        )
    )

    section("errors")
    lines.append(
        run_command(
            [
                "bash",
                "-lc",
                f"rg -n {ERROR_PATTERN!r} {scheduler_log_dir!s} {run_log_dir!s} -S | tail -n 80 || true",
            ],
            cwd=root,
            timeout=15.0,
        )
    )

    section("sync-dry-run")
    lines.append(
        run_command(
            [".venv/bin/python", "scripts/sync_eval_db_to_remote.py", "--score-id-min", str(score_id_min)],
            cwd=root,
            timeout=30.0,
            extra_env={
                "PG_HOST": env_value(env, "PG_HOST", "127.0.0.1"),
                "PG_PORT": env_value(env, "PG_PORT", "5432"),
                "PG_USER": env_value(env, "PG_USER", "postgres"),
                "PG_PASSWORD": env_value(env, "PG_PASSWORD", ""),
                "PG_DBNAME": env_value(env, "PG_DBNAME", "rwkv-eval"),
                "PG_SSLMODE": env_value(env, "PG_SSLMODE", "prefer"),
            },
        )
    )
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch local benchmark progress")
    parser.add_argument("--run-tag", default=os.environ.get("RUN_TAG", "local_6gpu_resume_20260622"))
    parser.add_argument("--interval-seconds", type=int, default=int(os.environ.get("INTERVAL_SECONDS", "300")))
    parser.add_argument("--score-id-min", type=int, default=int(os.environ.get("SCORE_ID_MIN", "182")))
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    env = load_env(root / ".env")
    log_dir = root / "logs" / "monitor" / args.run_tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "progress.log"
    while True:
        text = collect_once(root=root, run_tag=args.run_tag, score_id_min=args.score_id_min, env=env)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
        if args.once:
            return 0
        time.sleep(max(5, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
