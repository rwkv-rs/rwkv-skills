from __future__ import annotations

"""Summarize formal inference-swap eval results from the scheduler database."""

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from src.bin.run_infer_swap_eval import (
    DEFAULT_DATASETS,
    DEFAULT_DB_HOST,
    DEFAULT_DB_NAME,
    DEFAULT_DB_PORT,
    DEFAULT_DB_USER,
    DEFAULT_INFER_MODEL,
)
from src.eval.env_config import load_env_file
from src.eval.scheduler.dataset_utils import canonical_slug, split_benchmark_and_split
from src.eval.scheduler.models import normalize_model_name


@dataclass(slots=True, frozen=True)
class DatasetEvalSummary:
    dataset: str
    benchmark_name: str
    benchmark_split: str
    evaluator: str | None
    model: str
    status: str
    task_id: int | None = None
    task_status: str | None = None
    task_created_at: datetime | None = None
    log_path: str | None = None
    benchmark_num_samples: int | None = None
    completion_count: int = 0
    completed_completion_count: int = 0
    failed_completion_count: int = 0
    eval_count: int = 0
    passed_eval_count: int = 0
    score_count: int = 0
    score_id: int | None = None
    score_created_at: datetime | None = None
    cot_mode: str | None = None
    metrics: Mapping[str, Any] | None = None

    @property
    def has_score(self) -> bool:
        return self.score_count > 0 and self.metrics is not None


@dataclass(slots=True, frozen=True)
class InferSwapEvalSummary:
    model: str
    db_target: str
    datasets: tuple[DatasetEvalSummary, ...]

    @property
    def total_count(self) -> int:
        return len(self.datasets)

    @property
    def scored_count(self) -> int:
        return sum(1 for item in self.datasets if item.has_score)

    @property
    def task_count(self) -> int:
        return sum(1 for item in self.datasets if item.task_id is not None)

    @property
    def all_scored(self) -> bool:
        return self.total_count > 0 and self.scored_count == self.total_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "db_target": self.db_target,
            "total_count": self.total_count,
            "task_count": self.task_count,
            "scored_count": self.scored_count,
            "all_scored": self.all_scored,
            "datasets": [asdict(item) for item in self.datasets],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize inference-swap formal eval DB results")
    parser.add_argument("--model", default=DEFAULT_INFER_MODEL)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--db-host", default=DEFAULT_DB_HOST)
    parser.add_argument("--db-port", type=int, default=DEFAULT_DB_PORT)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--db-sslmode", default="prefer")
    parser.add_argument("--db-timeout-s", type=float, default=10.0)
    parser.add_argument("--output-json", help="Optional JSON summary path")
    parser.add_argument("--output-md", help="Optional Markdown evidence report path")
    parser.add_argument("--probe-json", help="Optional remote concurrency probe JSON to include in Markdown report")
    parser.add_argument("--watch", action="store_true", help="Poll until all datasets have real scores")
    parser.add_argument("--watch-interval-s", type=float, default=60.0, help="Polling interval for --watch")
    parser.add_argument(
        "--watch-timeout-s",
        type=float,
        default=0.0,
        help="Maximum watch duration; 0 means no timeout",
    )
    parser.add_argument(
        "--stdout",
        choices=("summary", "json", "none"),
        default="summary",
        help="Stdout format",
    )
    return parser.parse_args(argv)


def detect_formal_evaluator(dataset: str) -> str | None:
    from src.eval.scheduler.jobs import detect_job_from_dataset

    return detect_job_from_dataset(canonical_slug(dataset), is_cot=True)


def build_conninfo_from_args(args: argparse.Namespace) -> tuple[str, str]:
    load_env_file(Path(".env"))
    password = str(os.environ.get("PG_PASSWORD", "") or "")
    conninfo = make_conninfo(
        "",
        host=str(args.db_host),
        port=int(args.db_port),
        user=str(args.db_user),
        password=password,
        dbname=str(args.db_name),
        sslmode=str(args.db_sslmode),
        connect_timeout=max(1, int(float(args.db_timeout_s))),
    )
    target = f"{args.db_host}:{int(args.db_port)}/{args.db_name} user={args.db_user}"
    return conninfo, target


def fetch_dataset_summary(
    conn: psycopg.Connection[Any],
    *,
    dataset: str,
    model: str,
    evaluator: str | None,
) -> DatasetEvalSummary:
    canonical_dataset = canonical_slug(dataset)
    benchmark_name, benchmark_split = split_benchmark_and_split(canonical_dataset)
    params: list[Any] = [benchmark_name, benchmark_split, normalize_model_name(model)]
    evaluator_filter = ""
    if evaluator:
        evaluator_filter = "AND t.evaluator = %s"
        params.append(str(evaluator))

    query = f"""
        WITH latest_task AS (
            SELECT
                t.task_id,
                t.status AS task_status,
                t.created_at AS task_created_at,
                t.log_path,
                t.evaluator,
                b.num_samples AS benchmark_num_samples
            FROM task t
            JOIN benchmark b ON b.benchmark_id = t.benchmark_id
            JOIN model m ON m.model_id = t.model_id
            WHERE b.benchmark_name = %s
              AND b.benchmark_split = %s
              AND m.model_name = %s
              AND t.is_param_search = FALSE
              AND t.is_tmp = FALSE
              {evaluator_filter}
            ORDER BY t.created_at DESC, t.task_id DESC
            LIMIT 1
        )
        SELECT
            lt.task_id,
            lt.task_status,
            lt.task_created_at,
            lt.log_path,
            lt.evaluator,
            lt.benchmark_num_samples,
            COALESCE(cs.completion_count, 0) AS completion_count,
            COALESCE(cs.completed_completion_count, 0) AS completed_completion_count,
            COALESCE(cs.failed_completion_count, 0) AS failed_completion_count,
            COALESCE(es.eval_count, 0) AS eval_count,
            COALESCE(es.passed_eval_count, 0) AS passed_eval_count,
            CASE WHEN ss.score_id IS NULL THEN 0 ELSE 1 END AS score_count,
            ss.score_id,
            ss.score_created_at,
            ss.cot_mode,
            ss.metrics
        FROM latest_task lt
        LEFT JOIN LATERAL (
            SELECT
                COUNT(*)::INT AS completion_count,
                COUNT(*) FILTER (WHERE status = 'Completed')::INT AS completed_completion_count,
                COUNT(*) FILTER (WHERE status = 'Failed')::INT AS failed_completion_count
            FROM completions
            WHERE task_id = lt.task_id
        ) cs ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                COUNT(*)::INT AS eval_count,
                COUNT(*) FILTER (WHERE e.is_passed)::INT AS passed_eval_count
            FROM eval e
            JOIN completions c ON c.completions_id = e.completions_id
            WHERE c.task_id = lt.task_id
        ) es ON TRUE
        LEFT JOIN LATERAL (
            SELECT
                score_id,
                created_at AS score_created_at,
                cot_mode,
                metrics
            FROM scores
            WHERE task_id = lt.task_id
            ORDER BY created_at DESC, score_id DESC
            LIMIT 1
        ) ss ON TRUE
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, tuple(params))
        row = cur.fetchone()
    return summarize_dataset_row(
        dataset=canonical_dataset,
        benchmark_name=benchmark_name,
        benchmark_split=benchmark_split,
        evaluator=evaluator,
        model=normalize_model_name(model),
        row=dict(row) if row else None,
    )


def summarize_dataset_row(
    *,
    dataset: str,
    benchmark_name: str,
    benchmark_split: str,
    evaluator: str | None,
    model: str,
    row: Mapping[str, Any] | None,
) -> DatasetEvalSummary:
    if row is None:
        return DatasetEvalSummary(
            dataset=dataset,
            benchmark_name=benchmark_name,
            benchmark_split=benchmark_split,
            evaluator=evaluator,
            model=model,
            status="no_task",
        )

    score_count = int(row.get("score_count") or 0)
    task_status = str(row.get("task_status") or "")
    if score_count > 0:
        status = "scored"
    elif task_status == "Failed":
        status = "failed"
    elif task_status == "Running":
        status = "running"
    else:
        status = "pending_score"

    raw_metrics = row.get("metrics")
    metrics = raw_metrics if isinstance(raw_metrics, Mapping) else None
    return DatasetEvalSummary(
        dataset=dataset,
        benchmark_name=benchmark_name,
        benchmark_split=benchmark_split,
        evaluator=str(row.get("evaluator") or evaluator or "") or None,
        model=model,
        status=status,
        task_id=_optional_int(row.get("task_id")),
        task_status=task_status or None,
        task_created_at=_optional_datetime(row.get("task_created_at")),
        log_path=str(row.get("log_path") or "") or None,
        benchmark_num_samples=_optional_int(row.get("benchmark_num_samples")),
        completion_count=int(row.get("completion_count") or 0),
        completed_completion_count=int(row.get("completed_completion_count") or 0),
        failed_completion_count=int(row.get("failed_completion_count") or 0),
        eval_count=int(row.get("eval_count") or 0),
        passed_eval_count=int(row.get("passed_eval_count") or 0),
        score_count=score_count,
        score_id=_optional_int(row.get("score_id")),
        score_created_at=_optional_datetime(row.get("score_created_at")),
        cot_mode=str(row.get("cot_mode") or "") or None,
        metrics=metrics,
    )


def summarize_eval_results(
    conn: psycopg.Connection[Any],
    *,
    model: str,
    datasets: Sequence[str],
    db_target: str,
) -> InferSwapEvalSummary:
    normalized_model = normalize_model_name(model)
    items = tuple(
        fetch_dataset_summary(
            conn,
            dataset=dataset,
            model=normalized_model,
            evaluator=detect_formal_evaluator(dataset),
        )
        for dataset in datasets
    )
    return InferSwapEvalSummary(model=normalized_model, db_target=db_target, datasets=items)


def format_summary(summary: InferSwapEvalSummary) -> str:
    lines = [
        (
            f"model={summary.model} db={summary.db_target} "
            f"tasks={summary.task_count}/{summary.total_count} "
            f"scored={summary.scored_count}/{summary.total_count} "
            f"all_scored={str(summary.all_scored).lower()}"
        )
    ]
    for item in summary.datasets:
        task = f"task={item.task_id}" if item.task_id is not None else "task=none"
        evaluator = item.evaluator or "unknown"
        counts = f"completions={item.completed_completion_count}/{item.completion_count} eval={item.eval_count}"
        score = "score=none"
        if item.has_score:
            score = f"score={item.score_id} metrics={_format_metrics(item.metrics or {})}"
        lines.append(f"{item.dataset}: {item.status} {task} evaluator={evaluator} {counts} {score}")
    return "\n".join(lines)


def write_summary(path: Path, summary: InferSwapEvalSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def load_probe_payload(path: str | Path | None) -> Mapping[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else None


def format_markdown_report(
    summary: InferSwapEvalSummary,
    *,
    probe_payload: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        "# Inference Swap Eval Evidence Report",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Model | {_md_cell(summary.model)} |",
        f"| DB target | {_md_cell(summary.db_target)} |",
        f"| Dataset count | {summary.total_count} |",
        f"| Task count | {summary.task_count}/{summary.total_count} |",
        f"| Scored count | {summary.scored_count}/{summary.total_count} |",
        f"| All scored | {str(summary.all_scored).lower()} |",
        "",
    ]
    if probe_payload is not None:
        lines.extend(_format_probe_markdown(probe_payload))
        lines.append("")
    lines.extend(
        [
            "## Dataset Results",
            "",
            "| Dataset | Status | Task | Task Status | Completions | Eval | Score | Metrics |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in summary.datasets:
        metrics = "none"
        if item.has_score:
            metrics = _format_metrics(item.metrics or {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item.dataset),
                    _md_cell(item.status),
                    str(item.task_id) if item.task_id is not None else "none",
                    _md_cell(item.task_status or "none"),
                    f"{item.completed_completion_count}/{item.completion_count}",
                    str(item.eval_count),
                    str(item.score_id) if item.score_id is not None else "none",
                    _md_cell(metrics),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- `score=none` means the latest formal task has no DB `scores` row; no substitute score is shown.",
            "- Metrics in this report are copied from DB `scores.metrics` only.",
            "- Speedup design conclusions require completed benchmark scores plus runtime/probe evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown_report(
    path: Path,
    summary: InferSwapEvalSummary,
    *,
    probe_payload: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_markdown_report(summary, probe_payload=probe_payload), encoding="utf-8")


def run_summary_once(args: argparse.Namespace) -> InferSwapEvalSummary:
    conninfo, target = build_conninfo_from_args(args)
    with psycopg.connect(conninfo) as conn:
        return summarize_eval_results(
            conn,
            model=str(args.model),
            datasets=tuple(str(item) for item in args.datasets),
            db_target=target,
        )


def emit_summary(args: argparse.Namespace, summary: InferSwapEvalSummary) -> None:
    if args.stdout == "json":
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, default=_json_default), flush=True)
    elif args.stdout == "summary":
        print(format_summary(summary), flush=True)
    if args.output_json:
        write_summary(Path(args.output_json).expanduser(), summary)
    if args.output_md:
        write_markdown_report(
            Path(args.output_md).expanduser(),
            summary,
            probe_payload=load_probe_payload(getattr(args, "probe_json", None)),
        )


def watch_summary(
    args: argparse.Namespace,
    *,
    monotonic_fn=time.monotonic,
    sleep_fn=time.sleep,
) -> int:
    interval_s = float(args.watch_interval_s)
    timeout_s = float(args.watch_timeout_s)
    if interval_s <= 0:
        raise ValueError("--watch-interval-s must be positive")
    if timeout_s < 0:
        raise ValueError("--watch-timeout-s must be >= 0")

    start = monotonic_fn()
    deadline = None if timeout_s == 0 else start + timeout_s
    last_summary: InferSwapEvalSummary | None = None
    while True:
        summary = run_summary_once(args)
        last_summary = summary
        emit_summary(args, summary)
        if summary.all_scored:
            return 0
        now = monotonic_fn()
        if deadline is not None and now >= deadline:
            return 1
        sleep_for = interval_s
        if deadline is not None:
            sleep_for = min(sleep_for, max(0.0, deadline - now))
        if sleep_for <= 0:
            return 0 if last_summary and last_summary.all_scored else 1
        sleep_fn(sleep_for)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.watch):
        try:
            return watch_summary(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    summary = run_summary_once(args)
    emit_summary(args, summary)
    return 0 if summary.all_scored else 1


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    return None


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _format_metrics(metrics: Mapping[str, Any]) -> str:
    return json.dumps(dict(metrics), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _format_probe_markdown(payload: Mapping[str, Any]) -> list[str]:
    rows = [
        ("Base URL", payload.get("base_url")),
        ("Model", payload.get("model")),
        ("Protocol", payload.get("protocol")),
        ("Selected concurrency", payload.get("selected_concurrency")),
        ("Throughput best concurrency", payload.get("throughput_best_concurrency")),
        ("GPU full concurrency", payload.get("gpu_full_concurrency")),
        ("Largest successful concurrency", payload.get("largest_successful_concurrency")),
        ("Suggested infer workers", payload.get("suggested_infer_max_workers")),
        ("Suggested remote batch size", payload.get("suggested_remote_batch_size")),
        ("Target GPU utilization", payload.get("target_gpu_utilization")),
    ]
    lines = [
        "## Remote Concurrency Probe",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key, value in rows:
        lines.append(f"| {_md_cell(key)} | {_md_cell(_display_value(value))} |")
    points = payload.get("points")
    if isinstance(points, list):
        lines.extend(
            [
                "",
                "| Concurrency | Status | RPS | Output chars/s | Avg GPU | Peak GPU |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for point in points:
            if not isinstance(point, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(_display_value(point.get("concurrency"))),
                        _md_cell(_display_value(point.get("status"))),
                        _md_cell(_format_number(point.get("rps"))),
                        _md_cell(_format_number(point.get("output_chars_per_s"))),
                        _md_cell(_format_number(point.get("avg_gpu_utilization"))),
                        _md_cell(_format_number(point.get("peak_gpu_utilization"))),
                    ]
                )
                + " |"
            )
    return lines


def _format_number(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _display_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "DatasetEvalSummary",
    "InferSwapEvalSummary",
    "build_conninfo_from_args",
    "detect_formal_evaluator",
    "fetch_dataset_summary",
    "format_markdown_report",
    "format_summary",
    "load_probe_payload",
    "main",
    "parse_args",
    "emit_summary",
    "run_summary_once",
    "summarize_dataset_row",
    "summarize_eval_results",
    "watch_summary",
    "write_markdown_report",
    "write_summary",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
