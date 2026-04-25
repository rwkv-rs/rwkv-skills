from __future__ import annotations

"""Summarize prompt-ablation scores from the isolated prompt database."""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from sqlalchemy import create_engine, text

from src.eval.scheduler.config import DEFAULT_DB_CONFIG


TRIAL_RE = re.compile(r"/prompt_ablation/(?P<run_id>[^/]+)/(?P<trial>[^/]+)/configs/")


@dataclass(frozen=True)
class ScoreRow:
    run_id: str
    trial: str
    dataset: str
    evaluator: str
    task_id: int
    score_id: int
    metric_name: str
    metric_value: float
    metrics: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prompt-ablation scores")
    parser.add_argument("--run-id", required=True, help="Prompt ablation run id")
    parser.add_argument("--model", help="Filter by model_name")
    parser.add_argument("--benchmark", action="append", help="Filter by benchmark dataset name, e.g. gpqa_main")
    parser.add_argument("--format", choices=("markdown", "tsv", "json"), default="markdown")
    parser.add_argument("--show-all-metrics", action="store_true", help="Print raw metrics JSON in markdown/tsv output")
    return parser.parse_args(argv)


def _db_url() -> str:
    cfg = DEFAULT_DB_CONFIG
    return f"postgresql+psycopg://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.dbname}"


def _dataset_name(benchmark_name: str, benchmark_split: str) -> str:
    return benchmark_name if not benchmark_split else f"{benchmark_name}_{benchmark_split}"


def _trial_from_config_path(path: str) -> tuple[str, str] | None:
    match = TRIAL_RE.search(path or "")
    if not match:
        return None
    return match.group("run_id"), match.group("trial")


def _select_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    avg_keys = sorted(
        (key for key in metrics if key.startswith("avg@")),
        key=lambda item: float(item.removeprefix("avg@")),
    )
    for key in avg_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key in ("judge_accuracy", "exact_accuracy", "accuracy", "success_rate", "prompt_accuracy"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return "metric", 0.0


def _load_rows(args: argparse.Namespace) -> list[ScoreRow]:
    engine = create_engine(_db_url(), pool_pre_ping=True, future=True)
    query = text(
        """
        select
            t.task_id,
            t.config_path,
            t.evaluator,
            b.benchmark_name,
            b.benchmark_split,
            m.model_name,
            s.score_id,
            s.metrics
        from scores s
        join task t on t.task_id = s.task_id
        join benchmark b on b.benchmark_id = t.benchmark_id
        join model m on m.model_id = t.model_id
        where t.config_path like :run_pattern
          and (:model_name is null or m.model_name = :model_name)
        order by b.benchmark_name, b.benchmark_split, t.evaluator, t.config_path, s.score_id
        """
    )
    run_pattern = f"%/prompt_ablation/{args.run_id}/%"
    rows: list[ScoreRow] = []
    with engine.connect() as conn:
        result = conn.execute(query, {"run_pattern": run_pattern, "model_name": args.model})
        for row in result.mappings():
            parsed = _trial_from_config_path(str(row["config_path"] or ""))
            if parsed is None:
                continue
            run_id, trial = parsed
            dataset = _dataset_name(str(row["benchmark_name"]), str(row["benchmark_split"]))
            if args.benchmark and dataset not in set(args.benchmark):
                continue
            raw_metrics = row["metrics"]
            metrics = dict(raw_metrics) if isinstance(raw_metrics, dict) else json.loads(raw_metrics)
            metric_name, metric_value = _select_metric(metrics)
            rows.append(
                ScoreRow(
                    run_id=run_id,
                    trial=trial,
                    dataset=dataset,
                    evaluator=str(row["evaluator"]),
                    task_id=int(row["task_id"]),
                    score_id=int(row["score_id"]),
                    metric_name=metric_name,
                    metric_value=metric_value,
                    metrics=metrics,
                )
            )
    engine.dispose()
    return rows


def _print_markdown(rows: list[ScoreRow], *, show_all_metrics: bool) -> None:
    if not rows:
        print("No scores found.")
        return
    header = ["dataset", "trial", "metric", "score", "evaluator", "task_id"]
    if show_all_metrics:
        header.append("metrics")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        values = [
            row.dataset,
            row.trial,
            row.metric_name,
            f"{row.metric_value:.6g}",
            row.evaluator,
            str(row.task_id),
        ]
        if show_all_metrics:
            values.append(json.dumps(row.metrics, ensure_ascii=False, sort_keys=True))
        print("| " + " | ".join(values) + " |")
    print()
    _print_winners(rows)


def _print_tsv(rows: list[ScoreRow], *, show_all_metrics: bool) -> None:
    header = ["dataset", "trial", "metric", "score", "evaluator", "task_id", "score_id"]
    if show_all_metrics:
        header.append("metrics")
    print("\t".join(header))
    for row in rows:
        values = [
            row.dataset,
            row.trial,
            row.metric_name,
            f"{row.metric_value:.8g}",
            row.evaluator,
            str(row.task_id),
            str(row.score_id),
        ]
        if show_all_metrics:
            values.append(json.dumps(row.metrics, ensure_ascii=False, sort_keys=True))
        print("\t".join(values))


def _print_json(rows: list[ScoreRow]) -> None:
    payload = [
        {
            "run_id": row.run_id,
            "dataset": row.dataset,
            "trial": row.trial,
            "evaluator": row.evaluator,
            "task_id": row.task_id,
            "score_id": row.score_id,
            "metric": row.metric_name,
            "score": row.metric_value,
            "metrics": row.metrics,
        }
        for row in rows
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _print_winners(rows: list[ScoreRow]) -> None:
    grouped: dict[str, list[ScoreRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset].append(row)
    print("| dataset | best_trial | metric | score | control | delta |")
    print("| --- | --- | --- | --- | --- | --- |")
    for dataset in sorted(grouped):
        candidates = grouped[dataset]
        best = max(candidates, key=lambda row: row.metric_value)
        controls = [row for row in candidates if row.trial.startswith("control_")]
        control = controls[0] if controls else None
        if control is None:
            control_score = ""
            delta = ""
        else:
            control_score = f"{control.metric_value:.6g}"
            delta = f"{best.metric_value - control.metric_value:+.6g}"
        print(
            "| "
            + " | ".join(
                [
                    dataset,
                    best.trial,
                    best.metric_name,
                    f"{best.metric_value:.6g}",
                    control_score,
                    delta,
                ]
            )
            + " |"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _load_rows(args)
    if args.format == "json":
        _print_json(rows)
    elif args.format == "tsv":
        _print_tsv(rows, show_all_metrics=args.show_all_metrics)
    else:
        _print_markdown(rows, show_all_metrics=args.show_all_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
