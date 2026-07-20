#!/usr/bin/env python3
from __future__ import annotations

"""Print the current requested-benchmark score matrix from Postgres.

The report intentionally separates "has a score" from the latest task status:
when a dataset/model already has any score row, the missing-score backfill
scripts should skip it even if a newer fresh task is still running or failed.
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

DEFAULT_MODELS = (
    "rwkv7-g1f-7.2b-20260414-ctx8192",
    "rwkv7-g1f-13.3b-20260415-ctx8192",
    "rwkv7-g1g-7.2b-20260523-ctx8192",
    "rwkv7-g1g-13.3b-20260523-ctx8192",
    "rwkv7-g1h-preview3121-7.2b-20260701-ctx8192",
    "rwkv7-g1h-preview4673-2.9b-20260701-ctx8192",
)

DEFAULT_DATASETS = (
    "terminal_bench_2_1",
    "nl2repo",
    "deepswe",
    "widesearch",
    "deepsearchqa",
    "browsecomp",
    "browsecomp_zh",
    "browsecomp_plus",
    "complexfuncbench_official",
    "complexfuncbench_subset",
    "longbench",
    "longbench_qa",
    "longbench_qa_balanced",
    "mcp_atlas",
    "toolathlon",
    "apex_agents",
    "claweval",
    "wildclawbench",
    "skillsbench",
    "e_bench",
    "hle_with_tools",
    "hy_backend_2_0",
    "hy_swe_max",
    "hy_companybench",
    "hy_finmodelbench",
    "prodbench",
    "hy_skillsworld",
    "hy_euler_pro",
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    benchmark: str
    split: str
    domain: str
    integration: str
    source_kind: str
    benchmark_aliases: tuple[str, ...]


@dataclass(frozen=True)
class PairState:
    dataset: str
    benchmark: str
    split: str
    domain: str
    integration: str
    model: str
    data_rows: int | None
    data_path: str
    state: str
    score_value: str
    score_id: int | None
    score_task_id: int | None
    score_benchmark: str
    score_split: str
    latest_task_id: int | None
    latest_status: str
    latest_benchmark: str
    latest_split: str
    completions: int
    completed_completions: int
    total_samples: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root containing data/<dataset>/test.jsonl.")
    parser.add_argument("--dataset", action="append", help="Dataset to include. Repeatable; default is --suite requested40.")
    parser.add_argument(
        "--suite",
        choices=("requested40", "agent21", "agent_function_like"),
        default="requested40",
        help="Default dataset suite when --dataset is not supplied.",
    )
    parser.add_argument("--model", action="append", help="Model to include. Repeatable; default is g1f/g1g/g1h set.")
    parser.add_argument("--json", action="store_true", help="Emit JSON rows instead of TSV.")
    return parser.parse_args()


def connect_from_env():
    from src.eval.env_config import load_env_file
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    load_env_file()
    return psycopg.connect(
        host=DEFAULT_DB_CONFIG.host,
        port=DEFAULT_DB_CONFIG.port,
        user=DEFAULT_DB_CONFIG.user,
        password=DEFAULT_DB_CONFIG.password,
        dbname=DEFAULT_DB_CONFIG.dbname,
        row_factory=dict_row,
    )


def default_dataset_specs(suite: str) -> tuple[DatasetSpec, ...]:
    from src.eval.benchmark_sources import REQUESTED_BENCHMARK_SOURCES
    from src.eval.scheduler.dataset_utils import split_benchmark_and_split

    if suite == "requested40":
        sources = REQUESTED_BENCHMARK_SOURCES
    elif suite == "agent21":
        sources = tuple(source for source in REQUESTED_BENCHMARK_SOURCES if source.domain == "agent")
    else:
        names = set(DEFAULT_DATASETS)
        return tuple(manual_dataset_specs(tuple(name for name in DEFAULT_DATASETS if name in names)))

    specs: list[DatasetSpec] = []
    for source in sources:
        parsed_benchmark, parsed_split = split_benchmark_and_split(source.dataset_slug)
        benchmark = source.benchmark_name
        split = parsed_split or "test"
        aliases = tuple(dict.fromkeys((benchmark, parsed_benchmark)))
        specs.append(
            DatasetSpec(
                dataset=source.benchmark_name,
                benchmark=benchmark,
                split=split,
                domain=source.domain,
                integration=source.integration,
                source_kind=source.source_kind,
                benchmark_aliases=aliases,
            )
        )
    return tuple(specs)


def manual_dataset_specs(datasets: tuple[str, ...]) -> tuple[DatasetSpec, ...]:
    from src.eval.scheduler.dataset_utils import split_benchmark_and_split

    specs: list[DatasetSpec] = []
    for dataset in datasets:
        benchmark, split = split_benchmark_and_split(dataset)
        specs.append(
            DatasetSpec(
                dataset=dataset,
                benchmark=benchmark or dataset,
                split=split or "test",
                domain="manual",
                integration="manual",
                source_kind="manual",
                benchmark_aliases=tuple(dict.fromkeys((benchmark or dataset, dataset))),
            )
        )
    return tuple(specs)


def data_row_counts(repo_root: Path, specs: tuple[DatasetSpec, ...]) -> dict[str, tuple[int | None, str]]:
    counts: dict[str, tuple[int | None, str]] = {}
    for spec in specs:
        paths = _candidate_dataset_paths(repo_root, spec)
        path = next((candidate for candidate in paths if candidate.exists()), None)
        if path is None:
            counts[spec.dataset] = (None, "")
            continue
        with path.open("r", encoding="utf-8") as fh:
            counts[spec.dataset] = (sum(1 for line in fh if line.strip()), str(path))
    return counts


def _candidate_dataset_paths(repo_root: Path, spec: DatasetSpec) -> tuple[Path, ...]:
    names = tuple(dict.fromkeys((spec.benchmark, spec.dataset, *spec.benchmark_aliases)))
    splits = tuple(dict.fromkeys((spec.split, "test")))
    paths: list[Path] = []
    for name in names:
        for split in splits:
            paths.append(repo_root / "data" / name / f"{split}.jsonl")
    return tuple(dict.fromkeys(paths))


def load_db_rows(specs: tuple[DatasetSpec, ...], models: tuple[str, ...]) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    # Query the exact registered benchmark identity.  Cross-spelling and
    # dataset aliases can collide with a different benchmark and falsely make
    # a missing pair look scored.
    benchmark_names = sorted({spec.benchmark for spec in specs})
    splits = sorted({spec.split for spec in specs} | {"test"})
    with connect_from_env() as conn:
        latest_scores = {
            (str(row["benchmark_name"]), str(row["benchmark_split"]), str(row["model_name"])): dict(row)
            for row in conn.execute(
                """
                SELECT DISTINCT ON (b.benchmark_name, b.benchmark_split, m.model_name)
                    b.benchmark_name,
                    b.benchmark_split,
                    m.model_name,
                    s.score_id,
                    s.task_id AS score_task_id,
                    s.metrics,
                    s.created_at AS score_created_at
                FROM scores s
                JOIN task t ON t.task_id = s.task_id
                JOIN benchmark b ON b.benchmark_id = t.benchmark_id
                JOIN model m ON m.model_id = t.model_id
                WHERE coalesce(t.is_tmp, false) = false
                  AND b.benchmark_name = ANY(%s)
                  AND b.benchmark_split = ANY(%s)
                  AND m.model_name = ANY(%s)
                ORDER BY b.benchmark_name, b.benchmark_split, m.model_name, s.created_at DESC, s.score_id DESC
                """,
                (benchmark_names, splits, list(models)),
            )
        }
        latest_tasks = {
            (str(row["benchmark_name"]), str(row["benchmark_split"]), str(row["model_name"])): dict(row)
            for row in conn.execute(
                """
                WITH completion_counts AS (
                    SELECT
                        task_id,
                        count(*) AS completions,
                        count(*) FILTER (WHERE status = 'Completed') AS completed_completions
                    FROM completions
                    GROUP BY task_id
                )
                SELECT DISTINCT ON (b.benchmark_name, b.benchmark_split, m.model_name)
                    b.benchmark_name,
                    b.benchmark_split,
                    m.model_name,
                    t.task_id,
                    t.status,
                    b.num_samples,
                    coalesce(cc.completions, 0) AS completions,
                    coalesce(cc.completed_completions, 0) AS completed_completions,
                    t.created_at AS task_created_at
                FROM task t
                JOIN benchmark b ON b.benchmark_id = t.benchmark_id
                JOIN model m ON m.model_id = t.model_id
                LEFT JOIN completion_counts cc ON cc.task_id = t.task_id
                WHERE coalesce(t.is_tmp, false) = false
                  AND b.benchmark_name = ANY(%s)
                  AND b.benchmark_split = ANY(%s)
                  AND m.model_name = ANY(%s)
                ORDER BY b.benchmark_name, b.benchmark_split, m.model_name, t.created_at DESC, t.task_id DESC
                """,
                (benchmark_names, splits, list(models)),
            )
        }
    return latest_scores, latest_tasks


def metric_value(metrics: Any) -> str:
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            return metrics[:80]
    if not isinstance(metrics, dict):
        return ""
    for key in ("avg@1", "accuracy", "score", "pass@1"):
        if key in metrics:
            return str(metrics[key])
    return json.dumps(metrics, ensure_ascii=False, sort_keys=True)[:120]


def classify_pairs(repo_root: Path, specs: tuple[DatasetSpec, ...], models: tuple[str, ...]) -> list[PairState]:
    data_counts = data_row_counts(repo_root, specs)
    latest_scores, latest_tasks = load_db_rows(specs, models)
    states: list[PairState] = []
    for spec in specs:
        for model in models:
            score = _lookup_row(latest_scores, spec, model)
            task = _lookup_row(latest_tasks, spec, model)
            data_rows, data_path = data_counts[spec.dataset]
            if score:
                state = "scored"
            elif data_rows is None:
                state = "missing_dataset"
            elif task is None:
                state = "no_task"
            elif str(task.get("status") or "") == "Running":
                state = "running_no_score"
            elif int(task.get("completed_completions") or 0) <= 0:
                state = "failed_zero_completion"
            else:
                state = "failed_with_completions"
            states.append(
                PairState(
                    dataset=spec.dataset,
                    benchmark=spec.benchmark,
                    split=spec.split,
                    domain=spec.domain,
                    integration=spec.integration,
                    model=model,
                    data_rows=data_rows,
                    data_path=data_path,
                    state=state,
                    score_value=metric_value(score.get("metrics")) if score else "",
                    score_id=int(score["score_id"]) if score else None,
                    score_task_id=int(score["score_task_id"]) if score else None,
                    score_benchmark=str(score.get("benchmark_name") or "") if score else "",
                    score_split=str(score.get("benchmark_split") or "") if score else "",
                    latest_task_id=int(task["task_id"]) if task else None,
                    latest_status=str(task.get("status") or "") if task else "",
                    latest_benchmark=str(task.get("benchmark_name") or "") if task else "",
                    latest_split=str(task.get("benchmark_split") or "") if task else "",
                    completions=int(task.get("completions") or 0) if task else 0,
                    completed_completions=int(task.get("completed_completions") or 0) if task else 0,
                    total_samples=int(task["num_samples"]) if task and task.get("num_samples") is not None else None,
                )
            )
    return states


def _lookup_row(rows: dict[tuple[str, str, str], dict[str, Any]], spec: DatasetSpec, model: str) -> dict[str, Any] | None:
    splits = tuple(dict.fromkeys((spec.split, "test")))
    row = rows.get((spec.benchmark, spec.split, model))
    if row is not None:
        return row
    return None


def print_tsv(states: list[PairState]) -> None:
    print(
        "dataset\tbenchmark\tsplit\tdomain\tintegration\tdata_rows\tdata_path\tmodel\tstate\t"
        "score_value\tscore_id\tscore_task_id\tscore_benchmark\tscore_split\t"
        "latest_task_id\tlatest_status\tlatest_benchmark\tlatest_split\t"
        "completions\tcompleted_completions\ttotal_samples"
    )
    for state in states:
        print(
            f"{state.dataset}\t{state.benchmark}\t{state.split}\t{state.domain}\t{state.integration}\t"
            f"{state.data_rows if state.data_rows is not None else 'MISSING'}\t{state.data_path}\t"
            f"{state.model}\t{state.state}\t{state.score_value}\t{state.score_id or ''}\t"
            f"{state.score_task_id or ''}\t{state.score_benchmark}\t{state.score_split}\t"
            f"{state.latest_task_id or ''}\t{state.latest_status}\t{state.latest_benchmark}\t{state.latest_split}\t"
            f"{state.completions}\t{state.completed_completions}\t{state.total_samples or ''}"
        )


def print_summary(states: list[PairState]) -> None:
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for state in states:
        by_dataset[state.dataset][state.state] += 1
        by_domain[state.domain][state.state] += 1
    print("summary_by_domain")
    for domain in sorted(by_domain):
        counts = by_domain[domain]
        rendered = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
        print(f"{domain}\t{rendered}")
    print("summary_by_dataset")
    for dataset in sorted(by_dataset):
        counts = by_dataset[dataset]
        rendered = " ".join(f"{key}={counts[key]}" for key in sorted(counts))
        print(f"{dataset}\t{rendered}")
    print("summary_total")
    total = Counter(state.state for state in states)
    print(" ".join(f"{key}={total[key]}" for key in sorted(total)))


def main() -> int:
    opts = parse_args()
    specs = manual_dataset_specs(tuple(opts.dataset)) if opts.dataset else default_dataset_specs(str(opts.suite))
    models = tuple(opts.model or DEFAULT_MODELS)
    states = classify_pairs(Path(opts.repo_root), specs, models)
    if opts.json:
        print(json.dumps([state.__dict__ for state in states], ensure_ascii=False, indent=2))
    else:
        print_summary(states)
        print_tsv(states)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
