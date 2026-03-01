from __future__ import annotations

"""Audit/fix benchmark.num_samples by recounting dataset JSONL records."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from src.db.eval_db_repo import EvalDbRepository
from src.db.orm import Benchmark, get_session, init_orm
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_stats import count_jsonl_records
from src.eval.scheduler.datasets import DATASET_ROOTS, find_dataset_file
from src.eval.scheduler.dataset_utils import canonical_slug


@dataclass(slots=True)
class AuditStats:
    total: int = 0
    filtered: int = 0
    missing_dataset: int = 0
    count_errors: int = 0
    correct: int = 0
    mismatched: int = 0
    updated: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit benchmark.num_samples using current dataset parsing/counting logic."
    )
    parser.add_argument(
        "--benchmark-id",
        type=int,
        default=0,
        help="Only process one benchmark_id. Default: 0 (process all benchmarks).",
    )
    parser.add_argument(
        "--prepare-missing-dataset",
        action="store_true",
        help="Auto-prepare/download missing datasets before counting.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print mismatches/summary.",
    )
    return parser.parse_args(argv)


def _dataset_slug(benchmark_name: str, benchmark_split: str) -> str:
    if benchmark_split:
        return canonical_slug(f"{benchmark_name}_{benchmark_split}")
    return canonical_slug(benchmark_name)


def _resolve_dataset_path(dataset_slug: str, *, prepare_missing: bool) -> Path | None:
    found = find_dataset_file(dataset_slug, DATASET_ROOTS)
    if found is not None and found.exists():
        return found
    if not prepare_missing:
        return None
    try:
        return resolve_or_prepare_dataset(dataset_slug, verbose=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: failed to prepare dataset slug={dataset_slug}: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    only_id = int(args.benchmark_id) if int(args.benchmark_id) > 0 else 0

    init_orm(DEFAULT_DB_CONFIG)
    repo = EvalDbRepository()
    stats = AuditStats()

    with get_session() as session:
        stmt = select(Benchmark).order_by(Benchmark.benchmark_id.asc())
        benchmarks = list(session.execute(stmt).scalars())

        for benchmark in benchmarks:
            stats.total += 1
            benchmark_id = int(benchmark.benchmark_id)
            dataset_slug = _dataset_slug(benchmark.benchmark_name, benchmark.benchmark_split)

            if only_id > 0 and benchmark_id != only_id:
                stats.filtered += 1
                continue

            dataset_path = _resolve_dataset_path(
                dataset_slug,
                prepare_missing=bool(args.prepare_missing_dataset),
            )
            if dataset_path is None:
                stats.missing_dataset += 1
                print(f"WARN: dataset not found for benchmark_id={benchmark_id} dataset={dataset_slug}")
                continue

            try:
                expected = int(count_jsonl_records(dataset_path))
            except OSError as exc:
                stats.count_errors += 1
                print(f"WARN: failed to count dataset={dataset_slug} path={dataset_path}: {exc}")
                continue

            current = int(benchmark.num_samples or 0)
            if current == expected:
                stats.correct += 1
                if not args.quiet:
                    print(
                        f"OK: benchmark_id={benchmark_id} dataset={dataset_slug} "
                        f"num_samples={current} path={dataset_path}"
                    )
                continue

            stats.mismatched += 1
            print(
                f"DIFF: benchmark_id={benchmark_id} dataset={dataset_slug} "
                f"db={current} expected={expected} path={dataset_path}"
            )
            repo.update_benchmark_num_samples(
                session,
                benchmark_id=benchmark_id,
                num_samples=expected,
            )
            stats.updated += 1

    print(
        f"Summary (apply): total={stats.total} filtered={stats.filtered} "
        f"correct={stats.correct} mismatched={stats.mismatched} updated={stats.updated} "
        f"missing_dataset={stats.missing_dataset} count_errors={stats.count_errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
