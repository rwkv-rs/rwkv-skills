from __future__ import annotations

"""Reconcile benchmark.num_samples with local dataset JSONL record counts."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import select

from src.db.orm import Benchmark, get_session, init_orm
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_stats import count_jsonl_records
from src.eval.scheduler.dataset_utils import canonical_slug, infer_dataset_slug_from_path, split_benchmark_and_split
from src.eval.scheduler.datasets import DATASET_ROOTS


@dataclass(slots=True)
class ReconcileRow:
    benchmark_id: int
    slug: str
    old_samples: int
    new_samples: int
    path: Path


def _collect_dataset_files(roots: Iterable[Path]) -> dict[str, Path]:
    dataset_map: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*.jsonl"):
            slug = canonical_slug(infer_dataset_slug_from_path(str(candidate)))
            resolved = candidate.resolve()
            previous = dataset_map.get(slug)
            if previous is None:
                dataset_map[slug] = resolved
            elif previous.stem == "input_data" and resolved.stem != "input_data":
                dataset_map[slug] = resolved
    return dataset_map


def _slug_from_benchmark_row(row: Benchmark) -> str:
    if row.benchmark_split:
        return canonical_slug(f"{row.benchmark_name}_{row.benchmark_split}")
    return canonical_slug(row.benchmark_name)


def _safe_count(path: Path) -> int | None:
    try:
        value = count_jsonl_records(path)
    except OSError:
        return None
    return value if value > 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile benchmark.num_samples from local dataset files")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist changes (default is dry-run)",
    )
    parser.add_argument(
        "--insert-missing",
        action="store_true",
        help="Insert missing benchmark rows for local datasets",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Limit to dataset slug(s), e.g. aime25_test (can be repeated)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = {canonical_slug(item) for item in args.dataset if item}

    init_orm(DEFAULT_DB_CONFIG)
    dataset_files = _collect_dataset_files(DATASET_ROOTS)
    if requested:
        missing = sorted(item for item in requested if item not in dataset_files)
        if missing:
            print("⚠️ 以下数据集在本地 data/ 下找不到 JSONL：")
            for slug in missing:
                print(f"  - {slug}")

    updated_rows: list[ReconcileRow] = []
    inserted_count = 0
    unresolved: list[str] = []

    with get_session() as session:
        benchmarks = list(session.execute(select(Benchmark)).scalars().all())
        known_slugs: set[str] = set()

        for row in benchmarks:
            slug = _slug_from_benchmark_row(row)
            known_slugs.add(slug)
            if requested and slug not in requested:
                continue
            path = dataset_files.get(slug)
            if path is None:
                unresolved.append(slug)
                continue
            count = _safe_count(path)
            if count is None:
                unresolved.append(slug)
                continue
            if int(row.num_samples) != count:
                updated_rows.append(
                    ReconcileRow(
                        benchmark_id=int(row.benchmark_id),
                        slug=slug,
                        old_samples=int(row.num_samples),
                        new_samples=count,
                        path=path,
                    )
                )
                if args.apply:
                    row.num_samples = count

        if args.insert_missing:
            for slug, path in sorted(dataset_files.items()):
                if requested and slug not in requested:
                    continue
                if slug in known_slugs:
                    continue
                count = _safe_count(path)
                if count is None:
                    continue
                benchmark_name, benchmark_split = split_benchmark_and_split(slug)
                if args.apply:
                    session.add(
                        Benchmark(
                            benchmark_name=benchmark_name,
                            benchmark_split=benchmark_split,
                            url=None,
                            status="Todo",
                            num_samples=count,
                        )
                    )
                inserted_count += 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] benchmark rows checked: {len(updated_rows)} change(s), {inserted_count} insert(s)")
    if updated_rows:
        print("变更列表：")
        for item in updated_rows:
            print(
                f"  - id={item.benchmark_id} {item.slug}: {item.old_samples} -> {item.new_samples} "
                f"({item.path})"
            )
    if unresolved:
        uniq = sorted(set(unresolved))
        print(f"⚠️ 无法对齐（本地找不到文件或样本数=0）：{len(uniq)}")
        for slug in uniq:
            print(f"  - {slug}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
