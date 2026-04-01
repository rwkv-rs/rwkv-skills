from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import pyarrow.parquet as pq

T = TypeVar("T")


def collect_files_with_extension(root: str | Path, extension: str) -> list[Path]:
    base = Path(root).expanduser().resolve()
    if not base.exists():
        return []
    normalized = extension.lower().lstrip(".")
    files = [
        path
        for path in base.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") == normalized
    ]
    files.sort()
    return files


def iter_jsonl_items(
    path: str | Path,
    parse_item: Callable[[dict[str, Any]], T] | None = None,
) -> Iterator[T | dict[str, Any]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            yield parse_item(record) if parse_item is not None else record


def read_jsonl_items(
    path: str | Path,
    parse_item: Callable[[dict[str, Any]], T] | None = None,
) -> list[T | dict[str, Any]]:
    return list(iter_jsonl_items(path, parse_item=parse_item))


def iter_gzip_jsonl_items(
    path: str | Path,
    parse_item: Callable[[dict[str, Any]], T] | None = None,
) -> Iterator[T | dict[str, Any]]:
    with gzip.open(Path(path).expanduser().resolve(), "rt", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            yield parse_item(record) if parse_item is not None else record


def read_gzip_jsonl_items(
    path: str | Path,
    parse_item: Callable[[dict[str, Any]], T] | None = None,
) -> list[T | dict[str, Any]]:
    return list(iter_gzip_jsonl_items(path, parse_item=parse_item))


def read_csv_items(
    path: str | Path,
    parse_row: Callable[[dict[str, str]], T] | None = None,
) -> list[T | dict[str, str]]:
    rows: list[T | dict[str, str]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(parse_row(row) if parse_row is not None else row)
    return rows


def iter_parquet_items(
    path: str | Path,
    parse_row: Callable[[dict[str, Any]], T] | None = None,
) -> Iterator[T | dict[str, Any]]:
    parquet_file = pq.ParquetFile(Path(path).expanduser().resolve())
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            record = dict(row)
            yield parse_row(record) if parse_row is not None else record


def read_parquet_items(
    path: str | Path,
    parse_row: Callable[[dict[str, Any]], T] | None = None,
) -> list[T | dict[str, Any]]:
    return list(iter_parquet_items(path, parse_row=parse_row))


__all__ = [
    "collect_files_with_extension",
    "iter_gzip_jsonl_items",
    "iter_jsonl_items",
    "iter_parquet_items",
    "read_csv_items",
    "read_gzip_jsonl_items",
    "read_jsonl_items",
    "read_parquet_items",
]
