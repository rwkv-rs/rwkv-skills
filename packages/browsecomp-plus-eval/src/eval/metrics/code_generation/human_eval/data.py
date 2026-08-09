from __future__ import annotations

from typing import Iterable, Dict
import gzip
import json
import os
from pathlib import Path


def read_problems(evalset_file: str) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """Parse each jsonl line and yield as a dictionary (supports .gz)."""
    path = Path(filename)
    if path.suffix == ".gz":
        with path.open("rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False) -> None:
    """Write iterable of dicts to jsonl (optionally gz)."""
    mode = "ab" if append else "wb"
    filename = os.path.expanduser(filename)
    path = Path(filename)
    if path.suffix == ".gz":
        with path.open(mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with path.open(mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))
