from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from filelock import FileLock  # pyright: ignore[reportMissingImports]

from src.eval.scheduler.config import RESULTS_ROOT


def resolve_space_results_root() -> Path:
    override = os.environ.get("RWKV_SPACE_RESULTS_DIR") or os.environ.get("RUN_RESULTS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return RESULTS_ROOT.expanduser().resolve()


def resolve_score_index_path() -> Path:
    override = os.environ.get("RWKV_SPACE_SCORE_INDEX")
    if override:
        return Path(override).expanduser().resolve()
    return (resolve_space_results_root() / "space" / "score_index.jsonl").resolve()


def _lock_path(index_path: Path) -> Path:
    return index_path.with_suffix(index_path.suffix + ".lock")


def _normalize_task_id(task_id: str | int) -> int | str:
    try:
        return int(task_id)
    except (TypeError, ValueError):
        return str(task_id)


def build_score_index_entry(payload: Mapping[str, Any], *, task_id: str | int) -> dict[str, Any]:
    entry = dict(payload)
    entry["task_id"] = _normalize_task_id(task_id)
    entry.setdefault("indexed_at", datetime.now(UTC).isoformat())
    entry.setdefault("index_version", 1)
    return entry


def append_score_index_entry(payload: Mapping[str, Any], *, task_id: str | int) -> Path:
    index_path = resolve_score_index_path()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(build_score_index_entry(payload, task_id=task_id), ensure_ascii=False)
    with FileLock(str(_lock_path(index_path))):
        with index_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
    return index_path


def iter_score_index_payloads(path: str | Path | None = None) -> Iterator[dict[str, Any]]:
    index_path = resolve_score_index_path() if path is None else Path(path).expanduser().resolve()
    if not index_path.exists():
        return

    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            payload = raw_line.strip()
            if not payload:
                continue
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError(f"score index 第 {line_number} 行不是对象记录")
            yield parsed


__all__ = [
    "append_score_index_entry",
    "build_score_index_entry",
    "iter_score_index_payloads",
    "resolve_score_index_path",
    "resolve_space_results_root",
]

