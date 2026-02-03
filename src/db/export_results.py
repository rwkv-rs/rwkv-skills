from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from src.eval.scheduler.config import RESULTS_ROOT
from src.eval.scheduler.dataset_utils import safe_slug

from .eval_db_service import EvalDbService


def _isoformat(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for row in rows:
            fh.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def _normalize_jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return _isoformat(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized[str(key)] = _normalize_jsonable(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(v) for v in value]
    return value


def _table_jsonl_path(table: str, *, model_name: str, dataset_slug: str) -> Path:
    model_dir = safe_slug(model_name)
    filename = f"{safe_slug(dataset_slug)}.jsonl"
    return RESULTS_ROOT / table / model_dir / filename


def export_version_results(
    service: EvalDbService,
    *,
    task_id: str,
) -> None:
    bundle = service.get_task_bundle(task_id=task_id)
    if not bundle:
        return
    task_row = bundle.get("task") or {}
    model_row = bundle.get("model") or {}
    benchmark_row = bundle.get("benchmark") or {}
    model_name = str(model_row.get("model_name") or "")
    benchmark_name = str(benchmark_row.get("benchmark_name") or "")
    benchmark_split = str(benchmark_row.get("benchmark_split") or "")
    dataset_name = f"{benchmark_name}_{benchmark_split}" if benchmark_split else benchmark_name
    table_rows: dict[str, list[dict[str, Any]]] = {
        "benchmark": [benchmark_row],
        "model": [model_row],
        "task": [task_row],
        "completions": service.list_completions_rows(task_id=task_id),
        "eval": service.list_eval_rows(task_id=task_id),
        "scores": service.list_scores_rows(task_id=task_id),
    }
    for table, rows in table_rows.items():
        normalized = [_normalize_jsonable(row) for row in rows if isinstance(row, dict)]
        path = _table_jsonl_path(table, model_name=model_name, dataset_slug=dataset_name)
        _write_jsonl(path, normalized)

    return


__all__ = ["export_version_results"]
