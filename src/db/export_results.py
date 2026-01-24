from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from src.eval.results.layout import (
    COMPLETIONS_ROOT,
    EVAL_RESULTS_ROOT,
    SCORES_ROOT,
    CONSOLE_LOG_ROOT,
)
from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug

from .eval_db_service import EvalDbService


def _isoformat(value: Any) -> str | None:
    if isinstance(value, datetime):
        text = value.isoformat()
        return text.replace("+00:00", "Z")
    if isinstance(value, str):
        return value
    return None


def _versioned_path(root: Path, model_slug: str, stem: str, suffix: str) -> Path:
    target = root / model_slug / f"{stem}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for row in rows:
            fh.write(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def export_version_results(
    service: EvalDbService,
    *,
    version_id: str,
    is_param_search: bool,
) -> None:
    for path in (COMPLETIONS_ROOT, EVAL_RESULTS_ROOT, SCORES_ROOT, CONSOLE_LOG_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    score_payload = service.get_score_payload(version_id=version_id, is_param_search=is_param_search)
    if not score_payload:
        return
    dataset = score_payload.get("dataset")
    model = score_payload.get("model")
    is_cot = bool(score_payload.get("cot", False))
    if not isinstance(dataset, str) or not isinstance(model, str):
        return

    dataset_slug = canonical_slug(dataset)
    model_slug = safe_slug(model)
    stem = f"{dataset_slug}_v{version_id}"

    completions_payloads = service.list_completion_payloads(
        version_id=version_id,
        is_param_search=is_param_search,
    )
    eval_payloads = service.list_eval_payloads(
        version_id=version_id,
        is_param_search=is_param_search,
    )
    log_payloads = service.list_log_payloads(version_id=version_id)

    completions_path = _versioned_path(COMPLETIONS_ROOT, model_slug, stem, ".jsonl")
    eval_path = _versioned_path(EVAL_RESULTS_ROOT, model_slug, stem, "_results.jsonl")
    score_path = _versioned_path(SCORES_ROOT, model_slug, stem, ".json")
    logs_path = _versioned_path(CONSOLE_LOG_ROOT, model_slug, stem, ".jsonl")

    _write_jsonl(completions_path, completions_payloads)
    _write_jsonl(eval_path, eval_payloads)
    _write_jsonl(
        logs_path,
        [
            {
                "version_id": version_id,
                "event": row.get("event"),
                "job_id": row.get("job_id"),
                "payload": row.get("payload"),
                "created_at": _isoformat(row.get("created_at")),
            }
            for row in log_payloads
        ],
    )

    task_details = score_payload.get("task_details") if isinstance(score_payload.get("task_details"), dict) else {}
    task_details = dict(task_details)
    task_details["eval_details_path"] = str(eval_path)
    task_details.setdefault("check_details_path", None)
    score_payload_versioned = {
        "dataset": dataset_slug,
        "model": model,
        "cot": bool(is_cot),
        "metrics": score_payload.get("metrics") if isinstance(score_payload.get("metrics"), dict) else {},
        "samples": int(score_payload.get("samples", 0)),
        "problems": score_payload.get("problems"),
        "created_at": _isoformat(score_payload.get("created_at")),
        "log_path": str(completions_path),
        "task": score_payload.get("task"),
        "task_details": task_details,
    }
    _write_json(score_path, score_payload_versioned)

    return


__all__ = ["export_version_results"]
