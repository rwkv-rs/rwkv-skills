from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path


def _normalize_cot_mode(value, *, is_cot: bool):  # noqa: ANN001
    raw = str(value or "").strip().lower()
    mapping = {
        "no_cot": "no_cot",
        "nocot": "no_cot",
        "no-cot": "no_cot",
        "fake_cot": "fake_cot",
        "fakecot": "fake_cot",
        "fake-cot": "fake_cot",
        "cot": "cot",
    }
    if raw in mapping:
        return mapping[raw]
    return "cot" if is_cot else "no_cot"


def _normalize_jsonable(value):  # noqa: ANN001
    import numpy as np  # local import: avoids import cost in CLI startup

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_jsonable(v) for v in value]
    return value


def make_score_payload(
    dataset_slug: str,
    *,
    is_cot: bool,
    model_name: str,
    metrics: dict,
    samples: int,
    problems: int | None = None,
    task: str | None = None,
    task_details: dict | None = None,
    extra: dict | None = None,
) -> dict:
    created_at = datetime.now(ZoneInfo("Asia/Shanghai")).replace(microsecond=False, tzinfo=None).isoformat()
    explicit_cot_mode = None
    if isinstance(extra, dict):
        explicit_cot_mode = extra.get("cot_mode")
    if explicit_cot_mode is None and isinstance(task_details, dict):
        explicit_cot_mode = task_details.get("cot_mode")
    payload = {
        "dataset": dataset_slug,
        "model": model_name,
        "cot": bool(is_cot),
        "cot_mode": _normalize_cot_mode(explicit_cot_mode, is_cot=bool(is_cot)),
        "metrics": _normalize_jsonable(metrics),
        "samples": int(samples),
        "created_at": created_at,
    }
    if problems is not None:
        payload["problems"] = int(problems)
    if task:
        payload["task"] = task
    if task_details:
        payload["task_details"] = task_details
    if extra:
        payload.update(extra)
    return payload


__all__ = ["make_score_payload"]
