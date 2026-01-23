from __future__ import annotations

from datetime import datetime
from pathlib import Path


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
    payload = {
        "dataset": dataset_slug,
        "model": model_name,
        "cot": bool(is_cot),
        "metrics": _normalize_jsonable(metrics),
        "samples": int(samples),
        "created_at": datetime.utcnow().replace(microsecond=False).isoformat() + "Z",
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
