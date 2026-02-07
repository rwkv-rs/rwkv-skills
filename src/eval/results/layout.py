"""Canonical output layout for evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.eval.scheduler.config import (
    RESULTS_ROOT,
    DEFAULT_LOG_DIR,
    DEFAULT_COMPLETION_DIR,
    DEFAULT_EVAL_RESULT_DIR,
    DEFAULT_CHECK_RESULT_DIR,
    DEFAULT_RUN_LOG_DIR,
)
from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug


COMPLETIONS_ROOT = DEFAULT_COMPLETION_DIR
EVAL_RESULTS_ROOT = DEFAULT_EVAL_RESULT_DIR
CHECK_RESULTS_ROOT = DEFAULT_CHECK_RESULT_DIR
SCORES_ROOT = DEFAULT_LOG_DIR
CONSOLE_LOG_ROOT = DEFAULT_RUN_LOG_DIR
PARAM_SEARCH_ROOT = RESULTS_ROOT / "param_search"
PARAM_SEARCH_COMPLETIONS_ROOT = PARAM_SEARCH_ROOT / "completions"
PARAM_SEARCH_EVAL_RESULTS_ROOT = PARAM_SEARCH_ROOT / "eval"
PARAM_SEARCH_SCORES_ROOT = PARAM_SEARCH_ROOT / "scores"


def ensure_results_structure() -> None:
    for path in (
        RESULTS_ROOT,
        COMPLETIONS_ROOT,
        EVAL_RESULTS_ROOT,
        CHECK_RESULTS_ROOT,
        SCORES_ROOT,
        CONSOLE_LOG_ROOT,
        PARAM_SEARCH_ROOT,
        PARAM_SEARCH_COMPLETIONS_ROOT,
        PARAM_SEARCH_EVAL_RESULTS_ROOT,
        PARAM_SEARCH_SCORES_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _dataset_file_stem(dataset_slug: str, *, is_cot: bool) -> str:
    slug = canonical_slug(dataset_slug)
    return f"{slug}__cot" if is_cot else slug


def _model_dataset_relpath(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    model_dir = safe_slug(model_name)
    stem = _dataset_file_stem(dataset_slug, is_cot=is_cot)
    return Path(model_dir) / stem


def _materialize(base: Path, *, suffix: str, root: Path) -> Path:
    target = root / base.parent / f"{base.name}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def jsonl_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return _materialize(rel, suffix=".jsonl", root=COMPLETIONS_ROOT)


def scores_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return _materialize(rel, suffix=".json", root=SCORES_ROOT)


def eval_details_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    rel = _model_dataset_relpath(dataset_slug, is_cot=is_cot, model_name=model_name)
    return _materialize(rel, suffix="_results.jsonl", root=EVAL_RESULTS_ROOT)

def check_details_path(benchmark_name: str, *, model_name: str) -> Path:
    """Per-benchmark checker output (results/check/{model}/{benchmark}.jsonl)."""
    ensure_results_structure()
    model_dir = safe_slug(model_name)
    bench = safe_slug(benchmark_name)
    target = CHECK_RESULTS_ROOT / model_dir / f"{bench}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target

def param_search_dir(dataset_slug: str, *, model_name: str, root: Path) -> Path:
    ensure_results_structure()
    model_dir = safe_slug(model_name)
    benchmark_dir = canonical_slug(dataset_slug)
    directory = root / model_dir / benchmark_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def param_search_completion_trial_path(
    dataset_slug: str,
    *,
    model_name: str,
    trial_index: int,
) -> Path:
    directory = param_search_dir(dataset_slug, model_name=model_name, root=PARAM_SEARCH_COMPLETIONS_ROOT)
    return directory / f"trial_{int(trial_index)}.jsonl"

def param_search_eval_trial_path(
    dataset_slug: str,
    *,
    model_name: str,
    trial_index: int,
) -> Path:
    directory = param_search_dir(dataset_slug, model_name=model_name, root=PARAM_SEARCH_EVAL_RESULTS_ROOT)
    return directory / f"trial_{int(trial_index)}.jsonl"


def param_search_scores_trial_path(
    dataset_slug: str,
    *,
    model_name: str,
    trial_index: int,
) -> Path:
    directory = param_search_dir(dataset_slug, model_name=model_name, root=PARAM_SEARCH_SCORES_ROOT)
    return directory / f"trial_{int(trial_index)}.json"


__all__ = [
    "COMPLETIONS_ROOT",
    "EVAL_RESULTS_ROOT",
    "CHECK_RESULTS_ROOT",
    "CONSOLE_LOG_ROOT",
    "SCORES_ROOT",
    "PARAM_SEARCH_ROOT",
    "PARAM_SEARCH_COMPLETIONS_ROOT",
    "PARAM_SEARCH_EVAL_RESULTS_ROOT",
    "PARAM_SEARCH_SCORES_ROOT",
    "ensure_results_structure",
    "jsonl_path",
    "scores_path",
    "eval_details_path",
    "check_details_path",
    "param_search_dir",
    "param_search_completion_trial_path",
    "param_search_eval_trial_path",
    "param_search_scores_trial_path",
    "write_scores_json",
    "make_scores_payload",
    "write_scores_json_to_path",
]


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


def make_scores_payload(
    dataset_slug: str,
    *,
    is_cot: bool,
    model_name: str,
    metrics: dict,
    samples: int,
    problems: int | None = None,
    log_path: Path | str,
    task: str | None = None,
    task_details: dict | None = None,
    extra: dict | None = None,
) -> dict:
    ensure_results_structure()
    payload = {
        "dataset": dataset_slug,
        "model": model_name,
        "cot": bool(is_cot),
        "metrics": _normalize_jsonable(metrics),
        "samples": int(samples),
        "created_at": datetime.utcnow().replace(microsecond=False).isoformat() + "Z",
        "log_path": str(log_path),
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


def write_scores_json_to_path(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def write_scores_json(
    dataset_slug: str,
    *,
    is_cot: bool,
    model_name: str,
    metrics: dict,
    samples: int,
    problems: int | None = None,
    log_path: Path | str,
    task: str | None = None,
    task_details: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist aggregated metrics as JSON in the canonical scores directory."""

    ensure_results_structure()
    path = scores_path(dataset_slug, is_cot=is_cot, model_name=model_name)
    payload = make_scores_payload(
        dataset_slug,
        is_cot=is_cot,
        model_name=model_name,
        metrics=metrics,
        samples=samples,
        problems=problems,
        log_path=log_path,
        task=task,
        task_details=task_details,
        extra=extra,
    )
    return write_scores_json_to_path(path, payload)
