from __future__ import annotations

"""Helpers to build stable IDs for dispatcher jobs."""

from pathlib import Path
from typing import Union

from .dataset_utils import canonical_slug, safe_slug


ModelReference = Union[Path, str]


def _model_name(model_ref: ModelReference) -> str:
    if isinstance(model_ref, Path):
        return model_ref.stem
    text = str(model_ref).strip()
    if not text:
        return ""
    return Path(text).stem if text.endswith(".pth") or "/" in text or "\\" in text else text


def build_run_slug(model_ref: ModelReference, dataset_slug: str, *, is_cot: bool) -> str:
    dataset_part = canonical_slug(dataset_slug)
    cot_part = "cot" if is_cot else "nocot"
    base = f"{dataset_part}_{cot_part}_{_model_name(model_ref)}"
    return safe_slug(base)


def build_run_log_name(model_ref: ModelReference, dataset_slug: str, *, is_cot: bool) -> Path:
    """Return a stable relative path stem for artifacts of a run."""

    model_part = safe_slug(_model_name(model_ref))
    dataset_part = canonical_slug(dataset_slug)
    stem = f"{dataset_part}__cot" if is_cot else dataset_part
    return Path(model_part) / stem


__all__ = ["build_run_slug", "build_run_log_name"]
