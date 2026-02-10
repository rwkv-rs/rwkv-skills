from __future__ import annotations

"""Helper to resolve dataset arguments: existing path -> OK, otherwise auto-prepare.

This is intended for direct bin scripts so users don't have to pre-run prepare_dataset.
"""

from pathlib import Path

from .datasets import DATA_OUTPUT_ROOT, DATASET_ROOTS
from .dataset_utils import canonical_slug, infer_dataset_slug_from_path
from .dataset_stats import record_dataset_samples
from .jobs import locate_dataset


def resolve_or_prepare_dataset(dataset_arg: str, *, verbose: bool = True) -> Path:
    """Resolve dataset path, auto-preparing via registered preparers.

    Missing/failed prepare will raise so callers can fail fast.
    """

    candidate = Path(dataset_arg).expanduser()
    if candidate.exists():
        record_dataset_samples(candidate)
        return candidate

    slug = canonical_slug(infer_dataset_slug_from_path(dataset_arg))
    try:
        return locate_dataset(slug, search=DATASET_ROOTS, output_root=DATA_OUTPUT_ROOT)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"❌ 自动准备数据集失败：{slug} ({exc})")
        raise


__all__ = ["resolve_or_prepare_dataset"]
