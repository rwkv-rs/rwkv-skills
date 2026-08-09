from __future__ import annotations

"""Dataset discovery helpers shared by queue construction & dispatch."""

from pathlib import Path
from typing import Sequence

from .config import REPO_ROOT
from .dataset_utils import canonical_slug, infer_dataset_slug_from_path


DATASET_ROOTS: list[Path] = [
    (REPO_ROOT / "data").resolve(),
]

DATA_OUTPUT_ROOT = DATASET_ROOTS[0]

_dataset_index: dict[str, Path] = {}
_dataset_index_stale = True
_SKIP_INDEX_DIR_NAMES = {"cache", "__pycache__"}


def _is_indexable_dataset_path(candidate: Path, root: Path) -> bool:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        relative = candidate
    return not any(part.startswith(".") or part in _SKIP_INDEX_DIR_NAMES for part in relative.parts)


def refresh_dataset_index(roots: Sequence[Path] | None = None) -> None:
    global _dataset_index, _dataset_index_stale
    index: dict[str, Path] = {}
    search_roots = roots if roots is not None else DATASET_ROOTS
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.rglob("*.jsonl"):
            if not _is_indexable_dataset_path(candidate, root):
                continue
            slug = infer_dataset_slug_from_path(str(candidate))
            resolved = candidate.resolve()
            previous = index.get(slug)
            if previous is None:
                index[slug] = resolved
            elif previous.stem == "input_data" and resolved.stem != "input_data":
                index[slug] = resolved
    _dataset_index = index
    _dataset_index_stale = False


def _ensure_dataset_index(roots: Sequence[Path] | None = None) -> None:
    if _dataset_index_stale:
        refresh_dataset_index(roots)


def find_dataset_file(slug: str, roots: Sequence[Path] | None = None) -> Path | None:
    canonical = canonical_slug(slug)
    _ensure_dataset_index(roots)
    path = _dataset_index.get(canonical)
    if path and path.exists():
        return path
    refresh_dataset_index(roots)
    return _dataset_index.get(canonical)


def dataset_search_paths() -> tuple[str, ...]:
    return tuple(str(root) for root in DATASET_ROOTS)


__all__ = [
    "DATASET_ROOTS",
    "DATA_OUTPUT_ROOT",
    "dataset_search_paths",
    "find_dataset_file",
    "refresh_dataset_index",
]
