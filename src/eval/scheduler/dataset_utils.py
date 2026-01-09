from __future__ import annotations

"""Helpers for dataset slug normalisation compatible with the dispatcher."""

from pathlib import Path
from typing import Collection, Sequence


DATASET_SLUG_ALIASES: dict[str, str] = {
    "math500": "math_500_test",
    "math": "hendrycks_math_test",
    "input_data": "ifeval_test",
    "ceval_exam_test": "ceval_test",
    "mbpp": "mbpp_test",
    "humanevalplus": "human_eval_plus_test",
    "humaneval_plus": "human_eval_plus_test",
    "human_eval+": "human_eval_plus_test",
    "humanevalfix": "human_eval_fix_test",
    "humaneval_cn": "human_eval_cn_test",
    "lcb": "livecodebench_test",
    "mmmlu": "mmmlu_test",
    "cmmlu": "cmmlu_test",
}

_KNOWN_SPLIT_NAMES = {
    "train",
    "test",
    "validation",
    "val",
    "dev",
    "devtest",
    "main",
    "science",
    "verified",
    "all",
    "text",
    # HLE category splits (data/hle/{math,cs,...}.jsonl) should keep the parent prefix.
    "other",
    "human",
    "math",
    "phy",
    "cs",
    "bio",
    "chem",
    "eng",
}


def safe_slug(text: str) -> str:
    slug_chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {".", "_"}:
            slug_chars.append(char)
        else:
            slug_chars.append("_")
    return "".join(slug_chars).replace(".", "_")


def canonical_slug(text: str) -> str:
    slug = safe_slug(text).lower()
    return DATASET_SLUG_ALIASES.get(slug, slug)


def _strip_known_split_suffix(slug: str) -> str:
    for split in sorted(_KNOWN_SPLIT_NAMES, key=len, reverse=True):
        suffix = f"_{split}"
        if slug.endswith(suffix):
            return slug[: -len(suffix)]
    return slug


def split_benchmark_and_split(dataset_slug: str) -> tuple[str, str]:
    """Split a canonical dataset slug into (benchmark_name, dataset_split).

    The split suffix is detected using the same `_KNOWN_SPLIT_NAMES` logic that
    powers benchmark canonicalisation. If no known suffix is found, the split
    is returned as an empty string.
    """
    slug = canonical_slug(dataset_slug)
    # Artifact-only stems (e.g. results/.../xxx__cot.jsonl) may surface here; strip them.
    if slug.endswith("__cot"):
        slug = slug[: -len("__cot")]
    for split in sorted(_KNOWN_SPLIT_NAMES, key=len, reverse=True):
        suffix = f"_{split}"
        if slug.endswith(suffix):
            return slug[: -len(suffix)], split
    return slug, ""


def make_dataset_slug(name: str, split: str) -> str:
    return canonical_slug(f"{name}_{split}")


def infer_dataset_slug_from_path(dataset_path: str) -> str:
    path = Path(dataset_path)
    stem = path.stem
    parent = path.parent.name
    lower_stem = stem.lower()
    # Prefer `parent_stem` when the file name looks like a split (e.g. data/<bench>/<split>.jsonl).
    # Avoid prefixing the top-level data dir so files like data/math.jsonl can still be aliased.
    if lower_stem in _KNOWN_SPLIT_NAMES and parent and parent.lower() != "data":
        candidate = f"{parent}_{stem}"
    else:
        candidate = stem
    return canonical_slug(candidate)


def canonicalize_benchmark_list(
    names: Sequence[str] | None,
    *,
    known_slugs: Collection[str],
) -> tuple[str, ...]:
    """Map user-provided benchmark names (with or without split suffix) to canonical slugs."""

    if not names:
        return tuple()

    known_set = {canonical_slug(slug) for slug in known_slugs}
    base_map: dict[str, str] = {}
    for slug in known_set:
        base = _strip_known_split_suffix(slug)
        base_map.setdefault(base, slug)

    resolved: set[str] = set()
    unknown: list[str] = []
    for raw in names:
        slug = canonical_slug(raw)
        if slug in known_set:
            resolved.add(slug)
            continue
        base_candidate = _strip_known_split_suffix(slug)
        match = base_map.get(base_candidate)
        if match:
            resolved.add(match)
            continue
        unknown.append(raw)

    if unknown:
        missing = ", ".join(sorted({safe_slug(item) for item in unknown}))
        raise ValueError(f"未知的 benchmark 名称: {missing}")

    return tuple(sorted(resolved))


__all__ = [
    "DATASET_SLUG_ALIASES",
    "canonical_slug",
    "canonicalize_benchmark_list",
    "infer_dataset_slug_from_path",
    "make_dataset_slug",
    "safe_slug",
    "split_benchmark_and_split",
]
