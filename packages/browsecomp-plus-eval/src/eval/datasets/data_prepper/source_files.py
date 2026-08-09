from __future__ import annotations

"""Shared helpers for benchmark preppers that read user-provided source files.

Sources are resolved per dataset from an env override, a family source root, or
a repo-relative default root, and may be JSON/JSONL files or Hugging Face
dataset ids.
"""

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def per_dataset_source_env(prefix: str, dataset_name: str) -> str:
    suffix = re.sub(r"[^A-Z0-9]+", "_", dataset_name.upper()).strip("_")
    return f"{prefix}_{suffix}"


def resolve_source_path(
    dataset_name: str,
    split: str,
    *,
    env_prefix: str,
    root_envs: Sequence[str],
    default_root: Path,
) -> Path:
    env_path = os.environ.get(per_dataset_source_env(env_prefix, dataset_name))
    if env_path:
        return first_existing_or_default(source_candidates(Path(env_path).expanduser(), dataset_name, split))

    root_raw = next((os.environ[name] for name in root_envs if os.environ.get(name)), None)
    root = Path(root_raw).expanduser() if root_raw else default_root
    return first_existing_or_default(source_candidates(root, dataset_name, split))


def source_candidates(base: Path, dataset_name: str, split: str) -> tuple[Path, ...]:
    if base.suffix.lower() in {".jsonl", ".json"}:
        return (base,)
    return (
        base / dataset_name / f"{split}.jsonl",
        base / dataset_name / f"{split}.json",
        base / dataset_name / "test.jsonl",
        base / dataset_name / "test.json",
        base / f"{dataset_name}_{split}.jsonl",
        base / f"{dataset_name}_{split}.json",
        base / f"{dataset_name}.jsonl",
        base / f"{dataset_name}.json",
    )


def first_existing_or_default(candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def read_source_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path}: JSONL rows must be objects")
                rows.append(payload)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("rows", "examples", "instances", "data", "tasks", "items"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, Mapping)]
        return [payload]
    raise ValueError(f"unsupported benchmark source format: {path}")


def hf_dataset_id(source: str) -> str:
    marker = "huggingface.co/datasets/"
    if marker in source:
        return source.split(marker, 1)[1].strip("/")
    return source.strip()


_HF_SPLIT_FALLBACKS = ("test", "eval", "validation", "train")


def load_hf_rows(source: str, split: str) -> list[Mapping[str, Any]]:
    dataset_id = hf_dataset_id(source)
    if not dataset_id:
        raise FileNotFoundError(f"missing Hugging Face dataset id: {source!r}")
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("Install `datasets` to prepare Hugging Face-backed benchmarks") from exc
    candidates = (split, *(name for name in _HF_SPLIT_FALLBACKS if name != split))
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            dataset = load_dataset(dataset_id, split=candidate)
        except ValueError as exc:  # unknown split name; try the next conventional split
            last_error = exc
            continue
        return [dict(row) for row in dataset]
    raise ValueError(f"no usable split for Hugging Face dataset {dataset_id!r}: {last_error}")


__all__ = [
    "first_existing_or_default",
    "hf_dataset_id",
    "load_hf_rows",
    "per_dataset_source_env",
    "read_source_rows",
    "resolve_source_path",
    "source_candidates",
]
