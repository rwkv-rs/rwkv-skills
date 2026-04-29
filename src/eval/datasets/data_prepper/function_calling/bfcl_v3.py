from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling import load_bfcl_v3_rows_from_source

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "instruction", "tools")


def bfcl_v3_source_root() -> Path:
    override = (
        os.environ.get("RWKV_BFCL_V3_SOURCE")
        or os.environ.get("RWKV_BFCL_V3_ROOT")
        or os.environ.get("BFCL_V3_SOURCE")
        or os.environ.get("BFCL_V3_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    return (Path(__file__).resolve().parents[6] / "gorilla" / "berkeley-function-call-leaderboard").resolve()


def bfcl_v3_source_paths(split: str) -> tuple[Path, ...]:
    root = bfcl_v3_source_root()
    if root.is_file():
        return (root,)

    candidate_roots = [
        root,
        root / "data",
        root / "bfcl_eval" / "data",
        root / "berkeley-function-call-leaderboard",
        root / "berkeley-function-call-leaderboard" / "data",
        root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
    ]
    exact_names = (
        f"bfcl_v3_{split}.jsonl",
        f"bfcl_v3_{split}.json",
        f"{split}.jsonl",
        f"{split}.json",
        "bfcl_v3.jsonl",
        "bfcl_v3.json",
        "multi_turn.jsonl",
        "multi_turn.json",
    )
    for base in candidate_roots:
        for name in exact_names:
            candidate = base / name
            if candidate.is_file():
                return (candidate.resolve(),)

    fuzzy: list[Path] = []
    for base in candidate_roots:
        if not base.exists():
            continue
        for pattern in ("*bfcl*v3*.json", "*bfcl*v3*.jsonl", "*multi*turn*.json", "*multi*turn*.jsonl"):
            fuzzy.extend(sorted(base.rglob(pattern)))
    deduped = tuple(dict.fromkeys(path.resolve() for path in fuzzy if path.is_file()))
    if deduped:
        return deduped
    raise FileNotFoundError(
        f"could not locate BFCL V3 source under {root}; set RWKV_BFCL_V3_SOURCE or RWKV_BFCL_V3_ROOT"
    )


def bfcl_v3_source_path(split: str) -> Path:
    paths = bfcl_v3_source_paths(split)
    if len(paths) != 1:
        joined = ", ".join(str(path) for path in paths[:5])
        raise FileNotFoundError(
            f"multiple BFCL V3 source files matched; use bfcl_v3_source_paths() or set RWKV_BFCL_V3_SOURCE explicitly. Matches: {joined}"
        )
    return paths[0]


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_v3")
def prepare_bfcl_v3_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError("bfcl_v3 仅提供 test split")

    def _load() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source in bfcl_v3_source_paths(split):
            rows.extend(load_bfcl_v3_rows_from_source(source))
        return rows

    return LocalRowsDatasetSpec(
        "bfcl_v3",
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="local_bfcl_v3_source",
        required_paths=lambda: bfcl_v3_source_paths(split),
        load_local_records=_load,
    )


__all__ = [
    "bfcl_v3_source_path",
    "bfcl_v3_source_paths",
    "bfcl_v3_source_root",
    "prepare_bfcl_v3_spec",
]
