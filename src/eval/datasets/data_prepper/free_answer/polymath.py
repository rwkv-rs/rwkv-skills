from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec

_DATASET_ID = "Qwen/PolyMath"
_ALLOWED_SPLITS = ("all", "top", "high", "medium", "low")


def _polymath_config_names() -> list[str]:
    configure_hf_home()
    from datasets import get_dataset_config_names  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

    return sorted(name for name in get_dataset_config_names(_DATASET_ID) if name and name != "default")


def _polymath_source_splits(split: str) -> tuple[str, ...]:
    if split == "all":
        return ("top", "high", "medium", "low")
    if split in _ALLOWED_SPLITS:
        return (split,)
    raise ValueError("polymath 仅支持 all/top/high/medium/low split")


def _load_polymath_rows(config: str, split: str) -> Iterable[Mapping[str, Any]]:
    configure_hf_home()
    from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

    return load_dataset(_DATASET_ID, config, split=split)


def _iter_records(split: str) -> Iterable[dict[str, Any]]:
    for config in _polymath_config_names():
        for source_split in _polymath_source_splits(split):
            for index, row in enumerate(_load_polymath_rows(config, source_split)):
                payload: dict[str, Any] = {
                    "id": row.get("id") or f"{config}_{source_split}_{index}",
                    "problem": str(row.get("question") or row.get("problem") or ""),
                    "expected_answer": str(row.get("answer") or row.get("expected_answer") or ""),
                    "language": config,
                    "difficulty": source_split,
                    "source": "polymath",
                }
                for key in ("solution", "explanation", "subject", "topic"):
                    value = row.get(key)
                    if value is not None:
                        payload[key] = value
                yield payload


@FREE_ANSWER_REGISTRY.register_spec("polymath")
def prepare_polymath_spec(output_root: Path, split: str = "all") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "polymath",
        output_root,
        split,
        load_rows=_iter_records,
        source_kind="hf_load_dataset",
        manifest_extra_factory=lambda resolved_split: {
            "dataset_id": _DATASET_ID,
            "source_split": resolved_split,
        },
    )


__all__ = ["prepare_polymath_spec"]
