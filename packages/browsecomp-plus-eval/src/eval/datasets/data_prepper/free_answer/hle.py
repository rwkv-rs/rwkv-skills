from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import iter_hf_dataset, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY
from src.eval.datasets.runtime import DatasetManifest, MaterializingDatasetSpec
from src.eval.datasets.runtime.validators import validate_jsonl_file

HLE_CATEGORIES_MAP: dict[str, str] = {
    "Other": "other",
    "Humanities/Social Science": "human",
    "Math": "math",
    "Physics": "phy",
    "Computer Science/AI": "cs",
    "Biology/Medicine": "bio",
    "Chemistry": "chem",
    "Engineering": "eng",
}

HLE_REVERSE_MAP = {v: k for k, v in HLE_CATEGORIES_MAP.items()}
AVAILABLE_SPLITS = ("all", "text", *HLE_REVERSE_MAP.keys())
_REQUIRED_FIELDS = ("id", "problem", "expected_answer")


def _map_hle_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "problem": entry["question"],
        "expected_answer": entry["answer"],
        "answer_type": entry["answer_type"],
        "reference_solution": entry["rationale"],
        "raw_subject": entry["raw_subject"],
        "category": entry["category"],
        "author_name": entry["author_name"],
        "canary": entry["canary"],
    }


def _load_hle_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in iter_hf_dataset("cais/hle", split="test"):
        if entry.get("image"):
            continue
        rows.append(_map_hle_row(entry))
    return rows


def _category_for_split(split: str) -> str | None:
    if split in {"all", "text"}:
        return None
    category = HLE_REVERSE_MAP.get(split)
    if category is None:
        raise ValueError(f"未知的 HLE split: {split}")
    return category


def _filter_hle_rows(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    category = _category_for_split(split)
    if category is None:
        return list(rows)
    return [row for row in rows if row["category"] == category]


class HleDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str) -> None:
        super().__init__("hle", output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="hf_load_dataset")

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        if self.split not in AVAILABLE_SPLITS:
            raise ValueError(f"未知的 HLE split: {self.split}")
        return _load_hle_rows()

    def materialized_paths(self) -> list[Path]:
        if self.split == "all":
            return [self.artifact_dir / f"{name}.jsonl" for name in ("all", "text", *HLE_REVERSE_MAP.keys())]
        return [self.artifact_path]

    def validate_materialized_artifact(self) -> bool:
        try:
            for path in self.materialized_paths():
                validate_jsonl_file(path, self.required_fields)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return False
        return True

    def len(self) -> int:
        return len(_filter_hle_rows(self._records, split=self.split))

    def iter_records(self) -> Iterable[dict[str, Any]]:
        return _filter_hle_rows(self._records, split=self.split)

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "dataset_id": "cais/hle",
            "source_split": "test",
            "produced_splits": [path.stem for path in self.materialized_paths()],
        }

    def materialize(self) -> list[Path]:
        paths = self.materialized_paths()
        split_counts: dict[str, int] = {}
        for path in paths:
            split_name = path.stem
            rows = _filter_hle_rows(self._records, split=split_name)
            write_jsonl(path, rows)
            split_counts[split_name] = len(rows)
        manifest = DatasetManifest(
            dataset=self.name,
            split=self.split,
            row_count=split_counts.get(self.split, len(_filter_hle_rows(self._records, split=self.split))),
            source_kind=self.source_kind,
            artifact_path=str(self.artifact_path),
            cache_dir=str(self.cache_dir),
            prepared_at=datetime.now(UTC).isoformat(),
            extra={**self.manifest_extra(), "split_row_counts": split_counts},
        )
        self.manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return paths


@FREE_ANSWER_REGISTRY.register_spec("hle")
def prepare_hle_spec(output_root: Path, split: str = "all") -> HleDatasetSpec:
    return HleDatasetSpec(output_root, split)


__all__ = ["prepare_hle_spec"]
