from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling import (
    BrowseCompRecord,
    load_browsecomp_rows_from_csv,
    load_browsecomp_zh_rows_from_xlsx,
)

from .common import LocalRowsDatasetSpec, rwkv_rs_datasets_root

_REQUIRED_FIELDS = ("task_id", "question", "answer", "locale")


def _rows_from_records(records: list[BrowseCompRecord], *, source: Path) -> list[dict[str, str]]:
    return [
        {
            "task_id": record.task_id,
            "question": record.question,
            "answer": record.answer,
            "topic": record.topic or "",
            "locale": record.locale,
            "source_path": str(source),
        }
        for record in records
    ]


def _build_browsecomp_spec(
    *,
    dataset_name: str,
    output_root: Path,
    split: str,
    source: Path,
    load_records: Callable[[str | Path], list[BrowseCompRecord]],
) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    def _load() -> list[dict[str, str]]:
        return _rows_from_records(load_records(source), source=source)

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="rwkv_rs_local_manifest",
        required_paths=(source,),
        load_local_records=_load,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("browsecomp")
def prepare_browsecomp_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    source = rwkv_rs_datasets_root() / "browsecomp" / "browse_comp_test_set.csv"
    return _build_browsecomp_spec(
        dataset_name="browsecomp",
        output_root=output_root,
        split=split,
        source=source,
        load_records=load_browsecomp_rows_from_csv,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("browsecomp_zh")
def prepare_browsecomp_zh_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    source = rwkv_rs_datasets_root() / "browsecomp_zh" / "browsecomp-zh-encrypted.xlsx"
    return _build_browsecomp_spec(
        dataset_name="browsecomp_zh",
        output_root=output_root,
        split=split,
        source=source,
        load_records=load_browsecomp_zh_rows_from_xlsx,
    )


__all__ = [
    "prepare_browsecomp_spec",
    "prepare_browsecomp_zh_spec",
]
