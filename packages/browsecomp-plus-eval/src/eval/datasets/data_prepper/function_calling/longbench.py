from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.data_utils import iter_hf_dataset
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.tasks.function_calling.longbench import (
    LONG_BENCH_DATASETS,
    LONG_BENCH_QA_DATASETS,
    load_longbench_rows_from_source,
    normalize_longbench_manifest_row,
)

from .common import LocalRowsDatasetSpec, rwkv_rs_datasets_root

_REQUIRED_FIELDS = ("task_id", "dataset", "input", "context", "answers")
_LONGBENCH_HF_SOURCE = "THUDM/LongBench"


def longbench_root() -> Path:
    override = os.environ.get("RWKV_LONGBENCH_ROOT") or os.environ.get("LONGBENCH_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return (rwkv_rs_datasets_root() / "longbench").resolve()


def _build_longbench_spec(
    *,
    dataset_name: str,
    output_root: Path,
    split: str,
    include_datasets: set[str] | None = None,
    balance_by_dataset: bool = False,
    extra: dict[str, Any] | None = None,
) -> LocalRowsDatasetSpec:
    configured_source = longbench_root()

    def _load() -> list[dict[str, Any]]:
        source = longbench_root()
        if source.exists():
            rows = load_longbench_rows_from_source(source, split=split, include_datasets=include_datasets)
        else:
            rows = _load_hf_longbench_rows(split=split, include_datasets=include_datasets)
        return _round_robin_by_dataset(rows) if balance_by_dataset else rows

    def _required_paths() -> tuple[Path, ...]:
        source = longbench_root()
        return (source,) if source.exists() else ()

    payload_extra = {
        "benchmark": "LongBench",
        "official_source": _LONGBENCH_HF_SOURCE,
        "source": "local_or_huggingface",
        "source_mode": "local_root" if configured_source.exists() else "huggingface",
        "default_local_root": str(configured_source),
        **(extra or {}),
    }
    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="local_or_hf_longbench_manifest",
        required_paths=_required_paths,
        load_local_records=_load,
        extra=payload_extra,
    )


def _load_hf_longbench_rows(*, split: str, include_datasets: set[str] | None) -> list[dict[str, Any]]:
    selected = sorted(include_datasets or LONG_BENCH_DATASETS)
    rows: list[dict[str, Any]] = []
    for dataset in selected:
        source_path = f"hf://{_LONGBENCH_HF_SOURCE}/{dataset}/{split}"
        for index, row in enumerate(
            iter_hf_dataset(_LONGBENCH_HF_SOURCE, config=dataset, split=split, trust_remote_code=True)
        ):
            payload = dict(row)
            payload.setdefault("dataset", dataset)
            rows.append(
                normalize_longbench_manifest_row(
                    payload,
                    fallback_index=index,
                    source_path=source_path,
                )
            )
    if not rows:
        raise FileNotFoundError(f"no LongBench rows found from THUDM/LongBench split={split}")
    return rows


def _round_robin_by_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("dataset") or "unknown"), []).append(row)

    ordered: list[dict[str, Any]] = []
    dataset_names = sorted(buckets)
    index = 0
    while True:
        added = False
        for dataset in dataset_names:
            bucket = buckets[dataset]
            if index < len(bucket):
                ordered.append(bucket[index])
                added = True
        if not added:
            return ordered
        index += 1


@FUNCTION_CALLING_REGISTRY.register_spec("longbench")
def prepare_longbench_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _build_longbench_spec(
        dataset_name="longbench",
        output_root=output_root,
        split=split,
        extra={"subset_filter": "all"},
    )


@FUNCTION_CALLING_REGISTRY.register_spec("longbench_qa")
def prepare_longbench_qa_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _build_longbench_spec(
        dataset_name="longbench_qa",
        output_root=output_root,
        split=split,
        include_datasets=set(LONG_BENCH_QA_DATASETS),
        extra={"subset_filter": "qa"},
    )


@FUNCTION_CALLING_REGISTRY.register_spec("longbench_qa_balanced")
def prepare_longbench_qa_balanced_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _build_longbench_spec(
        dataset_name="longbench_qa_balanced",
        output_root=output_root,
        split=split,
        include_datasets=set(LONG_BENCH_QA_DATASETS),
        balance_by_dataset=True,
        extra={"subset_filter": "qa", "ordering": "round_robin_by_dataset"},
    )


__all__ = [
    "longbench_root",
    "prepare_longbench_qa_balanced_spec",
    "prepare_longbench_qa_spec",
    "prepare_longbench_spec",
]
