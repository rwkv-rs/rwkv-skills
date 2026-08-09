from __future__ import annotations

import contextlib
from pathlib import Path
from collections.abc import Iterable

from ..data_utils import dataset_cache_dir, download_file, read_jsonl
from src.eval.datasets.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext

DATA_URL = "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl"


def _iter_records(split: str, context: DatasetPrepareContext) -> Iterable[dict]:
    if split != "test":
        raise ValueError("ifbench 仅提供 test split")

    cache_dir = dataset_cache_dir(context.data_root, "ifbench")
    raw_path = cache_dir / "IFBench_test.jsonl"
    download_file(DATA_URL, raw_path)

    for payload in read_jsonl(raw_path):
        record = dict(payload)
        record["question"] = record.get("prompt", "")
        key = record.get("key")
        if isinstance(key, str):
            with contextlib.suppress(ValueError):
                record["key"] = int(key)
        yield record


@INSTRUCTION_FOLLOWING_REGISTRY.register_spec("ifbench")
def prepare_ifbench_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec("ifbench", output_root, split, load_rows=_iter_records, source_kind="url_download")


__all__ = ["prepare_ifbench_spec"]
