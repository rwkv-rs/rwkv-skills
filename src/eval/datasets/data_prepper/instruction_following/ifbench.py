from __future__ import annotations

import contextlib
from pathlib import Path
from typing import List
from collections.abc import Iterable

from ..data_utils import dataset_cache_dir, download_file, read_jsonl, write_jsonl
from src.dataset.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY

DATA_URL = "https://raw.githubusercontent.com/allenai/IFBench/refs/heads/main/data/IFBench_test.jsonl"


def _iter_records(split: str) -> Iterable[dict]:
    if split != "test":
        raise ValueError("ifbench 仅提供 test split")

    cache_dir = dataset_cache_dir(Path("data"), "ifbench")
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


@INSTRUCTION_FOLLOWING_REGISTRY.register("ifbench")
def prepare_ifbench(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "ifbench"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]


__all__ = ["prepare_ifbench"]
