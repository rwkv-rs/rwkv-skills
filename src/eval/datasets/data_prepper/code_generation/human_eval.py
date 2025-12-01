from __future__ import annotations

"""Prepare HumanEval dataset (downloads from upstream repo and writes plain JSONL)."""

import gzip
import json
from pathlib import Path
from typing import Iterator

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY

DATA_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def _records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("human_eval 仅提供 test split")
    cache_dir = dataset_cache_dir(Path("data"), "human_eval")
    source_path = cache_dir / "HumanEval.jsonl.gz"
    download_file(DATA_URL, source_path)

    with gzip.open(source_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            payload = json.loads(line)
            yield {
                "task_id": payload["task_id"],
                "prompt": payload["prompt"],
                "canonical_solution": payload.get("canonical_solution"),
                "test": payload.get("test"),
                "entry_point": payload.get("entry_point"),
            }


@CODE_GENERATION_REGISTRY.register("human_eval")
def prepare_human_eval(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "human_eval"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _records(split))
    return [target]
