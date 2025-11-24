from __future__ import annotations

import json
from pathlib import Path
from typing import List
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

DATA_URL = "https://github.com/openai/prm800k/raw/main/prm800k/math_splits/test.jsonl"


def _records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("math-500 仅提供 test split")
    cache_dir = dataset_cache_dir(Path("data"), "math_500")
    source_path = cache_dir / "math_500_test.jsonl"
    download_file(DATA_URL, source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            payload["expected_answer"] = payload.pop("answer")
            payload["reference_solution"] = payload.pop("solution")
            yield payload


@FREE_ANSWER_REGISTRY.register("math-500")
def prepare_math_500(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "math-500"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _records(split))
    return [target]
