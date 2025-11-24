from __future__ import annotations

import json
from pathlib import Path
from typing import List
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

DATA_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"


def _records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("svamp 仅提供 test split")
    cache_dir = dataset_cache_dir(Path("data"), "svamp")
    source_path = cache_dir / "SVAMP.json"
    download_file(DATA_URL, source_path)

    with source_path.open("r", encoding="utf-8") as handle:
        original = json.load(handle)
        for entry in original:
            answer = entry["Answer"]
            if isinstance(answer, (int, float)) and int(answer) == answer:
                answer = int(answer)
            yield {
                "problem": entry["Body"].rstrip(".") + ". " + entry["Question"],
                "expected_answer": answer,
                "reference_equation": entry["Equation"],
            }


@FREE_ANSWER_REGISTRY.register("svamp")
def prepare_svamp(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "svamp"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _records(split))
    return [target]
