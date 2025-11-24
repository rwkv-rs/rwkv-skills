from __future__ import annotations

import json
from pathlib import Path
from typing import List
from collections.abc import Iterator

from ..data_utils import dataset_cache_dir, download_file, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY

DATA_URL = "https://raw.githubusercontent.com/microsoft/ToRA/main/src/data/mawps/{subset}.jsonl"
SUBSETS = ("addsub", "singleeq", "singleop", "multiarith")


def _records(split: str) -> Iterator[dict]:
    if split != "test":
        raise ValueError("mawps 仅提供 test split")
    cache_dir = dataset_cache_dir(Path("data"), "mawps")
    for subset in SUBSETS:
        source_path = cache_dir / f"{subset}.jsonl"
        download_file(DATA_URL.format(subset=subset), source_path)
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                answer = payload["target"]
                try:
                    numeric = float(answer)
                    if int(numeric) == numeric:
                        answer = int(numeric)
                except Exception:
                    pass
                yield {
                    "problem": payload["input"],
                    "expected_answer": answer,
                    "type": subset,
                }


@FREE_ANSWER_REGISTRY.register("mawps")
def prepare_mawps(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "mawps"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _records(split))
    return [target]
