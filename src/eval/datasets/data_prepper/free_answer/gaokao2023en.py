from __future__ import annotations

from pathlib import Path
from typing import List

from ..data_utils import load_qwen_dataset, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY


@FREE_ANSWER_REGISTRY.register("gaokao2023en")
def prepare_gaokao2023en(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "gaokao2023en"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    rows = load_qwen_dataset("gaokao2023en", split)
    write_jsonl(target, rows)
    return [target]
