from __future__ import annotations

from pathlib import Path
from typing import List

from ..data_utils import load_qwen_dataset, write_jsonl
from src.dataset.data_prepper.prepper_registry import FREE_ANSWER_REGISTRY


def _register(dataset: str, name: str, split: str = "test") -> None:
    @FREE_ANSWER_REGISTRY.register(name)
    def _prepare(output_root: Path, split: str = "test") -> list[Path]:  # type: ignore[override]
        dataset_dir = output_root / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target = dataset_dir / f"{split}.jsonl"
        rows = load_qwen_dataset(dataset, split)
        write_jsonl(target, rows)
        return [target]


_register("math", "hendrycks_math")
_register("college_math", "college_math")
_register("minerva_math", "minerva_math")
_register("olympiadbench", "olympiadbench")
_register("amc23", "amc23")
