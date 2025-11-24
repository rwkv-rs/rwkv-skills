from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

from ..data_utils import configure_hf_home, write_jsonl
from src.dataset.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY

_DATASET_ID = "ceval/ceval-exam"
_ALLOWED_SPLITS: dict[str, str] = {
    "test": "test",
    "val": "val",
    "validation": "val",
    "dev": "dev",
}


def _iter_records(split: str) -> Iterable[dict]:
    configure_hf_home(Path("data/hf_cache"))
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 ceval-exam 数据集，请运行 `pip install datasets`"
        ) from exc

    split_key = _ALLOWED_SPLITS.get(split.lower())
    if split_key is None:
        raise ValueError(f"ceval 仅支持 split: {', '.join(sorted(_ALLOWED_SPLITS))}")

    for config in get_dataset_config_names(_DATASET_ID):
        dataset = load_dataset(_DATASET_ID, config, split=split_key)
        for row in dataset:
            yield {
                "question": row.get("question", "").strip(),
                "A": row.get("A", "").strip(),
                "B": row.get("B", "").strip(),
                "C": row.get("C", "").strip(),
                "D": row.get("D", "").strip(),
                "answer": str(row.get("answer", "")).strip().upper(),
                "subject": config,
                "dataset": "ceval",
                "id": row.get("id"),
                "explanation": row.get("explanation", ""),
            }


@MULTIPLE_CHOICE_REGISTRY.register("ceval")
def prepare_ceval(output_root: Path, split: str = "test") -> list[Path]:  # type: ignore[override]
    dataset_dir = output_root / "ceval"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]
