from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

from ..data_utils import configure_hf_home, write_jsonl
from src.dataset.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY


def _iter_records(split: str) -> Iterable[dict]:
    if split not in {"validation", "test"}:
        raise ValueError("mmlu-pro 仅支持 validation 与 test split")
    configure_hf_home()
    if load_dataset is None:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 mmlu-pro 数据集，请运行 `pip install datasets`"
        )
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    for row in dataset:
        category = row.get("category", "").replace(" ", "_")
        choices = [choice.strip() for choice in row.get("options", [])]
        answer_letter = str(row.get("answer", "")).strip().upper()
        if not answer_letter:
            raise ValueError("缺少答案字段")
        answer_idx = ord(answer_letter) - ord("A")
        if not (0 <= answer_idx < len(choices)):
            raise ValueError(f"答案索引越界: {answer_letter}")
        payload: dict[str, object] = {
            "question": row.get("question", "").strip(),
            "answer": answer_letter,
            "subject": category,
            "subset": category,
            "examples_type": f"mmlu_pro_few_shot_{category}",
        }
        for idx, choice in enumerate(choices):
            payload[chr(ord("A") + idx)] = choice
        yield payload


@MULTIPLE_CHOICE_REGISTRY.register("mmlu-pro")
def prepare_mmlu_pro(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "mmlu-pro"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]
