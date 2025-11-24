from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

from ..data_utils import configure_hf_home, write_jsonl
from src.dataset.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY


def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())


def _iter_records(split: str, seed: int) -> Iterable[dict]:
    configure_hf_home()
    if load_dataset is None:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 supergpqa 数据集，请运行 `pip install datasets`"
        )
    dataset = load_dataset("m-a-p/SuperGPQA")["train"]
    if split == "science":
        dataset = dataset.filter(lambda entry: entry.get("discipline") == "Science")
    elif split != "test":
        raise ValueError("supergpqa 支持 split: test 或 science")

    rng = random.Random(seed)
    for row in dataset:
        choices = [_norm(option) for option in row.get("options", [])]
        if len(choices) < 4:
            raise ValueError("SuperGPQA 至少需要四个选项")
        answer_letter = str(row.get("answer_letter", "")).strip().upper()
        if not answer_letter:
            raise ValueError("缺少 answer_letter 字段")
        answer_idx = ord(answer_letter) - ord("A")
        if not (0 <= answer_idx < len(choices)):
            raise ValueError("answer_letter 超出范围")
        correct_choice = choices[answer_idx]
        rng.shuffle(choices)
        answer_letter = chr(ord("A") + choices.index(correct_choice))

        payload: dict[str, object] = {
            "question": _norm(row.get("question")),
            "answer": answer_letter,
            "subject": row.get("discipline"),
            "subset": row.get("discipline"),
            "uuid": row.get("uuid"),
            "field": row.get("field"),
            "subfield": row.get("subfield"),
            "difficulty": row.get("difficulty"),
            "is_calculation": row.get("is_calculation"),
        }
        for idx, choice in enumerate(choices):
            payload[chr(ord("A") + idx)] = choice
        yield payload


@MULTIPLE_CHOICE_REGISTRY.register("supergpqa")
def prepare_supergpqa(output_root: Path, split: str = "test", seed: int = 42) -> list[Path]:
    dataset_dir = output_root / "supergpqa"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split, seed))
    return [target]
