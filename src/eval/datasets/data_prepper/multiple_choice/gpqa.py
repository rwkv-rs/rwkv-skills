from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List
from collections.abc import Iterable

try:
    from datasets import load_dataset  # type: ignore
    from datasets.exceptions import DatasetNotFoundError  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore
    DatasetNotFoundError = RuntimeError  # type: ignore

from ..data_utils import configure_hf_home, write_jsonl
from src.dataset.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY


def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())


def _iter_records(split: str, seed: int) -> Iterable[dict]:
    configure_hf_home()
    if split not in {"extended", "main", "diamond"}:
        raise ValueError("gpqa 支持 split: extended/main/diamond")
    if load_dataset is None:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 gpqa 数据集，请运行 `pip install datasets`"
        )
    try:
        dataset = load_dataset("Idavidrein/gpqa", f"gpqa_{split}")["train"]
    except DatasetNotFoundError as exc:
        raise RuntimeError(
            "GPQA 数据集为 Hugging Face 限制访问资源，请先在 https://huggingface.co/datasets/Idavidrein/gpqa "
            "页面申请并通过授权后，再运行数据准备脚本。"
        ) from exc
    rng = random.Random(seed)
    for row in dataset:
        choices = [
            _norm(row.get("Incorrect Answer 1")),
            _norm(row.get("Incorrect Answer 2")),
            _norm(row.get("Incorrect Answer 3")),
            _norm(row.get("Correct Answer")),
        ]
        correct = choices[-1]
        rng.shuffle(choices)
        correct_letter = chr(ord("A") + choices.index(correct))
        payload: dict[str, object] = {
            "question": _norm(row.get("Question")),
            "answer": correct_letter,
            "subject": row.get("Subdomain"),
            "subset": row.get("Subdomain"),
            "difficulty": row.get("Writer's Difficulty Estimate"),
            "explanation": _norm(row.get("Explanation")),
            "source": "gpqa",
        }
        for idx, choice in enumerate(choices):
            payload[chr(ord("A") + idx)] = choice
        yield payload


@MULTIPLE_CHOICE_REGISTRY.register("gpqa")
def prepare_gpqa(output_root: Path, split: str = "main", seed: int = 42) -> list[Path]:
    dataset_dir = output_root / "gpqa"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split, seed))
    return [target]
