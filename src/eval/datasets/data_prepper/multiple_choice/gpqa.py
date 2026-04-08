from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_DATASET_ID = "Idavidrein/gpqa"
_ALLOWED_SPLITS = {"extended", "main", "diamond"}
_DEFAULT_SEED = 42
_REQUIRED_FIELDS = ("question", "answer", "A", "B", "C", "D")

def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())


def _load_gpqa_rows(split: str) -> Iterable[Mapping[str, Any]]:
    configure_hf_home()
    if split not in _ALLOWED_SPLITS:
        raise ValueError("gpqa 支持 split: extended/main/diamond")
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 gpqa 数据集，请运行 `pip install datasets`"
        ) from exc
    try:
        dataset = load_dataset(_DATASET_ID, f"gpqa_{split}")["train"]
    except Exception as exc:
        if exc.__class__.__name__ == "DatasetNotFoundError":
            raise RuntimeError(
                "GPQA 数据集为 Hugging Face 限制访问资源，请先在 https://huggingface.co/datasets/Idavidrein/gpqa "
                "页面申请并通过授权后，再运行数据准备脚本。"
            ) from exc
        raise
    return dataset


def _iter_records(split: str, seed: int) -> Iterable[dict[str, object]]:
    rng = random.Random(seed)
    for row in _load_gpqa_rows(split):
        choices = [
            _norm(str(row.get("Incorrect Answer 1") or "")),
            _norm(str(row.get("Incorrect Answer 2") or "")),
            _norm(str(row.get("Incorrect Answer 3") or "")),
            _norm(str(row.get("Correct Answer") or "")),
        ]
        correct = choices[-1]
        rng.shuffle(choices)
        correct_letter = chr(ord("A") + choices.index(correct))
        payload: dict[str, object] = {
            "question": _norm(str(row.get("Question") or "")),
            "answer": correct_letter,
            "subject": row.get("Subdomain"),
            "subset": row.get("Subdomain"),
            "difficulty": row.get("Writer's Difficulty Estimate"),
            "explanation": _norm(str(row.get("Explanation") or "")),
            "source": "gpqa",
        }
        for idx, choice in enumerate(choices):
            payload[chr(ord("A") + idx)] = choice
        yield payload


class GpqaDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str, *, seed: int = _DEFAULT_SEED) -> None:
        super().__init__("gpqa", output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="hf_load_dataset")
        self._seed = seed

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(_iter_records(self.split, self._seed))

    def manifest_extra(self) -> dict[str, Any]:
        return {"dataset_id": _DATASET_ID, "source_split": self.split, "seed": self._seed}


@MULTIPLE_CHOICE_REGISTRY.register_spec("gpqa")
def prepare_gpqa_spec(output_root: Path, split: str = "main") -> GpqaDatasetSpec:
    return GpqaDatasetSpec(output_root, split)


__all__ = ["prepare_gpqa_spec"]
