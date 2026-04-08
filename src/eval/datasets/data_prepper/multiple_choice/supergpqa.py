from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_DATASET_ID = "m-a-p/SuperGPQA"
_DEFAULT_SEED = 42
_REQUIRED_FIELDS = ("question", "answer", "A", "B", "C", "D")

def _norm(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())


def _load_supergpqa_rows(split: str) -> Iterable[Mapping[str, Any]]:
    configure_hf_home()
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 supergpqa 数据集，请运行 `pip install datasets`"
        ) from exc
    dataset = load_dataset(_DATASET_ID)["train"]
    if split == "science":
        return dataset.filter(lambda entry: entry.get("discipline") == "Science")
    elif split != "test":
        raise ValueError("supergpqa 支持 split: test 或 science")
    return dataset


def _iter_records(split: str, seed: int) -> Iterable[dict[str, Any]]:
    rng = random.Random(seed)
    for row in _load_supergpqa_rows(split):
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


class SuperGpqaDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str, *, seed: int = _DEFAULT_SEED) -> None:
        super().__init__("supergpqa", output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="hf_load_dataset")
        self._seed = seed

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(_iter_records(self.split, self._seed))

    def manifest_extra(self) -> dict[str, Any]:
        return {"dataset_id": _DATASET_ID, "source_split": self.split, "seed": self._seed}


@MULTIPLE_CHOICE_REGISTRY.register_spec("supergpqa")
def prepare_supergpqa_spec(output_root: Path, split: str = "test") -> SuperGpqaDatasetSpec:
    return SuperGpqaDatasetSpec(output_root, split)


__all__ = ["prepare_supergpqa_spec"]
