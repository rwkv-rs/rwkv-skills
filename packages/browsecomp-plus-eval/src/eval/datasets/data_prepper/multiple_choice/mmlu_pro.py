from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_DATASET_ID = "TIGER-Lab/MMLU-Pro"
_ALLOWED_SPLITS = {"validation", "test"}
_REQUIRED_FIELDS = ("question", "answer", "A", "B", "C", "D")

def _load_mmlu_pro_rows(split: str) -> Iterable[Mapping[str, Any]]:
    if split not in _ALLOWED_SPLITS:
        raise ValueError("mmlu-pro 仅支持 validation 与 test split")
    configure_hf_home()
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 mmlu-pro 数据集，请运行 `pip install datasets`"
        ) from exc
    return load_dataset(_DATASET_ID, split=split)


def _iter_records(split: str) -> Iterable[dict[str, Any]]:
    for row in _load_mmlu_pro_rows(split):
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


class MmluProDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str) -> None:
        super().__init__("mmlu-pro", output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="hf_load_dataset")

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(_iter_records(self.split))

    def manifest_extra(self) -> dict[str, Any]:
        return {"dataset_id": _DATASET_ID, "source_split": self.split}


@MULTIPLE_CHOICE_REGISTRY.register_spec("mmlu_pro")
def prepare_mmlu_pro_spec(output_root: Path, split: str = "test") -> MmluProDatasetSpec:
    return MmluProDatasetSpec(output_root, split)


__all__ = ["prepare_mmlu_pro_spec"]
