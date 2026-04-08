from __future__ import annotations

"""Prepare MMMLU (OpenAI multilingual MMLU; 14 non-English languages).

This writes a single combined JSONL under `data/mmmlu/test.jsonl` that contains
all selected languages + all MMLU subjects.
"""

import os
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from .mmlu import _SUBCATEGORIES as _MMLU_SUBCATEGORIES
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec


_DATASET_ID = "giuliolovisotto/openai_multilingual_mmlu"
_REQUIRED_FIELDS = ("question", "answer", "A", "B", "C", "D")

# Default to the 14 non-English languages in OpenAI multilingual MMLU.
_DEFAULT_LANGUAGE_SPLITS: tuple[str, ...] = (
    "AR_XY",
    "BN_BD",
    "DE_DE",
    "ES_LA",
    "FR_FR",
    "HI_IN",
    "ID_ID",
    "IT_IT",
    "JA_JP",
    "KO_KR",
    "PT_BR",
    "SW_KE",
    "YO_NG",
    "ZH_CN",
)


def _parse_language_splits() -> tuple[str, ...]:
    raw = os.environ.get("RWKV_SKILLS_MMMLU_LANGS", "").strip()
    if not raw:
        return _DEFAULT_LANGUAGE_SPLITS
    parts = [part.strip() for part in raw.replace(";", ",").split(",")]
    return tuple(part for part in parts if part)


def _load_mmmlu_dataset_by_subject(subject: str) -> Mapping[str, Iterable[Mapping[str, Any]]]:
    configure_hf_home()
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("需要安装 `datasets` 才能准备 mmmlu 数据集，请运行 `pip install datasets`") from exc
    return load_dataset(_DATASET_ID, subject)


def _iter_records(split: str) -> Iterable[dict[str, Any]]:
    if split != "test":
        raise ValueError("mmmlu 目前仅提供 test split")
    language_splits = _parse_language_splits()
    subjects = tuple(sorted(_MMLU_SUBCATEGORIES.keys()))

    for subject in subjects:
        dataset_by_lang = _load_mmmlu_dataset_by_subject(subject)
        for language in language_splits:
            if language not in dataset_by_lang:
                raise KeyError(
                    f"mmmlu missing language split {language!r} for subject {subject!r} (available: {list(dataset_by_lang)})"
                )
            for row in dataset_by_lang[language]:
                question = str(row.get("Question", "") or "").strip()
                option_a = str(row.get("A", "") or "").strip()
                option_b = str(row.get("B", "") or "").strip()
                option_c = str(row.get("C", "") or "").strip()
                option_d = str(row.get("D", "") or "").strip()
                answer = str(row.get("Answer", "") or "").strip().upper()
                if not question or not option_a or not option_b or not option_c or not option_d or not answer:
                    continue
                yield {
                    "question": question,
                    "A": option_a,
                    "B": option_b,
                    "C": option_c,
                    "D": option_d,
                    "answer": answer,
                    "subject": subject,
                    "subset": _MMLU_SUBCATEGORIES.get(subject, "unknown"),
                    "language": language,
                    "source_dataset": _DATASET_ID,
                }


class MmmluDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str) -> None:
        super().__init__("mmmlu", output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="hf_load_dataset")

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(_iter_records(self.split))

    def manifest_extra(self) -> dict[str, Any]:
        return {"dataset_id": _DATASET_ID, "source_split": self.split, "languages": list(_parse_language_splits())}


@MULTIPLE_CHOICE_REGISTRY.register_spec("mmmlu")
def prepare_mmmlu_spec(output_root: Path, split: str = "test") -> MmmluDatasetSpec:
    return MmmluDatasetSpec(output_root, split)


__all__ = ["prepare_mmmlu_spec"]
