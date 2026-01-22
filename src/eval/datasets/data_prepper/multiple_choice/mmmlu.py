from __future__ import annotations

"""Prepare MMMLU (OpenAI multilingual MMLU; 14 non-English languages).

This writes a single combined JSONL under `data/mmmlu/test.jsonl` that contains
all selected languages + all MMLU subjects.
"""

import os
from pathlib import Path
from typing import Iterable

try:
    from datasets import load_dataset  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

from ..data_utils import configure_hf_home, write_jsonl
from .mmlu import _SUBCATEGORIES as _MMLU_SUBCATEGORIES
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY


_DATASET_ID = "giuliolovisotto/openai_multilingual_mmlu"

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


def _iter_records(split: str) -> Iterable[dict]:
    if split != "test":
        raise ValueError("mmmlu 目前仅提供 test split")
    if load_dataset is None:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError("需要安装 `datasets` 才能准备 mmmlu 数据集，请运行 `pip install datasets`")

    configure_hf_home()
    language_splits = _parse_language_splits()
    subjects = tuple(sorted(_MMLU_SUBCATEGORIES.keys()))

    for subject in subjects:
        dataset_by_lang = load_dataset(_DATASET_ID, subject)
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


@MULTIPLE_CHOICE_REGISTRY.register("mmmlu")
def prepare_mmmlu(output_root: Path, split: str = "test") -> list[Path]:
    dataset_dir = output_root / "mmmlu"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / f"{split}.jsonl"
    write_jsonl(target, _iter_records(split))
    return [target]


__all__ = ["prepare_mmmlu"]
