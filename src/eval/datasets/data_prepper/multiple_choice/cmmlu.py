from __future__ import annotations

"""Prepare CMMLU (Chinese MMLU) from lmlmcat/cmmlu release."""

import csv
import io
import zipfile
from pathlib import Path
from typing import Any
from collections.abc import Iterable

from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import UrlDownloadFile, UrlFilesJsonlDatasetSpec

DATA_URL = "https://hf-mirror.com/datasets/lmlmcat/cmmlu/resolve/main/cmmlu_v1_0_1.zip"
_CHOICE_LETTERS = ("A", "B", "C", "D")
_REQUIRED_FIELDS = ("question", "answer")


def _iter_rows(zip_path: Path, split: str) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix = f"{split}/"
        for member in sorted(name for name in zf.namelist() if name.startswith(prefix) and name.endswith(".csv")):
            subject = Path(member).stem
            with zf.open(member, "r") as handle:
                reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig"))
                for row in reader:
                    question = (row.get("Question") or "").strip()
                    answer = (row.get("Answer") or "").strip().upper()
                    options = [(row.get(letter) or "").strip() for letter in _CHOICE_LETTERS]
                    if not question or not answer:
                        continue
                    yield {
                        "question": question,
                        "answer": answer,
                        "subject": subject,
                        **{letter: text for letter, text in zip(_CHOICE_LETTERS, options) if text},
                    }


@MULTIPLE_CHOICE_REGISTRY.register_spec("cmmlu")
def prepare_cmmlu_spec(output_root: Path, split: str = "test") -> UrlFilesJsonlDatasetSpec:
    def _load(source_root: Path) -> list[dict[str, Any]]:
        return list(_iter_rows(source_root / "cmmlu_v1_0_1.zip", split))

    return UrlFilesJsonlDatasetSpec(
        "cmmlu",
        output_root,
        split,
        files=(UrlDownloadFile(Path("cmmlu_v1_0_1.zip"), DATA_URL),),
        load_downloaded_records=_load,
        required_fields=_REQUIRED_FIELDS,
        tasks=1,
    )


__all__ = ["prepare_cmmlu_spec"]
