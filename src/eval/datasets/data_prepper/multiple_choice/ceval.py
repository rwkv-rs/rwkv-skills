from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import MaterializingDatasetSpec

_DATASET_ID = "ceval/ceval-exam"
_ALLOWED_SPLITS: dict[str, str] = {
    "test": "test",
    "val": "val",
    "validation": "val",
    "dev": "dev",
}
_REQUIRED_FIELDS = ("question", "answer", "A", "B", "C", "D")


def _configure_ceval_hf_home(hf_root: Path) -> None:
    configure_hf_home(hf_root)


def _load_ceval_config_names() -> list[str]:
    try:
        from datasets import get_dataset_config_names  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 ceval-exam 数据集，请运行 `pip install datasets`"
        ) from exc
    return list(get_dataset_config_names(_DATASET_ID))


def _load_ceval_rows(config: str, split_key: str) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "需要安装 `datasets` 才能准备 ceval-exam 数据集，请运行 `pip install datasets`"
        ) from exc
    return load_dataset(_DATASET_ID, config, split=split_key)


def _iter_records(split: str, *, hf_root: Path) -> Iterable[dict[str, Any]]:
    split_key = _ALLOWED_SPLITS.get(split.lower())
    if split_key is None:
        raise ValueError(f"ceval 仅支持 split: {', '.join(sorted(_ALLOWED_SPLITS))}")

    _configure_ceval_hf_home(hf_root)
    for config in _load_ceval_config_names():
        for row in _load_ceval_rows(config, split_key):
            yield {
                "question": str(row.get("question", "") or "").strip(),
                "A": str(row.get("A", "") or "").strip(),
                "B": str(row.get("B", "") or "").strip(),
                "C": str(row.get("C", "") or "").strip(),
                "D": str(row.get("D", "") or "").strip(),
                "answer": str(row.get("answer", "")).strip().upper(),
                "subject": config,
                "dataset": "ceval",
                "id": row.get("id"),
                "explanation": str(row.get("explanation", "") or ""),
            }


class CevalDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str) -> None:
        super().__init__("ceval", output_root, split, required_fields=_REQUIRED_FIELDS, source_kind="hf_load_dataset")

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(_iter_records(self.split, hf_root=self.context.cache_root / "hf_cache"))

    def manifest_extra(self) -> dict[str, Any]:
        return {"dataset_id": _DATASET_ID, "source_split": _ALLOWED_SPLITS.get(self.split.lower(), self.split)}


@MULTIPLE_CHOICE_REGISTRY.register_spec("ceval")
def prepare_ceval_spec(output_root: Path, split: str = "test") -> CevalDatasetSpec:
    return CevalDatasetSpec(output_root, split)


__all__ = ["prepare_ceval_spec"]
