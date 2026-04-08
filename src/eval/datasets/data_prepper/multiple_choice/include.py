from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping

from ..data_utils import configure_hf_home
from src.eval.datasets.data_prepper.prepper_registry import MULTIPLE_CHOICE_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec

_DATASET_ID = "CohereForAI/include-base-44"
_REQUIRED_FIELDS = ("question", "answer", "A", "B", "C", "D")


def _include_config_names() -> list[str]:
    configure_hf_home()
    from datasets import get_dataset_config_names  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

    return sorted(name for name in get_dataset_config_names(_DATASET_ID) if name and name != "default")


def _load_include_rows(config: str, split: str) -> Iterable[Mapping[str, Any]]:
    configure_hf_home()
    from datasets import load_dataset  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

    return load_dataset(_DATASET_ID, config, split=split)


def _answer_letter(raw: object) -> str:
    if isinstance(raw, str):
        value = raw.strip().upper()
        if value in {"A", "B", "C", "D"}:
            return value
        raw = value
    index = int(raw)
    if index < 0 or index > 3:
        raise ValueError(f"include answer index out of range: {raw!r}")
    return chr(ord("A") + index)


def _iter_records(split: str) -> Iterable[dict[str, Any]]:
    if split not in {"validation", "test"}:
        raise ValueError("include 仅支持 validation 或 test split")

    for config in _include_config_names():
        for row in _load_include_rows(config, split):
            choices = row.get("options") or row.get("choices") or row.get("answers")
            if not isinstance(choices, (list, tuple)) or len(choices) != 4:
                raise ValueError(f"include:{config}:{split} 缺少四个候选项: {row}")

            payload: dict[str, Any] = {
                "question": str(row.get("question") or row.get("prompt") or row.get("input") or ""),
                "answer": _answer_letter(
                    row.get("answer")
                    if row.get("answer") is not None
                    else row.get("label")
                    if row.get("label") is not None
                    else row.get("target")
                ),
                "subject": row.get("subject") or row.get("category") or row.get("topic") or config,
                "subset": config,
                "source": "include-base-44",
            }
            for index, choice in enumerate(choices):
                payload[chr(ord("A") + index)] = str(choice)

            for key in (
                "id",
                "language",
                "country",
                "regional_feature",
                "explanation",
                "source_dataset",
                "domain",
            ):
                value = row.get(key)
                if value is not None:
                    payload[key] = value
            yield payload


@MULTIPLE_CHOICE_REGISTRY.register_spec("include")
def prepare_include_spec(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "include",
        output_root,
        split,
        load_rows=_iter_records,
        required_fields=_REQUIRED_FIELDS,
        source_kind="hf_load_dataset",
        manifest_extra_factory=lambda resolved_split: {
            "dataset_id": _DATASET_ID,
            "source_split": resolved_split,
        },
    )


__all__ = ["prepare_include_spec"]
