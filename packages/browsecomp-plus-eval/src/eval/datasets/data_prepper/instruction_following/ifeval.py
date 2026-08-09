from __future__ import annotations

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import INSTRUCTION_FOLLOWING_REGISTRY
from src.eval.datasets.runtime import UrlDownloadFile, UrlFilesJsonlDatasetSpec, read_jsonl_items

IFEVAL_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/instruction_following_eval/data/input_data.jsonl"
)


def _map_record(payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["question"] = record.get("prompt", "")
    return record


@INSTRUCTION_FOLLOWING_REGISTRY.register_spec("ifeval")
def prepare_ifeval_spec(output_root: Path, split: str = "test") -> UrlFilesJsonlDatasetSpec:
    if split != "test":
        raise ValueError("IFEval 数据集仅提供 test split")

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return read_jsonl_items(source_root / "input_data.jsonl", parse_item=_map_record)

    return UrlFilesJsonlDatasetSpec(
        "ifeval",
        output_root,
        split,
        files=(UrlDownloadFile(Path("input_data.jsonl"), IFEVAL_URL),),
        load_downloaded_records=_load,
        required_fields=("prompt", "question"),
    )


__all__ = ["prepare_ifeval_spec"]
