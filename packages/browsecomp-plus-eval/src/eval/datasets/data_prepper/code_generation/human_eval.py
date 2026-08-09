from __future__ import annotations

"""Prepare HumanEval dataset via the shared dataset runtime."""

from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import CODE_GENERATION_REGISTRY
from src.eval.datasets.runtime import UrlDownloadFile, UrlFilesJsonlDatasetSpec, read_gzip_jsonl_items

DATA_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def _map_record(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": payload["task_id"],
        "prompt": payload["prompt"],
        "canonical_solution": payload.get("canonical_solution"),
        "test": payload.get("test"),
        "entry_point": payload.get("entry_point"),
    }


@CODE_GENERATION_REGISTRY.register_spec("human_eval")
def prepare_human_eval_spec(output_root: Path, split: str = "test") -> UrlFilesJsonlDatasetSpec:
    if split != "test":
        raise ValueError("human_eval 仅提供 test split")

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return read_gzip_jsonl_items(source_root / "HumanEval.jsonl.gz", parse_item=_map_record)

    return UrlFilesJsonlDatasetSpec(
        "human_eval",
        output_root,
        split,
        files=(UrlDownloadFile(Path("HumanEval.jsonl.gz"), DATA_URL),),
        load_downloaded_records=_load,
        required_fields=("task_id", "prompt"),
    )
