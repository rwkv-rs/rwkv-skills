from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.function_calling.browsecomp_plus import (
    DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT,
    browsecomp_plus_index_path,
    browsecomp_plus_source_path,
    load_browsecomp_plus_rows_from_decrypted_jsonl,
)

from .common import LocalRowsDatasetSpec

_REQUIRED_FIELDS = ("task_id", "instruction", "answer")


def browsecomp_plus_official_root() -> Path:
    override = os.environ.get("RWKV_BROWSECOMP_PLUS_ROOT") or os.environ.get("BROWSECOMP_PLUS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT,
        Path("/tmp/ref-BrowseComp-Plus"),
    ):
        if (candidate / "data" / "browsecomp_plus_decrypted.jsonl").exists():
            return candidate.resolve()
    return DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT.resolve()


def _prepare_browsecomp_plus_spec(dataset_name: str, output_root: Path, split: str) -> LocalRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")
    root = browsecomp_plus_official_root()
    source_path = browsecomp_plus_source_path(root)
    index_path = browsecomp_plus_index_path(root)

    def _load() -> list[dict[str, Any]]:
        include_documents = str(os.environ.get("RWKV_BROWSECOMP_PLUS_EMBED_DOCS") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        return load_browsecomp_plus_rows_from_decrypted_jsonl(
            source_path,
            official_root=root,
            dataset_name=dataset_name,
            include_documents=include_documents,
        )

    return LocalRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_browsecomp_plus",
        required_paths=(source_path, index_path),
        load_local_records=_load,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("browsecomp_plus")
def prepare_browsecomp_plus_spec(output_root: Path, split: str = "test") -> LocalRowsDatasetSpec:
    return _prepare_browsecomp_plus_spec("browsecomp_plus", output_root, split)


__all__ = [
    "browsecomp_plus_official_root",
    "prepare_browsecomp_plus_spec",
]
