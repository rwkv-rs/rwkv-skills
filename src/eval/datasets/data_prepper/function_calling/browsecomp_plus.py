from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import download_hf_repo
from src.eval.function_calling.browsecomp_plus import (
    DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT,
    OFFICIAL_BROWSECOMP_PLUS_SOURCE,
    browsecomp_plus_index_path,
    browsecomp_plus_source_path,
    load_browsecomp_plus_rows_from_decrypted_jsonl,
)

from .common import OfficialRowsDatasetSpec, first_complete_source_root

_REQUIRED_FIELDS = ("task_id", "instruction", "answer")
_BROWSECOMP_PLUS_REVISION = "main"
_BROWSECOMP_PLUS_ROOT_NAME = "BrowseComp-Plus"


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


def _browsecomp_plus_required_paths(root: Path) -> tuple[Path, Path]:
    return (browsecomp_plus_source_path(root), browsecomp_plus_index_path(root))


def _browsecomp_plus_source_candidates() -> tuple[Path, ...]:
    candidates = [
        browsecomp_plus_official_root(),
        DEFAULT_OFFICIAL_BROWSECOMP_PLUS_ROOT,
        Path("/tmp/ref-BrowseComp-Plus"),
    ]
    return tuple(dict.fromkeys(candidates))


def _resolve_browsecomp_plus_root(spec: OfficialRowsDatasetSpec) -> Path:
    return first_complete_source_root(_browsecomp_plus_source_candidates, _browsecomp_plus_required_paths) or (
        spec.cache_dir / _BROWSECOMP_PLUS_ROOT_NAME
    )


def _download_browsecomp_plus_source(spec: OfficialRowsDatasetSpec) -> None:
    download_hf_repo(
        spec.cache_dir,
        OFFICIAL_BROWSECOMP_PLUS_SOURCE,
        revision=_BROWSECOMP_PLUS_REVISION,
        repo_type="dataset",
        root_name=_BROWSECOMP_PLUS_ROOT_NAME,
    )


def _prepare_browsecomp_plus_spec(dataset_name: str, output_root: Path, split: str) -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    def _load(root: Path) -> list[dict[str, Any]]:
        include_documents = str(os.environ.get("RWKV_BROWSECOMP_PLUS_EMBED_DOCS") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        return load_browsecomp_plus_rows_from_decrypted_jsonl(
            browsecomp_plus_source_path(root),
            official_root=root,
            dataset_name=dataset_name,
            include_documents=include_documents,
        )

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_browsecomp_plus_hf",
        official_source=OFFICIAL_BROWSECOMP_PLUS_SOURCE,
        resolve_source_root=_resolve_browsecomp_plus_root,
        required_paths=_browsecomp_plus_required_paths,
        load_official_records=_load,
        download_source=_download_browsecomp_plus_source,
        extra=lambda root: {
            "repo": OFFICIAL_BROWSECOMP_PLUS_SOURCE,
            "revision": _BROWSECOMP_PLUS_REVISION,
            "bm25_index_path": str(browsecomp_plus_index_path(root)),
            "bm25_required": True,
        },
    )


@FUNCTION_CALLING_REGISTRY.register_spec("browsecomp_plus")
def prepare_browsecomp_plus_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_browsecomp_plus_spec("browsecomp_plus", output_root, split)


__all__ = [
    "browsecomp_plus_official_root",
    "prepare_browsecomp_plus_spec",
]
