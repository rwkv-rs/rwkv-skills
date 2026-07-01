from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import download_git_repo
from src.eval.tasks.function_calling.api_bank import load_api_bank_rows_from_source
from src.eval.scheduler.config import REPO_ROOT

from .common import OfficialRowsDatasetSpec, first_complete_source_root

_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_tool_calls")
_API_BANK_REPO_URL = "https://github.com/AlibabaResearch/DAMO-ConvAI.git"
_API_BANK_REPO_REVISION = "main"
_API_BANK_REPO_ROOT_NAME = "DAMO-ConvAI"


def api_bank_source_root() -> Path:
    override = os.environ.get("API_BANK_SOURCE_ROOT") or os.environ.get("RWKV_API_BANK_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        REPO_ROOT / "references" / "API-Bank",
        REPO_ROOT / "references" / "api-bank",
        REPO_ROOT.parent / "API-Bank",
        REPO_ROOT.parent / "api-bank",
        Path("/tmp/rwkv-official-refs/DAMO-ConvAI/api-bank"),
        Path("/tmp/ref-DAMO-ConvAI/api-bank"),
    ):
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT.parent / "API-Bank").resolve()


def api_bank_lv1_lv2_dir() -> Path:
    return api_bank_source_root() / "lv1-lv2-samples" / "level-1-given-desc"


def _api_bank_required_paths(source_root: Path) -> tuple[Path, ...]:
    return (source_root / "lv1-lv2-samples" / "level-1-given-desc",)


def _api_bank_source_candidates() -> tuple[Path, ...]:
    candidates = [api_bank_source_root()]
    lv1_lv2_dir = api_bank_lv1_lv2_dir()
    if lv1_lv2_dir.name == "level-1-given-desc" and lv1_lv2_dir.parent.name == "lv1-lv2-samples":
        candidates.insert(0, lv1_lv2_dir.parent.parent)
    candidates.extend(
        [
            REPO_ROOT / "references" / "DAMO-ConvAI" / "api-bank",
            REPO_ROOT / "references" / "API-Bank",
            REPO_ROOT / "references" / "api-bank",
            REPO_ROOT.parent / "DAMO-ConvAI" / "api-bank",
            REPO_ROOT.parent / "API-Bank",
            REPO_ROOT.parent / "api-bank",
            Path("/tmp/rwkv-official-refs/DAMO-ConvAI/api-bank"),
            Path("/tmp/ref-DAMO-ConvAI/api-bank"),
        ]
    )
    return tuple(dict.fromkeys(candidates))


def _api_bank_downloaded_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return spec.cache_dir / _API_BANK_REPO_ROOT_NAME / "api-bank"


def _resolve_api_bank_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return first_complete_source_root(_api_bank_source_candidates, _api_bank_required_paths) or _api_bank_downloaded_source_root(spec)


def _download_api_bank_source(spec: OfficialRowsDatasetSpec) -> None:
    download_git_repo(
        spec.cache_dir,
        _API_BANK_REPO_URL,
        revision=_API_BANK_REPO_REVISION,
        root_name=_API_BANK_REPO_ROOT_NAME,
    )


def _prepare_api_bank_spec(dataset_name: str, output_root: Path, split: str, *, level: int) -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} only provides test split")

    def _load(source_root: Path) -> list[dict[str, Any]]:
        return load_api_bank_rows_from_source(
            source_root / "lv1-lv2-samples" / "level-1-given-desc",
            dataset_name=dataset_name,
            level=level,
        )

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_api_bank_git",
        official_source="AlibabaResearch/DAMO-ConvAI/api-bank",
        resolve_source_root=_resolve_api_bank_source_root,
        required_paths=_api_bank_required_paths,
        load_official_records=_load,
        download_source=_download_api_bank_source,
        extra={
            "source_repo_url": _API_BANK_REPO_URL,
            "source_revision": _API_BANK_REPO_REVISION,
            "canonical_datasets": ["apibank_level1", "apibank_level2"],
            "level": level,
        },
    )


@FUNCTION_CALLING_REGISTRY.register_spec("apibank_level1")
def prepare_apibank_level1_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_api_bank_spec("apibank_level1", output_root, split, level=1)


@FUNCTION_CALLING_REGISTRY.register_spec("apibank_level2")
def prepare_apibank_level2_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_api_bank_spec("apibank_level2", output_root, split, level=2)


@FUNCTION_CALLING_REGISTRY.register_spec("apibank_l1")
def prepare_apibank_l1_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_api_bank_spec("apibank_l1", output_root, split, level=1)


@FUNCTION_CALLING_REGISTRY.register_spec("apibank_l2")
def prepare_apibank_l2_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _prepare_api_bank_spec("apibank_l2", output_root, split, level=2)


__all__ = [
    "api_bank_lv1_lv2_dir",
    "api_bank_source_root",
    "prepare_apibank_l1_spec",
    "prepare_apibank_l2_spec",
    "prepare_apibank_level1_spec",
    "prepare_apibank_level2_spec",
]
