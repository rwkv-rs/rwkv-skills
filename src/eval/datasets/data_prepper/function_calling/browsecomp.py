from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.eval.datasets.runtime import UrlDownloadFile, download_url_files
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.tasks.function_calling import (
    BrowseCompRecord,
    load_browsecomp_rows_from_csv,
    load_browsecomp_zh_rows_from_xlsx,
)

from .common import OfficialRowsDatasetSpec, first_complete_source_root, rwkv_rs_datasets_root

_REQUIRED_FIELDS = ("task_id", "question", "answer", "locale")
_BROWSECOMP_CSV_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
_BROWSECOMP_ZH_REPO_URL = "https://github.com/PALIN2018/BrowseComp-ZH.git"
_BROWSECOMP_ZH_XLSX_URL = (
    "https://raw.githubusercontent.com/PALIN2018/BrowseComp-ZH/main/data/browsecomp-zh-encrypted.xlsx"
)
_BROWSECOMP_ZH_REPO_REVISION = "main"
_BROWSECOMP_ZH_REPO_ROOT_NAME = "BrowseComp-ZH"


def _rows_from_records(records: list[BrowseCompRecord], *, source: Path) -> list[dict[str, str]]:
    return [
        {
            "task_id": record.task_id,
            "question": record.question,
            "answer": record.answer,
            "topic": record.topic or "",
            "locale": record.locale,
            "source_path": str(source),
        }
        for record in records
    ]


def _build_browsecomp_spec(
    *,
    dataset_name: str,
    output_root: Path,
    split: str,
    resolve_source_root: Callable[[OfficialRowsDatasetSpec], Path],
    required_paths: Callable[[Path], tuple[Path, ...]],
    source_file: Callable[[Path], Path],
    load_records: Callable[[str | Path], list[BrowseCompRecord]],
    download_source: Callable[[OfficialRowsDatasetSpec], None],
    official_source: str,
    extra: dict[str, object],
) -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError(f"{dataset_name} 仅提供 test split")

    def _load(source_root: Path) -> list[dict[str, str]]:
        source = source_file(source_root)
        return _rows_from_records(load_records(source), source=source)

    return OfficialRowsDatasetSpec(
        dataset_name,
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_browsecomp_source",
        official_source=official_source,
        resolve_source_root=resolve_source_root,
        required_paths=required_paths,
        load_official_records=_load,
        download_source=download_source,
        extra=extra,
    )


def _browsecomp_required_paths(source_root: Path) -> tuple[Path, ...]:
    return (source_root / "browse_comp_test_set.csv",)


def _browsecomp_source_candidates() -> tuple[Path, ...]:
    candidates = [
        rwkv_rs_datasets_root() / "browsecomp",
        Path("/tmp/rwkv-official-refs/openai/simple-evals"),
    ]
    return tuple(dict.fromkeys(candidates))


def _resolve_browsecomp_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return first_complete_source_root(_browsecomp_source_candidates, _browsecomp_required_paths) or (
        spec.cache_dir / "openai-simple-evals"
    )


def _download_browsecomp_source(spec: OfficialRowsDatasetSpec) -> None:
    download_url_files(
        spec.cache_dir,
        "openai-simple-evals",
        [UrlDownloadFile(Path("browse_comp_test_set.csv"), _BROWSECOMP_CSV_URL)],
        tasks=1,
    )


def _browsecomp_zh_required_paths(source_root: Path) -> tuple[Path, ...]:
    return (source_root / "browsecomp-zh-encrypted.xlsx",)


def _browsecomp_zh_source_candidates() -> tuple[Path, ...]:
    candidates = [
        rwkv_rs_datasets_root() / "browsecomp_zh",
        Path("/tmp/rwkv-official-refs/BrowseComp-ZH/data"),
        Path("/tmp/ref-BrowseComp-ZH/data"),
    ]
    return tuple(dict.fromkeys(candidates))


def _resolve_browsecomp_zh_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return first_complete_source_root(_browsecomp_zh_source_candidates, _browsecomp_zh_required_paths) or (
        spec.cache_dir / _BROWSECOMP_ZH_REPO_ROOT_NAME / "data"
    )


def _download_browsecomp_zh_source(spec: OfficialRowsDatasetSpec) -> None:
    download_url_files(
        spec.cache_dir,
        f"{_BROWSECOMP_ZH_REPO_ROOT_NAME}/data",
        [UrlDownloadFile(Path("browsecomp-zh-encrypted.xlsx"), _BROWSECOMP_ZH_XLSX_URL)],
        tasks=1,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("browsecomp")
def prepare_browsecomp_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _build_browsecomp_spec(
        dataset_name="browsecomp",
        output_root=output_root,
        split=split,
        resolve_source_root=_resolve_browsecomp_source_root,
        required_paths=_browsecomp_required_paths,
        source_file=lambda root: root / "browse_comp_test_set.csv",
        load_records=load_browsecomp_rows_from_csv,
        download_source=_download_browsecomp_source,
        official_source="openai/simple-evals BrowseComp",
        extra={"source_url": _BROWSECOMP_CSV_URL},
    )


@FUNCTION_CALLING_REGISTRY.register_spec("browsecomp_zh")
def prepare_browsecomp_zh_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    return _build_browsecomp_spec(
        dataset_name="browsecomp_zh",
        output_root=output_root,
        split=split,
        resolve_source_root=_resolve_browsecomp_zh_source_root,
        required_paths=_browsecomp_zh_required_paths,
        source_file=lambda root: root / "browsecomp-zh-encrypted.xlsx",
        load_records=load_browsecomp_zh_rows_from_xlsx,
        download_source=_download_browsecomp_zh_source,
        official_source="PALIN2018/BrowseComp-ZH",
        extra={
            "source_repo_url": _BROWSECOMP_ZH_REPO_URL,
            "source_file_url": _BROWSECOMP_ZH_XLSX_URL,
            "source_revision": _BROWSECOMP_ZH_REPO_REVISION,
        },
    )


__all__ = [
    "prepare_browsecomp_spec",
    "prepare_browsecomp_zh_spec",
]
