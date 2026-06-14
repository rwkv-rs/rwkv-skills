from __future__ import annotations

import os
from pathlib import Path

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import CallableRowsDatasetSpec, DatasetPrepareContext, download_git_repo
from src.eval.function_calling.complexfuncbench import (
    DEFAULT_COMPLEXFUNC_MAX_ROWS,
    load_complexfuncbench_rows_from_source,
    require_complexfuncbench_official_root,
)
from src.eval.scheduler.config import REPO_ROOT

from ..data_utils import download_file

_COMPLEXFUNC_HF_URL = (
    "https://huggingface.co/datasets/zai-org/ComplexFuncBench/resolve/main/ComplexFuncBench.jsonl"
)
_COMPLEXFUNC_GIT_URL = "https://github.com/zai-org/ComplexFuncBench.git"
_COMPLEXFUNC_GIT_REVISION = "main"
_COMPLEXFUNC_GIT_ROOT_NAME = "ComplexFuncBench"


def complexfuncbench_official_root() -> Path | None:
    override = (
        os.environ.get("RWKV_COMPLEXFUNC_OFFICIAL_ROOT")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_OFFICIAL_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        REPO_ROOT / "references" / "ComplexFuncBench",
        REPO_ROOT.parent / "ComplexFuncBench",
        Path("/tmp/rwkv-official-refs/ComplexFuncBench"),
        Path("/tmp/ref-ComplexFuncBench"),
    ):
        try:
            return require_complexfuncbench_official_root(candidate)
        except FileNotFoundError:
            continue
    return None


def _resolve_or_download_complexfuncbench_official_root(
    context: DatasetPrepareContext,
    *,
    dataset_name: str,
) -> Path:
    existing = complexfuncbench_official_root()
    if existing is not None:
        return require_complexfuncbench_official_root(existing)
    return require_complexfuncbench_official_root(
        download_git_repo(
            context.cache_root / dataset_name,
            _COMPLEXFUNC_GIT_URL,
            revision=_COMPLEXFUNC_GIT_REVISION,
            root_name=_COMPLEXFUNC_GIT_ROOT_NAME,
        )
    )


def complexfuncbench_source_root() -> Path:
    override = (
        os.environ.get("RWKV_COMPLEXFUNC_SOURCE_ROOT")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_SOURCE_ROOT")
        or os.environ.get("COMPLEXFUNCBENCH_SOURCE_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (
        REPO_ROOT / "references" / "ComplexFuncBench",
        REPO_ROOT.parent / "ComplexFuncBench",
    ):
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / "references" / "ComplexFuncBench").resolve()


def complexfuncbench_source_path(output_root: Path, dataset_name: str) -> Path:
    override = (
        os.environ.get("RWKV_COMPLEXFUNCBENCH_SOURCE")
        or os.environ.get("RWKV_COMPLEXFUNC_SOURCE")
        or os.environ.get("COMPLEXFUNCBENCH_SOURCE")
    )
    if override:
        return Path(override).expanduser().resolve()

    source_root = complexfuncbench_source_root()
    official_root = complexfuncbench_official_root()
    official_candidates = (
        (official_root / "data" / "ComplexFuncBench.jsonl", official_root / "ComplexFuncBench.jsonl")
        if official_root is not None
        else ()
    )
    for candidate in (
        source_root / "data" / "ComplexFuncBench.jsonl",
        source_root / "ComplexFuncBench.jsonl",
        *official_candidates,
        REPO_ROOT / "data" / "cache" / dataset_name / "ComplexFuncBench.jsonl",
        REPO_ROOT / "data" / "cache" / "complexfuncbench_subset" / "ComplexFuncBench.jsonl",
    ):
        if candidate.exists():
            return candidate.resolve()
    return download_file(
        _COMPLEXFUNC_HF_URL,
        output_root / "cache" / dataset_name / "ComplexFuncBench.jsonl",
    )


def _load_complexfuncbench_official_rows(
    dataset_name: str,
    split: str,
    context: DatasetPrepareContext,
) -> list[dict[str, object]]:
    if split != "test":
        raise ValueError(f"{dataset_name} only provides test split")
    source_path = complexfuncbench_source_path(context.data_root, dataset_name)
    official_root = _resolve_or_download_complexfuncbench_official_root(context, dataset_name=dataset_name)
    max_rows = int(os.environ.get("RWKV_COMPLEXFUNCBENCH_MAX_ROWS", str(DEFAULT_COMPLEXFUNC_MAX_ROWS)))
    response_eval = str(os.environ.get("RWKV_COMPLEXFUNCBENCH_RESPONSE_EVAL", "1")).strip().lower()
    rows = load_complexfuncbench_rows_from_source(
        source_path,
        official_root=official_root,
        dataset_name=dataset_name,
        max_rows=max_rows,
        response_eval=response_eval not in {"0", "false", "no", "off"},
    )
    if not rows:
        raise ValueError(f"{source_path} did not yield any ComplexFuncBench official rows")
    return rows


@FUNCTION_CALLING_REGISTRY.register_spec("complexfuncbench_official")
def prepare_complexfuncbench_official(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "complexfuncbench_official",
        output_root,
        split,
        load_rows=lambda requested_split, context: _load_complexfuncbench_official_rows(
            "complexfuncbench_official",
            requested_split,
            context,
        ),
        source_kind="complexfuncbench_official",
        manifest_extra_factory=lambda _split, _context: {
            "official_source": "zai-org/ComplexFuncBench",
            "source_dataset_url": _COMPLEXFUNC_HF_URL,
            "official_repo_url": _COMPLEXFUNC_GIT_URL,
            "official_repo_revision": _COMPLEXFUNC_GIT_REVISION,
        },
    )


@FUNCTION_CALLING_REGISTRY.register_spec("complexfuncbench_subset")
def prepare_complexfuncbench_subset(output_root: Path, split: str = "test") -> CallableRowsDatasetSpec:
    return CallableRowsDatasetSpec(
        "complexfuncbench_subset",
        output_root,
        split,
        load_rows=lambda requested_split, context: _load_complexfuncbench_official_rows(
            "complexfuncbench_subset",
            requested_split,
            context,
        ),
        source_kind="complexfuncbench_official",
        manifest_extra_factory=lambda _split, _context: {
            "official_source": "zai-org/ComplexFuncBench",
            "source_dataset_url": _COMPLEXFUNC_HF_URL,
            "official_repo_url": _COMPLEXFUNC_GIT_URL,
            "official_repo_revision": _COMPLEXFUNC_GIT_REVISION,
            "subset": True,
        },
    )


__all__ = [
    "complexfuncbench_official_root",
    "complexfuncbench_source_root",
    "complexfuncbench_source_path",
    "prepare_complexfuncbench_official",
    "prepare_complexfuncbench_subset",
]
