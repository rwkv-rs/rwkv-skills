from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.runtime import download_git_repo
from src.eval.tasks.function_calling import load_bfcl_v3_rows_from_source
from src.eval.scheduler.config import REPO_ROOT

from .common import OfficialRowsDatasetSpec, first_complete_source_root

_REQUIRED_FIELDS = ("task_id", "instruction", "tools")
_BFCL_REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
_BFCL_REPO_REVISION = "main"
_BFCL_REPO_ROOT_NAME = "gorilla"
_BFCL_REPO_DATA_SUBDIR = ("berkeley-function-call-leaderboard", "bfcl_eval", "data")


def _bfcl_v3_source_override() -> str | None:
    return (
        os.environ.get("RWKV_BFCL_V3_SOURCE")
        or os.environ.get("RWKV_BFCL_V3_ROOT")
        or os.environ.get("BFCL_V3_SOURCE")
        or os.environ.get("BFCL_V3_ROOT")
    )


def bfcl_v3_source_root() -> Path:
    override = _bfcl_v3_source_override()
    if override:
        return Path(override).expanduser().resolve()
    repo_raw_root = REPO_ROOT / "data" / "bfcl_v3_raw"
    if repo_raw_root.exists():
        return repo_raw_root.resolve()
    reference_root = REPO_ROOT / "references" / "gorilla" / "berkeley-function-call-leaderboard"
    if reference_root.exists():
        return reference_root.resolve()
    return (REPO_ROOT.parent / "gorilla" / "berkeley-function-call-leaderboard").resolve()


def _bfcl_v3_candidate_roots(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        return (root.resolve(),)
    return tuple(
        dict.fromkeys(
            candidate.resolve()
            for candidate in (
                root,
                root / "data",
                root / "bfcl_eval" / "data",
                root / "berkeley-function-call-leaderboard",
                root / "berkeley-function-call-leaderboard" / "data",
                root / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
            )
        )
    )


def _bfcl_v3_source_paths_from_root(root: Path, split: str) -> tuple[Path, ...]:
    if root.is_file():
        return (root.resolve(),)
    direct_sources = tuple(
        dict.fromkeys(
            path.resolve()
            for pattern in ("BFCL_v3_multi_turn_*.json", "BFCL_v4_multi_turn_*.json")
            for path in sorted(root.glob(pattern))
            if path.is_file()
        )
    )
    if direct_sources:
        return direct_sources

    candidate_roots = _bfcl_v3_candidate_roots(root)
    exact_names = (
        f"bfcl_v3_{split}.jsonl",
        f"bfcl_v3_{split}.json",
        f"{split}.jsonl",
        f"{split}.json",
        "bfcl_v3.jsonl",
        "bfcl_v3.json",
        "multi_turn.jsonl",
        "multi_turn.json",
    )
    for base in candidate_roots:
        for name in exact_names:
            candidate = base / name
            if candidate.is_file():
                return (candidate.resolve(),)

    fuzzy: list[Path] = []
    for base in candidate_roots:
        if not base.exists():
            continue
        for pattern in ("*.json", "*.jsonl"):
            fuzzy.extend(sorted(base.rglob(pattern)))
    deduped = tuple(
        dict.fromkeys(
            path.resolve()
            for path in fuzzy
            if path.is_file()
            and (
                ("bfcl" in path.name.lower() and "v3" in path.name.lower())
                or path.name.lower().startswith("bfcl_v4_multi_turn_")
            )
            and not ({"possible_answer", "unused_datasets"} & {part.lower() for part in path.parts})
        )
    )
    if deduped:
        return deduped
    return ()


def bfcl_v3_source_paths(split: str) -> tuple[Path, ...]:
    root = bfcl_v3_source_root()
    roots = [root]
    if not _bfcl_v3_source_override():
        cache_root = (
            REPO_ROOT
            / "data"
            / "cache"
            / "bfcl_exec_multiple_ast"
            / "gorilla"
            / "berkeley-function-call-leaderboard"
        )
        if cache_root.exists():
            roots.append(cache_root.resolve())
        roots.extend(_bfcl_v3_source_candidates())
    for candidate_root in tuple(dict.fromkeys(roots)):
        paths = _bfcl_v3_source_paths_from_root(candidate_root, split)
        if paths:
            return paths
    raise FileNotFoundError(
        f"could not locate BFCL V3 source under {root}; set RWKV_BFCL_V3_SOURCE or RWKV_BFCL_V3_ROOT"
    )


def bfcl_v3_source_path(split: str) -> Path:
    paths = bfcl_v3_source_paths(split)
    if len(paths) != 1:
        joined = ", ".join(str(path) for path in paths[:5])
        raise FileNotFoundError(
            f"multiple BFCL V3 source files matched; use bfcl_v3_source_paths() or set RWKV_BFCL_V3_SOURCE explicitly. Matches: {joined}"
        )
    return paths[0]


def _bfcl_v3_source_candidates() -> tuple[Path, ...]:
    candidates = [
        bfcl_v3_source_root(),
        REPO_ROOT / "references" / "gorilla" / "berkeley-function-call-leaderboard",
        REPO_ROOT / "references" / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
        REPO_ROOT.parent / "gorilla" / "berkeley-function-call-leaderboard",
        REPO_ROOT.parent / "gorilla" / "berkeley-function-call-leaderboard" / "bfcl_eval" / "data",
        Path("/tmp/rwkv-official-refs/gorilla/berkeley-function-call-leaderboard"),
        Path("/tmp/rwkv-official-refs/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data"),
    ]
    return tuple(dict.fromkeys(candidates))


def _bfcl_v3_required_paths(split: str):
    def _required(source_root: Path) -> tuple[Path, ...]:
        paths = _bfcl_v3_source_paths_from_root(source_root, split)
        if paths:
            return paths
        return (source_root / "__missing_bfcl_v3_multi_turn_source__",)

    return _required


def _bfcl_v3_downloaded_source_root(spec: OfficialRowsDatasetSpec) -> Path:
    return spec.cache_dir.joinpath(_BFCL_REPO_ROOT_NAME, *_BFCL_REPO_DATA_SUBDIR)


def _resolve_bfcl_v3_source_root(spec: OfficialRowsDatasetSpec, *, split: str) -> Path:
    try:
        paths = bfcl_v3_source_paths(split)
    except FileNotFoundError:
        paths = ()
    if paths:
        if len(paths) == 1 and paths[0].is_file():
            return paths[0].parent.resolve()
        parents = {path.parent.resolve() for path in paths}
        if len(parents) == 1:
            return next(iter(parents))
    return first_complete_source_root(_bfcl_v3_source_candidates, _bfcl_v3_required_paths(split)) or _bfcl_v3_downloaded_source_root(spec)


def _download_bfcl_v3_source(spec: OfficialRowsDatasetSpec) -> None:
    download_git_repo(
        spec.cache_dir,
        _BFCL_REPO_URL,
        revision=_BFCL_REPO_REVISION,
        root_name=_BFCL_REPO_ROOT_NAME,
    )


@FUNCTION_CALLING_REGISTRY.register_spec("bfcl_v3")
def prepare_bfcl_v3_spec(output_root: Path, split: str = "test") -> OfficialRowsDatasetSpec:
    if split != "test":
        raise ValueError("bfcl_v3 仅提供 test split")

    def _load(source_root: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        sources = _bfcl_v3_source_paths_from_root(source_root, split)
        if not sources:
            sources = bfcl_v3_source_paths(split)
        for source in sources:
            rows.extend(load_bfcl_v3_rows_from_source(source))
        return rows

    return OfficialRowsDatasetSpec(
        "bfcl_v3",
        output_root,
        split,
        required_fields=_REQUIRED_FIELDS,
        source_kind="official_bfcl_v3_git",
        official_source="ShishirPatil/gorilla/berkeley-function-call-leaderboard",
        resolve_source_root=lambda spec: _resolve_bfcl_v3_source_root(spec, split=split),
        required_paths=_bfcl_v3_required_paths(split),
        load_official_records=_load,
        download_source=_download_bfcl_v3_source,
        extra={
            "source_repo_url": _BFCL_REPO_URL,
            "source_revision": _BFCL_REPO_REVISION,
        },
    )


__all__ = [
    "bfcl_v3_source_path",
    "bfcl_v3_source_paths",
    "bfcl_v3_source_root",
    "prepare_bfcl_v3_spec",
]
