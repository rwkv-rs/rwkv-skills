from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from src.eval.datasets.runtime import MaterializingDatasetSpec
from src.eval.scheduler.config import REPO_ROOT


def rwkv_rs_datasets_root() -> Path:
    override = os.environ.get("RWKV_RS_DATASETS_ROOT") or os.environ.get("RWKV_RS_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if root.name == "rwkv-rs":
            return root / "examples" / "rwkv-lm-eval" / "datasets"
        return root
    return (REPO_ROOT.parent / "rwkv-rs" / "examples" / "rwkv-lm-eval" / "datasets").resolve()


class LocalRowsDatasetSpec(MaterializingDatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        required_fields: tuple[str, ...] = (),
        source_kind: str = "local_rows",
        required_paths: Sequence[Path] | Callable[[], Sequence[Path]] = (),
        load_local_records: Callable[[], Iterable[dict[str, Any]]],
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind=source_kind)
        self._required_paths = required_paths
        self._load_local_records = load_local_records
        self._extra = dict(extra or {})

    def _resolve_required_paths(self) -> tuple[Path, ...]:
        paths = self._required_paths() if callable(self._required_paths) else self._required_paths
        return tuple(path.expanduser().resolve() for path in paths)

    def download(self) -> None:
        required_paths = self._resolve_required_paths()
        missing = [path for path in required_paths if not path.exists()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"missing local source paths for {self.name}: {joined}")

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(self._load_local_records())

    def manifest_extra(self) -> dict[str, Any]:
        extra = dict(self._extra)
        required_paths = self._resolve_required_paths()
        if required_paths:
            extra["source_paths"] = [str(path) for path in required_paths]
        return extra


class OfficialRowsDatasetSpec(MaterializingDatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        official_source: str,
        resolve_source_root: Callable[["OfficialRowsDatasetSpec"], Path],
        required_paths: Callable[[Path], Sequence[Path]],
        load_official_records: Callable[[Path], Iterable[dict[str, Any]]],
        download_source: Callable[["OfficialRowsDatasetSpec"], None] | None = None,
        required_fields: tuple[str, ...] = (),
        source_kind: str = "official_source",
        extra: dict[str, Any] | Callable[[Path], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind=source_kind)
        self.official_source = official_source
        self._resolve_source_root = resolve_source_root
        self._required_paths = required_paths
        self._load_official_records = load_official_records
        self._download_source = download_source
        self._extra = extra

    def source_root(self) -> Path:
        return self._resolve_source_root(self).expanduser().resolve()

    def source_paths(self) -> tuple[Path, ...]:
        root = self.source_root()
        return tuple(path.expanduser().resolve() for path in self._required_paths(root))

    def _missing_paths(self) -> list[Path]:
        return [path for path in self.source_paths() if not path.exists()]

    def _raise_missing(self, missing: Sequence[Path]) -> None:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing official source paths for {self.name}: {joined}")

    def download(self) -> None:
        missing = self._missing_paths()
        if not missing:
            return
        if self._download_source is None:
            self._raise_missing(missing)
        self._download_source(self)
        missing_after = self._missing_paths()
        if missing_after:
            self._raise_missing(missing_after)

    def load_records(self) -> Iterable[dict[str, Any]]:
        missing = self._missing_paths()
        if missing:
            self._raise_missing(missing)
        return list(self._load_official_records(self.source_root()))

    def manifest_extra(self) -> dict[str, Any]:
        root = self.source_root()
        extra: dict[str, Any] = {
            "official_source": self.official_source,
            "source_root": str(root),
            "source_paths": [str(path) for path in self.source_paths()],
        }
        if self._extra is not None:
            payload = self._extra(root) if callable(self._extra) else self._extra
            extra.update(dict(payload))
        return extra


def first_complete_source_root(
    candidates: Sequence[Path] | Callable[[], Sequence[Path]],
    required_paths: Callable[[Path], Sequence[Path]],
) -> Path | None:
    materialized = candidates() if callable(candidates) else candidates
    for candidate in materialized:
        root = candidate.expanduser().resolve()
        if all(path.expanduser().resolve().exists() for path in required_paths(root)):
            return root
    return None


__all__ = [
    "OfficialRowsDatasetSpec",
    "first_complete_source_root",
    "LocalRowsDatasetSpec",
    "rwkv_rs_datasets_root",
]
