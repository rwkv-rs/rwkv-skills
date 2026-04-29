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


__all__ = [
    "LocalRowsDatasetSpec",
    "rwkv_rs_datasets_root",
]
