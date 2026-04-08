from __future__ import annotations

import inspect
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock  # pyright: ignore[reportMissingImports]

from src.eval.datasets.data_prepper.data_utils import write_jsonl

from .download import UrlDownloadFile, download_hf_repo, download_url_files
from .hf import download_hf_parquet_splits
from .loaders import (
    collect_files_with_extension,
    read_gzip_jsonl_items,
    read_jsonl_items,
    read_parquet_items,
)
from .validators import validate_jsonl_file, validate_non_empty_records, validate_required_fields


@dataclass(slots=True, frozen=True)
class DatasetPrepareContext:
    data_root: Path
    cache_root: Path
    artifact_root: Path
    lock_root: Path

    @classmethod
    def from_output_root(cls, output_root: str | Path) -> "DatasetPrepareContext":
        root = Path(output_root).expanduser().resolve()
        return cls(
            data_root=root,
            cache_root=(root / "cache").resolve(),
            artifact_root=root,
            lock_root=(root / ".locks").resolve(),
        )


@dataclass(slots=True)
class DatasetManifest:
    dataset: str
    split: str
    row_count: int
    source_kind: str
    artifact_path: str
    cache_dir: str
    prepared_at: str
    extra: dict[str, Any] = field(default_factory=dict)


class DatasetSpec(ABC):
    name: str
    split: str
    source_kind: str
    required_fields: tuple[str, ...]

    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        required_fields: tuple[str, ...] = (),
        source_kind: str = "custom",
    ) -> None:
        self.name = name
        self.split = split
        self.context = DatasetPrepareContext.from_output_root(output_root)
        self.required_fields = required_fields
        self.source_kind = source_kind

    @property
    def root_dir(self) -> Path:
        return self.context.data_root

    @property
    def cache_dir(self) -> Path:
        path = (self.context.cache_root / self.name).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def artifact_dir(self) -> Path:
        path = (self.context.artifact_root / self.name).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def artifact_path(self) -> Path:
        return self.artifact_dir / f"{self.split}.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.artifact_path.with_suffix(self.artifact_path.suffix + ".manifest.json")

    @property
    def lock_path(self) -> Path:
        self.context.lock_root.mkdir(parents=True, exist_ok=True)
        return (self.context.lock_root / f"{self.name}__{self.split}.lock").resolve()

    def materialized_paths(self) -> list[Path]:
        return [self.artifact_path]

    def validate_materialized_artifact(self) -> bool:
        try:
            for path in self.materialized_paths():
                validate_jsonl_file(path, self.required_fields)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return False
        return True

    @abstractmethod
    def load(self) -> bool:
        """Return True when local source data is missing or invalid."""

    @abstractmethod
    def check(self) -> bool:
        """Return True when loaded data fails validation."""

    @abstractmethod
    def download(self) -> None:
        """Fetch or materialize the local source data."""

    @abstractmethod
    def len(self) -> int:
        pass

    @abstractmethod
    def iter_records(self) -> Iterable[dict[str, Any]]:
        pass

    def manifest_extra(self) -> dict[str, Any]:
        return {}

    def materialize(self) -> list[Path]:
        write_jsonl(self.artifact_path, self.iter_records())
        manifest = DatasetManifest(
            dataset=self.name,
            split=self.split,
            row_count=self.len(),
            source_kind=self.source_kind,
            artifact_path=str(self.artifact_path),
            cache_dir=str(self.cache_dir),
            prepared_at=datetime.now(UTC).isoformat(),
            extra=self.manifest_extra(),
        )
        self.manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return self.materialized_paths()


@contextmanager
def _dataset_runtime_env(context: DatasetPrepareContext) -> Iterable[None]:
    hf_home = (context.cache_root / "hf_cache").resolve()
    overrides = {
        "RWKV_SKILLS_DATA_ROOT": str(context.data_root),
        "RWKV_SKILLS_HF_HOME": str(hf_home),
        "HF_HOME": str(hf_home),
        "HUGGINGFACE_HUB_CACHE": str((hf_home / "hub").resolve()),
        "HF_DATASETS_CACHE": str((hf_home / "datasets").resolve()),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _call_with_optional_context(
    callback: Callable[..., Any],
    split: str,
    context: DatasetPrepareContext,
) -> Any:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(split)

    parameters = tuple(signature.parameters.values())
    positional_count = sum(
        1
        for parameter in parameters
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    has_varargs = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters)
    if has_varargs or positional_count >= 2:
        return callback(split, context)
    return callback(split)


class LegacyPreparerDatasetSpec(DatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        preparer: Callable[[Path, str], list[Path]],
    ) -> None:
        super().__init__(name, output_root, split, source_kind="legacy_prepper")
        self._preparer = preparer
        self._prepared_paths: list[Path] = []
        self._row_count = 0

    def _candidate_paths(self) -> list[Path]:
        return self._prepared_paths or [self.artifact_path]

    def materialized_paths(self) -> list[Path]:
        return self._candidate_paths()

    def load(self) -> bool:
        try:
            self._row_count = validate_jsonl_file(self.artifact_path, self.required_fields)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return True
        return False

    def check(self) -> bool:
        try:
            counts = [validate_jsonl_file(path, self.required_fields) for path in self._candidate_paths()]
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return True
        self._row_count = counts[0] if counts else 0
        return False

    def download(self) -> None:
        self._prepared_paths = [path.expanduser().resolve() for path in self._preparer(self.context.artifact_root, self.split)]

    def len(self) -> int:
        return self._row_count

    def iter_records(self) -> Iterable[dict[str, Any]]:
        return read_jsonl_items(self.artifact_path)

    def materialize(self) -> list[Path]:
        paths = self._candidate_paths()
        manifest = DatasetManifest(
            dataset=self.name,
            split=self.split,
            row_count=self._row_count,
            source_kind=self.source_kind,
            artifact_path=str(self.artifact_path),
            cache_dir=str(self.cache_dir),
            prepared_at=datetime.now(UTC).isoformat(),
            extra={"prepared_paths": [str(path) for path in paths]},
        )
        self.manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return paths


class MaterializingDatasetSpec(DatasetSpec, ABC):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        required_fields: tuple[str, ...] = (),
        source_kind: str,
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind=source_kind)
        self._records: list[dict[str, Any]] = []

    def load(self) -> bool:
        try:
            with _dataset_runtime_env(self.context):
                self._records = list(self.load_records())
            validate_non_empty_records(self._records, self.name)
        except FileNotFoundError:
            self._records = []
            return True
        return False

    def check(self) -> bool:
        try:
            validate_non_empty_records(self._records, self.name)
            validate_required_fields(self._records, self.required_fields, self.name)
        except ValueError:
            return True
        return False

    def len(self) -> int:
        return len(self._records)

    def iter_records(self) -> Iterable[dict[str, Any]]:
        return self._records

    @abstractmethod
    def load_records(self) -> Iterable[dict[str, Any]]:
        pass


class StaticRowsDatasetSpec(MaterializingDatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        rows: Iterable[dict[str, Any]],
        required_fields: tuple[str, ...] = (),
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind="static_rows")
        self._static_rows = [dict(row) for row in rows]

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        return list(self._static_rows)


class CallableRowsDatasetSpec(MaterializingDatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        load_rows: Callable[[str], Iterable[dict[str, Any]]],
        required_fields: tuple[str, ...] = (),
        source_kind: str = "callable_rows",
        manifest_extra_factory: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind=source_kind)
        self._load_rows = load_rows
        self._manifest_extra_factory = manifest_extra_factory

    def download(self) -> None:
        return None

    def load_records(self) -> Iterable[dict[str, Any]]:
        with _dataset_runtime_env(self.context):
            return list(_call_with_optional_context(self._load_rows, self.split, self.context))

    def manifest_extra(self) -> dict[str, Any]:
        if self._manifest_extra_factory is None:
            return {}
        with _dataset_runtime_env(self.context):
            payload = _call_with_optional_context(self._manifest_extra_factory, self.split, self.context)
        return dict(payload)


class UrlFilesJsonlDatasetSpec(MaterializingDatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        files: Iterable[UrlDownloadFile],
        load_downloaded_records: Callable[[Path], Iterable[dict[str, Any]]],
        required_fields: tuple[str, ...] = (),
        tasks: int = 4,
        source_root_name: str = "source",
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind="url_files")
        self._files = list(files)
        self._load_downloaded_records = load_downloaded_records
        self._tasks = tasks
        self._source_root_name = source_root_name

    @property
    def source_root(self) -> Path:
        return self.cache_dir / self._source_root_name

    def download(self) -> None:
        download_url_files(self.cache_dir, self._source_root_name, self._files, self._tasks)

    def load_records(self) -> Iterable[dict[str, Any]]:
        if not self.source_root.exists():
            raise FileNotFoundError(self.source_root)
        return list(self._load_downloaded_records(self.source_root))

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "files": [
                {"relative_path": str(file.relative_path), "url": file.url}
                for file in self._files
            ]
        }


class HfParquetJsonlDatasetSpec(MaterializingDatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        dataset_id: str,
        config: str,
        source_splits: tuple[str, ...],
        parse_row: Callable[[dict[str, Any]], dict[str, Any]],
        required_fields: tuple[str, ...] = (),
        tasks: int = 8,
        source_root_name: str = "source",
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind="hf_parquet")
        self.dataset_id = dataset_id
        self.config = config
        self.source_splits = source_splits
        self._parse_row = parse_row
        self._tasks = tasks
        self._source_root_name = source_root_name

    @property
    def source_root(self) -> Path:
        return self.cache_dir / self._source_root_name

    def download(self) -> None:
        download_hf_parquet_splits(
            self.cache_dir,
            self._source_root_name,
            self.dataset_id,
            self.config,
            list(self.source_splits),
            self._tasks,
        )

    def load_records(self) -> Iterable[dict[str, Any]]:
        if not self.source_root.exists():
            raise FileNotFoundError(self.source_root)
        rows: list[dict[str, Any]] = []
        for path in collect_files_with_extension(self.source_root, "parquet"):
            rows.extend(read_parquet_items(path, parse_row=self._parse_row))
        return rows

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "config": self.config,
            "source_splits": list(self.source_splits),
        }


class HfRepoJsonlDatasetSpec(MaterializingDatasetSpec):
    def __init__(
        self,
        name: str,
        output_root: str | Path,
        split: str,
        *,
        repo: str,
        revision: str,
        load_downloaded_records: Callable[[Path], Iterable[dict[str, Any]]],
        required_fields: tuple[str, ...] = (),
        repo_type: str = "dataset",
        source_root_name: str = "source",
        allow_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(name, output_root, split, required_fields=required_fields, source_kind="hf_repo")
        self.repo = repo
        self.revision = revision
        self.repo_type = repo_type
        self._load_downloaded_records = load_downloaded_records
        self._source_root_name = source_root_name
        self._allow_patterns = allow_patterns

    @property
    def source_root(self) -> Path:
        return self.cache_dir / self._source_root_name

    def download(self) -> None:
        download_hf_repo(
            self.cache_dir,
            self.repo,
            revision=self.revision,
            repo_type=self.repo_type,
            root_name=self._source_root_name,
            allow_patterns=self._allow_patterns,
        )

    def load_records(self) -> Iterable[dict[str, Any]]:
        if not self.source_root.exists():
            raise FileNotFoundError(self.source_root)
        return list(self._load_downloaded_records(self.source_root))

    def manifest_extra(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "repo_type": self.repo_type,
        }


def ensure_dataset_materialized(spec: DatasetSpec) -> list[Path]:
    with FileLock(str(spec.lock_path)):
        if spec.validate_materialized_artifact():
            return spec.materialized_paths()
        if spec.load():
            spec.download()
            if spec.load():
                raise RuntimeError(f"dataset {spec.name}:{spec.split} is still unavailable after download()")
        if spec.check():
            raise RuntimeError(f"dataset {spec.name}:{spec.split} failed check()")
        return spec.materialize()


__all__ = [
    "CallableRowsDatasetSpec",
    "DatasetManifest",
    "DatasetPrepareContext",
    "DatasetSpec",
    "HfParquetJsonlDatasetSpec",
    "HfRepoJsonlDatasetSpec",
    "LegacyPreparerDatasetSpec",
    "MaterializingDatasetSpec",
    "StaticRowsDatasetSpec",
    "UrlFilesJsonlDatasetSpec",
    "ensure_dataset_materialized",
]
