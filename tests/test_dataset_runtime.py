from __future__ import annotations

import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.eval.datasets.data_prepper.data_utils import write_jsonl
from src.eval.datasets.data_prepper.prepper_registry import DatasetRegistry
from src.eval.datasets.runtime import (
    LegacyPreparerDatasetSpec,
    StaticRowsDatasetSpec,
    collect_files_with_extension,
    ensure_dataset_materialized,
    read_csv_items,
    read_gzip_jsonl_items,
    read_jsonl_items,
    read_parquet_items,
)


def test_runtime_loader_helpers_cover_jsonl_csv_gzip_and_parquet(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True)

    jsonl_path = root / "items.jsonl"
    write_jsonl(jsonl_path, [{"value": 1}, {"value": 2}])

    gzip_path = root / "items.jsonl.gz"
    with gzip.open(gzip_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"value": 3}) + "\n")
        handle.write(json.dumps({"value": 4}) + "\n")

    csv_path = root / "items.csv"
    csv_path.write_text("value,name\n5,alpha\n6,beta\n", encoding="utf-8")

    parquet_path = root / "items.parquet"
    pq.write_table(pa.Table.from_pylist([{"value": 7}, {"value": 8}]), parquet_path)

    assert collect_files_with_extension(root, "jsonl") == [jsonl_path]
    assert read_jsonl_items(jsonl_path) == [{"value": 1}, {"value": 2}]
    assert read_gzip_jsonl_items(gzip_path) == [{"value": 3}, {"value": 4}]
    assert read_csv_items(csv_path) == [
        {"value": "5", "name": "alpha"},
        {"value": "6", "name": "beta"},
    ]
    assert read_parquet_items(parquet_path) == [{"value": 7}, {"value": 8}]


def test_static_rows_spec_materializes_jsonl_and_manifest(tmp_path: Path) -> None:
    spec = StaticRowsDatasetSpec(
        "demo_dataset",
        tmp_path,
        "test",
        rows=[{"question": "Q", "answer": "A"}],
        required_fields=("question", "answer"),
    )

    paths = ensure_dataset_materialized(spec)

    assert paths == [tmp_path / "demo_dataset" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [{"question": "Q", "answer": "A"}]

    manifest = json.loads((tmp_path / "demo_dataset" / "test.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "demo_dataset"
    assert manifest["split"] == "test"
    assert manifest["row_count"] == 1
    assert manifest["source_kind"] == "static_rows"


def test_legacy_preparer_spec_wraps_existing_jsonl_preparer(tmp_path: Path) -> None:
    def _prepare(output_root: Path, split: str) -> list[Path]:
        path = output_root / "legacy_dataset" / f"{split}.jsonl"
        write_jsonl(path, [{"question": "hello", "answer": "world"}])
        return [path]

    spec = LegacyPreparerDatasetSpec("legacy_dataset", tmp_path, "test", preparer=_prepare)

    paths = ensure_dataset_materialized(spec)

    assert paths == [tmp_path / "legacy_dataset" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [{"question": "hello", "answer": "world"}]
    manifest = json.loads((tmp_path / "legacy_dataset" / "test.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "legacy_prepper"
    assert manifest["row_count"] == 1


def test_dataset_registry_supports_spec_factories() -> None:
    registry = DatasetRegistry("runtime_test")

    @registry.register_spec("demo")
    def _factory(output_root: Path, split: str) -> StaticRowsDatasetSpec:
        return StaticRowsDatasetSpec("demo", output_root, split, rows=[{"id": 1}])

    assert registry.names() == ("demo",)
    assert registry.get("demo") is None
    assert registry.get_spec_factory("demo") is _factory
