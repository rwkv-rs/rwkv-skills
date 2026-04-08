from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.eval.datasets.data_prepper.data_manager import prepare_dataset
from src.eval.datasets.data_prepper.data_utils import write_jsonl
from src.eval.datasets.data_prepper.data_manager import _ensure_registries
from src.eval.datasets.data_prepper.prepper_registry import (
    CODE_GENERATION_REGISTRY,
    DatasetRegistry,
    FREE_ANSWER_REGISTRY,
    FUNCTION_CALLING_REGISTRY,
    INSTRUCTION_FOLLOWING_REGISTRY,
    MULTIPLE_CHOICE_REGISTRY,
)
from src.eval.datasets.runtime import (
    LegacyPreparerDatasetSpec,
    StaticRowsDatasetSpec,
    CallableRowsDatasetSpec,
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


def test_dataset_registry_can_wrap_legacy_preparer_as_spec_factory(tmp_path: Path) -> None:
    registry = DatasetRegistry("runtime_test")

    @registry.register_legacy_spec("legacy")
    def _prepare(output_root: Path, split: str) -> list[Path]:
        path = output_root / "legacy" / f"{split}.jsonl"
        write_jsonl(path, [{"question": "q", "answer": "a"}])
        return [path]

    factory = registry.get_spec_factory("legacy")

    assert registry.get("legacy") is None
    assert factory is not None
    spec = factory(tmp_path, "test")
    paths = ensure_dataset_materialized(spec)
    assert paths == [tmp_path / "legacy" / "test.jsonl"]


def test_callable_rows_spec_receives_runtime_context_and_env_overrides(tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def _load_rows(split: str, context) -> list[dict[str, object]]:
        seen["split"] = split
        seen["data_root"] = str(context.data_root)
        seen["hf_home"] = os.environ["RWKV_SKILLS_HF_HOME"]
        seen["datasets_cache"] = os.environ["HF_DATASETS_CACHE"]
        return [{"id": 1}]

    spec = CallableRowsDatasetSpec("runtime_ctx", tmp_path, "test", load_rows=_load_rows)

    paths = ensure_dataset_materialized(spec)

    assert paths == [tmp_path / "runtime_ctx" / "test.jsonl"]
    assert seen["split"] == "test"
    assert seen["data_root"] == str(tmp_path.resolve())
    assert seen["hf_home"] == str((tmp_path / "cache" / "hf_cache").resolve())
    assert seen["datasets_cache"] == str((tmp_path / "cache" / "hf_cache" / "datasets").resolve())


def test_prepare_dataset_routes_ifbench_cache_into_output_root(tmp_path: Path, monkeypatch) -> None:
    def _fake_download(_url: str, target_path: Path) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps({"prompt": "Follow this", "key": "7"}) + "\n", encoding="utf-8")
        return target_path

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.instruction_following.ifbench.download_file",
        _fake_download,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("ifbench", output_root, "test")

    assert paths == [output_root / "ifbench" / "test.jsonl"]
    assert (output_root / "cache" / "ifbench" / "IFBench_test.jsonl").exists()
    assert read_jsonl_items(paths[0]) == [{"prompt": "Follow this", "question": "Follow this", "key": 7}]


def test_benchmark_family_registries_are_now_spec_only(tmp_path: Path) -> None:
    _ensure_registries()

    registries = (
        MULTIPLE_CHOICE_REGISTRY,
        FREE_ANSWER_REGISTRY,
        INSTRUCTION_FOLLOWING_REGISTRY,
        CODE_GENERATION_REGISTRY,
        FUNCTION_CALLING_REGISTRY,
    )

    assert all(not getattr(registry, "_preparers") for registry in registries)
    assert all(
        registry.get_spec_factory(name)(tmp_path, "test").source_kind != "legacy_prepper"
        for registry in registries
        for name in registry.names()
        if registry.get_spec_factory(name) is not None
    )
