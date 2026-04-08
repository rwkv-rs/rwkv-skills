from __future__ import annotations

from pathlib import Path

from src.eval.datasets.data_prepper.data_manager import available_function_calling_datasets, prepare_dataset
from src.eval.datasets.runtime import read_jsonl_items
from src.eval.function_calling import BrowseCompRecord, McpBenchItem, McpBenchTaskSpec


def test_available_function_calling_datasets_lists_registered_specs() -> None:
    names = set(available_function_calling_datasets())

    assert "browsecomp" in names
    assert "browsecomp_zh" in names
    assert "mcp_bench" in names
    assert "tau_bench_retail" in names
    assert "tau2_bench_airline" in names


def test_prepare_dataset_materializes_browsecomp_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "rwkv_rs"
    source = source_root / "browsecomp" / "browse_comp_test_set.csv"
    source.parent.mkdir(parents=True)
    source.write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.browsecomp.rwkv_rs_datasets_root",
        lambda: source_root,
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.browsecomp.load_browsecomp_rows_from_csv",
        lambda _path: [
            BrowseCompRecord(
                task_id="browsecomp_0000",
                question="What is the answer?",
                answer="42",
                locale="en",
                topic="demo",
            )
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("browsecomp", output_root, "test")

    assert paths == [output_root / "browsecomp" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "browsecomp_0000",
            "question": "What is the answer?",
            "answer": "42",
            "topic": "demo",
            "locale": "en",
            "source_path": str(source),
        }
    ]


def test_prepare_dataset_materializes_mcp_bench_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "rwkv_rs"
    tasks_root = source_root / "mcp_bench" / "tasks"
    runtime_root = source_root / "mcp_bench" / "runtime"
    tasks_root.mkdir(parents=True)
    runtime_root.mkdir(parents=True)

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.mcp_bench.rwkv_rs_datasets_root",
        lambda: source_root,
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.mcp_bench.load_mcp_bench_task_items",
        lambda _tasks_root, _runtime_root: [
            McpBenchItem(
                task_file="tasks.json",
                server_name="calendar",
                combination_name="calendar_only",
                combination_type="single",
                servers=("calendar",),
                task=McpBenchTaskSpec(
                    task_id="task-1",
                    task_description="Schedule the meeting",
                    fuzzy_description="Book the meeting",
                    dependency_analysis="none",
                    distraction_servers=(),
                ),
                runtime_root=str(runtime_root),
            )
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("mcp_bench", output_root, "test")

    assert paths == [output_root / "mcp_bench" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "task-1",
            "instruction": "Book the meeting",
            "task_file": "tasks.json",
            "server_name": "calendar",
            "combination_name": "calendar_only",
            "combination_type": "single",
            "servers": ["calendar"],
            "task": {
                "task_id": "task-1",
                "task_description": "Schedule the meeting",
                "fuzzy_description": "Book the meeting",
                "dependency_analysis": "none",
                "distraction_servers": [],
            },
            "runtime_root": str(runtime_root),
            "tasks_root": str(tasks_root),
            "task_assets_commit_hint": "local_rwkv_rs_snapshot",
        }
    ]


def test_prepare_dataset_materializes_tau2_base_split_to_standard_path(tmp_path: Path, monkeypatch) -> None:
    tau2_data_root = tmp_path / "tau2_data"
    tau2_data_root.mkdir(parents=True)

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.tau_bench.TAU_V2_DATA_ROOT",
        tau2_data_root,
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.tau_bench.load_tau_v2_tasks",
        lambda *, domain, split: [
            {
                "task_id": f"{domain}-{split}-0",
                "domain": domain,
                "index": 0,
                "instruction": "Resolve the ticket",
                "task": {"id": "ticket-0"},
                "benchmark_version": "tau_v2",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("tau2_bench_airline", output_root, "base")

    assert paths == [output_root / "tau2_bench_airline" / "base.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "airline-base-0",
            "domain": "airline",
            "index": 0,
            "instruction": "Resolve the ticket",
            "task": {"id": "ticket-0"},
            "benchmark_version": "tau_v2",
        }
    ]
