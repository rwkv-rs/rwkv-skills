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
    assert "apibank_level1" in names
    assert "apibank_level2" in names
    assert "agentbench_db" in names
    assert "agentbench_kg" in names
    assert "bfcl_simple_python" in names
    assert "bfcl_exec_multiple_ast" in names
    assert "bfcl_exec_multiple" in names
    assert "bfcl_exec_parallel" in names
    assert "bfcl_exec_parallel_multiple" in names
    assert "bfcl_v3" in names
    assert "toolalpaca_eval_simulated" in names
    assert "toolalpaca_eval_real" in names
    assert "tau_bench_retail" in names
    assert "tau2_bench_airline" in names
    assert "tau3_bench_banking_knowledge" in names


def test_prepare_dataset_materializes_api_bank_level1_spec(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "api-bank" / "lv1-lv2-samples" / "level-1-given-desc"
    source_dir.mkdir(parents=True)
    (source_dir / "Demo-level-1-1.jsonl").write_text(
        "\n".join(
            [
                '{"role":"User","text":"What is 2 plus 2?"}',
                '{"role":"API","api_name":"Calculator","param_dict":{"formula":"2+2"},'
                '"result":{"api_name":"Calculator","input":{"formula":"2+2"},"output":"4","exception":null}}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.api_bank.api_bank_lv1_lv2_dir",
        lambda: source_dir,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("apibank_level1", output_root, "test")

    assert paths == [output_root / "apibank_level1" / "test.jsonl"]
    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "apibank_level1__Demo-level-1-1_001"
    assert row["instruction"] == "User: What is 2 plus 2?"
    assert row["expected_tool_calls"] == [
        {
            "name": "Calculator",
            "arguments": {"formula": "2+2"},
            "argument_options": {"formula": ["2+2"]},
        }
    ]
    assert row["metadata"]["source_format"] == "official_api_bank"


def test_prepare_dataset_materializes_agentbench_specs(tmp_path: Path, monkeypatch) -> None:
    db_file = tmp_path / "standard.jsonl"
    db_file.write_text('{"description":"q1"}\n{"description":"q2"}\n', encoding="utf-8")
    kg_file = tmp_path / "std.json"
    kg_file.write_text('[{"question":"q","entities":[],"answer":[]}]', encoding="utf-8")

    def _data_file(dataset_name: str) -> Path:
        return db_file if dataset_name == "agentbench_db" else kg_file

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.agentbench.agentbench_data_file",
        _data_file,
    )

    output_root = tmp_path / "prepared"
    db_paths = prepare_dataset("agentbench_db", output_root, "test")
    kg_paths = prepare_dataset("agentbench_kg", output_root, "test")

    assert len(read_jsonl_items(db_paths[0])) == 2
    assert read_jsonl_items(db_paths[0])[0]["task_name"] == "dbbench-std"
    assert read_jsonl_items(kg_paths[0]) == [
        {
            "task_id": "agentbench_kg__00000",
            "task_name": "kg-std",
            "index": 0,
            "metadata": {
                "source_format": "official_agentbench_controller",
                "source_path": str(kg_file),
                "task_name": "kg-std",
            },
        }
    ]


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


def test_prepare_dataset_materializes_mcp_bench_runtime_tasks_layout(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "rwkv_rs"
    runtime_root = source_root / "mcp_bench" / "runtime"
    tasks_root = runtime_root / "tasks"
    tasks_root.mkdir(parents=True)

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
                task=McpBenchTaskSpec(task_id="task-1", task_description="Schedule the meeting"),
                runtime_root=str(runtime_root),
            )
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("mcp_bench", output_root, "test")

    item = read_jsonl_items(paths[0])[0]
    assert item["tasks_root"] == str(tasks_root)
    assert item["runtime_root"] == str(runtime_root)


def test_prepare_dataset_materializes_bfcl_v3_spec(tmp_path: Path, monkeypatch) -> None:
    source_a = tmp_path / "BFCL_v3_multi_turn_base.json"
    source_b = tmp_path / "BFCL_v3_multi_turn_miss_func.json"
    source_a.write_text('{"id":"a"}\n', encoding="utf-8")
    source_b.write_text('{"id":"b"}\n', encoding="utf-8")

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_v3.bfcl_v3_source_paths",
        lambda _split: (source_a, source_b),
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_v3.load_bfcl_v3_rows_from_source",
        lambda path: [
            {
                "task_id": f"bfcl_v3_{Path(path).stem}",
                "instruction": f"Instruction from {Path(path).name}",
                "tools": [
                    {
                        "name": "search_flights",
                        "description": "Search flights",
                        "parameters": {"type": "object", "properties": {"from": {"type": "string"}}},
                    }
                ],
                "expected_tool_calls": [
                    {
                        "name": "search_flights",
                        "arguments": {"from": "SFO"},
                        "result": {"flight_id": "F1"},
                        "error": None,
                        "state_updates": {"selected_flight": "F1"},
                        "optional": False,
                    }
                ],
                "expected_final_answers": ["Booked flight F1"],
                "expected_state": {"selected_flight": "F1"},
                "initial_state": {},
                "metadata": {"source_path": str(path)},
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("bfcl_v3", output_root, "test")

    assert paths == [output_root / "bfcl_v3" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "bfcl_v3_BFCL_v3_multi_turn_base",
            "instruction": f"Instruction from {source_a.name}",
            "tools": [
                {
                    "name": "search_flights",
                    "description": "Search flights",
                    "parameters": {"type": "object", "properties": {"from": {"type": "string"}}},
                }
            ],
            "expected_tool_calls": [
                {
                    "name": "search_flights",
                    "arguments": {"from": "SFO"},
                    "result": {"flight_id": "F1"},
                    "error": None,
                    "state_updates": {"selected_flight": "F1"},
                    "optional": False,
                }
            ],
            "expected_final_answers": ["Booked flight F1"],
            "expected_state": {"selected_flight": "F1"},
            "initial_state": {},
            "metadata": {"source_path": str(source_a)},
        },
        {
            "task_id": "bfcl_v3_BFCL_v3_multi_turn_miss_func",
            "instruction": f"Instruction from {source_b.name}",
            "tools": [
                {
                    "name": "search_flights",
                    "description": "Search flights",
                    "parameters": {"type": "object", "properties": {"from": {"type": "string"}}},
                }
            ],
            "expected_tool_calls": [
                {
                    "name": "search_flights",
                    "arguments": {"from": "SFO"},
                    "result": {"flight_id": "F1"},
                    "error": None,
                    "state_updates": {"selected_flight": "F1"},
                    "optional": False,
                }
            ],
            "expected_final_answers": ["Booked flight F1"],
            "expected_state": {"selected_flight": "F1"},
            "initial_state": {},
            "metadata": {"source_path": str(source_b)},
        }
    ]


def test_bfcl_v3_source_root_prefers_repo_raw_source(tmp_path: Path, monkeypatch) -> None:
    import src.eval.datasets.data_prepper.function_calling.bfcl_v3 as bfcl_v3_prepper

    repo_raw_root = tmp_path / "data" / "bfcl_v3_raw"
    reference_root = tmp_path / "references" / "gorilla" / "berkeley-function-call-leaderboard"
    repo_raw_root.mkdir(parents=True)
    reference_root.mkdir(parents=True)
    for name in ("RWKV_BFCL_V3_SOURCE", "RWKV_BFCL_V3_ROOT", "BFCL_V3_SOURCE", "BFCL_V3_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bfcl_v3_prepper, "REPO_ROOT", tmp_path)

    assert bfcl_v3_prepper.bfcl_v3_source_root() == repo_raw_root


def test_bfcl_v3_source_root_falls_back_to_reference_clone(tmp_path: Path, monkeypatch) -> None:
    import src.eval.datasets.data_prepper.function_calling.bfcl_v3 as bfcl_v3_prepper

    reference_root = tmp_path / "references" / "gorilla" / "berkeley-function-call-leaderboard"
    reference_root.mkdir(parents=True)
    for name in ("RWKV_BFCL_V3_SOURCE", "RWKV_BFCL_V3_ROOT", "BFCL_V3_SOURCE", "BFCL_V3_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bfcl_v3_prepper, "REPO_ROOT", tmp_path)

    assert bfcl_v3_prepper.bfcl_v3_source_root() == reference_root


def test_prepare_dataset_materializes_bfcl_small_ast_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "bfcl_data"
    possible_root = source_root / "possible_answer"
    possible_root.mkdir(parents=True)
    question_path = source_root / "BFCL_v4_simple_python.json"
    answer_path = possible_root / "BFCL_v4_simple_python.json"
    question_path.write_text(
        '{"id":"simple_python_0","question":[[{"role":"user","content":"Find the area."}]],'
        '"function":[{"name":"calculate_area","description":"Calculate area",'
        '"parameters":{"type":"dict","properties":{"base":{"type":"integer"},"height":{"type":"integer"}},'
        '"required":["base","height"]}}]}\n',
        encoding="utf-8",
    )
    answer_path.write_text(
        '{"id":"simple_python_0","ground_truth":[{"calculate_area":{"base":[10],"height":[5],"unit":["units",""]}}]}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_small.bfcl_small_source_root",
        lambda: source_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("bfcl_simple_python", output_root, "test")

    assert paths == [output_root / "bfcl_simple_python" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "simple_python_0",
            "instruction": "User: Find the area.",
            "tools": [
                {
                    "name": "calculate_area",
                    "description": "Calculate area",
                    "parameters": {
                        "type": "object",
                        "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
                        "required": ["base", "height"],
                    },
                }
            ],
            "expected_tool_calls": [
                {
                    "name": "calculate_area",
                    "arguments": {"base": 10, "height": 5, "unit": "units"},
                    "argument_options": {"base": [10], "height": [5], "unit": ["units", ""]},
                }
            ],
            "metadata": {
                "source_format": "official_bfcl_v4_ast",
                "category": "simple_python",
                "source_path": str(question_path.resolve()),
                "possible_answer_path": str(answer_path.resolve()),
                "execution_result_type": [],
            },
        }
    ]


def test_prepare_dataset_materializes_toolalpaca_spec(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "toolalpaca"
    source_root.mkdir(parents=True)
    source = source_root / "eval_simulated.json"
    source.write_text(
        """[
  {
    "Name": "DemoAPI",
    "Function_Projection": {"lookup": ["/lookup", "get"]},
    "Function_Description": {
      "lookup": "Lookup a value.\\nParameters: {\\"query\\": \\"Required. String. Search query.\\"}\\nOutput: object",
      "components": ""
    },
    "Instructions": ["Look up alpha"],
    "Golden_Answers": [[{"Action": "lookup", "Action_Input": "{\\"query\\": \\"alpha\\"}"}]]
  }
]""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.toolalpaca.toolalpaca_source_root",
        lambda: source_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("toolalpaca_eval_simulated", output_root, "test")

    assert paths == [output_root / "toolalpaca_eval_simulated" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "toolalpaca_eval_simulated__demoapi_000",
            "instruction": "Look up alpha",
            "tools": [
                {
                    "name": "lookup",
                    "description": (
                        "Lookup a value.\n"
                        'Parameters: {"query": "Required. String. Search query."}\n'
                        "Output: object"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Required. String. Search query."}},
                        "required": ["query"],
                    },
                    "metadata": {"path": "/lookup", "method": "get", "api_name": "DemoAPI"},
                }
            ],
            "expected_tool_calls": [
                {
                    "name": "lookup",
                    "arguments": {"query": "alpha"},
                    "argument_options": {"query": ["alpha"]},
                }
            ],
            "metadata": {
                "source_format": "official_toolalpaca",
                "api_name": "DemoAPI",
                "api_index": 0,
                "question_index": 0,
                "source_path": str(source),
                "execution_backend": "toolalpaca_simulator",
            },
        }
    ]


def test_prepare_dataset_materializes_tau_bench_from_vendor_test_split(tmp_path: Path, monkeypatch) -> None:
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
    paths = prepare_dataset("tau_bench_airline", output_root, "test")

    assert paths == [output_root / "tau_bench_airline" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "airline-test-0",
            "domain": "airline",
            "index": 0,
            "instruction": "Resolve the ticket",
            "task": {"id": "ticket-0"},
            "benchmark_version": "tau_bench",
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


def test_prepare_dataset_materializes_tau3_base_split_to_standard_path(tmp_path: Path, monkeypatch) -> None:
    tau2_data_root = tmp_path / "tau3_data"
    (tau2_data_root / "tau2" / "domains" / "banking_knowledge").mkdir(parents=True)
    monkeypatch.setenv("TAU3_DATA_ROOT", str(tau2_data_root))

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
    paths = prepare_dataset("tau3_bench_banking_knowledge", output_root, "base")

    assert paths == [output_root / "tau3_bench_banking_knowledge" / "base.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "banking_knowledge-base-0",
            "domain": "banking_knowledge",
            "index": 0,
            "instruction": "Resolve the ticket",
            "task": {"id": "ticket-0"},
            "benchmark_version": "tau_v3",
        }
    ]
