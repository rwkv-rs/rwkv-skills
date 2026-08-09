from __future__ import annotations

import json
from pathlib import Path
import zipfile

from src.eval.datasets.data_prepper.data_manager import (
    available_code_generation_datasets,
    available_free_answer_datasets,
    available_function_calling_datasets,
    prepare_dataset,
)
from src.eval.datasets.runtime import read_jsonl_items
from src.eval.tasks.function_calling import BrowseCompRecord, McpBenchItem, McpBenchTaskSpec


_MCP_BENCH_SPLIT_FILES = (
    "mcpbench_tasks_single_runner_format.json",
    "mcpbench_tasks_multi_2server_runner_format.json",
    "mcpbench_tasks_multi_3server_runner_format.json",
)


def _stage_mcp_bench_split_files(tasks_root: Path, file_names: tuple[str, ...] = _MCP_BENCH_SPLIT_FILES) -> None:
    tasks_root.mkdir(parents=True, exist_ok=True)
    payload = {"server_tasks": []}
    for file_name in file_names:
        (tasks_root / file_name).write_text(json.dumps(payload), encoding="utf-8")


def test_available_function_calling_datasets_lists_registered_specs() -> None:
    names = set(available_function_calling_datasets())

    assert "browsecomp" in names
    assert "browsecomp_zh" in names
    assert "complexfuncbench_official" in names
    assert "longbench" in names
    assert "longbench_qa" in names
    assert "longbench_qa_balanced" in names
    assert "longcodeqa" in names
    assert "mcp_bench" in names
    assert "mcp_bench_single" in names
    assert "mcp_bench_multi_2server" in names
    assert "mcp_bench_multi_3server" in names
    assert "apibank_l1" in names
    assert "apibank_l2" in names
    assert "apibank_level1" in names
    assert "apibank_level2" in names
    assert "agentbench_db" in names
    assert "agentbench_kg" in names
    assert "widesearch" in names
    assert "mcp_atlas" in names
    assert "terminal_bench_2_1" in names
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
    assert "tau3_bench_mock" in names
    assert "tau3_bench_mock_long_context" in names
    assert "terminal_bench_2_1" in names
    assert "nl2repo" in names
    assert "deep_swe" in names
    assert "wide_search" in names
    assert "deepsearchqa" in names
    assert "mcp_atlas" in names
    assert "toolathlon" in names
    assert "apex_agents" in names
    assert "claw_eval" in names
    assert "wildclawbench" in names
    assert "skillsbench" in names
    assert "hle_tools" in names
    assert "hy_backend_2_0" in names
    assert "hy_swe_max" in names
    assert "hy_companybench" in names
    assert "e_bench" in names
    assert "hy_finmodelbench" in names
    assert "prodbench" in names
    assert "hy_skillsworld" in names
    assert "hy_euler_pro" in names


def test_available_free_answer_datasets_lists_hy_math() -> None:
    names = set(available_free_answer_datasets())

    assert "hy_math" in names


def test_available_code_generation_datasets_lists_swe_bench_expansions() -> None:
    names = set(available_code_generation_datasets())

    assert "swe_bench_multilingual" in names
    assert "swe_bench_pro" in names


def test_prepare_dataset_materializes_source_required_agent_slot(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "official-agent-data"
    terminal_source = source_root / "terminal_bench_2_1"
    terminal_source.mkdir(parents=True)
    (terminal_source / "tasks.jsonl").write_text(
        json.dumps(
            {
                "id": "terminal-demo",
                "prompt": "Create a file named done.txt.",
                "answer": "done.txt exists",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_AGENT_BENCHMARKS_ROOT", str(source_root))

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("terminal_bench_2_1", output_root, "test")

    assert paths == [output_root / "terminal_bench_2_1" / "test.jsonl"]
    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "terminal-demo"
    assert row["instruction"] == "Create a file named done.txt."
    assert row["answer"] == "done.txt exists"
    assert row["metadata"]["official_sandbox_required"] is True
    assert row["metadata"]["official_sandbox_env"] == "RWKV_TERMINAL_BENCH_2_1_SANDBOX_ROOT"
    assert row["official_payload"]["prompt"] == "Create a file named done.txt."


def test_prepare_dataset_materializes_terminal_bench_official_checkout(tmp_path: Path, monkeypatch) -> None:
    official_root = tmp_path / "terminal-bench"
    task_dir = official_root / "original-tasks" / "fix-demo"
    task_dir.mkdir(parents=True)
    (task_dir / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n", encoding="utf-8")
    (task_dir / "run-tests.sh").write_text("test -f /app/done.txt\n", encoding="utf-8")
    (task_dir / "task.yaml").write_text(
        "\n".join(
            [
                "instruction: |-",
                "  Create /app/done.txt.",
                "difficulty: easy",
                "category: filesystem",
                "parser_name: pytest",
                "max_agent_timeout_sec: 120",
                "max_test_timeout_sec: 30",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_TERMINAL_BENCH_ROOT", str(official_root))

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("terminal_bench_2_1", output_root, "test")

    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "fix-demo"
    assert row["instruction"] == "Create /app/done.txt."
    assert row["executor"]["kind"] == "shell_sandbox"
    assert row["executor"]["config"]["backend"] == "docker"
    assert row["executor"]["config"]["dockerfile_context"] == str(task_dir)
    assert row["verifier"]["kind"] == "terminal_bench_official"
    assert row["verifier"]["config"]["official_task_id"] == "fix-demo"
    assert row["metadata"]["official_task_id"] == "fix-demo"


def test_prepare_dataset_materializes_terminal_bench_compose_client_build(
    tmp_path: Path,
    monkeypatch,
) -> None:
    official_root = tmp_path / "terminal-bench"
    task_dir = official_root / "original-tasks" / "compose-demo"
    client_dir = task_dir / "client"
    client_dir.mkdir(parents=True)
    (client_dir / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n", encoding="utf-8")
    (task_dir / "docker-compose.yaml").write_text(
        "\n".join(
            [
                "services:",
                "  client:",
                "    build:",
                "      context: client",
                "      dockerfile: Dockerfile",
                "    image: ${T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME}",
                "    container_name: ${T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME}",
            ]
        ),
        encoding="utf-8",
    )
    (task_dir / "run-tests.sh").write_text("test -f /app/done.txt\n", encoding="utf-8")
    (task_dir / "task.yaml").write_text("instruction: Create /app/done.txt.\n", encoding="utf-8")
    monkeypatch.setenv("RWKV_TERMINAL_BENCH_ROOT", str(official_root))

    [path] = prepare_dataset("terminal_bench_2_1", tmp_path / "prepared", "test")
    [row] = read_jsonl_items(path)

    assert row["task_id"] == "compose-demo"
    assert row["executor"]["config"]["dockerfile_context"] == str(client_dir.resolve())
    assert row["executor"]["config"]["dockerfile_path"] == str((client_dir / "Dockerfile").resolve())
    assert row["executor"]["config"]["docker_compose_file"] == str(task_dir / "docker-compose.yaml")
    assert row["metadata"]["docker_compose_file"] == str(task_dir / "docker-compose.yaml")


def test_prepare_dataset_materializes_nl2repo_official_checkout(tmp_path: Path, monkeypatch) -> None:
    official_root = tmp_path / "NL2RepoBench"
    project_dir = official_root / "test_files" / "demo_project"
    project_dir.mkdir(parents=True)
    (project_dir / "start.md").write_text("Create a package named demo_project.", encoding="utf-8")
    (project_dir / "test_commands.json").write_text(json.dumps(["pip install -e .", "pytest tests"]), encoding="utf-8")
    (project_dir / "test_files.json").write_text(json.dumps(["tests"]), encoding="utf-8")
    (project_dir / "test_case_count.txt").write_text("3", encoding="utf-8")
    monkeypatch.setenv("RWKV_NL2REPO_ROOT", str(official_root))

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("nl2repo", output_root, "test")

    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "demo_project"
    assert row["instruction"].startswith("Create a package named demo_project.")
    assert row["executor"]["kind"] == "shell_sandbox"
    assert row["executor"]["config"]["backend"] == "subprocess"
    assert row["verifier"]["kind"] == "nl2repo_official"
    assert row["verifier"]["config"]["official_task_id"] == "demo_project"
    assert row["metadata"]["official_task_id"] == "demo_project"
    assert row["metadata"]["test_case_count"] == 3


def test_prepare_dataset_materializes_deepswe_official_checkout(tmp_path: Path, monkeypatch) -> None:
    official_root = tmp_path / "deep-swe"
    task_dir = official_root / "tasks" / "fix-demo"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Fix the demo bug.", encoding="utf-8")
    (task_dir / "pre_artifacts.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (task_dir / "tests" / "grader.py").write_text("# grader\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                "[metadata]",
                'task_id = "fix-demo"',
                'display_title = "Fix demo"',
                'repository_url = "https://example.test/repo.git"',
                'language = "python"',
                'category = "bugfix"',
                'base_commit_hash = "abc123"',
                "[environment]",
                'docker_image = "example/deepswe:fix-demo"',
                "[verifier]",
                "timeout_sec = 123",
                "[agent]",
                "timeout_sec = 456",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_DEEPSWE_ROOT", str(official_root))

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("deepswe", output_root, "test")

    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "fix-demo"
    assert row["executor"]["kind"] == "shell_sandbox"
    assert row["executor"]["config"]["backend"] == "docker"
    assert row["executor"]["config"]["image"] == "example/deepswe:fix-demo"
    assert row["executor"]["config"]["docker_copy_paths"][0]["dst"] == "/tests"
    assert row["verifier"]["kind"] == "repo_tests_official"
    assert row["verifier"]["config"]["test_command"] == "bash /pre_artifacts.sh && bash /tests/test.sh"
    assert row["verifier"]["config"]["test_timeout_s"] == 123.0
    assert row["metadata"]["official_task_id"] == "fix-demo"


def test_prepare_dataset_materializes_agent_loop_answer_rows(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "sources"
    source_dir = source_root / "widesearch"
    source_dir.mkdir(parents=True)
    (source_dir / "test.jsonl").write_text(
        json.dumps(
            {
                "id": "wide-1",
                "question": "Who wrote The Left Hand of Darkness?",
                "answer": ["Ursula K. Le Guin", "Ursula Le Guin"],
                "context": "Use the provided search tools if needed.",
                "topic": "books",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_AGENT_LOOP_SOURCE_ROOT", str(source_root))

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("widesearch", output_root, "test")

    assert paths == [output_root / "widesearch" / "test.jsonl"]
    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "wide-1"
    assert row["instruction"].startswith("Context:\nUse the provided search tools")
    assert row["executor"]["kind"] == "web_search"
    assert row["verifier"]["kind"] == "widesearch_official"
    assert row["expected_tool_calls"][0]["name"] == "final_answer"
    assert row["metadata"]["source_format"] == "rwkvc_agent_loop"
    assert row["metadata"]["source_benchmark"] == "widesearch"


def test_prepare_dataset_materializes_agent_loop_explicit_manifest(tmp_path: Path, monkeypatch) -> None:
    source_file = tmp_path / "mcp_atlas.jsonl"
    source_file.write_text(
        json.dumps(
            {
                "task_id": "atlas-1",
                "instruction": "Find the weather in Tokyo.",
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
                "expected_tool_calls": [
                    {
                        "name": "get_weather",
                        "arguments": {"location": "Tokyo"},
                        "argument_options": {"location": ["Tokyo", "Tokyo, Japan"]},
                    }
                ],
                "executor": {"kind": "manifest_replay", "config": {}},
                "verifier": {"kind": "expected_tool_calls", "config": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RWKV_AGENT_LOOP_SOURCE_MCP_ATLAS", str(source_file))

    paths = prepare_dataset("mcp_atlas", tmp_path / "prepared", "test")

    [row] = read_jsonl_items(paths[0])
    assert row["tools"][0]["name"] == "get_weather"
    assert row["expected_tool_calls"][0]["name"] == "get_weather"
    assert row["executor"]["kind"] == "manifest_replay"
    assert row["verifier"]["kind"] == "expected_tool_calls"
    assert row["metadata"]["source_benchmark"] == "mcp_atlas"


def test_prepare_dataset_materializes_agent_loop_hf_source(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RWKV_AGENT_LOOP_SOURCE_ROOT", str(tmp_path / "missing-agent-loop-sources"))
    monkeypatch.delenv("RWKV_AGENT_LOOP_SOURCE_DEEPSEARCHQA", raising=False)
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.agent_loop.load_hf_rows",
        lambda source, split: [
            {
                "id": f"{source}:{split}:0",
                "question": "Find the cited answer.",
                "answer": "reference value",
            },
            {
                "id": f"{source}:{split}:1",
                "question": "This official row has no released answer.",
                "answer": None,
            }
        ],
    )

    paths = prepare_dataset("deepsearchqa", tmp_path / "prepared", "test")

    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "https://huggingface.co/datasets/google/deepsearchqa:test:0"
    assert row["executor"]["kind"] == "web_search"
    assert row["verifier"]["kind"] == "llm_rubric_judge"
    assert row["verifier"]["config"]["reference_answer"] == "reference value"
    assert row["metadata"]["source_benchmark"] == "deepsearchqa"
    assert row["metadata"]["source_path"] == "https://huggingface.co/datasets/google/deepsearchqa"
    assert len(read_jsonl_items(paths[0])) == 1


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
    assert row["instruction"] == 'Conversation transcript JSON:\n[{"role":"user","content":"What is 2 plus 2?"}]'
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


def test_prepare_dataset_materializes_longbench_qa_spec(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "longbench"
    source.mkdir()
    (source / "hotpotqa.jsonl").write_text(
        '{"_id":"hp1","input":"Who wrote the book?","context":"Alice wrote the book.",'
        '"answers":["Alice"],"length":24}\n',
        encoding="utf-8",
    )
    (source / "passage_count.jsonl").write_text(
        '{"_id":"pc1","input":"How many passages?","context":"passage one",'
        '"answers":["1"],"length":11}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.longbench.longbench_root",
        lambda: source,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("longbench_qa", output_root, "test")

    assert paths == [output_root / "longbench_qa" / "test.jsonl"]
    assert read_jsonl_items(paths[0]) == [
        {
            "task_id": "hp1",
            "dataset": "hotpotqa",
            "input": "Who wrote the book?",
            "context": "Alice wrote the book.",
            "answers": ["Alice"],
            "all_classes": [],
            "language": "en",
            "length": 24,
            "category": "multi_doc_qa",
            "source_path": str((source / "hotpotqa.jsonl").resolve()),
        }
    ]


def test_prepare_dataset_materializes_longbench_qa_balanced_spec(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "longbench"
    source.mkdir()
    (source / "hotpotqa.jsonl").write_text(
        "\n".join(
            [
                '{"_id":"hp1","input":"Question hp1","context":"Context hp1","answers":["A"],"length":11}',
                '{"_id":"hp2","input":"Question hp2","context":"Context hp2","answers":["B"],"length":11}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "qasper.jsonl").write_text(
        "\n".join(
            [
                '{"_id":"qa1","input":"Question qa1","context":"Context qa1","answers":["C"],"length":11}',
                '{"_id":"qa2","input":"Question qa2","context":"Context qa2","answers":["D"],"length":11}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.longbench.longbench_root",
        lambda: source,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("longbench_qa_balanced", output_root, "test")

    assert paths == [output_root / "longbench_qa_balanced" / "test.jsonl"]
    rows = read_jsonl_items(paths[0])
    assert [row["task_id"] for row in rows] == ["hp1", "qa1", "hp2", "qa2"]
    assert [row["dataset"] for row in rows] == ["hotpotqa", "qasper", "hotpotqa", "qasper"]


def test_prepare_dataset_materializes_longcodeqa_spec(tmp_path: Path, monkeypatch) -> None:
    archive = tmp_path / "LongCodeQA.zip"
    row = {
        "prompt_goal": "Answer with a letter.",
        "repo_text": "Repository:\n[start of a.py]\nVALUE = 1",
        "question": "Question:\nWhat is VALUE?\nA) 0\nB) 1\n",
        "prompt": "Answer with a letter.\nRepository: Repository:\n[start of a.py]\nVALUE = 1\nQuestion:\nWhat is VALUE?\nA) 0\nB) 1\n",
        "correct_letter": "B",
        "repo": "demo/repo",
        "is_hard": "No",
    }
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("LQA/32K.json", json.dumps([row]))

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.longcodebench.longcodebench_source",
        lambda: archive,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("longcodeqa", output_root, "test")

    assert paths == [output_root / "longcodeqa" / "test.jsonl"]
    [parsed] = read_jsonl_items(paths[0])
    assert parsed["task_id"] == "longcodeqa_32k_00000"
    assert parsed["repo"] == "demo/repo"
    assert parsed["context_bucket"] == "32K"
    assert parsed["correct_letter"] == "B"


def test_prepare_dataset_materializes_mcp_bench_spec(tmp_path: Path, monkeypatch) -> None:
    datasets_root = tmp_path / "rwkv_rs"
    source_root = datasets_root / "mcp_bench"
    tasks_root = source_root / "tasks"
    runtime_root = source_root / "runtime"
    _stage_mcp_bench_split_files(tasks_root)
    runtime_root.mkdir(parents=True)

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.mcp_bench.rwkv_rs_datasets_root",
        lambda: datasets_root,
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
            "task_assets_commit_hint": "official_accenture_mcp_bench",
            "official_source_root": str(source_root),
        }
    ]


def test_prepare_dataset_materializes_mcp_bench_split_specs(tmp_path: Path, monkeypatch) -> None:
    datasets_root = tmp_path / "rwkv_rs"
    source_root = datasets_root / "mcp_bench"
    tasks_root = source_root / "tasks"
    runtime_root = source_root / "runtime"
    _stage_mcp_bench_split_files(tasks_root, ("mcpbench_tasks_single_runner_format.json",))
    runtime_root.mkdir(parents=True)
    seen_file_names: list[tuple[str, ...] | None] = []

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.mcp_bench.rwkv_rs_datasets_root",
        lambda: datasets_root,
    )

    def _load_items(_tasks_root, _runtime_root, *, file_names=None):
        seen_file_names.append(tuple(file_names) if file_names is not None else None)
        return [
            McpBenchItem(
                task_file=(file_names or ("all.json",))[0],
                server_name="calendar",
                combination_name="calendar_only",
                combination_type="single",
                servers=("calendar",),
                task=McpBenchTaskSpec(task_id="task-1", task_description="Schedule the meeting"),
                runtime_root=str(runtime_root),
            )
        ]

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.mcp_bench.load_mcp_bench_task_items",
        _load_items,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("mcp_bench_single", output_root, "test")

    assert paths == [output_root / "mcp_bench_single" / "test.jsonl"]
    [row] = read_jsonl_items(paths[0])
    assert row["task_file"] == "mcpbench_tasks_single_runner_format.json"
    assert row["task_assets_commit_hint"] == "official_accenture_mcp_bench"
    assert row["official_source_root"] == str(source_root)
    assert seen_file_names == [("mcpbench_tasks_single_runner_format.json",)]


def test_prepare_dataset_materializes_mcp_bench_runtime_tasks_layout(tmp_path: Path, monkeypatch) -> None:
    datasets_root = tmp_path / "rwkv_rs"
    source_root = datasets_root / "mcp_bench"
    runtime_root = source_root / "runtime"
    tasks_root = runtime_root / "tasks"
    _stage_mcp_bench_split_files(tasks_root)

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.mcp_bench.rwkv_rs_datasets_root",
        lambda: datasets_root,
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
    assert item["task_assets_commit_hint"] == "official_accenture_mcp_bench"
    assert item["official_source_root"] == str(source_root)


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
                "bfcl_official_function": [
                    {
                        "name": "calculate_area",
                        "description": "Calculate area",
                        "parameters": {
                            "type": "dict",
                            "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
                            "required": ["base", "height"],
                        },
                    }
                ],
                "bfcl_official_ground_truth": [
                    {"calculate_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
                ],
                "bfcl_official_language": "python",
            },
        }
    ]


def test_bfcl_small_rows_keep_official_root_metadata(tmp_path: Path, monkeypatch) -> None:
    official_root = tmp_path / "berkeley-function-call-leaderboard"
    source_root = official_root / "bfcl_eval" / "data"
    source_root.mkdir(parents=True)
    question_path = source_root / "BFCL_v4_simple_python.json"
    answer_path = source_root / "possible_answer" / "BFCL_v4_simple_python.json"
    answer_path.parent.mkdir()
    question_path.write_text('{"id":"simple_python_0"}\n', encoding="utf-8")
    answer_path.write_text('{"id":"simple_python_0"}\n', encoding="utf-8")

    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_small.bfcl_small_source_root",
        lambda: source_root,
    )
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.bfcl_small.load_bfcl_ast_rows_from_sources",
        lambda _question, _answer, *, category: [
            {
                "task_id": "simple_python_0",
                "instruction": "User: Find the area.",
                "tools": [],
                "expected_tool_calls": [],
                "metadata": {"category": category},
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("bfcl_simple_python", output_root, "test")

    [row] = read_jsonl_items(paths[0])
    assert row["metadata"]["official_root"] == str(official_root.resolve())
    assert row["metadata"]["official_source"] == "gorilla/berkeley-function-call-leaderboard"


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


def test_prepare_dataset_materializes_tau3_lightweight_mock_without_tau3_source(tmp_path: Path, monkeypatch) -> None:
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
                "task_id": "create_task_1_with_env_assertions",
                "domain": domain,
                "index": 0,
                "instruction": "Create task",
                "task": {
                    "id": "create_task_1_with_env_assertions",
                    "ticket": "Create task",
                    "evaluation_criteria": {
                        "nl_assertions": ["The user is told the task was created."],
                        "reward_basis": ["DB", "ENV_ASSERTION"],
                    },
                },
                "benchmark_version": "tau_v2",
            }
        ],
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("tau3_bench_mock", output_root, "base")
    [row] = read_jsonl_items(paths[0])

    assert paths == [output_root / "tau3_bench_mock" / "base.jsonl"]
    assert row["domain"] == "mock"
    assert row["benchmark_version"] == "tau_v3_light"
    assert row["task"]["evaluation_criteria"]["reward_basis"] == ["DB", "ENV_ASSERTION"]
    assert "nl_assertions" not in row["task"]["evaluation_criteria"]


def test_prepare_dataset_materializes_tau3_lightweight_long_context(tmp_path: Path, monkeypatch) -> None:
    tau2_data_root = tmp_path / "tau2_data"
    tau2_data_root.mkdir(parents=True)
    monkeypatch.setattr(
        "src.eval.datasets.data_prepper.function_calling.tau_bench.TAU_V2_DATA_ROOT",
        tau2_data_root,
    )

    output_root = tmp_path / "prepared"
    paths = prepare_dataset("tau3_bench_mock_long_context", output_root, "base")
    rows = read_jsonl_items(paths[0])

    assert paths == [output_root / "tau3_bench_mock_long_context" / "base.jsonl"]
    assert [row["task_id"] for row in rows] == [
        "mock_long_context_create_task",
        "mock_long_context_update_task",
    ]
    assert rows[0]["benchmark_version"] == "tau_v3_light"
    history = rows[0]["task"]["initial_state"]["message_history"]
    assert len(history[0]["content"]) > 6000
    assert "Important Meeting" in history[0]["content"]
