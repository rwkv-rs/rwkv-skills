from __future__ import annotations

import json

from src.eval.tasks.function_calling.mcp_bench_worker import _is_transient_evaluator_error, load_server_configs


def test_load_server_configs_overrides_package_index_env(tmp_path, monkeypatch) -> None:
    runtime_root = tmp_path / "runtime"
    servers_root = runtime_root / "mcp_servers"
    servers_root.mkdir(parents=True)
    runtime_python = runtime_root / ".venv" / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    (servers_root / "commands.json").write_text(
        json.dumps(
            {
                "BioMCP": {
                    "cmd": "uv run biomcp run",
                    "env": ["NCI_API_KEY"],
                    "cwd": "../biomcp",
                }
            }
        ),
        encoding="utf-8",
    )
    (servers_root / "api_key").write_text("NCI_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://broken.example/simple")
    monkeypatch.setenv("UV_INDEX_URL", "https://broken.example/simple")
    monkeypatch.setenv("PIP_INDEX_URL", "https://broken.example/simple")
    monkeypatch.setenv("UV_PYTHON", "3.14")
    monkeypatch.setenv("RWKV_MCP_PACKAGE_INDEX_URL", "https://pypi.org/simple")

    config = load_server_configs(runtime_root, ["BioMCP"])[0]

    assert config["command"] == ["uv", "run", "biomcp", "run"]
    assert config["cwd"] == str((runtime_root / "mcp_servers" / "biomcp").resolve())
    assert config["env"]["NCI_API_KEY"] == "test-key"
    assert config["env"]["UV_DEFAULT_INDEX"] == "https://pypi.org/simple"
    assert config["env"]["UV_INDEX_URL"] == "https://pypi.org/simple"
    assert config["env"]["PIP_INDEX_URL"] == "https://pypi.org/simple"
    assert config["env"]["UV_PYTHON"] == str(runtime_python)
    assert config["env"]["UV_NO_DEV"] == "1"


def test_transient_evaluator_error_accepts_judge_database_error() -> None:
    assert _is_transient_evaluator_error(
        RuntimeError("official evaluator returned a non-dict payload: LLM judge evaluation failed: Database error")
    )
