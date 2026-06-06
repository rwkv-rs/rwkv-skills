from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types

import pytest

from src.eval.datasets.data_prepper.data_manager import prepare_dataset
from src.eval.datasets.runtime import read_jsonl_items
from src.eval.function_calling.complexfuncbench import (
    AgentObservation,
    COMPLEXFUNCBENCH_FINAL_SCHEMA,
    ComplexFuncBenchOfficialEnv,
    ToolAction,
    build_complexfuncbench_format_bridge,
    build_complexfuncbench_prompt,
    load_complexfuncbench_manifest_records,
    parse_complexfuncbench_tool_calls,
    summarize_complexfuncbench_official_payloads,
)
from src.eval.function_calling import complexfuncbench as complexfuncbench_module
from src.eval.function_calling.tool_router import ToolRoutingConfig, route_tools_for_prompt


def _official_source_row() -> dict:
    return {
        "id": "case-1",
        "functions": [
            {
                "name": "SearchHotel",
                "description": "Search hotels.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "adults": {"type": "integer"}},
                    "required": ["city", "adults"],
                },
            },
            {
                "name": "BookHotel",
                "description": "Book hotels.",
                "parameters": {
                    "type": "object",
                    "properties": {"hotel_id": {"type": "string"}},
                    "required": ["hotel_id"],
                },
            },
        ],
        "conversations": [
            {"role": "user", "content": "Find a hotel in Paris for two adults."},
            {
                "role": "assistant",
                "function_call": [
                    {"name": "SearchHotel", "arguments": {"city": "Paris", "adults": 2}},
                    {"name": "BookHotel", "arguments": {"hotel_id": "h1"}},
                ],
            },
            {"role": "observation", "content": [{"hotel_id": "h1"}, {"status": "booked"}]},
        ],
    }


def _write_official_root(path: Path) -> Path:
    for relative in (
        "runner/base_runner.py",
        "runner/response_runner.py",
        "utils/compare_method.py",
        "utils/rapidapi.py",
        "utils/tool_info.json",
        "utils/exact_match_values.json",
        "models/gpt.py",
        "prompts/compare.py",
        "prompts/response.py",
        "prompts/prompts.py",
    ):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}" if target.suffix == ".json" else "# test stub\n", encoding="utf-8")
    return path


def test_prepare_dataset_materializes_complexfuncbench_official(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "ComplexFuncBench.jsonl"
    source.write_text(json.dumps(_official_source_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    official_root = _write_official_root(tmp_path / "ComplexFuncBench")
    monkeypatch.setenv("RWKV_COMPLEXFUNCBENCH_SOURCE", str(source))
    monkeypatch.setenv("RWKV_COMPLEXFUNC_OFFICIAL_ROOT", str(official_root))

    paths = prepare_dataset("complexfuncbench_official", tmp_path / "out", "test")

    assert paths == [tmp_path / "out" / "complexfuncbench_official" / "test.jsonl"]
    [row] = read_jsonl_items(paths[0])
    assert row["task_id"] == "complexfuncbench_official__case-1"
    assert row["env"]["type"] == "complexfuncbench_official"
    assert row["scorer"]["type"] == "complexfuncbench_official"
    assert row["metadata"]["complexfuncbench_total_call_num"] == 2
    assert any(tool["name"] == COMPLEXFUNCBENCH_FINAL_SCHEMA["name"] for tool in row["tools"])


def test_prepare_dataset_requires_complexfuncbench_official_root(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "ComplexFuncBench.jsonl"
    source.write_text(json.dumps(_official_source_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    monkeypatch.setenv("RWKV_COMPLEXFUNCBENCH_SOURCE", str(source))
    monkeypatch.delenv("RWKV_COMPLEXFUNC_OFFICIAL_ROOT", raising=False)
    monkeypatch.delenv("RWKV_COMPLEXFUNCBENCH_OFFICIAL_ROOT", raising=False)

    with pytest.raises(ValueError, match="official sandbox"):
        prepare_dataset("complexfuncbench_official", tmp_path / "out", "test")


def test_complexfuncbench_prompt_uses_routed_tool_window(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "ComplexFuncBench.jsonl"
    source.write_text(json.dumps(_official_source_row(), ensure_ascii=False) + "\n", encoding="utf-8")
    official_root = _write_official_root(tmp_path / "ComplexFuncBench")
    monkeypatch.setenv("RWKV_COMPLEXFUNCBENCH_SOURCE", str(source))
    monkeypatch.setenv("RWKV_COMPLEXFUNC_OFFICIAL_ROOT", str(official_root))
    [path] = prepare_dataset("complexfuncbench_official", tmp_path / "out", "test")
    [record] = load_complexfuncbench_manifest_records(path)

    route = route_tools_for_prompt(
        record.tools,
        [{"role": "user", "content": record.instruction}],
        config=ToolRoutingConfig(mode="lexical", max_tools=1, trigger_tool_count=1, trigger_catalog_chars=1),
        control_tool_names=(COMPLEXFUNCBENCH_FINAL_SCHEMA["name"],),
    )
    prompt = build_complexfuncbench_prompt(record, [], AgentObservation(record.instruction), 0, tool_route=route)

    assert "Tool router trace" in prompt
    assert "final_answer" in prompt
    assert "JSON array" in prompt
    assert prompt.endswith("Assistant: ```json\n{")
    assert route.trace_payload()["routed"] is True


def test_complexfuncbench_parser_accepts_prefill_and_agentic_formats() -> None:
    prefilled = parse_complexfuncbench_tool_calls('"name":"SearchHotel","arguments":{"city":"Paris","adults":2}}')
    agentic = parse_complexfuncbench_tool_calls('**Tool Call:** SearchHotel(city="Paris", adults=2)')

    assert [(call.name, call.arguments) for call in prefilled] == [
        ("SearchHotel", {"city": "Paris", "adults": 2})
    ]
    assert [(call.name, call.arguments) for call in agentic] == [
        ("SearchHotel", {"city": "Paris", "adults": 2})
    ]


def test_complexfuncbench_official_env_accepts_parallel_calls_and_final_answer() -> None:
    class FakeCompare:
        def __init__(self, runner):
            self.runner = runner
            self.predict_lengths: list[int] = []

        def add_free_function(self, _convs):
            return None

        def compare_turn_prediction(self, _functions, _messages, function_calls, _golden_fcs, _golden_obs):
            self.predict_lengths.append(len(function_calls))
            return None, {index: {"ok": True} for index in range(len(function_calls))}, list(function_calls), {}

    class FakeRunner:
        def __init__(self):
            self.CompareClass = FakeCompare(self)
            self.unexpect_call_resp = {"error": "unexpected"}
            self.fc_chain = []
            self.turn_id = 0
            self.correct_count = 0
            self.golden_fcs = []
            self.golden_obs = []

        def init_golden(self, convs):
            self.fc_chain = []
            self.obs_chain = []
            for turn in convs:
                if "function_call" in turn:
                    self.fc_chain.append(turn["function_call"])
                elif turn.get("role") == "observation":
                    self.obs_chain.append(turn["content"])
            self.golden_fcs = list(self.fc_chain[0])
            self.golden_obs = list(self.obs_chain[0])

        def process_matches(self, success_matched):
            for matched in success_matched:
                if matched in self.golden_fcs:
                    index = self.golden_fcs.index(matched)
                    self.golden_fcs.pop(index)
                    self.golden_obs.pop(index)
            if success_matched:
                self.turn_id += 1

        def get_success_turn(self, _remain_fcs, _total_fcs):
            return self.turn_id

        def return_result(self, messages, error_info=None):
            if error_info:
                return messages, error_info, self.turn_id, self.correct_count
            if self.turn_id >= len(self.fc_chain) and not self.golden_fcs:
                return messages, "Success.", len(self.fc_chain), self.correct_count
            return messages, {"error_type": "stop_early"}, self.turn_id, self.correct_count

    class FakeSandbox:
        def __init__(self):
            self.runner = FakeRunner()

        def create_model_runner(self):
            return self.runner

        def run_response_eval(self, _official_row, _final_response):
            return {"complete": {"score": 2}, "correct": {"score": 2}}

    row = {
        "task_id": "complexfuncbench_official__case-1",
        "instruction": "Find a hotel in Paris for two adults.",
        "tools": [*_official_source_row()["functions"], COMPLEXFUNCBENCH_FINAL_SCHEMA],
        "expected_tool_calls": [],
        "env": {"type": "complexfuncbench_official", "official_root": "/tmp/not-used"},
        "scorer": {"type": "complexfuncbench_official"},
        "max_steps": 4,
        "metadata": {
            "official_id": "case-1",
            "complexfuncbench_functions": _official_source_row()["functions"],
            "complexfuncbench_conversations": _official_source_row()["conversations"],
        },
    }
    [record] = load_complexfuncbench_manifest_records(_write_jsonl_tmp(row))

    sandbox = FakeSandbox()
    env = ComplexFuncBenchOfficialEnv(record, sandbox=sandbox)
    env.reset()
    observation = env.step_many(
        [
            ToolAction("SearchHotel", {"city": "Paris", "adults": 2}),
            ToolAction("BookHotel", {"hotel_id": "h1"}),
        ]
    )
    result = env.step_many([ToolAction("final_answer", {"answer": "Booked."})])

    assert "final_answer" in observation.observation.content
    assert result.success is True
    assert result.score == 1.0
    assert sandbox.runner.CompareClass.predict_lengths == [2]
    assert result.details["count_dict"]["correct_call_num"] == 2
    assert result.details["resp_eval"] == {"complete": {"score": 2}, "correct": {"score": 2}}
    assert result.details["official_response_eval_input"]["final_response"] == "Booked."


def test_complexfuncbench_format_bridge_records_all_conversion_layers() -> None:
    actions = [
        ToolAction("SearchHotel", {"city": "Paris", "adults": 2}),
        ToolAction("BookHotel", {"hotel_id": "h1"}),
    ]
    bridge = build_complexfuncbench_format_bridge(
        '```json\n[{"name":"SearchHotel","arguments":{"city":"Paris","adults":2}},'
        '{"name":"BookHotel","arguments":{"hotel_id":"h1"}}]\n```',
        actions,
    )

    assert bridge["rwkv_output_format"] == "json_tool_call_object_or_array"
    assert bridge["internal_format"] == "ToolAction(name, arguments)"
    assert bridge["official_sandbox_format"] == "list[dict(name, arguments)]"
    assert [item["name"] for item in bridge["internal_tool_actions"]] == ["SearchHotel", "BookHotel"]
    assert bridge["official_sandbox_calls"] == [
        {"name": "SearchHotel", "arguments": {"city": "Paris", "adults": 2}},
        {"name": "BookHotel", "arguments": {"hotel_id": "h1"}},
    ]


def test_complexfuncbench_newapi_connection_sets_official_openai_env(tmp_path: Path, monkeypatch) -> None:
    official_root = _write_official_root(tmp_path / "ComplexFuncBench")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv(
        "RWKV_COMPLEXFUNCBENCH_OPENAI_CONN",
        json.dumps(
            {
                "_type": "newapi_channel_conn",
                "key": "sk-test",
                "url": "https://next-token.cc",
            }
        ),
    )

    with complexfuncbench_module._official_import_context(official_root):
        assert os.environ["OPENAI_API_KEY"] == "sk-test"
        assert os.environ["OPENAI_BASE_URL"] == "https://next-token.cc/v1"

    assert "OPENAI_API_KEY" not in os.environ
    assert "OPENAI_BASE_URL" not in os.environ


def test_complexfuncbench_openai_model_map_patches_official_gpt(monkeypatch) -> None:
    class FakeGPTModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

    module = types.SimpleNamespace(GPTModel=FakeGPTModel)
    monkeypatch.setitem(sys.modules, "runner.response_runner", module)

    complexfuncbench_module._patch_official_gpt_model_aliases(
        {"model_map": {"gpt-4o-2024-08-06": "gpt-4o-2024-11-20"}}
    )

    model = module.GPTModel("gpt-4o-2024-08-06")
    assert model.model_name == "gpt-4o-2024-11-20"
    assert model.rwkv_complexfuncbench_original_model_name == "gpt-4o-2024-08-06"


def test_complexfuncbench_official_metrics_aggregate_success_and_call_accuracy() -> None:
    metrics = summarize_complexfuncbench_official_payloads(
        [
            {
                "success": True,
                "agent_details": {
                    "final_env_details": {
                        "message": "Success.",
                        "count_dict": {"correct_call_num": 2, "total_call_num": 2},
                        "resp_eval": {"complete": {"score": 2}, "correct": {"score": 1}},
                    }
                },
            },
            {
                "success": False,
                "agent_details": {
                    "final_env_details": {
                        "message": {"error_type": "stop_early"},
                        "count_dict": {"correct_call_num": 1, "total_call_num": 2},
                        "resp_eval": {"complete": {"score": 1}, "correct": {"score": 0}},
                    }
                },
            },
            {
                "success": False,
                "metadata": {"complexfuncbench_total_call_num": 4},
            },
        ]
    )

    assert metrics.success_rate == 1 / 3
    assert metrics.call_accuracy == 3 / 8
    assert metrics.completeness == 1.5
    assert metrics.correctness == 0.5


def _write_jsonl_tmp(row: dict) -> Path:
    path = Path(os.environ.get("PYTEST_TMPDIR", "/tmp")) / "complexfuncbench_test_row.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
