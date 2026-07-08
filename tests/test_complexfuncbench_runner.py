from __future__ import annotations

import json
from types import SimpleNamespace

import src.eval.tasks.function_calling.complexfuncbench as complexfuncbench_module
from src.eval.tasks.function_calling.complexfuncbench import (
    AgentObservation,
    AgentStepResult,
    _trim_stop_suffixes,
    load_complexfuncbench_manifest_records,
    normalize_complexfuncbench_source_row,
    parse_complexfuncbench_calls,
    run_complexfuncbench_local_episode,
)
from src.eval.tasks.function_calling.tool_router import ToolRoutingConfig


def _official_row() -> dict:
    return {
        "id": "case-1",
        "functions": [
            {
                "name": "SearchHotel",
                "description": "Search hotels by city and guest count.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "adults": {"type": "integer"}},
                    "required": ["city", "adults"],
                },
            },
            {
                "name": "BookHotel",
                "description": "Book a hotel by hotel id.",
                "parameters": {
                    "type": "object",
                    "properties": {"hotel_id": {"type": "string"}},
                    "required": ["hotel_id"],
                },
            },
            {
                "name": "CancelFlight",
                "description": "Cancel an unrelated flight booking.",
                "parameters": {"type": "object", "properties": {"flight_id": {"type": "string"}}},
            },
        ],
        "conversations": [
            {"role": "user", "content": "Find a hotel in Paris for two adults, then book h1."},
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


def _record_from_row(row: dict, tmp_path) -> object:
    path = tmp_path / "complexfuncbench_official" / "test.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return load_complexfuncbench_manifest_records(path)[0]


def _write_official_root(path) -> object:
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


def test_complexfuncbench_rows_require_official_sandbox_metadata(tmp_path) -> None:
    official_root = _write_official_root(tmp_path / "ComplexFuncBench")
    row = normalize_complexfuncbench_source_row(_official_row(), index=0, official_root=official_root)
    assert row is not None
    record = _record_from_row(row, tmp_path)

    assert record.task_id == "complexfuncbench_official__case-1"
    assert record.env["type"] == "complexfuncbench_official"
    assert record.env["response_eval"] is True
    assert record.scorer["type"] == "complexfuncbench_official"
    assert record.metadata["complexfuncbench_official_root"] == str(official_root.resolve())
    assert "complexfuncbench_source_path" not in record.metadata


def test_complexfuncbench_episode_uses_lexical_router_and_records_format_bridge(tmp_path, monkeypatch) -> None:
    official_root = _write_official_root(tmp_path / "ComplexFuncBench")
    row = normalize_complexfuncbench_source_row(_official_row(), index=0, official_root=official_root)
    assert row is not None
    record = _record_from_row(row, tmp_path)
    outputs = iter(
        [
            '[{"name":"SearchHotel","arguments":{"city":"Paris","adults":2}},'
            '{"name":"BookHotel","arguments":{"hotel_id":"h1"}}]',
            '{"name":"final_answer","arguments":{"answer":"Booked h1."}}',
        ]
    )

    class FakeEngine:
        model_name = "fake"

        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.generate_kwargs: list[dict] = []

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            self.generate_kwargs.append(dict(kwargs))
            self.prompts.extend(str(prompt) for prompt in prompts)
            return [SimpleNamespace(text=next(outputs), finish_reason="stop")]

    class FakeEnv:
        def __init__(self) -> None:
            self.step_count = 0

        def reset(self):
            return AgentObservation(record.instruction, {"benchmark": "complexfuncbench"})

        def step_many(self, actions):  # noqa: ANN001
            self.step_count += 1
            if self.step_count == 1:
                return AgentStepResult(
                    AgentObservation(
                        "Official sandbox observation: []\nAll required official function calls matched. "
                        "Call final_answer with your final response.",
                        {"benchmark": "complexfuncbench"},
                    ),
                    done=False,
                    details={
                        "finish_reason": "official_observation",
                        "count_dict": {
                            "success_turn_num": 1,
                            "total_turn_num": 1,
                            "correct_call_num": len(actions),
                            "total_call_num": len(actions),
                            "real_turn_num": 1,
                        },
                        "call_accuracy": 1.0,
                        "resp_eval": None,
                    },
                )
            return AgentStepResult(
                AgentObservation("Final response recorded.", {"benchmark": "complexfuncbench", "done": True}),
                done=True,
                score=1.0,
                success=True,
                details={
                    "finish_reason": "final_answer",
                    "message": "Success.",
                    "count_dict": {
                        "success_turn_num": 1,
                        "total_turn_num": 1,
                        "correct_call_num": 2,
                        "total_call_num": 2,
                        "real_turn_num": 1,
                    },
                    "call_accuracy": 1.0,
                    "resp_eval": {"complete": {"score": 2}, "correct": {"score": 2}},
                    "final_response": actions[0].arguments["answer"],
                },
            )

    monkeypatch.setattr(complexfuncbench_module, "create_complexfuncbench_env", lambda _record: FakeEnv())
    engine = FakeEngine()
    episode = run_complexfuncbench_local_episode(
        record,
        engine=engine,
        sampling=object(),
        tool_routing_config=ToolRoutingConfig(
            mode="lexical",
            max_tools=2,
            trigger_tool_count=1,
            trigger_catalog_chars=1,
            context_chars=2000,
        ),
    )

    assert episode.success is True
    assert episode.count_dict["correct_call_num"] == 2
    assert episode.call_accuracy == 1.0
    assert episode.final_response == "Booked h1."
    assert len(episode.stages) == 2
    assert [bridge["official_sandbox_calls"] for bridge in episode.format_bridges] == [
        [
            {"name": "SearchHotel", "arguments": {"city": "Paris", "adults": 2}},
            {"name": "BookHotel", "arguments": {"hotel_id": "h1"}},
        ],
        [{"name": "final_answer", "arguments": {"answer": "Booked h1."}}],
    ]
    assert any(route["mode"] == "lexical" and route["routed"] for route in episode.tool_routes)
    assert engine.prompts[0].startswith("System: Tools:\n")
    assert "Available tools:" not in engine.prompts[0]
    assert "Tool router trace" not in engine.prompts[0]
    assert "Function output:" in engine.prompts[1]
    assert "CancelFlight" not in engine.prompts[0]
    assert all(kwargs.get("show_progress") is False for kwargs in engine.generate_kwargs)


def test_parse_complexfuncbench_calls_accepts_json_array() -> None:
    calls = parse_complexfuncbench_calls(
        '[{"name":"SearchHotel","arguments":{"city":"Paris"}},{"name":"BookHotel","arguments":{"hotel_id":"h1"}}]'
    )

    assert [call.name for call in calls] == ["SearchHotel", "BookHotel"]
    assert calls[0].arguments == {"city": "Paris"}


def test_trim_stop_suffixes_preserves_opening_json_fence() -> None:
    text = '```json\n{"name":"SearchHotel","arguments":{"city":"Paris"}}\n```'

    assert _trim_stop_suffixes(text, ("\n```", "```")).strip() == '```json\n{"name":"SearchHotel","arguments":{"city":"Paris"}}'
