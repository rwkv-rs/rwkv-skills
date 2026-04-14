from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import src.eval.function_calling.bfcl_v3 as bfcl_v3_mod
from src.eval.function_calling.bfcl_v3 import (
    BfclTaskRecord,
    BfclTurn,
    BfclToolCallExpectation,
    apply_bfcl_tool_call,
    build_bfcl_ask_prompt,
    build_bfcl_cot_prompt,
    build_bfcl_decision_prompt,
    build_bfcl_handoff_prompt,
    build_bfcl_router_prompt,
    build_bfcl_system_prompt,
    build_bfcl_tool_prompt,
    build_bfcl_tool_result_message,
    build_bfcl_turn_context,
    build_bfcl_user_block,
    collect_bfcl_dataset_issues,
    decode_bfcl_exec_response,
    extract_bfcl_cot_hidden_summary,
    execute_bfcl_official_tool_call,
    evaluate_bfcl_v3_episode,
    interpret_bfcl_assistant_output,
    load_bfcl_v3_rows_from_source,
    normalize_bfcl_decision_output,
    normalize_bfcl_tool_output_safe,
    normalize_bfcl_v3_source_row,
    parse_bfcl_assistant_output,
    parse_bfcl_router_output,
    render_bfcl_assistant_tool_message,
    render_bfcl_recent_tool_window,
    render_bfcl_state_delta,
    reconstruct_bfcl_tool_output,
    start_bfcl_runtime,
)
from src.eval.function_calling.context_budget import trim_history, trim_message_history
from src.eval.function_calling.tau_bench import TauToolCall
from src.infer.constraints import build_bfcl_tool_call_constraint


def _write_support_assets(
    tmp_path: Path,
    *,
    source_name: str = "BFCL_v3_multi_turn_base.json",
    task_id: str = "multi_turn_base_59",
    include_holdout: bool = False,
) -> Path:
    source = tmp_path / source_name
    source.write_text("", encoding="utf-8")
    possible_answer_root = tmp_path / "possible_answer"
    possible_answer_root.mkdir()
    possible_answer_rows = [
        {
            "id": task_id,
            "ground_truth": [
                ["measureDistance(fromLocation='San Francisco', toLocation='Rivermist')"],
                ["getFuelLevel()"],
            ],
        }
    ]
    if include_holdout:
        possible_answer_rows[0]["ground_truth"].append(["fillFuelTank(fuelAmount=5.0)"])
    (possible_answer_root / source_name).write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in possible_answer_rows) + "\n",
        encoding="utf-8",
    )
    func_doc_root = tmp_path / "multi_turn_func_doc"
    func_doc_root.mkdir()
    tools = [
        {
            "name": "measureDistance",
            "description": "Measure the distance between two places.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "fromLocation": {"type": "string"},
                    "toLocation": {"type": "string"},
                },
            },
            "response": {"type": "dict"},
        },
        {
            "name": "getFuelLevel",
            "description": "Get the fuel level.",
            "parameters": {"type": "dict", "properties": {}},
            "response": {"type": "dict"},
        },
    ]
    if include_holdout:
        tools.append(
            {
                "name": "fillFuelTank",
                "description": "Fill the tank.",
                "parameters": {
                    "type": "dict",
                    "properties": {"fuelAmount": {"type": "float"}},
                },
                "response": {"type": "dict"},
            }
        )
    (func_doc_root / "vehicle_control.json").write_text(
        "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools) + "\n",
        encoding="utf-8",
    )
    return source


def test_normalize_bfcl_v3_source_row_accepts_canonical_shape() -> None:
    row = normalize_bfcl_v3_source_row(
        {
            "task_id": "demo-1",
            "instruction": "Search and confirm",
            "tools": [
                {
                    "name": "search",
                    "description": "Search flights",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
            "expected_tool_calls": [
                {
                    "name": "search",
                    "arguments": {"q": "SFO"},
                    "result": {"flight_id": "F1"},
                    "state_updates": {"selected": "F1"},
                }
            ],
            "expected_final_answer": "Booked F1",
        },
        index=0,
    )

    assert row["task_id"] == "demo-1"
    assert row["instruction"] == "Search and confirm"
    assert row["tools"][0]["name"] == "search"
    assert row["expected_tool_calls"][0]["arguments"] == {"q": "SFO"}
    assert row["expected_final_answers"] == ["Booked F1"]


def test_load_bfcl_v3_rows_from_source_accepts_jsonl_content_with_json_suffix(tmp_path: Path) -> None:
    source = tmp_path / "BFCL_v3_multi_turn_base.json"
    source.write_text(
        "\n".join(
            [
                json.dumps({"id": "row-1", "question": "Search and confirm"}),
                json.dumps({"id": "row-2", "question": "Book and summarize"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_bfcl_v3_rows_from_source(source)

    assert [row["task_id"] for row in rows] == ["row-1", "row-2"]
    assert [row["instruction"] for row in rows] == ["Search and confirm", "Book and summarize"]


def test_normalize_bfcl_v3_source_row_recovers_official_multi_turn_shape(tmp_path: Path) -> None:
    source = _write_support_assets(tmp_path)
    row = normalize_bfcl_v3_source_row(
        {
            "id": "multi_turn_base_59",
            "question": [
                [{"role": "user", "content": "How far is San Francisco from Rivermist?"}],
                [{"role": "user", "content": "What is the gasoline level in liters?"}],
            ],
            "initial_config": {"VehicleControlAPI": {"fuelLevel": 20}},
            "path": ["VehicleControlAPI.measureDistance", "VehicleControlAPI.getFuelLevel"],
            "involved_classes": ["VehicleControlAPI"],
        },
        index=59,
        source_path=source,
    )

    assert row["task_id"] == "multi_turn_base_59"
    assert row["instruction"].startswith("Multi-turn requests:")
    assert "Turn 1:" in row["instruction"]
    assert "User: How far is San Francisco from Rivermist?" in row["instruction"]
    assert [tool["name"] for tool in row["tools"]] == ["measureDistance", "getFuelLevel"]
    assert [step["name"] for step in row["expected_tool_calls"]] == ["measureDistance", "getFuelLevel"]
    assert row["turns"][0]["ground_truth"] == ["measureDistance(fromLocation='San Francisco', toLocation='Rivermist')"]
    assert row["initial_state"] == {"VehicleControlAPI": {"fuelLevel": 20}}
    assert row["metadata"]["source_format"] == "official_bfcl_v3_multi_turn"
    assert row["metadata"]["possible_answer_path"].endswith("BFCL_v3_multi_turn_base.json")


def test_normalize_bfcl_v3_source_row_recovers_from_marshaled_instruction_and_metadata_path(tmp_path: Path) -> None:
    source = _write_support_assets(tmp_path)
    manifest_path = tmp_path / "prepared.jsonl"
    row = normalize_bfcl_v3_source_row(
        {
            "task_id": "multi_turn_base_59",
            "instruction": (
                "[[{'role': 'user', 'content': 'How far is San Francisco from Rivermist?'}], "
                "[{'role': 'user', 'content': 'What is the gasoline level in liters?'}]]"
            ),
            "tools": [],
            "expected_tool_calls": [],
            "metadata": {
                "path": ["VehicleControlAPI.measureDistance", "VehicleControlAPI.getFuelLevel"],
                "initial_config": {"VehicleControlAPI": {"fuelLevel": 20}},
                "source_path": str(source),
            },
        },
        index=0,
        source_path=manifest_path,
    )

    assert row["instruction"].startswith("Multi-turn requests:")
    assert "Turn 2:" in row["instruction"]
    assert [tool["name"] for tool in row["tools"]] == ["measureDistance", "getFuelLevel"]
    assert [step["name"] for step in row["expected_tool_calls"]] == ["measureDistance", "getFuelLevel"]
    assert row["initial_state"] == {"VehicleControlAPI": {"fuelLevel": 20}}
    assert row["metadata"]["source_path"] == str(source)
    assert row["metadata"]["manifest_path"] == str(manifest_path)


def test_normalize_bfcl_v3_source_row_excludes_holdout_tools_until_turn(tmp_path: Path) -> None:
    source = _write_support_assets(tmp_path, source_name="BFCL_v3_multi_turn_miss_func.json", include_holdout=True)
    row = normalize_bfcl_v3_source_row(
        {
            "id": "multi_turn_base_59",
            "question": [
                [{"role": "user", "content": "How far is San Francisco from Rivermist?"}],
                [{"role": "user", "content": "What is the gasoline level in liters?"}],
                [],
            ],
            "initial_config": {"VehicleControlAPI": {"fuelLevel": 20}},
            "path": [
                "VehicleControlAPI.measureDistance",
                "VehicleControlAPI.getFuelLevel",
                "VehicleControlAPI.fillFuelTank",
            ],
            "involved_classes": ["VehicleControlAPI"],
            "missed_function": {"2": ["fillFuelTank"]},
        },
        index=59,
        source_path=source,
    )

    assert [tool["name"] for tool in row["tools"]] == ["measureDistance", "getFuelLevel"]
    assert row["turns"][2]["tool_additions"][0]["name"] == "fillFuelTank"


def test_apply_bfcl_tool_call_updates_state_for_matching_step() -> None:
    record = BfclTaskRecord(
        task_id="demo-1",
        instruction="Search and confirm",
        tools=(
            {
                "name": "search",
                "description": "Search flights",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        ),
        expected_tool_calls=(
            BfclToolCallExpectation(
                name="search",
                arguments={"q": "SFO"},
                result={"flight_id": "F1"},
                state_updates={"selected": "F1"},
            ),
        ),
        expected_state={"selected": "F1"},
    )
    runtime = start_bfcl_runtime(record)

    execution = apply_bfcl_tool_call(record, runtime, TauToolCall(name="search", arguments={"q": "SFO"}))

    assert execution.success is True
    assert execution.result == {"flight_id": "F1"}
    assert runtime.current_state == {"selected": "F1"}
    assert runtime.completed_required_steps == 1


def test_apply_bfcl_tool_call_allows_skipping_optional_prefix_step() -> None:
    record = BfclTaskRecord(
        task_id="demo-optional",
        instruction="Confirm directly",
        expected_tool_calls=(
            BfclToolCallExpectation(name="search", optional=True),
            BfclToolCallExpectation(
                name="confirm",
                arguments={"id": "F1"},
                result={"status": "confirmed"},
                state_updates={"confirmed": True},
            ),
        ),
        expected_state={"confirmed": True},
    )
    runtime = start_bfcl_runtime(record)

    execution = apply_bfcl_tool_call(record, runtime, TauToolCall(name="confirm", arguments={"id": "F1"}))

    assert execution.matched_expectation is True
    assert execution.success is True
    assert runtime.current_state == {"confirmed": True}
    assert runtime.completed_required_steps == 1
    assert runtime.next_tool_index == 2


def test_apply_bfcl_tool_call_keeps_expected_tool_error_out_of_runner_mismatches() -> None:
    record = BfclTaskRecord(
        task_id="demo-error",
        instruction="Lookup then explain failure",
        expected_tool_calls=(
            BfclToolCallExpectation(
                name="lookup",
                arguments={"id": "missing"},
                error="not found",
            ),
        ),
    )
    runtime = start_bfcl_runtime(record)

    execution = apply_bfcl_tool_call(record, runtime, TauToolCall(name="lookup", arguments={"id": "missing"}))

    assert execution.matched_expectation is True
    assert execution.success is False
    assert execution.error == "not found"
    assert runtime.completed_required_steps == 1
    assert runtime.unexpected_tool_calls == 0


def test_evaluate_bfcl_v3_episode_requires_steps_state_and_answer() -> None:
    record = BfclTaskRecord(
        task_id="demo-1",
        instruction="Search and confirm",
        tools=(
            {
                "name": "search",
                "description": "Search flights",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        ),
        expected_tool_calls=(
            BfclToolCallExpectation(
                name="search",
                arguments={"q": "SFO"},
                result={"flight_id": "F1"},
                state_updates={"selected": "F1"},
            ),
        ),
        expected_final_answers=("Booked F1",),
        expected_state={"selected": "F1"},
    )
    runtime = start_bfcl_runtime(record)
    _ = apply_bfcl_tool_call(record, runtime, TauToolCall(name="search", arguments={"q": "SFO"}))

    evaluation = evaluate_bfcl_v3_episode(
        record,
        runtime,
        "Booked F1",
        termination_reason="agent_stop",
    )

    assert evaluation.is_passed is True
    assert evaluation.reward == 1.0


def test_evaluate_bfcl_v3_episode_fails_closed_without_supervision() -> None:
    record = BfclTaskRecord(task_id="demo-empty", instruction="Do something")
    runtime = start_bfcl_runtime(record)

    evaluation = evaluate_bfcl_v3_episode(
        record,
        runtime,
        "I did something.",
        termination_reason="agent_stop",
    )

    assert evaluation.is_passed is False
    assert evaluation.reward == 0.0
    assert evaluation.details["dataset_ok"] is False
    assert evaluation.details["dataset_issue"] == "missing BFCL supervision targets"
    assert "missing BFCL supervision targets" in evaluation.fail_reason


def test_execute_bfcl_official_tool_call_uses_official_runtime(monkeypatch) -> None:
    record = BfclTaskRecord(
        task_id="multi_turn_base_59",
        instruction="Official task",
        tools=(
            {
                "name": "getFuelLevel",
                "description": "Get fuel level",
                "parameters": {"type": "dict", "properties": {}},
            },
        ),
        turns=(BfclTurn(messages=({"role": "user", "content": "fuel?"},), ground_truth=("getFuelLevel()",)),),
        involved_classes=("VehicleControlAPI",),
        initial_state={"VehicleControlAPI": {"fuelLevel": 20}},
        metadata={"official_root": "/tmp/fake-official"},
    )
    runtime = start_bfcl_runtime(record)
    runtime.official_model_name = "demo-model"

    class DummyVehicle:
        def __init__(self) -> None:
            self.fuelLevel = 25.0
            self.engine_state = "stopped"

    def _fake_execute(**_kwargs):
        return ['{"fuelLevel": 25.0}'], {"VehicleControlAPI": DummyVehicle()}

    monkeypatch.setattr(
        bfcl_v3_mod,
        "_load_bfcl_official_runtime",
        lambda _root: SimpleNamespace(
            execute_multi_turn_func_call=_fake_execute,
            multi_turn_checker=lambda *_args, **_kwargs: {"valid": True},
        ),
    )

    execution = execute_bfcl_official_tool_call(
        record,
        runtime,
        TauToolCall(name="getFuelLevel", arguments={}),
    )

    assert execution.success is True
    assert execution.result == {"fuelLevel": 25.0}
    assert runtime.current_state["VehicleControlAPI"]["fuelLevel"] == 25.0


def test_evaluate_bfcl_v3_episode_uses_official_checker(monkeypatch) -> None:
    record = BfclTaskRecord(
        task_id="multi_turn_base_59",
        instruction="Official task",
        tools=(
            {
                "name": "measureDistance",
                "description": "Measure distance",
                "parameters": {"type": "dict", "properties": {"fromLocation": {}, "toLocation": {}}},
            },
        ),
        turns=(
            BfclTurn(
                messages=({"role": "user", "content": "distance?"},),
                ground_truth=("measureDistance(fromLocation='San Francisco', toLocation='Rivermist')",),
            ),
        ),
        involved_classes=("VehicleControlAPI",),
        initial_state={"VehicleControlAPI": {"fuelLevel": 20}},
        metadata={"official_root": "/tmp/fake-official"},
    )
    runtime = start_bfcl_runtime(record)
    runtime.official_model_name = "demo-model"
    runtime.decoded_turn_outputs = [[["measureDistance(fromLocation='San Francisco', toLocation='Rivermist')"]]]

    monkeypatch.setattr(
        bfcl_v3_mod,
        "_load_bfcl_official_runtime",
        lambda _root: SimpleNamespace(
            execute_multi_turn_func_call=lambda **_kwargs: ([], {}),
            multi_turn_checker=lambda *_args, **_kwargs: {"valid": True},
        ),
    )

    evaluation = evaluate_bfcl_v3_episode(
        record,
        runtime,
        "",
        termination_reason="agent_stop",
    )

    assert evaluation.is_passed is True
    assert evaluation.reward == 1.0
    assert evaluation.details["tool_sequence_ok"] is True
    assert evaluation.details["state_ok"] is True


def test_collect_bfcl_dataset_issues_reports_missing_support_assets() -> None:
    issues = collect_bfcl_dataset_issues(
        [
            BfclTaskRecord(
                task_id="multi_turn_base_59",
                instruction="Official task",
                turns=(BfclTurn(messages=({"role": "user", "content": "hi"},), ground_truth=()),),
                metadata={"normalization_error": "missing BFCL possible_answer file"},
            )
        ]
    )

    assert issues == ("multi_turn_base_59: missing BFCL possible_answer file",)


def test_trim_message_history_keeps_tail_with_notice() -> None:
    messages = [
        {"role": "user", "content": "A" * 100},
        {"role": "assistant", "content": "B" * 100},
        {"role": "user", "content": "C" * 100},
    ]

    trimmed = trim_message_history(messages, max_chars=200)

    assert trimmed[0]["content"].startswith("[Earlier conversation history truncated]")
    assert trimmed[-1]["content"] == "C" * 100


def test_trim_message_history_truncates_single_oversized_message_to_budget() -> None:
    trimmed = trim_message_history([{"role": "user", "content": "x" * 100}], max_chars=40)

    assert sum(len(message["role"]) + len(message["content"]) + 4 for message in trimmed) <= 40
    assert trimmed == [{"role": "user", "content": "x" * 32}]


def test_trim_history_never_exceeds_budget() -> None:
    trimmed = trim_history("x" * 100, 20)

    assert len(trimmed) == 20
    assert trimmed == "x" * 20


def test_build_bfcl_user_block_uses_rwkv_style_sections() -> None:
    content = build_bfcl_user_block(
        "Line 1\r\n\r\nLine 2  ",
        previous_tool_result={"ok": True, "tool": "lookup", "result": {"id": "A1"}},
        current_state_snapshot={"selected": "A1"},
    )

    assert content.startswith("User: Request:\nLine 1\nLine 2")
    assert "Previous tool result:" in content
    assert "Current structured state snapshot:" in content
    assert "\r" not in content


def test_build_bfcl_cot_and_decision_prompts_are_separated() -> None:
    system_prompt = build_bfcl_system_prompt(
        (
            {
                "name": "lookup",
                "description": "Lookup a value",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            },
        )
    )

    cot_prompt = build_bfcl_cot_prompt(
        system_prompt,
        user_request="Find A1",
        current_state_snapshot={"selected": "A0"},
        previous_tool_result={"ok": True, "tool": "lookup", "result": {"id": "A0"}},
    )
    decision_prompt = build_bfcl_decision_prompt(
        system_prompt,
        user_request="Find A1",
        current_state_snapshot={"selected": "A0"},
        previous_tool_result={"ok": True, "tool": "lookup", "result": {"id": "A0"}},
    )

    assert cot_prompt.endswith("Assistant: <think>\n")
    assert decision_prompt.endswith("Assistant:")
    assert "<|completions_of_cot|>" not in decision_prompt
    assert "</think>" not in decision_prompt


def test_build_bfcl_router_and_branch_prompts_use_hidden_summary_and_tool_prefix() -> None:
    system_prompt = build_bfcl_system_prompt(
        (
            {
                "name": "lookup",
                "description": "Lookup a value",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]},
            },
        )
    )

    router_prompt = build_bfcl_router_prompt(
        system_prompt,
        user_request="Find A1",
        cot_hidden_summary="Need one lookup before answering.",
        recent_tool_window=(
            {"name": "lookup", "arguments": {"id": "A0"}, "success": True, "result": {"id": "A0"}},
        ),
        current_state_snapshot={"selected": "A1"},
        previous_state_snapshot={"selected": "A0"},
    )
    tool_prompt = build_bfcl_tool_prompt(
        system_prompt,
        user_request="Find A1",
        cot_hidden_summary="Need one lookup before answering.",
        current_state_snapshot={"selected": "A1"},
        previous_state_snapshot={"selected": "A0"},
    )
    ask_prompt = build_bfcl_ask_prompt(
        system_prompt,
        user_request="Find A1",
        cot_hidden_summary="Need missing id.",
    )
    handoff_prompt = build_bfcl_handoff_prompt(
        system_prompt,
        user_request="Find A1",
        cot_hidden_summary="No tool is needed.",
    )

    assert "Private reasoning summary" in router_prompt
    assert "Current structured state delta" in router_prompt
    assert '"selected": "A1"' in router_prompt
    assert "```" not in router_prompt
    assert tool_prompt.endswith('Assistant: <tool_call>{"requestor":"assistant","name":"')
    assert ask_prompt.endswith("Assistant:")
    assert handoff_prompt.endswith("Assistant:")


def test_build_bfcl_tool_result_message_omits_request_replay() -> None:
    content = build_bfcl_tool_result_message(
        {"ok": True, "tool": "lookup", "result": {"id": "A1"}},
    )

    assert content.startswith("User: Previous tool result:")
    assert "Request:" not in content


def test_parse_bfcl_assistant_output_accepts_single_tool_call_block() -> None:
    decision = parse_bfcl_assistant_output(
        '<tool_call>{"requestor":"assistant","name":"lookup","arguments":{"id":"A1"}}</tool_call>'
    )

    assert decision.is_tool_call is True
    assert decision.tool_call is not None
    assert decision.tool_call.name == "lookup"
    assert decision.tool_call.arguments == {"id": "A1"}


def test_parse_bfcl_assistant_output_requires_explicit_requestor() -> None:
    try:
        parse_bfcl_assistant_output('<tool_call>{"name":"lookup","arguments":{"id":"A1"}}</tool_call>')
    except ValueError as exc:
        assert "exactly requestor, name, and arguments" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected strict requestor validation")


def test_parse_bfcl_router_output_accepts_only_known_labels() -> None:
    assert parse_bfcl_router_output("tool") == "TOOL"
    assert parse_bfcl_router_output("ASK") == "ASK"

    try:
        parse_bfcl_router_output("TOOL NOW")
    except ValueError as exc:
        assert "TOOL, ASK, HANDOFF" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected router validation failure")


def test_parse_bfcl_assistant_output_rejects_non_rwkv_tool_protocols() -> None:
    bad_outputs = [
        '```json\n{"tool_calls":[{"name":"lookup","arguments":{}}]}\n```',
        '<tool_call>{"requestor":"assistant","name":"lookup","arguments":{}}</tool_call>\n'
        '<tool_call>{"requestor":"assistant","name":"lookup","arguments":{"id":"B2"}}</tool_call>',
    ]

    for text in bad_outputs:
        try:
            parse_bfcl_assistant_output(text)
        except ValueError as exc:
            assert any(token in str(exc) for token in ("forbidden", "unexpected", "multiple"))
        else:  # pragma: no cover - defensive
            raise AssertionError(f"expected ValueError for {text!r}")


def test_interpret_bfcl_assistant_output_rejects_unknown_tool_name() -> None:
    try:
        interpret_bfcl_assistant_output(
            '<tool_call>{"requestor":"assistant","name":"unknown","arguments":{}}</tool_call>',
            tools=[{"name": "lookup"}],
        )
    except ValueError as exc:
        assert "unknown BFCL tool name" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected unknown-tool rejection")


def test_decode_bfcl_exec_response_rejects_invalid_arguments() -> None:
    try:
        decode_bfcl_exec_response(
            '<tool_call>{"requestor":"assistant","name":"lookup","arguments":{}}</tool_call>',
            tools=[
                {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                }
            ],
        )
    except ValueError as exc:
        assert "invalid arguments for BFCL tool lookup" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected schema validation failure")


def test_bfcl_tool_call_constraint_enforces_root_argument_schema() -> None:
    tools = [
        {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
                "additionalProperties": False,
            },
        }
    ]

    valid = build_bfcl_tool_call_constraint(tools)
    assert valid.feed_text('lookup","arguments":{"id":"A1"}}</tool_call>')
    assert valid.is_complete() is True

    invalid = build_bfcl_tool_call_constraint(tools)
    assert invalid.feed_text('lookup","arguments":{"bad"') is False


def test_normalize_bfcl_decision_output_restores_tool_call_suffix() -> None:
    normalized = normalize_bfcl_decision_output(
        '<tool_call>{"requestor":"assistant","name":"lookup","arguments":{"id":"A1"}}'
    )

    assert normalized.endswith("</tool_call>")


def test_normalize_bfcl_tool_output_safe_repairs_missing_requestor_and_string_arguments() -> None:
    repaired = normalize_bfcl_tool_output_safe(
        reconstruct_bfcl_tool_output('{"name":"lookup","arguments":"{\\"id\\":\\"A1\\"}"}')
    )
    decision = parse_bfcl_assistant_output(repaired)

    assert decision.is_tool_call is True
    assert decision.tool_call is not None
    assert decision.tool_call.requestor == "assistant"
    assert decision.tool_call.arguments == {"id": "A1"}


def test_extract_hidden_summary_and_state_delta_are_compact() -> None:
    summary = extract_bfcl_cot_hidden_summary("<think>\nPlan lookup then answer.\n</think>")
    delta = render_bfcl_state_delta(
        {"selected": "A1", "done": True},
        previous_state={"selected": "A0"},
    )

    assert summary == "Plan lookup then answer."
    assert '"selected": "A1"' in delta
    assert '"done": true' in delta
    assert "A0" not in delta


def test_render_bfcl_recent_tool_window_keeps_last_call_only() -> None:
    rendered = render_bfcl_recent_tool_window(
        (
            {"name": "old", "arguments": {"id": "A0"}, "success": True},
            {"name": "lookup", "arguments": {"id": "A1"}, "success": False, "error": "not found"},
        )
    )

    assert "lookup" in rendered
    assert "A1" in rendered
    assert "old" not in rendered


def test_render_bfcl_assistant_tool_message_contains_only_tool_call() -> None:
    rendered = render_bfcl_assistant_tool_message(TauToolCall(name="lookup", arguments={"id": "A1"}))

    assert rendered.startswith("<tool_call>")
    assert rendered.endswith("</tool_call>")
    assert "<think>" not in rendered


def test_build_bfcl_turn_context_includes_state_snapshot() -> None:
    record = BfclTaskRecord(task_id="demo-1", instruction="Search and confirm")
    runtime = start_bfcl_runtime(record)
    runtime.current_state["selected"] = "F1"

    context = build_bfcl_turn_context(
        "System prompt",
        [{"role": "user", "content": "Search and confirm"}],
        runtime,
        history_max_chars=500,
    )

    assert "Current structured state snapshot" in context
    assert '"selected": "F1"' in context


def test_build_bfcl_turn_context_uses_rwkv_dialogue_skeleton() -> None:
    tool = {
        "name": "lookup",
        "description": "Lookup a record",
        "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
    }
    context = build_bfcl_turn_context(
        build_bfcl_system_prompt([tool]),
        [
            {"role": "user", "content": "Find A1"},
            {
                "role": "assistant",
                "content": render_bfcl_assistant_tool_message(TauToolCall(name="lookup", arguments={"id": "A1"})),
            },
        ],
        start_bfcl_runtime(BfclTaskRecord(task_id="demo-1", instruction="Find A1")),
        history_max_chars=4000,
    )

    assert context.startswith("System: You are solving a BFCL V3 multi-turn function-calling task.")
    assert "\nUser: Request:\nFind A1\n" in context
    assert "\nAssistant: <tool_call>" in context
    assert context.endswith("Assistant: <think>\n<|completions_of_cot|>")
