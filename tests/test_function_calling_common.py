from __future__ import annotations

import types

from src.eval.evaluating import RunContext, RunMode, TaskExecutionState
from src.eval.execution_plan import AttemptKey
from src.eval.function_calling import common as function_calling_common
from src.eval.function_calling.common import (
    build_pending_attempts,
    compute_function_calling_metrics,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.context_budget import normalize_rwkv_text
from src.eval.function_calling.rwkv_prompt import extract_json_call_value_text
from src.eval.function_calling.mcp_bench import (
    McpBenchItem,
    McpBenchTaskSpec,
    build_final_answer_prompt,
    build_planning_json_call_prompt,
    parse_planning_decision,
)
from src.eval.function_calling.simple_tool_call import (
    SimpleToolCallRecord,
    ToolCallExpectation,
    build_simple_tool_call_prompt,
)
from src.infer.sampling import SamplingConfig


def test_build_pending_attempts_filters_skip_keys() -> None:
    attempt_keys = (
        AttemptKey(0, 0, 0),
        AttemptKey(1, 0, 0),
        AttemptKey(2, 1, 0),
    )
    records = ["a", "b", "c"]

    pending = build_pending_attempts(attempt_keys, records, skip_keys={(1, 0, 0)})

    assert pending == [
        (AttemptKey(0, 0, 0), "a"),
        (AttemptKey(2, 1, 0), "c"),
    ]


def test_repeat_probe_entries_repeats_to_batch_size() -> None:
    repeated = repeat_probe_entries([1, 2], batch_size=5)

    assert repeated == [1, 2, 1, 2, 1]


def test_simple_tool_call_prompt_uses_rwkv_json_function_call_shape() -> None:
    record = SimpleToolCallRecord(
        task_id="demo",
        instruction='Translate "Will it rain tomorrow?" into Japanese.',
        tools=(
            {
                "name": "translate_text",
                "description": "Translate text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "target_language": {"type": "string"},
                    },
                    "required": ["text", "target_language"],
                },
            },
        ),
        expected_tool_calls=(
            ToolCallExpectation(
                name="translate_text",
                arguments={"text": "Will it rain tomorrow?", "target_language": "Japanese"},
                argument_options={},
            ),
        ),
        metadata={},
    )

    prompt = build_simple_tool_call_prompt(record, history_max_chars=4000)

    assert prompt.startswith("System: Tools:\n[")
    assert '"name": "translate_text"' in prompt
    assert '"arguments": {' in prompt
    assert '"required": [' in prompt
    assert prompt.index('"name": "translate_text"') < prompt.index('"arguments": {')
    assert '"parameters"' not in prompt
    assert "Output JSON schema:" in prompt
    assert '"oneOf": [' in prompt
    assert "return a JSON array containing every required call" in prompt
    assert "Do not copy tool schemas" in prompt
    assert "Available tools:" not in prompt
    assert '\n\nUser: Translate "Will it rain tomorrow?" into Japanese.\n\nAssistant: <think>\n</think>\n```json\n' in prompt


def test_api_bank_prompt_documents_missing_year_convention() -> None:
    record = SimpleToolCallRecord(
        task_id="api-bank-0",
        instruction="Conversation history:\nUser: Book it on September 21st.\nReturn the next API call only.",
        tools=(),
        expected_tool_calls=(),
        metadata={"source_format": "official_api_bank"},
    )

    prompt = build_simple_tool_call_prompt(record, history_max_chars=4000)

    assert "API-Bank date convention" in prompt
    assert "use year 2023" in prompt


def test_function_calling_sampling_removes_raw_role_stop_tokens() -> None:
    sampling = SamplingConfig(max_generate_tokens=4096, stop_tokens=(0, 261, 24281))

    adjusted = function_calling_common.clamp_function_calling_sampling(sampling, 768)

    assert adjusted.max_generate_tokens == 768
    assert adjusted.stop_tokens == (0,)


def test_normalize_rwkv_text_strips_crlf_and_blank_lines() -> None:
    assert normalize_rwkv_text("  Line 1\r\n\r\nLine 2\n\n\nLine 3  ") == "Line 1\nLine 2\nLine 3"


def test_extract_json_call_value_accepts_sft_safe_wrappers() -> None:
    assert (
        extract_json_call_value_text(
            'Assistant: <think>\n</think>\n```json\n{"name":"lookup","arguments":{"id":"A1"}}\n```'
        )
        == '{"name":"lookup","arguments":{"id":"A1"}}'
    )


def test_mcp_prompts_use_rwkv_sections_without_blank_lines() -> None:
    item = McpBenchItem(
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
        runtime_root="/tmp/runtime",
    )
    tools = {
        "calendar.search": {
            "server": "calendar",
            "name": "search",
            "description": "Find calendar events.\n\nReturns matching IDs.",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    }

    planning = build_planning_json_call_prompt(
        item,
        tools,
        ({"role": "user", "content": "Task:\nBook the meeting"},),
        history_max_chars=4000,
    )
    final = build_final_answer_prompt(item, "Found A.\n\nFound B.")

    assert planning.startswith("System: Tools:")
    assert '"name": "calendar:search"' in planning
    assert "Output JSON schema:" in planning
    assert "\nUser: Task:\nBook the meeting" in planning
    assert planning.endswith("Assistant: <think>\n</think>\n```json\n")
    assert final.endswith("Assistant:")


def test_mcp_parse_planning_accepts_openai_tool_call_shape() -> None:
    decision = parse_planning_decision(
        '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"calendar:search","arguments":"{\\"query\\":\\"budget\\"}"}}]}'
    )

    assert decision.should_continue is True
    assert len(decision.tool_calls) == 1
    assert decision.tool_calls[0].full_name == "calendar:search"
    assert decision.tool_calls[0].arguments == {"query": "budget"}


def test_compute_function_calling_metrics_reports_success_rate_and_avg_key() -> None:
    payloads = [
        {"sample_index": 0, "repeat_index": 0, "pass_index": 0, "is_passed": True},
        {"sample_index": 1, "repeat_index": 0, "pass_index": 0, "is_passed": False},
    ]

    metrics = compute_function_calling_metrics(payloads, avg_k=1.0)

    assert metrics["success_rate"] == 0.5
    assert metrics["avg@1"] == 0.5


def test_prepare_function_calling_run_uses_explicit_run_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_service = object()
    fake_runtime = types.SimpleNamespace(create_writer=lambda max_queue: ("writer", max_queue))

    monkeypatch.setattr(function_calling_common, "init_db", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(function_calling_common, "EvalDbService", lambda: fake_service)

    def _fake_prepare_task_execution(**kwargs):
        captured.update(kwargs)
        return TaskExecutionState(
            task_id="task-1",
            run_mode=RunMode.RESUME,
            resume_context=types.SimpleNamespace(completed_keys=[(1, 0, 0)]),
        )

    monkeypatch.setattr(function_calling_common, "prepare_task_execution", _fake_prepare_task_execution)
    monkeypatch.setattr(
        function_calling_common.TaskRunState,
        "from_task_execution",
        lambda execution_state, attempt_keys, expected_attempt_count: types.SimpleNamespace(task_id=execution_state.task_id),
    )
    monkeypatch.setattr(function_calling_common, "TaskRunController", lambda service, state: fake_runtime)
    monkeypatch.setattr(function_calling_common, "set_task_env", lambda _task_id: None)

    run_context = RunContext(job_name="function_tau_bench", run_mode=RunMode.RESUME)
    ctx = prepare_function_calling_run(
        dataset_slug="tau_bench_retail_test",
        model_name="demo-model",
        job_name="ignored-default",
        attempt_keys=(),
        expected_attempt_count=0,
        sampling_payload={},
        avg_k=1.0,
        effective_sample_count=1,
        db_write_queue=8,
        run_context=run_context,
    )

    assert captured["job_name"] == "function_tau_bench"
    assert captured["run_mode"] is RunMode.RESUME
    assert ctx.service is fake_service
    assert ctx.runtime is fake_runtime
    assert ctx.writer == ("writer", 8)
    assert ctx.skip_keys == frozenset({(1, 0, 0)})
