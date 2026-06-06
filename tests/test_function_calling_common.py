from __future__ import annotations

import types

from src.eval.evaluating import RunContext, RunMode, TaskExecutionState
from src.eval.execution_plan import AttemptKey
from src.eval.function_calling import common as function_calling_common
from src.eval.function_calling.common import (
    FunctionCallingRunContext,
    attach_function_calling_context_metadata,
    build_pending_attempts,
    compute_function_calling_diagnostics,
    compute_function_calling_metrics,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.context_budget import normalize_rwkv_text
from src.eval.function_calling.rwkv_prompt import build_rwkv_json_call_prompt, extract_json_call_value_text
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
from src.eval.function_calling.tool_router import ToolRoutingConfig
from src.eval.long_doc_evidence import LongDocEvidenceConfig


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
    assert '\n\nUser: Translate "Will it rain tomorrow?" into Japanese.\n\nAssistant: ```json\n{' in prompt
    assert "<think>" not in prompt


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


def test_extract_json_call_value_accepts_prefilled_object_continuation() -> None:
    assert (
        extract_json_call_value_text('"name":"lookup","arguments":{"id":"A1"}}')
        == '{"name":"lookup","arguments":{"id":"A1"}}'
    )


def test_extract_json_call_value_accepts_rwkv_agentic_tool_call_format() -> None:
    assert (
        extract_json_call_value_text('**Tool Call:** lookup(id="A1")')
        == '{"name":"lookup","arguments":{"id":"A1"}}'
    )


def test_rwkv_json_call_prompt_collapses_history_to_single_user_and_assistant_turn() -> None:
    prompt = build_rwkv_json_call_prompt(
        "Use exactly one tool call.",
        [
            {"role": "user", "content": "Find A1"},
            {"role": "assistant", "content": '{"name":"lookup","arguments":{"id":"A1"}}'},
            {"role": "user", "content": "Function output:\n{\"ok\":true}"},
        ],
        history_max_chars=4000,
    )

    assert sum(1 for line in prompt.splitlines() if line.startswith("User:")) == 1
    assert sum(1 for line in prompt.splitlines() if line.startswith("Assistant:")) == 1
    assert "Conversation transcript JSON:" in prompt
    assert prompt.endswith('Assistant: ```json\n{')


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
    assert planning.endswith("Assistant: ```json\n{")
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


def test_compute_function_calling_diagnostics_reports_ablation_metrics() -> None:
    payloads = [
        {
            "sample_index": 0,
            "repeat_index": 0,
            "pass_index": 0,
            "prompt1": "System: x\n[Long document compacted: label=demo]",
            "completion1": "{}",
            "agent_result": {"num_turns": 2, "error": None},
            "agent_trace": [
                {
                    "tool_route": {
                        "routed": True,
                        "reason": "lexical",
                        "selected_names": ["refund_order"],
                        "total_tool_count": 20,
                        "catalog_chars": 9000,
                    }
                }
            ],
        },
        {
            "sample_index": 1,
            "repeat_index": 0,
            "pass_index": 0,
            "prompt1": "System: y",
            "prompt2": "Followup",
            "completion1": "bad json",
            "agent_result": {"num_turns": 1, "error": "unknown tool name"},
        },
    ]

    metrics = compute_function_calling_diagnostics(payloads)

    assert metrics["avg_stage_prompt_chars"] == metrics["avg_sample_prompt_chars"]
    assert metrics["avg_sample_total_prompt_chars"] > metrics["avg_sample_prompt_chars"]
    assert metrics["long_doc_prompt_rate"] == 1 / 3
    assert metrics["agent_error_rate"] == 0.5
    assert metrics["unknown_tool_error_rate"] == 0.5
    assert metrics["tool_route_count"] == 1.0
    assert metrics["tool_route_avg_selected_tools"] == 1.0
    assert metrics["tool_route_routed_rate"] == 1.0


def test_finalize_function_calling_run_records_score_when_checker_fails() -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.recorded_score: dict[str, object] | None = None

        def complete_attempt_stage(self, _writer: object, *, timeout_s: float | None) -> list[dict[str, object]]:
            assert timeout_s is None
            return [{"sample_index": 0, "repeat_index": 0, "pass_index": 0, "prompt1": "x"}]

        def ingest_eval_payloads(self, _payloads: list[dict[str, object]]) -> None:
            return None

        def run_checker(self, *, model_name: str) -> None:
            assert model_name == "model"
            raise RuntimeError("checker unavailable")

        def record_score(self, payload: dict[str, object]) -> None:
            self.recorded_score = payload

    runtime = FakeRuntime()
    ctx = FunctionCallingRunContext(
        service=object(),
        runtime=runtime,
        writer=object(),
        task_id="task",
        skip_keys=frozenset(),
    )

    _completions, _evals, metrics = finalize_function_calling_run(
        ctx=ctx,
        completion_to_eval=lambda _item: {
            "sample_index": 0,
            "repeat_index": 0,
            "pass_index": 0,
            "is_passed": True,
        },
        model_name="model",
        avg_k=1.0,
        timeout_s=None,
        build_score_payload=lambda _completions, _evals, score_metrics: {"metrics": dict(score_metrics)},
    )

    assert metrics["checker_failed"] == 1.0
    assert runtime.recorded_score is not None
    assert runtime.recorded_score["metrics"]["success_rate"] == 1.0


def test_attach_function_calling_context_metadata_distinguishes_ablations() -> None:
    payload = attach_function_calling_context_metadata(
        {"1": {"temperature": 0.0}},
        long_doc_config=LongDocEvidenceConfig(enabled=False, min_long_text_chars=1200),
        tool_routing_config=ToolRoutingConfig(mode="lexical", max_tools=8),
        prompt_max_chars=8192,
    )

    assert payload["long_context"]["prompt_max_chars"] == 8192
    assert payload["long_context"]["long_doc"]["enabled"] is False
    assert payload["long_context"]["long_doc"]["mode"] == "lexical"
    assert payload["long_context"]["long_doc"]["model_max_tokens"] > 0
    assert payload["long_context"]["long_doc"]["model_parallel_batch_size"] > 0
    assert payload["long_context"]["tool_router"]["mode"] == "lexical"
    assert payload["long_context"]["tool_router"]["max_tools"] == 8
    assert payload["long_context"]["tool_router"]["description_chars"] > 0
    assert payload["long_context"]["tool_router"]["parallel_chunk_tools"] > 0
    assert payload["long_context"]["tool_router"]["parallel_batch_size"] > 0


def test_prepare_function_calling_run_uses_explicit_run_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_service = object()
    fake_runtime = types.SimpleNamespace(create_writer=lambda max_queue: ("writer", max_queue))

    monkeypatch.setattr(function_calling_common, "init_eval_store", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(function_calling_common, "create_eval_service", lambda: fake_service)

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
