from __future__ import annotations

import types

from src.eval.evaluating import RunContext, RunMode, TaskExecutionState
from src.eval.execution_plan import AttemptKey
from src.eval.scheduler.config import DBConfig
from src.eval.tasks.function_calling import browsecomp_plus as browsecomp_plus_module
from src.eval.tasks.function_calling import common as function_calling_common
from src.eval.tasks.function_calling.browsecomp import (
    BrowseCompJudgeConfig,
    BrowseCompJudgeOutcome,
    build_browsecomp_answer_prompt,
)
from src.eval.tasks.function_calling.browsecomp_plus import BrowseCompPlusRecord, build_browsecomp_plus_budgeted_prompt
from src.eval.tasks.function_calling.common import (
    FunctionCallingRunContext,
    attach_function_calling_context_metadata,
    build_pending_attempts,
    compute_function_calling_diagnostics,
    compute_function_calling_metrics,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.tasks.function_calling.context_budget import normalize_rwkv_text
from src.eval.tasks.function_calling.final_answer import (
    build_final_answer_json_call_prompt,
    parse_final_answer_call,
    render_final_answer_call,
)
from src.eval.tasks.function_calling.rwkv_prompt import build_rwkv_json_call_prompt, extract_json_call_value_text
from src.eval.tasks.function_calling.mcp_bench import (
    McpBenchItem,
    McpBenchTaskSpec,
    build_final_answer_prompt,
    build_planning_json_call_prompt,
    clean_mcp_final_answer,
    parse_planning_decision,
    resolve_mcp_context_budget,
)
from src.eval.tasks.function_calling.simple_tool_call import (
    SimpleToolCallRecord,
    ToolCallExpectation,
    _auto_candidate_router_config,
    build_simple_tool_call_messages,
    build_simple_tool_call_prompt,
    decode_simple_tool_call_response,
)
from src.eval.tasks.function_calling.tool_call_contract import parse_tool_call_text, parse_tool_calls_text
from src.infer.sampling import SamplingConfig
from src.eval.tasks.function_calling.tool_router import ToolRoutingConfig
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


def test_mcp_context_budget_respects_g1h_10k_and_legacy_8k_contexts() -> None:
    args = types.SimpleNamespace(history_max_chars=24000, decision_max_tokens=2048, final_max_tokens=3072)

    g1h_budget = resolve_mcp_context_budget(args, "rwkv7-g1h-7.2b-20260710-ctx10240")
    old_budget = resolve_mcp_context_budget(args, "rwkv7-g1g-7.2b-20260523-ctx8192")

    assert g1h_budget == {
        "context_tokens": 10240,
        "history_max_chars": 14000,
        "final_history_max_chars": 11000,
        "decision_max_tokens": 896,
        "final_max_tokens": 1536,
    }
    assert old_budget == {
        "context_tokens": 8192,
        "history_max_chars": 11000,
        "final_history_max_chars": 8500,
        "decision_max_tokens": 768,
        "final_max_tokens": 1024,
    }


def test_clean_mcp_final_answer_strips_role_fence_and_final_answer_call() -> None:
    raw = 'Assistant: ```json\n{"name":"final_answer","arguments":{"answer":"answer text"}}\n```'

    assert clean_mcp_final_answer(raw) == "answer text"


def test_mcp_planning_decision_accepts_parallel_tool_call_array() -> None:
    decision = parse_planning_decision(
        """
        [
          {"name":"maps:directions","arguments":{"origin":"A","destination":"B"}},
          {"name":"calendar:search","arguments":{"query":"meeting"}}
        ]
        """
    )

    assert decision.should_continue is True
    assert [call.full_name for call in decision.tool_calls] == ["maps:directions", "calendar:search"]


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
    assert "Output JSON schema:" not in prompt
    assert '"oneOf": [' not in prompt
    assert "Return only a JSON function call." in prompt
    assert "return a JSON array containing every required call" in prompt
    assert "Output JSON schema:" not in prompt
    assert "Available tools:" not in prompt
    assert '\n\nUser: Translate "Will it rain tomorrow?" into Japanese.\n\nAssistant: ```json' in prompt
    assert prompt.endswith("Assistant: ```json\n")
    assert not prompt.endswith("{")
    assert "<think>" not in prompt


def test_simple_tool_call_native_messages_keep_tools_out_of_prompt() -> None:
    record = SimpleToolCallRecord(
        task_id="demo",
        instruction="What is the weather in Paris?",
        tools=(
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ),
        expected_tool_calls=(),
        metadata={},
    )

    messages = build_simple_tool_call_messages(record)

    assert messages == [
        {
            "role": "system",
            "content": (
                "Use the provided tools when a function call is needed.\n"
                "Call only provided tool names and supply valid JSON arguments that match the tool schema.\n"
                "For multiple required tool calls, return every required call in execution order.\n"
                "If no tool is needed, answer directly.\n"
                "For dates and times, use only dates/times stated or implied by the conversation or function outputs; do not use the real current date."
            ),
        },
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    assert "get_weather" not in messages[0]["content"]
    assert '"parameters"' not in messages[0]["content"]


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
    assert "User message:" not in prompt


def test_api_bank_prompt_converts_legacy_role_headers_to_transcript_json() -> None:
    record = SimpleToolCallRecord(
        task_id="api-bank-legacy",
        instruction=(
            "User: Can you help me?\n"
            "Assistant: Sure. What do you need?\n"
            "User: Check my balance.\n"
            "API: [GetUserToken(username='foo')] Response: {'output': {'token': 't1'}}\n"
            "Assistant: I have your token."
        ),
        tools=(),
        expected_tool_calls=(),
        metadata={"source_format": "official_api_bank"},
    )

    prompt = build_simple_tool_call_prompt(record, history_max_chars=4000)

    assert "Conversation transcript JSON:" in prompt
    assert '"role":"user","content":"Can you help me?"' in prompt
    assert '"role":"assistant","content":"Sure. What do you need?"' in prompt
    assert '"role":"api","content":"[GetUserToken(username=' in prompt
    assert "User message:" not in prompt
    assert "Assistant message:" not in prompt


def test_simple_tool_call_auto_candidate_router_triggers_for_long_context() -> None:
    args = types.SimpleNamespace(
        candidate_router_chunk_tools=2,
        candidate_router_batch_size=4,
        candidate_router_context_chars=400,
        candidate_router_prompt_max_chars=8192,
        candidate_router_candidate_max_tokens=128,
        candidate_router_aggregate_max_tokens=128,
        candidate_router_max_candidates=8,
        candidate_router_tool_schema_mode="compact",
        candidate_router_evidence_chars=120,
        candidate_router_policy_chars=600,
        disable_candidate_router_grounding=False,
    )
    record = SimpleToolCallRecord(
        task_id="long",
        instruction="Find the answer.\n" + ("context " * 100),
        tools=(
            {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        ),
        expected_tool_calls=(),
        metadata={},
    )
    small = SimpleToolCallRecord(
        task_id="small",
        instruction="Find the answer.",
        tools=record.tools,
        expected_tool_calls=(),
        metadata={},
    )

    assert _auto_candidate_router_config(args, record) is not None
    assert _auto_candidate_router_config(args, small) is None


def test_function_calling_sampling_removes_raw_role_stop_tokens() -> None:
    sampling = SamplingConfig(max_generate_tokens=4096, stop_tokens=(0, 261, 24281))

    adjusted = function_calling_common.clamp_function_calling_sampling(sampling, 768)

    assert adjusted.max_generate_tokens == 768
    assert adjusted.stop_tokens == (0,)


def test_normalize_rwkv_text_strips_crlf_and_blank_lines() -> None:
    assert normalize_rwkv_text("  Line 1\r\n\r\nLine 2\n\n\nLine 3  ") == "Line 1\nLine 2\nLine 3"


def test_rwkv_prompt_embedded_role_headers_become_transcript_json() -> None:
    prompt = build_rwkv_json_call_prompt(
        "Tools:\n[]\nReturn only a JSON function call.",
        [
            {
                "role": "user",
                "content": (
                    "Conversation history:\n"
                    "User: Can you help me?\n"
                    "Assistant: Sure.\n"
                    "User: Call lookup now."
                ),
            }
        ],
        history_max_chars=4000,
    )

    assert "Conversation transcript JSON:" in prompt
    assert '"role":"user","content":"Can you help me?"' in prompt
    assert '"role":"assistant","content":"Sure."' in prompt
    assert "User message:" not in prompt
    assert "Assistant message:" not in prompt


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


def test_extract_json_call_value_ignores_next_turn_after_complete_json() -> None:
    assert (
        extract_json_call_value_text('{"name":"lookup","arguments":{"id":"A1"}}\n```\n\nUser: next')
        == '{"name":"lookup","arguments":{"id":"A1"}}'
    )


def test_extract_json_call_value_recovers_complete_prefix_from_truncated_array() -> None:
    assert (
        extract_json_call_value_text(
            '[{"name":"lookup","arguments":{"id":"A1"}},{"name":"lookup","arguments":{"id":'
        )
        == '[{"name":"lookup","arguments":{"id":"A1"}}]'
    )


def test_decode_simple_tool_call_response_recovers_truncated_json_object() -> None:
    calls = decode_simple_tool_call_response(
        '{"name":"lookup","arguments":"{\\"id\\":\\"A1\\"}","id":"call_eeeeeeee'
    )

    assert calls == [{"name": "lookup", "arguments": {"id": "A1"}}]


def test_tool_call_contract_accepts_openai_tool_calls_shape() -> None:
    calls = parse_tool_calls_text(
        '{"tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\\"id\\":\\"A1\\"}"}}]}'
    )

    assert [(call.name, call.arguments) for call in calls] == [("lookup", {"id": "A1"})]


def test_tool_call_contract_does_not_recover_plain_prose_prefix() -> None:
    try:
        parse_tool_call_text('I will call lookup now {"name":"lookup","arguments":{"id":"A1"}}')
    except ValueError as exc:
        assert "JSON function call object" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected prose-prefixed tool call to fail")


def test_tool_call_contract_does_not_recover_unclosed_think_prefix() -> None:
    try:
        parse_tool_call_text('<think>I should call {"name":"lookup","arguments":{"id":"A1"}}')
    except ValueError as exc:
        assert "JSON function call object" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected unclosed think output to fail")


def test_parse_final_answer_call_recovers_truncated_metadata_tail() -> None:
    final_call = parse_final_answer_call(
        '{"name":"final_answer","arguments":"{\\"answer\\":\\"Alice\\"}","id":"call_eeeeeeee'
    )

    assert final_call.answer == "Alice"
    assert final_call.call["arguments"] == {"answer": "Alice"}


def test_final_answer_helper_renders_and_parses_rwkv_call() -> None:
    prompt = build_final_answer_json_call_prompt("Question: Who?", history_max_chars=4000)
    rendered = render_final_answer_call("Alice")
    parsed = parse_final_answer_call('```json\n{"name":"final_answer","arguments":{"answer":"Alice"},"id":"call_7"}\n```')
    openai_shape = parse_final_answer_call(
        '{"tool_calls":[{"id":"call_8","type":"function","function":{"name":"final_answer","arguments":"{\\"answer\\":\\"Bob\\"}"}}]}'
    )

    assert prompt.endswith("Assistant: ```json\n{")
    assert '"name": "final_answer"' in prompt
    assert rendered == '{"name":"final_answer","arguments":{"answer":"Alice"},"id":"final_answer"}'
    assert parsed.answer == "Alice"
    assert parsed.call_id == "call_7"
    assert parsed.call["id"] == "call_7"
    assert parsed.call["name"] == "final_answer"
    assert openai_shape.answer == "Bob"
    assert openai_shape.call_id == "call_8"


def test_browsecomp_answer_prompt_requests_final_answer_json_call() -> None:
    prompt = build_browsecomp_answer_prompt("User: question\n\nAssistant: <think>", "reasoning", locale="en")

    assert "final_answer" in prompt
    assert '"id":"final_answer"' in prompt
    assert prompt.endswith("Assistant: ```json\n{")


def test_browsecomp_plus_budgeted_prompt_compacts_long_tool_output() -> None:
    long_output = "\n".join(
        [f"irrelevant evidence row {index}" for index in range(80)]
        + ["target evidence: Zurich is the answer"]
        + [f"archive evidence row {index}" for index in range(80)]
    )

    prompt, trace = build_browsecomp_plus_budgeted_prompt(
        [
            {"role": "user", "content": "Question: Where is the answer?"},
            {"role": "assistant", "content": '{"name":"search","arguments":{"query":"answer"}}'},
            {"role": "user", "content": "Function output:\n" + long_output},
        ],
        history_max_chars=12000,
        prompt_max_chars=5000,
        long_doc_config=LongDocEvidenceConfig(
            enabled=True,
            mode="lexical",
            max_chunk_chars=240,
            overlap_lines=0,
            min_long_text_chars=400,
            max_evidence_chunks=2,
            max_evidence_chars=500,
        ),
        tool_routing_config=ToolRoutingConfig(
            mode="lexical",
            max_tools=1,
            trigger_tool_count=1,
            trigger_catalog_chars=1,
        ),
    )

    assert "Long document compacted" in prompt
    assert "Conversation transcript JSON" not in prompt
    assert "target evidence: Zurich is the answer" in prompt
    assert '"name": "search"' in prompt
    assert '"name": "final_answer"' in prompt
    assert '"name": "get_document_chunks"' not in prompt
    assert trace["compacted_message_count"] == 1
    assert trace["tool_route"]["routed"] is True
    assert trace["tool_route"]["selected_names"] == ["search", "final_answer"]
    assert trace["prompt_chars"] <= 5000


def test_browsecomp_plus_prompt_uses_agent_state_shape() -> None:
    prompt = browsecomp_plus_module.build_browsecomp_plus_prompt(
        [
            {"role": "user", "content": "Question: Who?"},
            {"role": "assistant", "content": '{"name":"search","arguments":{"query":"Who"}}'},
            {"role": "user", "content": "Function output:\n[]"},
        ],
        history_max_chars=4000,
    )

    assert "Conversation transcript JSON" not in prompt
    assert '"name": "get_document"' in prompt
    assert '"name":"final_answer","arguments":{"answer":"<exact answer>"}' in prompt
    assert "Do not use reason, reasoning, explanation, output, or response keys" in prompt
    assert 'Assistant action: {"name":"search"' in prompt
    assert "Current observation:\nFunction output:\n[]" in prompt
    assert prompt.endswith("Assistant: <think>\n</think>\n```json")


def test_browsecomp_plus_prefers_record_documents_over_bm25(monkeypatch, tmp_path) -> None:
    index_path = tmp_path / "bm25"
    index_path.mkdir()

    def _unexpected_bm25(*args, **kwargs):
        raise AssertionError("BM25 should not be used by default")

    monkeypatch.delenv("RWKV_BROWSECOMP_PLUS_RETRIEVER", raising=False)
    monkeypatch.setattr(browsecomp_plus_module, "_pyserini_available", lambda: True)
    monkeypatch.setattr(browsecomp_plus_module, "_search_bm25", _unexpected_bm25)

    env = browsecomp_plus_module.BrowseCompPlusEnv(
        BrowseCompPlusRecord(
            task_id="bc-plus-1",
            query_id="q1",
            question="Which city is the answer?",
            answer="Zurich",
            metadata={
                "browsecomp_plus_bm25_index_path": str(index_path),
                "browsecomp_plus_documents": [
                    {"docid": "doc-1", "text": "Zurich is the answer."},
                    {"docid": "doc-2", "text": "Paris is unrelated."},
                ],
            },
        )
    )

    chunks = env.search("Zurich", 1)

    assert chunks[0]["docid"] == "doc-1"
    assert env._details()["retriever"] == "record_documents"


def test_browsecomp_plus_attempt_keeps_raw_completion_separate_from_sandbox_return(monkeypatch) -> None:
    raw_completion = '```json\n{"name":"final_answer","arguments":{"answer":"Zurich"},"id":"call_42"}\n```'

    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(
            self,
            prompts,
            sampling,
            batch_size,
            progress_desc,
            prompt_seeds=None,
            prompt_stop_suffixes=None,
            constraints=None,
            constraint_mode="off",
        ):
            self.calls.append(
                {
                    "prompts": list(prompts),
                    "sampling": sampling,
                    "batch_size": batch_size,
                    "progress_desc": progress_desc,
                    "prompt_seeds": prompt_seeds,
                    "prompt_stop_suffixes": prompt_stop_suffixes,
                    "constraints": constraints,
                    "constraint_mode": constraint_mode,
                }
            )
            return [types.SimpleNamespace(text=raw_completion, finish_reason="stop")]

    def _fake_judge(inputs, config):
        assert config.model == "judge"
        assert [(record.task_id, answer) for record, answer in inputs] == [("bc-plus-1", "Zurich")]
        return [BrowseCompJudgeOutcome(is_passed=True, reason="matched")]

    monkeypatch.setattr(browsecomp_plus_module, "judge_browsecomp_answers", _fake_judge)

    engine = FakeEngine()
    payload = browsecomp_plus_module._run_one_browsecomp_plus_attempt(
        args=types.SimpleNamespace(max_steps=1),
        run=types.SimpleNamespace(engine=engine, benchmark_name="browsecomp_plus", dataset_split="test"),
        record=BrowseCompPlusRecord(
            task_id="bc-plus-1",
            query_id="q1",
            question="Which city is the answer?",
            answer="Zurich",
            metadata={},
        ),
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        sampling=object(),
        sampling_payload={"max_generate_tokens": 64},
        history_max_chars=4000,
        prompt_max_chars=4000,
        long_doc_config=LongDocEvidenceConfig(enabled=False),
        tool_routing_config=ToolRoutingConfig(mode="off"),
        judge=BrowseCompJudgeConfig(api_key="", model="judge"),
    )

    sandbox_return = '{"name":"final_answer","arguments":{"answer":"Zurich"},"id":"call_42"}'
    assert payload["completion1"] == raw_completion
    assert payload["agent_info"]["final_answer"] == "Zurich"
    assert payload["agent_info"]["final_answer_call"] == sandbox_return
    assert payload["agent_info"]["decoded_final_answer_call"] == {
        "name": "final_answer",
        "arguments": {"answer": "Zurich"},
        "id": "call_42",
    }
    assert payload["agent_trace"][0]["decision_completion"] == raw_completion
    assert payload["agent_trace"][0]["sandbox_return"] == sandbox_return


def test_browsecomp_plus_attempt_recovers_final_answer_format_variants(monkeypatch) -> None:
    class FakeEngine:
        def __init__(self, raw_completion: str) -> None:
            self.raw_completion = raw_completion

        def generate(
            self,
            prompts,
            sampling,
            batch_size,
            progress_desc,
            prompt_seeds=None,
            prompt_stop_suffixes=None,
            constraints=None,
            constraint_mode="off",
        ):
            return [types.SimpleNamespace(text=self.raw_completion, finish_reason="stop")]

    def _fake_judge(inputs, config):
        return [BrowseCompJudgeOutcome(is_passed=True, reason="matched")]

    def _run(raw_completion: str) -> dict[str, object]:
        monkeypatch.setattr(browsecomp_plus_module, "judge_browsecomp_answers", _fake_judge)
        return browsecomp_plus_module._run_one_browsecomp_plus_attempt(
            args=types.SimpleNamespace(max_steps=1),
            run=types.SimpleNamespace(engine=FakeEngine(raw_completion), benchmark_name="browsecomp_plus", dataset_split="test"),
            record=BrowseCompPlusRecord(
                task_id="bc-plus-1",
                query_id="q1",
                question="Which city is the answer?",
                answer="Zurich",
                metadata={},
            ),
            sample_index=0,
            repeat_index=0,
            pass_index=0,
            sampling=object(),
            sampling_payload={"max_generate_tokens": 64},
            history_max_chars=4000,
            prompt_max_chars=4000,
            long_doc_config=LongDocEvidenceConfig(enabled=False),
            tool_routing_config=ToolRoutingConfig(mode="off"),
            judge=BrowseCompJudgeConfig(api_key="", model="judge"),
        )

    trailing = '{"name":"final_answer","arguments":"{\\"answer\\":\\"Zurich\\"}","id":"call_42"},{"docid":"x"}'
    bare_answer = '{"answer":"Zurich"}'
    malformed_arguments = (
        '{\n'
        '  "name": "final_answer",\n'
        '  "arguments": "{\\"answer\\": \\"Zurich\\", \\"evidence\\": [\\"doc\\"]", "is_output": true}",\n'
        '  "id": "call_bad"\n'
        '}'
    )

    trailing_payload = _run(trailing)
    bare_payload = _run(bare_answer)
    malformed_payload = _run(malformed_arguments)

    assert trailing_payload["agent_info"]["final_answer"] == "Zurich"
    assert trailing_payload["agent_info"]["fail_reason"] == ""
    assert bare_payload["agent_info"]["final_answer"] == "Zurich"
    assert bare_payload["agent_info"]["fail_reason"] == ""
    assert malformed_payload["agent_info"]["final_answer"] == "Zurich"
    assert malformed_payload["agent_info"]["fail_reason"] == ""


def test_rwkv_json_call_prompt_renders_multi_turn_dialog_by_default() -> None:
    prompt = build_rwkv_json_call_prompt(
        "Use exactly one tool call.",
        [
            {"role": "user", "content": "Find A1"},
            {"role": "assistant", "content": '{"name":"lookup","arguments":{"id":"A1"}}'},
            {"role": "user", "content": "Function output:\n{\"ok\":true}"},
        ],
        history_max_chars=4000,
    )

    assert sum(1 for line in prompt.splitlines() if line.startswith("User:")) == 2
    assert sum(1 for line in prompt.splitlines() if line.startswith("Assistant:")) == 2
    assert "Conversation transcript JSON:" not in prompt
    assert "Function output:" in prompt
    assert prompt.endswith('Assistant: ```json\n{')


def test_rwkv_json_call_prompt_can_collapse_history_to_single_user_turn() -> None:
    prompt = build_rwkv_json_call_prompt(
        "Use exactly one tool call.",
        [
            {"role": "user", "content": "Find A1"},
            {"role": "assistant", "content": '{"name":"lookup","arguments":{"id":"A1"}}'},
            {"role": "user", "content": "Function output:\n{\"ok\":true}"},
        ],
        history_max_chars=4000,
        single_user_turn=True,
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
    final = build_final_answer_prompt(item, "Found A.\n\nFound B.", history_max_chars=4000)

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
    assert payload["long_context"]["tool_router"]["mode"] == "lexical"
    assert payload["long_context"]["tool_router"]["max_tools"] == 8
    assert payload["long_context"]["tool_router"]["description_chars"] > 0


def test_prepare_function_calling_run_uses_explicit_run_context(monkeypatch) -> None:
    captured: dict[str, object] = {}
    init_configs: list[object] = []
    fake_service = object()
    fake_runtime = types.SimpleNamespace(create_writer=lambda max_queue: ("writer", max_queue))

    monkeypatch.setattr(function_calling_common, "init_eval_store", lambda config=None: init_configs.append(config))
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
    db_config = DBConfig(host="127.0.0.1", port=15432, user="test", dbname="isolated")
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
        db_config=db_config,
    )

    assert init_configs == [db_config]
    assert captured["job_name"] == "function_tau_bench"
    assert captured["run_mode"] is RunMode.RESUME
    assert ctx.service is fake_service
    assert ctx.runtime is fake_runtime
    assert ctx.writer == ("writer", 8)
    assert ctx.skip_keys == frozenset({(1, 0, 0)})
