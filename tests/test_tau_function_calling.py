from __future__ import annotations

from types import SimpleNamespace

from src.eval.function_calling import (
    build_tau_system_prompt,
    parse_tool_call_or_final_answer,
    render_tau_user_prompt,
)
from src.eval.agent_bench.tau_official import (
    RWKVTauOfficialAgent,
    build_tau_official_agent_system_prompt,
    configure_tau_nl_assertions_judge,
    _parse_tau_agent_decision,
    _normalize_tau_arguments,
)
from src.eval.env_config import OpenAIModelConfig, normalize_openai_base_url
from src.eval.function_calling.tau_runner import _tau_official_completion_payload
from src.eval.long_doc_evidence import LongDocEvidenceConfig


def test_render_tau_user_prompt_prefers_ticket() -> None:
    prompt = render_tau_user_prompt(
        {
            "ticket": "Customer needs a refund for order #123",
            "user_scenario": {"instructions": "ignored"},
        }
    )

    assert prompt == "Customer needs a refund for order #123"


def test_parse_tau_tool_call_from_json_function_call() -> None:
    decision = parse_tool_call_or_final_answer(
        '{"name":"user.inspect_order","arguments":{"order_id":"123"}}'
    )

    assert decision.is_tool_call
    assert decision.tool_call is not None
    assert decision.tool_call.requestor == "user"
    assert decision.tool_call.name == "inspect_order"
    assert decision.tool_call.arguments == {"order_id": "123"}


def test_parse_tau_tool_call_from_prefixed_name() -> None:
    decision = parse_tool_call_or_final_answer(
        '{"name":"assistant.inspect_order","arguments":{"order_id":"123"}}'
    )

    assert decision.is_tool_call
    assert decision.tool_call is not None
    assert decision.tool_call.requestor == "assistant"
    assert decision.tool_call.name == "inspect_order"


def test_parse_tau_final_answer_function_call() -> None:
    decision = parse_tool_call_or_final_answer(
        '{"name":"final_answer","arguments":{"answer":"Done"}}'
    )

    assert not decision.is_tool_call
    assert decision.final_answer == "Done"


def test_parse_tau_tool_call_from_json_code_fence() -> None:
    decision = parse_tool_call_or_final_answer(
        '```json\n{"name":"assistant.inspect_order","arguments":{"order_id":"123"}}\n```'
    )

    assert decision.is_tool_call
    assert decision.tool_call is not None
    assert decision.tool_call.name == "inspect_order"


def test_parse_tau_tool_call_accepts_openai_chat_tool_call_shape() -> None:
    decision = parse_tool_call_or_final_answer(
        '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"assistant.inspect_order","arguments":"{\\"order_id\\":\\"123\\"}"}}]}'
    )

    assert decision.is_tool_call
    assert decision.tool_call is not None
    assert decision.tool_call.requestor == "assistant"
    assert decision.tool_call.name == "inspect_order"
    assert decision.tool_call.arguments == {"order_id": "123"}


def test_parse_tau_tool_call_accepts_openai_response_function_call_shape() -> None:
    decision = parse_tool_call_or_final_answer(
        '{"id":"fc_1","call_id":"call_1","type":"function_call","name":"user.inspect_order","arguments":"{\\"order_id\\":\\"123\\"}"}'
    )

    assert decision.is_tool_call
    assert decision.tool_call is not None
    assert decision.tool_call.requestor == "user"
    assert decision.tool_call.name == "inspect_order"
    assert decision.tool_call.arguments == {"order_id": "123"}


def test_tau_official_parser_accepts_action_input_shape() -> None:
    name, arguments = _parse_tau_agent_decision(
        '{"action":"inspect_order","action_input":{"order_id":"123"}}'
    )

    assert name == "inspect_order"
    assert arguments == {"order_id": "123"}


def test_tau_official_parser_accepts_top_level_content_as_arguments() -> None:
    name, arguments = _parse_tau_agent_decision(
        '{"name":"respond","content":"Done ###STOP###"}'
    )

    assert name == "respond"
    assert arguments == {"content": "Done ###STOP###"}


def test_tau_official_transfer_to_human_keeps_only_summary_argument() -> None:
    arguments = _normalize_tau_arguments(
        "transfer_to_human_agents",
        {"content": "Please help with cancellation.", "summary": "Cancellation needs human support."},
    )

    assert arguments == {"summary": "Cancellation needs human support."}


def test_parse_tau_rejects_plain_text_final_answer() -> None:
    try:
        parse_tool_call_or_final_answer("The refund has been submitted successfully.")
    except ValueError as exc:
        assert "JSON function call object" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected strict JSON function-call validation")


def test_build_tau_system_prompt_lists_assistant_and_user_tools() -> None:
    prompt = build_tau_system_prompt(
        "Follow the refund policy.",
        assistant_tools=(
            {
                "name": "refund_order",
                "description": "Refund an order",
                "parameters": {"properties": {"order_id": {"type": "string"}}},
            },
        ),
        user_tools=(
            {
                "name": "view_email",
                "description": "Read a confirmation email",
                "parameters": {"properties": {"message_id": {"type": "string"}}},
            },
        ),
    )

    assert "assistant.refund_order" in prompt
    assert "user.view_email" in prompt
    assert "final_answer" in prompt
    assert "Return only a JSON function call." in prompt
    assert "Follow the refund policy." in prompt


def test_build_tau_official_agent_system_prompt_uses_respond_and_real_tools() -> None:
    tool = {
        "name": "refund_order",
        "description": "Refund an order",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
    }

    prompt = build_tau_official_agent_system_prompt("Follow the refund policy.", [tool])

    assert '"name": "refund_order"' in prompt
    assert '"name": "respond"' in prompt
    assert "Use a real tool call when you need information or need to change state." in prompt
    assert "include ###STOP###" in prompt
    assert "Follow the refund policy." in prompt


def test_tau_official_agent_prompt_compacts_long_tool_outputs() -> None:
    long_tool_output = "\n".join(
        [f"unrelated order row {index:03d}" for index in range(80)]
        + ["order ORD-77 refund eligibility approved evidence"]
        + [f"archive order row {index:03d}" for index in range(80)]
    )
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=[],
        domain_policy="Follow the order policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
        long_doc_config=LongDocEvidenceConfig(
            max_chunk_chars=240,
            overlap_lines=1,
            min_long_text_chars=400,
            max_evidence_chunks=1,
            max_evidence_chars=320,
        ),
    )

    prompt = agent._build_prompt(  # noqa: SLF001 - prompt compaction is the behavior under test.
        [
            {"role": "user", "content": "Check refund eligibility for order ORD-77."},
            {"role": "assistant", "content": '{"name":"get_order","arguments":{"order_id":"ORD-77"}}'},
            {"role": "user", "content": long_tool_output},
        ]
    )

    assert "Long document compacted" in prompt
    assert "order ORD-77 refund eligibility approved evidence" in prompt
    assert "unrelated order row 000" not in prompt
    assert len(prompt) <= 5000


def test_tau_nl_assertions_judge_config_uses_custom_model_and_base_url() -> None:
    cfg = OpenAIModelConfig(
        api_key="test-key",
        model_name="gpt-5.4",
        base_url="https://api.ablai.top/v1/chat/completions",
    )

    configure_tau_nl_assertions_judge(cfg)

    from tau2.evaluator import evaluator_nl_assertions

    assert normalize_openai_base_url(cfg.base_url) == "https://api.ablai.top/v1"
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS == "gpt-5.4"
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_key"] == "test-key"
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS["api_base"] == "https://api.ablai.top/v1"
    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS["response_format"] == {"type": "json_object"}


def test_tau_official_payload_fails_strictly_on_parse_error() -> None:
    record = SimpleNamespace(domain="airline", task_id="task-1", benchmark_version="tau_v2")
    simulation = SimpleNamespace(task_id="task-1", agent_cost=0.0, user_cost=0.0, messages=[])
    evaluation = SimpleNamespace(reward=1.0, is_passed=True, details={"reward": 1.0})
    agent = SimpleNamespace(stages=[], parse_errors=["tau agent decision missing name"])

    payload = _tau_official_completion_payload(
        record=record,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        simulation=simulation,
        evaluation=evaluation,
        agent=agent,
        benchmark_name="tau2_bench_airline",
        dataset_split="base",
        sampling_payload={},
    )

    assert payload["agent_result"]["reward"] == 0.0
    assert payload["agent_result"]["is_passed"] is False
    assert payload["agent_info"]["official_reward"] == 1.0
    assert payload["agent_info"]["official_is_passed"] is True
