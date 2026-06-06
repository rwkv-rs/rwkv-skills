from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.eval.agent_bench import deps as tau_deps
from src.eval.agent_bench import tasks as tau_tasks
from src.eval.function_calling import (
    build_tau_system_prompt,
    parse_tool_call_or_final_answer,
    render_tau_user_prompt,
)
from src.eval.function_calling.rwkv_prompt import assistant_json_prefix, render_assistant_json_block
from src.eval.agent_bench.tau_official import (
    RWKVTauOfficialAgent,
    _apply_tau_json_object_prefill,
    _apply_tau_retail_progressive_tool_disclosure,
    _build_tau_tool_facts_message,
    build_tau_official_agent_system_prompt,
    configure_tau_nl_assertions_judge,
    normalize_tau_official_task_payload,
    _parse_tau_agent_decision,
    _recover_tau_agent_decision_from_text,
    _tau_litellm_model_name,
    _tau_llm_timeout_args,
)
from src.eval.env_config import OpenAIModelConfig, normalize_openai_base_url
from src.eval.function_calling.tau_runner import _tau_official_completion_payload
from src.eval.function_calling.tau_runner import _run_tau_official_attempt
from src.eval.function_calling.tau_runner import _tau_runtime_model_metadata
from src.eval.function_calling.tau_runner import _requires_tau_user_model, _requires_tau_v3_source
from src.eval.function_calling.tau_bench import TauManifestRecord
from src.eval.function_calling.tool_router import ToolRoutingConfig
from src.eval.agent_bench.tau_official import TauOfficialRuntime
from src.eval.long_doc_evidence import LongDocEvidenceConfig
from src.infer.sampling import GenerationOutput, SamplingConfig


def test_tau_v2_paths_prefer_repo_relative_reference_root(monkeypatch, tmp_path: Path) -> None:
    reference_root = tmp_path / "references" / "tau2-bench"
    (reference_root / "src" / "tau2").mkdir(parents=True)
    (reference_root / "data" / "tau2" / "domains" / "banking_knowledge").mkdir(parents=True)
    for name in (
        "RWKV_TAU3_BENCH_ROOT",
        "TAU3_BENCH_ROOT",
        "RWKV_TAU2_BENCH_ROOT",
        "TAU2_BENCH_ROOT",
        "RWKV_TAU3_DATA_ROOT",
        "TAU3_DATA_ROOT",
        "RWKV_TAU2_DATA_ROOT",
        "TAU2_DATA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tau_tasks, "TAU_V2_REFERENCE_ROOT", reference_root)
    monkeypatch.setattr(tau_deps, "_TAU_V2_REFERENCE_ROOT", reference_root)

    assert tau_tasks.tau_v2_vendor_root() == reference_root / "src"
    assert tau_tasks.tau_v2_data_root() == reference_root / "data"
    assert tau_tasks.tau_v3_source_available() is True
    assert tau_deps._tau_v2_vendor_root() == reference_root / "src"
    assert tau_deps._tau_v2_data_root() == reference_root / "data"


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


def test_tau_official_parser_recovers_runaway_id_after_arguments_string() -> None:
    name, arguments = _parse_tau_agent_decision(
        '{"name":"cancel_reservation","arguments":"{\\"reservation_id\\":\\"EHGLP3\\"}",'
        '"id":"toolu_013YZ1o8z1q1v8X8j6p6s8f8g8h8i8j'
    )

    assert name == "cancel_reservation"
    assert arguments == {"reservation_id": "EHGLP3"}


def test_tau_official_parser_recovers_runaway_id_after_arguments_object() -> None:
    name, arguments = _parse_tau_agent_decision(
        '{"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"},'
        '"id":"toolu_013YZ1o8z1q1v8X8j6p6s8f8g8h8i8j'
    )

    assert name == "get_reservation_details"
    assert arguments == {"reservation_id": "EHGLP3"}


def test_tau_official_parser_keeps_name_when_arguments_string_is_truncated() -> None:
    name, arguments = _parse_tau_agent_decision(
        '```json\n{"name":"get_user_details","arguments":"{\\"user_id\\":\\"yusuf_rossi_962'
    )

    assert name == "get_user_details"
    assert arguments == {}


def test_tau_official_parser_accepts_top_level_content_as_arguments() -> None:
    name, arguments = _parse_tau_agent_decision(
        '{"name":"respond","content":"Done ###STOP###"}'
    )

    assert name == "respond"
    assert arguments == {"content": "Done ###STOP###"}


def test_tau_official_parser_accepts_json_object_prefill_continuation() -> None:
    name, arguments = _parse_tau_agent_decision(
        _apply_tau_json_object_prefill(
            '"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"}}'
        )
    )

    assert name == "get_reservation_details"
    assert arguments == {"reservation_id": "EHGLP3"}


def test_tau_official_recovery_accepts_prefilled_json_with_trailing_text() -> None:
    name, arguments = _recover_tau_agent_decision_from_text(
        _apply_tau_json_object_prefill(
            '"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"}}\nI will now look it up.'
        ),
        [{"role": "user", "content": "Cancel reservation EHGLP3."}],
        available_tool_names={"get_reservation_details", "respond"},
    )

    assert name == "get_reservation_details"
    assert arguments == {"reservation_id": "EHGLP3"}


def test_tau_official_parser_keeps_semantic_tool_alias_unmodified() -> None:
    name, arguments = _parse_tau_agent_decision(
        '{"name":"search_flights","arguments":{"origin":"JFK","destination":"LAX","departure_date":"2024-05-01","return_date":null,"cabin":"economy"}}'
    )

    assert name == "search_flights"
    assert arguments == {
        "origin": "JFK",
        "destination": "LAX",
        "departure_date": "2024-05-01",
        "return_date": None,
        "cabin": "economy",
    }


def test_tau_official_parser_keeps_wrapper_tool_unmodified() -> None:
    name, arguments = _parse_tau_agent_decision(
        '{"name":"airline_agent_tool","arguments":{"action_details":{"action":"cancel","reservation_id":"IFOYYZ"},"user_id":"aarav_ahmed_6699","query":"Cancel reservation IFOYYZ"}}'
    )

    assert name == "airline_agent_tool"
    assert arguments["action_details"] == {"action": "cancel", "reservation_id": "IFOYYZ"}
    assert arguments["user_id"] == "aarav_ahmed_6699"


def test_tau_official_recovers_natural_language_reservation_lookup() -> None:
    name, arguments = _recover_tau_agent_decision_from_text(
        "The user wants to cancel reservation EHGLP3. I'll call get_reservation_details to inspect it.</think>",
        [{"role": "user", "content": "Please cancel reservation EHGLP3."}],
        available_tool_names={"get_reservation_details", "cancel_reservation", "respond"},
    )

    assert name == "get_reservation_details"
    assert arguments == {"reservation_id": "EHGLP3"}


def test_tau_official_recovers_search_reservation_alias_without_hallucinated_id() -> None:
    name, arguments = _recover_tau_agent_decision_from_text(
        '<tool_call>{"name":"search_reservation","arguments":{"reservation_id":"ZFA04Y"}}</tool_call>',
        [{"role": "user", "content": "I need to cancel my trip from Philadelphia to LaGuardia."}],
        available_tool_names={"get_reservation_details", "respond"},
    )

    assert name == "respond"
    assert "reservation ID" in arguments["content"]


def test_tau_official_recovers_ask_user_alias() -> None:
    name, arguments = _recover_tau_agent_decision_from_text(
        '{"name":"ask_user","arguments":{"question":"What is your user ID?"}}',
        [{"role": "user", "content": "I want to book a flight from San Francisco to New York."}],
        available_tool_names={"respond"},
    )

    assert name == "respond"
    assert arguments == {"content": "What is your user ID?"}


def test_tau_official_recovers_retail_product_search_language() -> None:
    name, arguments = _recover_tau_agent_decision_from_text(
        "The user wants to exchange the mechanical keyboard for a similar one with clicky switches. "
        "I need to find a product that is similar, so I should search products in the keyboard category.</think>",
        [{"role": "user", "content": "I want to exchange my mechanical keyboard for one with clicky switches."}],
        available_tool_names={"list_all_product_types", "get_product_details", "respond"},
    )

    assert name == "list_all_product_types"
    assert arguments == {}


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
    assert "Never invent wrapper or pseudo tools" in prompt
    assert "Do not invent ids/emails" in prompt
    assert "detail tools need exact IDs" in prompt
    assert "include ###STOP###" in prompt
    assert "Follow the refund policy." in prompt


def test_tau_official_agent_prompt_uses_plain_json_prefill_without_think() -> None:
    tool = {
        "name": "get_reservation_details",
        "description": "Get reservation details",
        "parameters": {
            "type": "object",
            "properties": {"reservation_id": {"type": "string"}},
            "required": ["reservation_id"],
        },
    }
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=[tool],
        domain_policy="Follow airline policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )

    prompt = agent._build_prompt([{"role": "user", "content": "Cancel reservation EHGLP3."}])  # noqa: SLF001

    assert prompt.endswith("Assistant: ```json\n{")
    assert "\n\nAssistant: <think>\n</think>\n```json\n" not in prompt


def test_assistant_json_object_prefill_does_not_double_render_history_object() -> None:
    rendered = render_assistant_json_block(
        '{"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"}}',
        assistant_prefix=assistant_json_prefix(enable_think=False, prefill_object=True),
    )

    assert rendered.startswith('Assistant: ```json\n{"name"')
    assert "{{" not in rendered


def test_tau_official_agent_preserves_hash_prefixed_user_ids() -> None:
    tools = [
        {
            "name": "get_order_details",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_order_details"}  # noqa: SLF001

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_order_details",
        {"order_id": "W2378156"},
        prompt_messages=[{"role": "user", "content": "I received order #W2378156 and need an exchange."}],
    )

    assert message.tool_calls[0].arguments == {"order_id": "#W2378156"}


def test_tau_official_agent_does_not_add_hash_without_exact_user_token() -> None:
    tools = [
        {
            "name": "get_order_details",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_order_details"}  # noqa: SLF001

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_order_details",
        {"order_id": "W2378156"},
        prompt_messages=[{"role": "user", "content": "I received order W2378156 and need an exchange."}],
    )

    assert message.tool_calls[0].arguments == {"order_id": "W2378156"}


def test_tau_official_retail_repeated_read_guard_stops_exact_repeat_when_enabled() -> None:
    tools = [
        {
            "name": "get_order_details",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
        retail_repeated_read_guard=True,
    )
    agent._current_tool_names = {"get_order_details"}  # noqa: SLF001
    messages = [
        {"role": "user", "content": "I received order #W2378156 and need an exchange."},
        {"role": "agent", "content": '{"name":"get_order_details","arguments":{"order_id":"#W2378156"}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps({"order_id": "#W2378156", "status": "delivered"}),
                },
                separators=(",", ":"),
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_order_details",
        {"order_id": "W2378156"},
        prompt_messages=messages,
    )

    assert message.content is not None
    assert "###STOP###" in message.content


def test_tau_official_retail_repeated_read_guard_is_off_by_default() -> None:
    tools = [
        {
            "name": "get_order_details",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_order_details"}  # noqa: SLF001
    messages = [
        {"role": "user", "content": "I received order #W2378156 and need an exchange."},
        {"role": "agent", "content": '{"name":"get_order_details","arguments":{"order_id":"#W2378156"}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps({"order_id": "#W2378156", "status": "delivered"}),
                },
                separators=(",", ":"),
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_order_details",
        {"order_id": "W2378156"},
        prompt_messages=messages,
    )

    assert message.tool_calls[0].name == "get_order_details"
    assert message.tool_calls[0].arguments == {"order_id": "#W2378156"}


def test_tau_official_retail_tool_use_guard_routes_product_name_to_list_tool() -> None:
    tools = [
        {
            "name": "get_product_details",
            "description": "Get product details",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
        {
            "name": "list_all_product_types",
            "description": "List product types",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
        retail_tool_use_guard=True,
    )
    agent._current_tool_names = {"get_product_details", "list_all_product_types"}  # noqa: SLF001

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_product_details",
        {"product_id": "t-shirt"},
        prompt_messages=[{"role": "user", "content": "How many t-shirt options are available?"}],
    )

    assert message.tool_calls[0].name == "list_all_product_types"
    assert message.tool_calls[0].arguments == {}


def test_tau_official_retail_tool_use_guard_repairs_invalid_order_id_from_user_hash_id() -> None:
    tools = [
        {
            "name": "get_order_details",
            "description": "Get order details",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
        retail_tool_use_guard=True,
    )
    agent._current_tool_names = {"get_order_details"}  # noqa: SLF001

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_order_details",
        {"order_id": "d_9513926"},
        prompt_messages=[{"role": "user", "content": "I received order #W2378156 and need an exchange."}],
    )

    assert message.tool_calls[0].name == "get_order_details"
    assert message.tool_calls[0].arguments == {"order_id": "#W2378156"}


def test_tau_official_retail_tool_use_guard_routes_repeated_catalog_to_product_detail() -> None:
    tools = [
        {
            "name": "list_all_product_types",
            "description": "List product types",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_product_details",
            "description": "Get product details",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
        retail_tool_use_guard=True,
    )
    agent._current_tool_names = {"get_product_details"}  # noqa: SLF001
    messages = [
        {"role": "user", "content": "How many t-shirt options are available?"},
        {"role": "agent", "content": '{"name":"list_all_product_types","arguments":{}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps({"T-Shirt": "9523456873"}),
                },
                separators=(",", ":"),
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "list_all_product_types",
        {},
        prompt_messages=messages,
    )

    assert message.tool_calls[0].name == "get_product_details"
    assert message.tool_calls[0].arguments == {"product_id": "9523456873"}


def test_tau_official_retail_tool_use_guard_prefers_earliest_catalog_match() -> None:
    tools = [
        {
            "name": "list_all_product_types",
            "description": "List product types",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_product_details",
            "description": "Get product details",
            "parameters": {
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow retail policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
        retail_tool_use_guard=True,
    )
    agent._current_tool_names = {"get_product_details"}  # noqa: SLF001
    messages = [
        {
            "role": "user",
            "content": "How many t-shirt options are available? I also need to return headphones.",
        },
        {"role": "agent", "content": '{"name":"list_all_product_types","arguments":{}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps({"Headphones": "6992792935", "T-Shirt": "9523456873"}),
                },
                separators=(",", ":"),
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "list_all_product_types",
        {},
        prompt_messages=messages,
    )

    assert message.tool_calls[0].name == "get_product_details"
    assert message.tool_calls[0].arguments == {"product_id": "9523456873"}


def test_tau_retail_progressive_disclosure_hides_order_lookup_for_product_question() -> None:
    tools = [
        {"name": "get_order_details", "description": "Get order details"},
        {"name": "get_product_details", "description": "Get product details"},
        {"name": "list_all_product_types", "description": "List product types"},
        {"name": "transfer_to_human_agents", "description": "Transfer to human"},
    ]

    result = _apply_tau_retail_progressive_tool_disclosure(
        tools,
        selected_tools=tools,
        prompt_messages=[{"role": "user", "content": "How many t-shirt options are available?"}],
        max_tools=8,
    )
    names = [tool["name"] for tool in result.selected_tools]

    assert result.trace["phase"] == "product_catalog_lookup"
    assert "list_all_product_types" in names
    assert "get_order_details" not in names
    assert "get_product_details" not in names


def test_tau_retail_progressive_disclosure_orders_lookup_before_catalog_when_order_id_present() -> None:
    tools = [
        {"name": "get_order_details", "description": "Get order details"},
        {"name": "list_all_product_types", "description": "List product types"},
        {"name": "transfer_to_human_agents", "description": "Transfer to human"},
    ]

    result = _apply_tau_retail_progressive_tool_disclosure(
        tools,
        selected_tools=tools,
        prompt_messages=[{"role": "user", "content": "I received order #W2378156 and need an exchange."}],
        max_tools=8,
    )
    names = [tool["name"] for tool in result.selected_tools]

    assert result.trace["phase"] == "order_lookup"
    assert "get_order_details" in names
    assert "list_all_product_types" not in names


def test_tau_retail_progressive_disclosure_exposes_product_details_after_catalog() -> None:
    tools = [
        {"name": "get_order_details", "description": "Get order details"},
        {"name": "get_product_details", "description": "Get product details"},
        {"name": "list_all_product_types", "description": "List product types"},
        {"name": "transfer_to_human_agents", "description": "Transfer to human"},
    ]
    messages = [
        {"role": "user", "content": "How many t-shirt options are available?"},
        {"role": "agent", "content": '{"name":"list_all_product_types","arguments":{}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps({"T-shirt": "6086499569"}),
                },
                separators=(",", ":"),
            ),
        },
    ]

    result = _apply_tau_retail_progressive_tool_disclosure(
        tools,
        selected_tools=tools,
        prompt_messages=messages,
        max_tools=8,
    )
    names = [tool["name"] for tool in result.selected_tools]

    assert result.trace["phase"] == "product_detail_lookup"
    assert "get_product_details" in names
    assert "get_order_details" not in names


def test_tau_retail_progressive_disclosure_uses_order_user_id_before_writes() -> None:
    tools = [
        {"name": "get_order_details", "description": "Get order details"},
        {"name": "get_user_details", "description": "Get user details"},
        {"name": "find_user_id_by_email", "description": "Find user by email"},
        {"name": "exchange_delivered_order_items", "description": "Exchange delivered items"},
        {"name": "transfer_to_human_agents", "description": "Transfer to human"},
    ]
    messages = [
        {"role": "user", "content": "I received order #W2378156 and need to exchange the blue headphones."},
        {"role": "agent", "content": '{"name":"get_order_details","arguments":{"order_id":"#W2378156"}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps(
                        {
                            "order_id": "#W2378156",
                            "status": "delivered",
                            "user_id": "yusuf_rossi_9626",
                        }
                    ),
                },
                separators=(",", ":"),
            ),
        },
    ]

    result = _apply_tau_retail_progressive_tool_disclosure(
        tools,
        selected_tools=tools,
        prompt_messages=messages,
        max_tools=8,
    )
    names = [tool["name"] for tool in result.selected_tools]

    assert result.trace["phase"] == "order_known_user_lookup"
    assert "get_user_details" in names
    assert "find_user_id_by_email" not in names
    assert "exchange_delivered_order_items" not in names


def test_tau_retail_progressive_disclosure_exposes_exchange_after_order_and_user() -> None:
    tools = [
        {"name": "get_order_details", "description": "Get order details"},
        {"name": "get_user_details", "description": "Get user details"},
        {"name": "exchange_delivered_order_items", "description": "Exchange delivered items"},
        {"name": "return_delivered_order_items", "description": "Return delivered items"},
        {"name": "transfer_to_human_agents", "description": "Transfer to human"},
    ]
    messages = [
        {"role": "user", "content": "I received order #W2378156 and need to exchange the blue headphones."},
        {"role": "agent", "content": '{"name":"get_order_details","arguments":{"order_id":"#W2378156"}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps(
                        {
                            "order_id": "#W2378156",
                            "status": "delivered",
                            "user_id": "yusuf_rossi_9626",
                            "items": [{"name": "Headphones", "product_id": "6992792935", "item_id": "4202497723"}],
                        }
                    ),
                },
                separators=(",", ":"),
            ),
        },
        {"role": "agent", "content": '{"name":"get_user_details","arguments":{"user_id":"yusuf_rossi_9626"}}'},
        {
            "role": "user",
            "content": "Function output:\n"
            + json.dumps(
                {
                    "requestor": "assistant",
                    "ok": True,
                    "output": json.dumps({"user_id": "yusuf_rossi_9626", "payment_methods": {"credit_card_9513926": {}}}),
                },
                separators=(",", ":"),
            ),
        },
    ]

    result = _apply_tau_retail_progressive_tool_disclosure(
        tools,
        selected_tools=tools,
        prompt_messages=messages,
        max_tools=8,
    )
    names = [tool["name"] for tool in result.selected_tools]

    assert result.trace["phase"] == "action_or_detail"
    assert "exchange_delivered_order_items" in names
    assert "return_delivered_order_items" not in names
    assert "get_order_details" not in names


def test_tau_tool_facts_message_extracts_airline_reservation_lists_and_payments() -> None:
    message = {
        "role": "user",
        "content": (
            'Function output:\n{"requestor":"assistant","ok":true,"output":'
            '"{\\"user_id\\": \\"raj_sanchez_7340\\", '
            '\\"reservations\\": [\\"MZDDS4\\", \\"60RX9E\\", \\"S5IK51\\", \\"OUEA45\\"], '
            '\\"payment_methods\\": {\\"credit_card_7891819\\": {\\"source\\": \\"credit_card\\"}}}"}'
        ),
    }

    facts = _build_tau_tool_facts_message([message])

    assert facts is not None
    assert "user_id: raj_sanchez_7340" in facts["content"]
    assert "reservations: MZDDS4, 60RX9E, S5IK51, OUEA45" in facts["content"]
    assert "payment_methods: credit_card_7891819" in facts["content"]


def test_tau_tool_facts_message_preserves_retail_item_summaries() -> None:
    order_output = {
        "order_id": "#W2378156",
        "items": [
            {
                "name": "Headphones",
                "product_id": "6992792935",
                "item_id": "4202497723",
                "options": {"color": "blue", "connectivity": "wireless"},
            },
            {
                "name": "Smart Watch",
                "product_id": "6945232052",
                "item_id": "9408160950",
                "options": {"display": "LCD", "band material": "leather"},
            },
        ],
        "payment_history": [{"payment_method_id": "credit_card_9513926"}],
    }
    message = {
        "role": "user",
        "content": "Function output:\n"
        + json.dumps(
            {"requestor": "assistant", "ok": True, "output": json.dumps(order_output)},
            separators=(",", ":"),
        ),
    }

    facts = _build_tau_tool_facts_message([message], max_chars=1200)

    assert facts is not None
    assert "order_id: #W2378156" in facts["content"]
    assert "item: Headphones | product_id=6992792935 | item_id=4202497723" in facts["content"]
    assert "options: color=blue, connectivity=wireless" in facts["content"]
    assert "item: Smart Watch | product_id=6945232052 | item_id=9408160950" in facts["content"]
    assert "payment_method_id: credit_card_9513926" in facts["content"]


def test_tau_official_agent_repeated_user_lookup_continues_reservation_scan() -> None:
    tools = [
        {
            "name": "get_user_details",
            "description": "Get user details",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
        {
            "name": "get_reservation_details",
            "description": "Get reservation details",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow airline policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_user_details", "get_reservation_details"}  # noqa: SLF001
    messages = [
        {
            "role": "user",
            "content": "I need help canceling a reservation. It is the trip from Philadelphia to LaGuardia.",
        },
        {"role": "user", "content": "My user ID is raj_sanchez_7340."},
        {"role": "agent", "content": '{"name":"get_user_details","arguments":{"user_id":"raj_sanchez_7340"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"user_id\\": \\"raj_sanchez_7340\\", '
                '\\"reservations\\": [\\"MZDDS4\\", \\"60RX9E\\", \\"S5IK51\\", \\"OUEA45\\", \\"Q69X3R\\"]}"}'
            ),
        },
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"MZDDS4"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"MZDDS4\\", \\"origin\\": \\"MIA\\", \\"destination\\": \\"LAX\\"}"}'
            ),
        },
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"60RX9E"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"60RX9E\\", \\"origin\\": \\"MSP\\", \\"destination\\": \\"EWR\\"}"}'
            ),
        },
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"S5IK51"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"S5IK51\\", \\"origin\\": \\"EWR\\", \\"destination\\": \\"MCO\\"}"}'
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_user_details",
        {"user_id": "raj_sanchez1048"},
        prompt_messages=messages,
    )

    assert message.tool_calls[0].name == "get_reservation_details"
    assert message.tool_calls[0].arguments == {"reservation_id": "OUEA45"}


def test_tau_official_agent_repeated_user_lookup_stops_after_reservation_scan_exhausted() -> None:
    tools = [
        {
            "name": "get_user_details",
            "description": "Get user details",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
        {
            "name": "get_reservation_details",
            "description": "Get reservation details",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow airline policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_user_details", "get_reservation_details"}  # noqa: SLF001
    messages = [
        {"role": "user", "content": "Cancel my reservation from Philadelphia to LaGuardia."},
        {"role": "user", "content": "My user ID is emma_kim_9957."},
        {"role": "agent", "content": '{"name":"get_user_details","arguments":{"user_id":"emma_kim_9957"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"user_id\\": \\"emma_kim_9957\\", '
                '\\"reservations\\": [\\"EHGLP3\\", \\"66EEUA\\"]}"}'
            ),
        },
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"EHGLP3\\", \\"origin\\": \\"MIA\\", \\"destination\\": \\"LAX\\"}"}'
            ),
        },
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"66EEUA"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"66EEUA\\", \\"origin\\": \\"MSP\\", \\"destination\\": \\"EWR\\"}"}'
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_user_details",
        {"user_id": "emma_kim_9957"},
        prompt_messages=messages,
    )

    assert message.content is not None
    assert "already checked" in message.content
    assert "###STOP###" in message.content


def test_tau_official_agent_repeated_user_lookup_answers_baggage_from_requested_reservation() -> None:
    tools = [
        {
            "name": "get_user_details",
            "description": "Get user details",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
        {
            "name": "get_reservation_details",
            "description": "Get reservation details",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow airline policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_user_details", "get_reservation_details"}  # noqa: SLF001
    messages = [
        {
            "role": "user",
            "content": (
                "I am trying to figure out how many suitcases my reservation allows. "
                "My confirmation number is JMO1MG, and my user ID is anya_garcia_5901."
            ),
        },
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"JMO1MG"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"JMO1MG\\", \\"user_id\\": \\"anya_garcia_5901\\", '
                '\\"total_baggages\\": 1, \\"nonfree_baggages\\": 0}"}'
            ),
        },
        {"role": "agent", "content": '{"name":"get_user_details","arguments":{"user_id":"anya_garcia_5901"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"user_id\\": \\"anya_garcia_5901\\", '
                '\\"reservations\\": [\\"82K959\\", \\"JMO1MG\\"]}"}'
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_user_details",
        {"user_id": "anya_garcia_5901"},
        prompt_messages=messages,
    )

    assert message.content is not None
    assert "JMO1MG" in message.content
    assert "1 total suitcases" in message.content
    assert "###STOP###" in message.content


def test_tau_official_agent_repeated_user_lookup_cancels_requested_reservation_instead_of_scanning_all() -> None:
    tools = [
        {
            "name": "get_user_details",
            "description": "Get user details",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
        {
            "name": "get_reservation_details",
            "description": "Get reservation details",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
        {
            "name": "cancel_reservation",
            "description": "Cancel reservation",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow airline policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_user_details", "get_reservation_details", "cancel_reservation"}  # noqa: SLF001
    messages = [
        {"role": "user", "content": "Hi, I need to cancel a reservation. The confirmation code is EHGLP3."},
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"EHGLP3\\", \\"user_id\\": \\"emma_kim_9957\\", '
                '\\"origin\\": \\"PHX\\", \\"destination\\": \\"JFK\\"}"}'
            ),
        },
        {"role": "agent", "content": '{"name":"get_user_details","arguments":{"user_id":"emma_kim_9957"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"user_id\\": \\"emma_kim_9957\\", '
                '\\"reservations\\": [\\"EHGLP3\\", \\"66EEUA\\", \\"MDCLVA\\"]}"}'
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "get_user_details",
        {"user_id": "emma_kim_9957"},
        prompt_messages=messages,
    )

    assert message.tool_calls[0].name == "cancel_reservation"
    assert message.tool_calls[0].arguments == {"reservation_id": "EHGLP3"}


def test_tau_official_agent_cancel_intent_redirects_wrong_write_to_read_then_cancel() -> None:
    tools = [
        {
            "name": "get_reservation_details",
            "description": "Get reservation details",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
        {
            "name": "cancel_reservation",
            "description": "Cancel reservation",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
        {
            "name": "book_reservation",
            "description": "Book reservation",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow airline policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_reservation_details", "cancel_reservation", "book_reservation"}  # noqa: SLF001
    before_details = [
        {"role": "user", "content": "Please cancel my reservation. The confirmation code is EHGLP3."},
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "book_reservation",
        {"user_id": "emma_kim_9957"},
        prompt_messages=before_details,
    )

    assert message.tool_calls[0].name == "get_reservation_details"
    assert message.tool_calls[0].arguments == {"reservation_id": "EHGLP3"}

    after_details = [
        *before_details,
        {"role": "agent", "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"EHGLP3"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"EHGLP3\\", \\"user_id\\": \\"emma_kim_9957\\"}"}'
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "book_reservation",
        {"user_id": "emma_kim_9957"},
        prompt_messages=after_details,
    )

    assert message.tool_calls[0].name == "cancel_reservation"
    assert message.tool_calls[0].arguments == {"reservation_id": "EHGLP3"}


def test_tau_official_agent_cancel_intent_does_not_repeat_successful_cancel() -> None:
    tools = [
        {
            "name": "get_reservation_details",
            "description": "Get reservation details",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
        {
            "name": "cancel_reservation",
            "description": "Cancel reservation",
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string"}},
                "required": ["reservation_id"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow airline policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
    )
    agent._current_tool_names = {"get_reservation_details", "cancel_reservation"}  # noqa: SLF001
    messages = [
        {"role": "user", "content": "Please cancel my reservation EHGLP3."},
        {"role": "agent", "content": '{"name":"cancel_reservation","arguments":{"reservation_id":"EHGLP3"}}'},
        {
            "role": "user",
            "content": (
                'Function output:\n{"requestor":"assistant","ok":true,"output":'
                '"{\\"reservation_id\\": \\"EHGLP3\\", \\"status\\": \\"cancelled\\", '
                '\\"payment_history\\": [{\\"payment_id\\": \\"credit_card_5832574\\", \\"amount\\": 208}, '
                '{\\"payment_id\\": \\"credit_card_5832574\\", \\"amount\\": -208}]}"}'
            ),
        },
    ]

    message = agent._decision_to_assistant_message(  # noqa: SLF001
        "cancel_reservation",
        {"reservation_id": "EHGLP3"},
        prompt_messages=messages,
    )

    assert message.content is not None
    assert "EHGLP3" in message.content
    assert "###STOP###" in message.content


def test_tau_official_budget_compacts_tool_schemas_without_dropping_tools() -> None:
    tools = [
        {
            "name": f"tool_{index}",
            "description": f"Tool {index} " + ("very long description " * 80),
            "parameters": {
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "Identifier " * 30,
                    }
                },
                "required": ["record_id"],
            },
        }
        for index in range(8)
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow the policy.",
        history_max_chars=1200,
        prompt_max_chars=4096,
    )

    prompt = agent._build_prompt([{"role": "user", "content": "Use the right tool for record R-1."}])  # noqa: SLF001

    assert len(prompt) <= 4096
    for index in range(8):
        assert f'"name":"tool_{index}"' in prompt
        assert f"tool_{index}" in agent._current_tool_names  # noqa: SLF001
    assert agent.tool_routes[-1]["emitted_tool_schema_mode"] == "compact"
    assert "emitted_names" not in agent.tool_routes[-1]


def test_tau_official_budget_uses_minimal_schema_instead_of_truncating_prompt() -> None:
    tools = [
        {
            "name": f"wide_tool_{index}",
            "description": f"Wide tool {index} " + ("description " * 120),
            "parameters": {
                "type": "object",
                "properties": {
                    f"field_{field}": {
                        "type": "string",
                        "description": "Long field description " * 20,
                    }
                    for field in range(16)
                },
                "required": [f"field_{field}" for field in range(4)],
            },
        }
        for index in range(6)
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow the policy.",
        history_max_chars=1200,
        prompt_max_chars=4096,
    )

    prompt = agent._build_prompt([{"role": "user", "content": "Handle record R-1."}])  # noqa: SLF001

    assert len(prompt) <= 4096
    assert agent.tool_routes[-1]["emitted_tool_schema_mode"] == "minimal"
    assert "Policy:\nFollow the policy." in prompt
    for index in range(6):
        assert f'"name":"wide_tool_{index}"' in prompt
        assert f"wide_tool_{index}" in agent._current_tool_names  # noqa: SLF001


def test_tau_official_budget_allows_3k_routed_prompt() -> None:
    tools = [
        {
            "name": "get_reservation_details",
            "description": "Get airline reservation details " + ("description " * 80),
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string", "description": "reservation id"}},
                "required": ["reservation_id"],
            },
        },
        {
            "name": "cancel_reservation",
            "description": "Cancel an airline reservation " + ("description " * 80),
            "parameters": {
                "type": "object",
                "properties": {"reservation_id": {"type": "string", "description": "reservation id"}},
                "required": ["reservation_id"],
            },
        },
        {
            "name": "search_direct_flight",
            "description": "Search direct flights " + ("description " * 80),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow the airline cancellation policy.",
        history_max_chars=3200,
        prompt_max_chars=3000,
        tool_routing_config=ToolRoutingConfig(
            mode="lexical",
            max_tools=2,
            trigger_tool_count=1,
            trigger_catalog_chars=1,
        ),
    )

    prompt = agent._build_prompt([{"role": "user", "content": "Cancel reservation EHGLP3."}])  # noqa: SLF001

    assert len(prompt) <= 3000
    assert '"name":"cancel_reservation"' in prompt
    assert '"name":"search_direct_flight"' not in prompt
    assert agent.tool_routes[-1]["routed"] is True


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


def test_tau_official_agent_compacts_domain_policy_with_model_parallel_router() -> None:
    class _EvidenceEngine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001
            assert kwargs["batch_size"] > 1
            return [
                SimpleNamespace(
                    text='{"relevant":true,"score":3}'
                    if "waiver code ALPHA7" in prompt
                    else '{"relevant":false,"score":0}',
                    finish_reason="stop",
                )
                for prompt in prompts
            ]

    long_policy = "\n".join(
        [f"irrelevant policy row {index:03d}" for index in range(40)]
        + ["waiver code ALPHA7 requires supervisor approval before refund"]
        + [f"archive policy row {index:03d}" for index in range(40)]
    )
    agent = RWKVTauOfficialAgent(
        engine=_EvidenceEngine(),
        sampling=SimpleNamespace(),
        tools=[],
        domain_policy=long_policy,
        history_max_chars=12000,
        prompt_max_chars=5000,
        long_doc_config=LongDocEvidenceConfig(
            mode="model_parallel",
            max_chunk_chars=240,
            overlap_lines=1,
            min_long_text_chars=400,
            max_evidence_chunks=1,
            max_evidence_chars=320,
            model_parallel_batch_size=8,
        ),
    )

    prompt = agent._build_prompt([{"role": "user", "content": "Can this refund use the special waiver?"}])  # noqa: SLF001

    assert "mode=model_parallel" in prompt
    assert "waiver code ALPHA7 requires supervisor approval before refund" in prompt
    assert "irrelevant policy row 000" not in prompt
    assert len(prompt) <= 5000


def test_tau_official_agent_routes_tool_catalog_per_turn() -> None:
    tools = [
        {
            "name": "refund_order",
            "description": "Refund an order",
            "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
        },
        {
            "name": "lookup_weather",
            "description": "Read city weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    ]
    agent = RWKVTauOfficialAgent(
        engine=SimpleNamespace(),
        sampling=SimpleNamespace(),
        tools=tools,
        domain_policy="Follow the refund policy.",
        history_max_chars=12000,
        prompt_max_chars=5000,
        tool_routing_config=ToolRoutingConfig(
            mode="lexical",
            max_tools=1,
            trigger_tool_count=1,
            trigger_catalog_chars=1,
        ),
    )

    prompt = agent._build_prompt([{"role": "user", "content": "Refund order ORD-7."}])  # noqa: SLF001

    assert '"name": "refund_order"' in prompt
    assert '"name": "lookup_weather"' not in prompt
    assert agent.tool_routes[-1]["selected_names"] == ["refund_order"]
    try:
        agent._decision_to_assistant_message("lookup_weather", {"city": "SFO"})  # noqa: SLF001
    except ValueError as exc:
        assert "not in routed tool window" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected routed tool-window validation")


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


def test_tau_litellm_model_name_prefixes_official_deepseek() -> None:
    cfg = OpenAIModelConfig(
        api_key="test-key",
        model_name="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )

    assert _tau_litellm_model_name(cfg) == "deepseek/deepseek-v4-flash"


def test_tau_litellm_model_name_keeps_explicit_provider() -> None:
    cfg = OpenAIModelConfig(
        api_key="test-key",
        model_name="deepseek/deepseek-v4-flash",
        base_url="https://api.deepseek.com",
    )

    assert _tau_litellm_model_name(cfg) == "deepseek/deepseek-v4-flash"


def test_tau_llm_timeout_args_uses_first_positive_env(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_TAU_LLM_TIMEOUT_S", "45.5")
    monkeypatch.setenv("RWKV_TAU_USER_TIMEOUT_S", "60")

    assert _tau_llm_timeout_args() == {"timeout": 45.5}


def test_tau_llm_timeout_args_ignores_missing_or_invalid_env(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_TAU_LLM_TIMEOUT_S", raising=False)
    monkeypatch.setenv("RWKV_TAU_USER_TIMEOUT_S", "not-a-number")
    monkeypatch.setenv("RWKV_LLM_TIMEOUT_S", "0")

    assert _tau_llm_timeout_args() == {}


def test_tau_runtime_model_metadata_records_user_and_judge_without_keys() -> None:
    user_model = OpenAIModelConfig(
        api_key="user-secret",
        model_name="gpt-5.4-mini",
        base_url="https://next-token.cc/v1",
    )
    judge_model = OpenAIModelConfig(
        api_key="judge-secret",
        model_name="gpt-5.4",
        base_url="https://next-token.cc/v1",
    )

    metadata = _tau_runtime_model_metadata(user_model, judge_model)

    assert metadata == {
        "user_model": "gpt-5.4-mini",
        "user_base_url": "https://next-token.cc/v1",
        "judge_model": "gpt-5.4",
        "judge_base_url": "https://next-token.cc/v1",
        "static_user": False,
        "judge_configured": True,
    }
    assert "secret" not in str(metadata)


def test_tau_nl_assertions_judge_config_uses_timeout_env(monkeypatch) -> None:
    monkeypatch.setenv("RWKV_TAU_LLM_TIMEOUT_S", "12")
    cfg = OpenAIModelConfig(
        api_key="test-key",
        model_name="gpt-5.4",
        base_url="https://api.ablai.top/v1",
    )

    configure_tau_nl_assertions_judge(cfg)

    from tau2.evaluator import evaluator_nl_assertions

    assert evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS["timeout"] == 12.0


def test_tau_official_payload_preserves_official_reward_with_parse_diagnostics() -> None:
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

    assert payload["agent_result"]["reward"] == 1.0
    assert payload["agent_result"]["is_passed"] is True
    assert payload["agent_result"]["error"] == "tau agent decision missing name"
    assert payload["agent_info"]["official_reward"] == 1.0
    assert payload["agent_info"]["official_is_passed"] is True


def test_tau_official_attempt_records_runtime_error_as_failed_payload() -> None:
    record = SimpleNamespace(
        task_id="task-1",
        domain="airline",
        benchmark_version="tau_v2",
        task={"id": "task-1"},
    )

    payload = _run_tau_official_attempt(
        args=SimpleNamespace(),
        run=SimpleNamespace(
            engine=SimpleNamespace(),
            benchmark_name="tau2_bench_airline",
            dataset_split="base",
        ),
        record=record,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        runtime_env=_RaisingTauRuntime("UserMessage must have either content or tool_calls"),
        user_model=SimpleNamespace(),
        judge_model=SimpleNamespace(),
        sampling=SamplingConfig(max_generate_tokens=64),
        sampling_payload={},
        history_max_chars=12000,
        prompt_max_chars=8192,
        long_doc_config=LongDocEvidenceConfig(min_long_text_chars=1000),
        max_steps=6,
        max_tool_errors=3,
    )

    assert payload["agent_result"]["reward"] == 0.0
    assert payload["agent_result"]["is_passed"] is False
    assert "UserMessage must have either content or tool_calls" in payload["agent_result"]["error"]
    assert "runtime_error" in payload["agent_info"]


def test_tau3_lightweight_records_do_not_require_external_tau3_or_user_model() -> None:
    records = [
        TauManifestRecord(
            task_id="mock_long_context_create_task",
            domain="mock",
            instruction="Create task",
            task={},
            benchmark_version="tau_v3_light",
        )
    ]

    assert not _requires_tau_v3_source(records)
    assert not _requires_tau_user_model(records)


def test_tau_official_task_payload_normalizes_reward_type_strings() -> None:
    payload = {
        "id": "mock_unit_create_task",
        "evaluation_criteria": {
            "reward_basis": ["RewardType.DB", "RewardType.ENV_ASSERTION", "ACTION"],
        },
    }

    normalized = normalize_tau_official_task_payload(payload)

    assert normalized["evaluation_criteria"]["reward_basis"] == ["DB", "ENV_ASSERTION", "ACTION"]
    assert payload["evaluation_criteria"]["reward_basis"] == [
        "RewardType.DB",
        "RewardType.ENV_ASSERTION",
        "ACTION",
    ]


def test_tau3_lightweight_mock_attempt_runs_without_user_llm() -> None:
    task = {
        "id": "mock_unit_create_task",
        "description": {"purpose": "unit test"},
        "user_scenario": {
            "persona": "Direct user",
            "instructions": "Create a task called Important Meeting for user_1.",
        },
        "ticket": "Create a task called Important Meeting for user_1.",
        "initial_state": {
            "message_history": [
                {
                    "role": "user",
                    "content": "Please create a task titled Important Meeting for user_1.",
                    "turn_idx": 0,
                }
            ]
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "action_id": "create_important_meeting",
                    "name": "create_task",
                    "arguments": {"user_id": "user_1", "title": "Important Meeting"},
                }
            ],
            "env_assertions": [
                {
                    "env_type": "assistant",
                    "func_name": "assert_task_status",
                    "arguments": {"task_id": "task_2", "expected_status": "pending"},
                }
            ],
            "reward_basis": ["RewardType.DB", "RewardType.ENV_ASSERTION", "RewardType.ACTION"],
        },
    }
    record = TauManifestRecord(
        task_id="mock_unit_create_task",
        domain="mock",
        instruction="Create task",
        task=task,
        benchmark_version="tau_v3_light",
    )
    engine = _SequencedEngine(
        [
            '{"name":"create_task","arguments":{"user_id":"user_1","title":"Important Meeting"}}',
            '{"name":"respond","arguments":{"content":"Created. ###STOP###"}}',
        ]
    )

    payload = _run_tau_official_attempt(
        args=SimpleNamespace(),
        run=SimpleNamespace(
            engine=engine,
            benchmark_name="tau3_bench_mock",
            dataset_split="base",
        ),
        record=record,
        sample_index=0,
        repeat_index=0,
        pass_index=1,
        runtime_env=TauOfficialRuntime(domain="mock"),
        user_model=None,
        judge_model=None,
        sampling=SamplingConfig(max_generate_tokens=64),
        sampling_payload={},
        history_max_chars=12000,
        prompt_max_chars=8192,
        long_doc_config=LongDocEvidenceConfig(min_long_text_chars=1000),
        max_steps=6,
        max_tool_errors=3,
    )

    assert payload["agent_result"]["is_passed"] is True
    assert payload["agent_result"]["reward"] == 1.0
    assert payload["agent_result"]["num_turns"] == 2


class _SequencedEngine:
    model_name = "fake-rwkv"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

    def generate(self, prompts, **_kwargs):
        return [
            GenerationOutput(
                prompt_index=index,
                prompt=prompt,
                token_ids=[],
                text=self.outputs.pop(0),
                finish_reason="stop",
            )
            for index, prompt in enumerate(prompts)
        ]


class _FakeTauEnvironment:
    def get_tools(self):
        return []

    def get_policy(self):
        return "Follow policy."


class _RaisingTauRuntime:
    def __init__(self, message: str) -> None:
        self.message = message

    def load_task(self, task):
        return task

    def create_environment(self, solo_mode: bool = False):
        return _FakeTauEnvironment()

    def build_user(self, **_kwargs):
        return SimpleNamespace()

    def build_orchestrator(self, **_kwargs):
        message = self.message

        class _RaisingOrchestrator:
            def run(self):
                raise RuntimeError(message)

        return _RaisingOrchestrator()
