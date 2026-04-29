from __future__ import annotations

from src.eval.function_calling import (
    build_tau_system_prompt,
    parse_tool_call_or_final_answer,
    render_tau_user_prompt,
)


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
