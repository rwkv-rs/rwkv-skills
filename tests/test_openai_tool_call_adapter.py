from __future__ import annotations

import json

from src.bin.run_openai_tool_call_adapter import (
    create_app,
    normalize_chat_completion_response,
    parse_text_tool_calls,
)


def test_parse_text_tool_calls_accepts_rwkv_direct_call_with_trailing_fence() -> None:
    tool_calls = parse_text_tool_calls(
        '{"name":"add","arguments":"{\\"a\\":2,\\"b\\":3}","id":"chatcmpl-tool-demo"}\n```'
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "chatcmpl-tool-demo"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "add"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}


def test_parse_text_tool_calls_accepts_tool_calls_envelope() -> None:
    tool_calls = parse_text_tool_calls(
        "```json\n"
        '{"type":"tool_calls","tool_calls":[{"name":"lookup","arguments":{"id":"A1"}}]}'
        "\n```"
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "lookup"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"id": "A1"}


def test_parse_text_tool_calls_accepts_prefilled_object_continuation() -> None:
    tool_calls = parse_text_tool_calls('"name":"lookup","arguments":{"id":"A1"}}')

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "lookup"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"id": "A1"}


def test_parse_text_tool_calls_accepts_rwkv_agentic_tool_call_label() -> None:
    tool_calls = parse_text_tool_calls('**Tool Call:** lookup(id="A1")')

    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "lookup"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"id": "A1"}


def test_parse_text_tool_calls_leaves_plain_text_alone() -> None:
    assert parse_text_tool_calls("The task is complete.") == []


def test_parse_text_tool_calls_ignores_invalid_tool_call_shape() -> None:
    assert parse_text_tool_calls('{"tool_calls":[{"arguments":{"id":"A1"}}]}') == []


def test_normalize_chat_completion_response_only_converts_tool_requests() -> None:
    response = {
        "id": "chatcmpl-demo",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"name":"lookup","arguments":{"id":"A1"}}',
                },
                "finish_reason": "stop",
            }
        ],
    }

    without_tools = normalize_chat_completion_response(response, request_payload={})
    assert without_tools["choices"][0]["message"]["content"] is not None

    with_tools = normalize_chat_completion_response(response, request_payload={"tools": [{"type": "function"}]})
    message = with_tools["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "lookup"
    assert with_tools["choices"][0]["finish_reason"] == "tool_calls"


def test_create_app_registers_adapter_routes() -> None:
    app = create_app(
        "http://127.0.0.1:19081/v1",
        model_ids=("rwkv7-g1f-7.2b-20260414-ctx8192",),
    )

    paths = {route.path for route in app.routes}

    assert "/healthz" in paths
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths
    assert "/v1/completions" in paths
