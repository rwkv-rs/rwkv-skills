from __future__ import annotations

import pytest

from src.eval.tasks.function_calling.native_tool_calls import (
    TOOL_CALL_IO_OPENAI_TOOLS,
    run_native_tool_call_decision,
)
from src.infer.sampling import ChatToolCall, ToolCallGenerationOutput


class _NativeEngine:
    def generate_tool_calls(self, message_batches, tools_batches, **kwargs):  # noqa: ANN001
        return [
            ToolCallGenerationOutput(
                prompt_index=0,
                messages=[dict(item) for item in message_batches[0]],
                tools=[dict(item) for item in tools_batches[0]],
                content="",
                tool_calls=[ChatToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})],
                finish_reason="tool_calls",
                raw_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                        }
                    ],
                },
                response_source="tool_calls",
            )
        ]


class _LegacyEngine:
    pass


def test_native_tool_call_decision_uses_openai_tools_contract() -> None:
    decision = run_native_tool_call_decision(
        engine=_NativeEngine(),
        messages=[{"role": "user", "content": "weather?"}],
        tools=[
            {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        sampling=object(),
        progress_desc="test",
    )

    assert TOOL_CALL_IO_OPENAI_TOOLS == "openai-tools"
    assert decision.decoded_calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    assert decision.trace["tool_call_io"] == "openai-tools"


def test_native_tool_call_decision_requires_native_backend() -> None:
    with pytest.raises(NotImplementedError, match="native chat tool calls"):
        run_native_tool_call_decision(
            engine=_LegacyEngine(),
            messages=[{"role": "user", "content": "weather?"}],
            tools=[],
            sampling=object(),
            progress_desc="test",
        )
