from __future__ import annotations

from types import SimpleNamespace

from src.eval.tasks.agent_bench.chat_bridge import RWKVChatBridge


class _FakeEngine:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompts, sampling, batch_size, progress_desc):
        self.prompts.extend(str(prompt) for prompt in prompts)
        return [
            SimpleNamespace(
                prompt_index=index,
                text=self.text,
                finish_reason="stop",
            )
            for index, _prompt in enumerate(prompts)
        ]


def _tools_schema() -> list[dict[str, object]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup a record",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
        }
    ]


def test_rwkv_chat_bridge_prefills_json_and_collapses_role_headers() -> None:
    engine = _FakeEngine('"name":"lookup","arguments":{"id":"A1"}}')
    bridge = RWKVChatBridge(engine=engine, default_sampling=object())

    result = bridge.chat(
        [
            {"role": "system", "content": "Follow policy."},
            {"role": "user", "content": "Find A1"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_0", "function": {"name": "lookup", "arguments": '{"id":"A0"}'}}],
            },
            {"role": "tool", "tool_call_id": "call_0", "name": "lookup", "content": '{"ok": true}'},
            {"role": "user", "content": "Now find A1"},
        ],
        tools_schema=_tools_schema(),
    )

    prompt = engine.prompts[0]
    assert sum(1 for line in prompt.splitlines() if line.startswith("User:")) == 1
    assert sum(1 for line in prompt.splitlines() if line.startswith("Assistant:")) == 1
    assert "Conversation transcript JSON:" in prompt
    assert prompt.endswith("Assistant: ```json\n{")
    assert [call.name for call in result.tool_calls] == ["lookup"]
    assert result.tool_calls[0].arguments == {"id": "A1"}


def test_rwkv_chat_bridge_accepts_agentic_tool_call_label() -> None:
    engine = _FakeEngine('**Tool Call:** lookup(id="A1")')
    bridge = RWKVChatBridge(engine=engine, default_sampling=object())

    result = bridge.chat([{"role": "user", "content": "Find A1"}], tools_schema=_tools_schema())

    assert [call.name for call in result.tool_calls] == ["lookup"]
    assert result.tool_calls[0].arguments == {"id": "A1"}
