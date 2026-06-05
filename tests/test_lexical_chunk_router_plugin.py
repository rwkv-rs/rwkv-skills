from __future__ import annotations

from types import SimpleNamespace

from lexical_chunk_router import (
    LongDocConfig,
    ToolRouterConfig,
    chunk_text,
    compact_messages,
    compact_text,
    route_tools,
    summarize_tool,
)
from src.plugins import lexical_chunk_router as compat_lexical_chunk_router


def _tool(name: str, description: str, *properties: str) -> dict[str, object]:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in properties},
            "required": list(properties[:1]),
        },
    }


def test_plugin_chunk_text_preserves_overlap() -> None:
    chunks = chunk_text("line1\nline2\nline3\nline4\n", max_chars=12, overlap_lines=1)

    assert [chunk.line_start for chunk in chunks] == [1, 2]
    assert [chunk.line_end for chunk in chunks] == [2, 4]
    assert chunks[1].overlap_lines == 1
    assert chunks[1].text.startswith("line2\nline3\n")


def test_plugin_compact_text_selects_query_relevant_chunk() -> None:
    text = "\n".join(
        [f"noise row {index:03d}" for index in range(40)]
        + ["invoice INV-42 status paid evidence"]
        + [f"archive row {index:03d}" for index in range(40)]
    )

    result = compact_text(
        text,
        query="What is the status of invoice INV-42?",
        config=LongDocConfig(
            max_chunk_chars=160,
            overlap_lines=1,
            min_long_text_chars=200,
            max_evidence_chunks=1,
            max_evidence_chars=240,
        ),
        label="invoice-log",
    )

    assert result.compacted is True
    assert result.chunk_count > 1
    assert "invoice INV-42 status paid evidence" in result.text
    assert "noise row 000" not in result.text
    assert "mode=lexical" in result.text
    assert result.trace_payload()["selected_chunk_ids"] == list(result.selected_chunk_ids)


def test_plugin_compact_messages_uses_recent_short_user_query() -> None:
    task = "Find invoice INV-42 status."
    long_tool_output = "\n".join(
        [f"unrelated ledger row {index:03d}" for index in range(30)]
        + ["invoice INV-42 status paid evidence"]
        + [f"archive row {index:03d}" for index in range(30)]
    )
    messages = [
        {"role": "user", "content": task},
        {"role": "assistant", "content": '{"name":"lookup_invoice","arguments":{"id":"INV-42"}}'},
        {"role": "user", "content": long_tool_output},
    ]

    result = compact_messages(
        messages,
        config=LongDocConfig(
            max_chunk_chars=160,
            overlap_lines=1,
            min_long_text_chars=200,
            max_evidence_chunks=1,
            max_evidence_chars=240,
        ),
    )

    assert result.compacted_message_count == 1
    assert "invoice INV-42 status paid evidence" in result.messages[-1]["content"]


def test_plugin_route_tools_selects_lexical_window() -> None:
    tools = [
        _tool("lookup_weather", "Read city weather forecast", "city"),
        _tool("book_flight", "Book an airline ticket", "origin", "destination"),
        _tool("cancel_order", "Cancel a retail order", "order_id"),
    ]

    route = route_tools(
        tools,
        [{"role": "user", "content": "Please book a flight from SFO to LAX."}],
        config=ToolRouterConfig(max_tools=1, trigger_tool_count=1, trigger_catalog_chars=1),
    )

    assert route.routed is True
    assert route.selected_names == ("book_flight",)
    assert route.trace_payload()["reason"] == "lexical"


def test_plugin_compact_text_model_parallel_uses_rwkv_backend_protocol() -> None:
    class _Backend:
        def __init__(self) -> None:
            self.prompt_count = 0
            self.batch_size = 0

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            self.prompt_count = len(prompts)
            self.batch_size = kwargs["batch_size"]
            return [
                SimpleNamespace(
                    text='{"relevant":true,"score":3}'
                    if "special-policy ALPHA7" in prompt
                    else '{"relevant":false,"score":0}',
                    finish_reason="stop",
                )
                for prompt in prompts
            ]

    backend = _Backend()
    text = "\n".join(
        [f"noise policy row {index:03d}" for index in range(20)]
        + ["special-policy ALPHA7 requires supervisor approval"]
        + [f"archive policy row {index:03d}" for index in range(20)]
    )

    result = compact_text(
        text,
        query="What is the approval requirement?",
        config=LongDocConfig(
            mode="model_parallel",
            max_chunk_chars=160,
            overlap_lines=1,
            min_long_text_chars=200,
            max_evidence_chunks=1,
            max_evidence_chars=240,
            model_parallel_batch_size=8,
        ),
        backend=backend,
        sampling=SimpleNamespace(),
    )

    assert backend.prompt_count > 1
    assert backend.batch_size > 1
    assert result.trace_payload()["mode"] == "model_parallel"
    assert "special-policy ALPHA7 requires supervisor approval" in result.text
    assert "noise policy row 000" not in result.text


def test_plugin_route_tools_model_parallel_uses_rwkv_backend_protocol() -> None:
    class _Backend:
        def __init__(self) -> None:
            self.prompt_count = 0
            self.batch_size = 0

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            self.prompt_count = len(prompts)
            self.batch_size = kwargs["batch_size"]
            assert all("Tool catalog:" in prompt for prompt in prompts)
            return [
                SimpleNamespace(text='{"selected_tools":["lookup_weather"]}', finish_reason="stop"),
                SimpleNamespace(text='{"selected_tools":["refund_order"]}', finish_reason="stop"),
            ]

    backend = _Backend()
    tools = [
        _tool("lookup_weather", "Read city weather forecast", "city"),
        _tool("book_flight", "Book an airline ticket", "origin", "destination"),
        _tool("refund_order", "Refund a retail order", "order_id"),
        _tool("cancel_order", "Cancel a retail order", "order_id"),
    ]

    route = route_tools(
        tools,
        [{"role": "user", "content": "Check weather, then refund order ORD-7."}],
        config=ToolRouterConfig(
            mode="model_parallel",
            max_tools=3,
            trigger_tool_count=1,
            trigger_catalog_chars=1,
            parallel_chunk_tools=2,
            parallel_batch_size=4,
        ),
        backend=backend,
        sampling=SimpleNamespace(),
    )

    assert backend.prompt_count == 2
    assert backend.batch_size == 2
    assert route.model_names == ("lookup_weather", "refund_order")
    assert route.trace_payload()["parallel_chunk_count"] == 2


def test_plugin_route_tools_keeps_retail_state_hints() -> None:
    tools = [
        _tool("calculate", "Calculate the result of a mathematical expression.", "expression"),
        _tool("exchange_delivered_order_items", "Exchange items in a delivered order", "order_id"),
        _tool("modify_pending_order_items", "Modify items in a pending order", "order_id"),
        _tool("return_delivered_order_items", "Return items in a delivered order", "order_id"),
        _tool("get_order_details", "Get the status and details of an order.", "order_id"),
        _tool("list_all_product_types", "List the name and product id of all product types."),
        _tool("get_product_details", "Get the inventory details of a product.", "product_id"),
        _tool("get_item_details", "Get the inventory details of an item.", "item_id"),
    ]

    route = route_tools(
        tools,
        [{"role": "user", "content": "I received order #W2378156 and need to exchange a keyboard."}],
        config=ToolRouterConfig(max_tools=8, trigger_tool_count=1, trigger_catalog_chars=1),
    )

    assert route.selected_names[:4] == (
        "get_order_details",
        "list_all_product_types",
        "get_product_details",
        "get_item_details",
    )
    assert "exchange_delivered_order_items" in route.selected_names
    assert "calculate" not in route.selected_names


def test_plugin_summarize_tool_accepts_openai_function_shape() -> None:
    summary = summarize_tool(
        {
            "type": "function",
            "function": _tool("book_flight", "Book an airline ticket", "origin", "destination"),
        },
        description_chars=20,
    )

    assert summary["name"] == "book_flight"
    assert summary["required"] == ["origin"]
    assert summary["properties"] == ["destination", "origin"]


def test_src_plugins_import_path_remains_compatible() -> None:
    assert compat_lexical_chunk_router.LongDocConfig is LongDocConfig
    assert compat_lexical_chunk_router.route_tools is route_tools
