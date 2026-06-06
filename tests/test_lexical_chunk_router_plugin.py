from __future__ import annotations

from lexical_chunk_router import (
    LongDocConfig,
    ToolRouterConfig,
    chunk_text,
    compact_messages,
    compact_text,
    route_tools,
    summarize_tool,
)
from rwkv_agent_eval_plugin import agent_plugin_config_from_sources, route_agent_prompt_inputs
from src.plugins import lexical_chunk_router as compat_lexical_chunk_router
from src.plugins import rwkv_agent_eval_plugin as compat_agent_eval_plugin


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


def test_rwkv_agent_eval_plugin_routes_policy_messages_and_tools() -> None:
    policy = "\n".join(
        [f"policy noise row {index:03d}" for index in range(30)]
        + ["invoice INV-42 policy evidence: paid invoices should be returned with receipt details."]
        + [f"policy archive row {index:03d}" for index in range(30)]
    )
    long_tool_output = "\n".join(
        [f"ledger noise row {index:03d}" for index in range(30)]
        + ["invoice INV-42 status paid evidence"]
        + [f"ledger archive row {index:03d}" for index in range(30)]
    )
    tools = [
        _tool("lookup_weather", "Read city weather forecast", "city"),
        _tool("lookup_invoice", "Lookup invoice payment status", "invoice_id"),
    ]

    config = agent_plugin_config_from_sources(
        {
            "agent_plugin_enabled": True,
            "tool_router_max_tools": 1,
            "tool_router_trigger_tool_count": 1,
            "tool_router_trigger_catalog_chars": 1,
            "tool_router_enable_domain_hints": False,
            "long_context_min_chars": 200,
            "long_context_chunk_chars": 160,
            "long_context_overlap_lines": 1,
            "long_context_max_evidence_chunks": 1,
            "long_context_max_evidence_chars": 260,
            "long_context_query_chars": 500,
            "prompt_max_chars": 3072,
        }
    )

    route = route_agent_prompt_inputs(
        domain_policy=policy,
        tools=tools,
        messages=[
            {"role": "user", "content": "Need invoice INV-42 status."},
            {"role": "assistant", "content": '{"name":"lookup_invoice","arguments":{"invoice_id":"INV-42"}}'},
            {"role": "user", "content": long_tool_output},
        ],
        config=config,
    )

    assert route.tool_route is not None
    assert route.tool_route.selected_names == ("lookup_invoice",)
    assert "invoice INV-42 status paid evidence" in route.messages[-1]["content"]
    assert "ledger noise row 000" not in route.messages[-1]["content"]
    assert "invoice INV-42 policy evidence" in route.domain_policy
    assert route.long_context_trace is not None
    assert route.long_context_trace["compacted_message_count"] == 1
    assert route.long_context_trace["policy_compacted"] is True
    assert config.prompt_max_chars == 3072
    assert config.long_context.max_chunk_chars == 160
    assert compat_agent_eval_plugin.route_agent_prompt_inputs is route_agent_prompt_inputs
