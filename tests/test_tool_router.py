from __future__ import annotations

from types import SimpleNamespace

from src.eval.function_calling.tool_router import (
    ToolRoutingConfig,
    parse_tool_router_response,
    route_tools_for_prompt,
    summarize_tool_for_router,
)


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


def test_lexical_tool_router_selects_relevant_tool_window() -> None:
    tools = [
        _tool("lookup_weather", "Read city weather forecast", "city"),
        _tool("book_flight", "Book an airline ticket", "origin", "destination"),
        _tool("cancel_order", "Cancel a retail order", "order_id"),
    ]

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "Please book a flight from SFO to LAX."}],
        config=ToolRoutingConfig(mode="lexical", max_tools=1, trigger_tool_count=1, trigger_catalog_chars=1),
    )

    assert route.routed is True
    assert route.selected_names == ("book_flight",)
    assert route.trace_payload()["reason"] == "lexical"


def test_model_tool_router_merges_model_and_lexical_names() -> None:
    class _Engine:
        model_name = "router-test"

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            assert "Tool catalog:" in prompts[0]
            return [
                SimpleNamespace(
                    text='{"selected_tools":["lookup_weather"]}',
                    finish_reason="stop",
                )
            ]

    tools = [
        _tool("lookup_weather", "Read city weather forecast", "city"),
        _tool("book_flight", "Book an airline ticket", "origin", "destination"),
        _tool("cancel_order", "Cancel a retail order", "order_id"),
    ]

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "Book a flight, but check the city weather first."}],
        config=ToolRoutingConfig(mode="model", max_tools=2, trigger_tool_count=1, trigger_catalog_chars=1),
        engine=_Engine(),
        sampling=SimpleNamespace(),
    )

    assert route.selected_names == ("lookup_weather", "book_flight")
    assert route.model_names == ("lookup_weather",)
    assert "book_flight" in route.lexical_names


def test_model_parallel_tool_router_shards_catalog_and_batches_calls() -> None:
    class _Engine:
        model_name = "router-test"

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

    engine = _Engine()
    tools = [
        _tool("lookup_weather", "Read city weather forecast", "city"),
        _tool("book_flight", "Book an airline ticket", "origin", "destination"),
        _tool("refund_order", "Refund a retail order", "order_id"),
        _tool("cancel_order", "Cancel a retail order", "order_id"),
    ]

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "Check weather, then refund order ORD-7."}],
        config=ToolRoutingConfig(
            mode="model_parallel",
            max_tools=3,
            trigger_tool_count=1,
            trigger_catalog_chars=1,
            parallel_chunk_tools=2,
            parallel_batch_size=4,
        ),
        engine=engine,
        sampling=SimpleNamespace(),
    )

    assert engine.prompt_count == 2
    assert engine.batch_size == 2
    assert route.model_names == ("lookup_weather", "refund_order")
    assert route.selected_names[:2] == ("lookup_weather", "refund_order")
    assert route.trace_payload()["parallel_chunk_count"] == 2


def test_model_parallel_tool_router_recovers_names_from_runaway_json() -> None:
    class _Engine:
        model_name = "router-test"

        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text='{"name":"get_reservation_details","arguments":"{\\"reservation_id\\":\\"HXDUBJ\\"}","id":"unterminated',
                    finish_reason="length",
                )
            ]

    tools = [
        _tool("get_reservation_details", "Get reservation details", "reservation_id"),
        _tool("cancel_reservation", "Cancel a reservation", "reservation_id"),
    ]

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "Reservation HXDUBJ needs review."}],
        config=ToolRoutingConfig(
            mode="model_parallel",
            max_tools=1,
            trigger_tool_count=1,
            trigger_catalog_chars=1,
            parallel_chunk_tools=2,
        ),
        engine=_Engine(),
        sampling=SimpleNamespace(),
    )

    assert route.model_names == ("get_reservation_details",)
    assert route.selected_names == ("get_reservation_details",)


def test_tool_router_prioritizes_tau_state_anchor_over_noisy_model_choice() -> None:
    class _Engine:
        model_name = "router-test"

        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [SimpleNamespace(text='{"selected_tools":["search_direct_flight"]}', finish_reason="stop")]

    tools = [
        _tool("get_reservation_details", "Get reservation details", "reservation_id"),
        _tool("search_direct_flight", "Search direct flights", "origin", "destination", "date"),
        _tool("book_reservation", "Book a reservation", "user_id"),
    ]

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "Please check reservation HXDUBJ before searching anything else."}],
        config=ToolRoutingConfig(mode="model", max_tools=1, trigger_tool_count=1, trigger_catalog_chars=1),
        engine=_Engine(),
        sampling=SimpleNamespace(),
    )

    assert route.selected_names == ("get_reservation_details",)
    assert route.model_names == ("search_direct_flight",)
    assert route.lexical_names[0] == "get_reservation_details"


def test_tool_router_prioritizes_retail_read_tools_before_write_tools() -> None:
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

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "I received order #W2378156 and need to exchange a keyboard."}],
        config=ToolRoutingConfig(mode="lexical", max_tools=8, trigger_tool_count=1, trigger_catalog_chars=1),
    )

    assert route.selected_names[:4] == (
        "get_order_details",
        "list_all_product_types",
        "get_product_details",
        "get_item_details",
    )
    assert "exchange_delivered_order_items" in route.selected_names
    assert "calculate" not in route.selected_names


def test_tool_router_keeps_complexfuncbench_attraction_location_tool() -> None:
    tools = [
        _tool("Search_Hotels", "Search hotels with destination id.", "dest_id"),
        _tool("Search_Hotel_Destination", "Find hotel destination by city or place.", "query"),
        _tool("Search_Attractions", "Search attractions by destination location.", "location_id"),
        _tool("Search_Attraction_Location", "Find attraction location by place name.", "query"),
        _tool("Get_Seat_Map", "Get aircraft seat map.", "flight_id"),
        _tool("Get_Packages", "Get travel packages.", "package_id"),
    ]

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "Find a hotel and nearby tourist attractions in Paris."}],
        config=ToolRoutingConfig(mode="lexical", max_tools=4, trigger_tool_count=1, trigger_catalog_chars=1),
    )

    assert "Search_Hotel_Destination" in route.selected_names
    assert "Search_Attraction_Location" in route.selected_names
    assert "Search_Attractions" in route.selected_names
    assert "Get_Seat_Map" not in route.selected_names


def test_tool_router_triggers_on_large_catalog_even_with_few_tools() -> None:
    tools = [
        _tool("refund_order", "Refund an order " + ("details " * 80), "order_id"),
        _tool("lookup_weather", "Read city weather " + ("forecast " * 80), "city"),
    ]

    route = route_tools_for_prompt(
        tools,
        [{"role": "user", "content": "Refund order ORD-7."}],
        config=ToolRoutingConfig(
            mode="lexical",
            max_tools=8,
            trigger_tool_count=16,
            trigger_catalog_chars=200,
        ),
    )

    assert route.routed is True
    assert route.selected_names == ("refund_order",)


def test_parse_tool_router_response_accepts_common_shapes() -> None:
    assert parse_tool_router_response('{"selected_tools":["a","b"]}') == ["a", "b"]
    assert parse_tool_router_response('```json\n{"tools":"a, b"}\n```') == ["a", "b"]
    assert parse_tool_router_response('{"name":"get_reservation_details","arguments":{"reservation_id":"R1"}}') == [
        "get_reservation_details"
    ]
    assert parse_tool_router_response('{"name":"get_reservation_details","id":"unterminated') == [
        "get_reservation_details"
    ]
    assert parse_tool_router_response(
        '{"tool_calls":[{"function":{"name":"cancel_reservation","arguments":"{}"}}]}'
    ) == ["cancel_reservation"]


def test_tool_router_summary_keeps_schema_shape_small() -> None:
    summary = summarize_tool_for_router(
        _tool("book_flight", "Book an airline ticket", "origin", "destination"),
        description_chars=20,
    )

    assert summary["name"] == "book_flight"
    assert summary["required"] == ["origin"]
    assert summary["properties"] == ["destination", "origin"]
