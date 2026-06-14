from __future__ import annotations

from types import SimpleNamespace

from src.eval.experiments.parallel_candidate_router import (
    CandidateToolCall,
    ParallelCandidateRouterConfig,
    parse_candidate_tool_call,
    route_parallel_candidate_tool_call,
)


def _tool(name: str, description: str, *required: str) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in required},
            "required": list(required),
        },
    }


def test_parse_candidate_tool_call_keeps_only_candidate_layer_fields() -> None:
    candidate = parse_candidate_tool_call(
        '{"name":"assistant.get_order_details","arguments":{"order_id":"#A1234567"},'
        '"confidence":85,"evidence":"user provided #A1234567"}'
    )

    assert candidate == CandidateToolCall(
        name="get_order_details",
        arguments={"order_id": "#A1234567"},
        confidence=0.85,
        evidence="user provided #A1234567",
    )
    assert tuple(candidate.layer_payload()) == ("name", "arguments", "confidence", "evidence")


def test_parse_candidate_tool_call_recovers_prefilled_truncated_object() -> None:
    candidate = parse_candidate_tool_call(
        '"name": "get_reservation_details",\n'
        '  "arguments": "{\\"reservation_id\\": \\"EHGLP3\\"}",'
    )

    assert candidate.name == "get_reservation_details"
    assert candidate.arguments == {"reservation_id": "EHGLP3"}


def test_parallel_candidate_router_shards_candidates_and_aggregates() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, prompts, **kwargs):  # noqa: ANN001
            self.calls.append((list(prompts), dict(kwargs)))
            if len(prompts) == 2:
                return [
                    SimpleNamespace(
                        text='{"name":"lookup_weather","arguments":{"city":"Paris"},"confidence":0.45,"evidence":"weather mentioned"}',
                        finish_reason="stop",
                    ),
                    SimpleNamespace(
                        text='{"name":"refund_order","arguments":{"order_id":"ORD-7"},"confidence":0.92,"evidence":"refund order ORD-7"}',
                        finish_reason="stop",
                    ),
                ]
            return [
                SimpleNamespace(
                    text='{"name":"refund_order","arguments":{"order_id":"ORD-7"},"confidence":0.95,"evidence":"highest supported candidate"}',
                    finish_reason="stop",
                )
            ]

    engine = _Engine()
    route = route_parallel_candidate_tool_call(
        tools=[
            _tool("lookup_weather", "Read city weather", "city"),
            _tool("book_flight", "Book a flight", "origin", "destination"),
            _tool("refund_order", "Refund an order", "order_id"),
            _tool("cancel_order", "Cancel an order", "order_id"),
        ],
        messages=[{"role": "user", "content": "Please refund order ORD-7."}],
        domain_policy="Refund orders only after reading the order id.",
        domain="retail",
        facts_text=None,
        engine=engine,
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=2, batch_size=8),
    )

    assert len(engine.calls) == 2
    assert len(engine.calls[0][0]) == 2
    assert engine.calls[0][1]["batch_size"] == 2
    assert "Candidates:" in engine.calls[1][0][0]
    assert route.selected is not None
    assert route.selected.layer_payload() == {
        "name": "refund_order",
        "arguments": {"order_id": "ORD-7"},
        "confidence": 0.95,
        "evidence": "highest supported candidate",
    }
    assert route.trace_payload()["candidate_count"] == 2


def test_parallel_candidate_router_falls_back_to_highest_confidence_candidate() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            if len(prompts) == 2:
                return [
                    SimpleNamespace(
                        text='{"name":"lookup_weather","arguments":{"city":"Paris"},"confidence":0.2,"evidence":"weak"}',
                        finish_reason="stop",
                    ),
                    SimpleNamespace(
                        text='{"name":"refund_order","arguments":{"order_id":"ORD-7"},"confidence":0.8,"evidence":"strong"}',
                        finish_reason="stop",
                    ),
                ]
            return [SimpleNamespace(text='{"name":"made_up_tool","arguments":{},"confidence":1,"evidence":"bad"}')]

    route = route_parallel_candidate_tool_call(
        tools=[
            _tool("lookup_weather", "Read city weather", "city"),
            _tool("book_flight", "Book a flight", "origin", "destination"),
            _tool("refund_order", "Refund an order", "order_id"),
            _tool("cancel_order", "Cancel an order", "order_id"),
        ],
        messages=[{"role": "user", "content": "Please refund order ORD-7."}],
        domain_policy="Refund orders only after reading the order id.",
        domain="retail",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=2),
    )

    assert route.fallback_used is True
    assert route.selected is not None
    assert route.selected.name == "refund_order"
    assert route.aggregate_error is not None


def test_parallel_candidate_router_normalizes_common_tool_name_aliases() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text='{"name":"get_reservation","arguments":{"reservation_id":"EHGLP3"},"confidence":0.8,"evidence":"lookup alias"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("get_reservation_details", "Get reservation", "reservation_id")],
        messages=[{"role": "user", "content": "Please inspect reservation EHGLP3."}],
        domain_policy="Use listed tools.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected == CandidateToolCall(
        name="get_reservation_details",
        arguments={"reservation_id": "EHGLP3"},
        confidence=0.8,
        evidence="lookup alias",
    )


def test_parallel_candidate_router_fallback_prefers_matching_update_over_repeated_read() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            if len(prompts) == 2:
                return [
                    SimpleNamespace(
                        text='{"name":"get_reservation","arguments":{"reservation_id":"YAX4DR"},"confidence":0.5,"evidence":"repeat read"}',
                        finish_reason="stop",
                    ),
                    SimpleNamespace(
                        text=(
                            '{"name":"update_reservation","arguments":{"reservation_id":"YAX4DR",'
                            '"total_baggages":2,"nonfree_baggages":0,"payment_id":"credit_card_4938634"},'
                            '"confidence":0.5,"evidence":"add checked bags"}'
                        ),
                        finish_reason="stop",
                    ),
                ]
            return [SimpleNamespace(text='{"name":"made_up_tool","arguments":{},"confidence":1,"evidence":"bad"}')]

    route = route_parallel_candidate_tool_call(
        tools=[
            _tool("get_reservation_details", "Get reservation", "reservation_id"),
            _tool(
                "update_reservation_baggages",
                "Update bags",
                "reservation_id",
                "total_baggages",
                "nonfree_baggages",
                "payment_id",
            ),
        ],
        messages=[
            {
                "role": "user",
                "content": "Please add 2 checked bags to reservation YAX4DR using my card.",
            },
            {
                "role": "assistant",
                "content": '{"name":"get_reservation_details","arguments":{"reservation_id":"YAX4DR"}}',
            },
            {
                "role": "user",
                "content": (
                    'Function output:\n{"requestor":"assistant","ok":true,"output":'
                    '"{\\"reservation_id\\": \\"YAX4DR\\", '
                    '\\"payment_methods\\": {\\"credit_card_4938634\\": {\\"source\\": \\"credit_card\\"}}}"}'
                ),
            },
        ],
        domain_policy="Use exact IDs from successful tool outputs.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.fallback_used is True
    assert route.selected is not None
    assert route.selected.name == "update_reservation_baggages"
    assert route.selected.arguments == {
        "reservation_id": "YAX4DR",
        "total_baggages": 2,
        "nonfree_baggages": 0,
        "payment_id": "credit_card_4938634",
    }


def test_parallel_candidate_router_filters_candidates_missing_required_arguments() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            if len(prompts) == 2:
                return [
                    SimpleNamespace(
                        text='{"name":"book_flight","arguments":{},"confidence":0.9,"evidence":"missing args"}',
                        finish_reason="stop",
                    ),
                    SimpleNamespace(
                        text='{"name":"lookup_weather","arguments":{"city":"Paris"},"confidence":0.6,"evidence":"complete args"}',
                        finish_reason="stop",
                    ),
                ]
            return [
                SimpleNamespace(
                    text='{"name":"book_flight","arguments":{},"confidence":1,"evidence":"bad aggregate"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[
            _tool("book_flight", "Book a flight", "origin", "destination"),
            _tool("lookup_weather", "Read city weather", "city"),
        ],
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        domain_policy="Use complete tool arguments.",
        domain=None,
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected == CandidateToolCall(
        name="lookup_weather",
        arguments={"city": "Paris"},
        confidence=0.6,
        evidence="complete args",
    )
    assert route.fallback_used is True
    assert route.chunks[0].candidate is None
    assert route.chunks[0].error is not None
    assert "missing required arguments" in route.chunks[0].error


def test_parallel_candidate_router_filters_ungrounded_identifier_arguments() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text='{"name":"get_user_details","arguments":{"user_id":"EHGLP3"},"confidence":0.9,"evidence":"wrong id type"}',
                    finish_reason="stop",
                ),
                SimpleNamespace(
                    text='{"name":"update_reservation_baggages","arguments":{"reservation_id":"EHGLP3","total_baggages":1,"nonfree_baggages":0,"payment_id":"AMXZQW"},"confidence":0.8,"evidence":"fake payment"}',
                    finish_reason="stop",
                ),
            ]

    route = route_parallel_candidate_tool_call(
        tools=[
            _tool("get_user_details", "Get airline user", "user_id"),
            _tool(
                "update_reservation_baggages",
                "Update bags",
                "reservation_id",
                "total_baggages",
                "nonfree_baggages",
                "payment_id",
            ),
        ],
        messages=[{"role": "user", "content": "Please cancel my reservation. The confirmation code is EHGLP3."}],
        domain_policy="IDs must be grounded.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected is not None
    assert route.selected.name == "respond"
    assert "missing detail" in route.selected.arguments["content"]
    assert route.aggregate_error == "no valid candidate tool calls"
    assert all(chunk.candidate is None for chunk in route.chunks)
    errors = "\n".join(str(chunk.error or "") for chunk in route.chunks)
    assert "user_id='EHGLP3' (user)" in errors
    assert "payment_id='AMXZQW' (payment)" in errors


def test_parallel_candidate_router_filters_ungrounded_search_dates() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text='{"name":"search_direct_flight","arguments":{"origin":"SFO","destination":"JFK","date":"2025-06-01"},"confidence":0.7,"evidence":"date guessed"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("search_direct_flight", "Search direct flights", "origin", "destination", "date")],
        messages=[{"role": "user", "content": "I need to book a flight from San Francisco to New York."}],
        domain_policy="Do not invent travel dates.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected is not None
    assert route.selected.name == "respond"
    assert route.chunks[0].candidate is None
    assert route.chunks[0].error is not None
    assert "date='2025-06-01' (date)" in route.chunks[0].error


def test_parallel_candidate_router_prunes_undeclared_tool_arguments() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            return [
                SimpleNamespace(
                    text='{"name":"list_all_airports","arguments":{"origin":"SF","destination":"NY","date":"2025-04-01"},"confidence":0.7,"evidence":"needs airport codes"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("list_all_airports", "List all airports")],
        messages=[{"role": "user", "content": "I need to book from San Francisco to New York."}],
        domain_policy="Use list_all_airports with no arguments.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected == CandidateToolCall(
        name="list_all_airports",
        arguments={},
        confidence=0.7,
        evidence="needs airport codes",
    )


def test_parallel_candidate_router_rejects_cancel_without_cancel_intent() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text='{"name":"cancel_reservation","arguments":{"reservation_id":"4OG6T3"},"confidence":0.8,"evidence":"reservation is grounded"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("cancel_reservation", "Cancel reservation", "reservation_id")],
        messages=[
            {"role": "user", "content": "I need to book a flight from San Francisco to New York."},
            {
                "role": "user",
                "content": (
                    'Function output:\n{"requestor":"assistant","ok":true,"output":'
                    '"{\\"reservation_id\\": \\"4OG6T3\\"}"}'
                ),
            },
        ],
        domain_policy="Only cancel when requested.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected is not None
    assert route.selected.name == "respond"
    assert route.chunks[0].candidate is None
    assert route.chunks[0].error == "candidate 'cancel_reservation' lacks explicit user cancellation intent"


def test_parallel_candidate_router_allows_cancel_with_cancel_intent() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            return [
                SimpleNamespace(
                    text='{"name":"cancel_reservation","arguments":{"reservation_id":"EHGLP3"},"confidence":0.8,"evidence":"user asked to cancel EHGLP3"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("cancel_reservation", "Cancel reservation", "reservation_id")],
        messages=[{"role": "user", "content": "Please cancel my reservation EHGLP3."}],
        domain_policy="Cancel only requested reservations.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected == CandidateToolCall(
        name="cancel_reservation",
        arguments={"reservation_id": "EHGLP3"},
        confidence=0.8,
        evidence="user asked to cancel EHGLP3",
    )


def test_parallel_candidate_router_rejects_cancel_for_tool_output_only_reservation_id() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text='{"name":"cancel_reservation","arguments":{"reservation_id":"MZDDS4"},"confidence":0.8,"evidence":"reservation came from profile scan"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("cancel_reservation", "Cancel reservation", "reservation_id")],
        messages=[
            {"role": "user", "content": "I want to cancel my trip from Philadelphia to LaGuardia."},
            {"role": "user", "content": "My user ID is raj_sanchez_7340."},
            {
                "role": "user",
                "content": (
                    'Function output:\n{"requestor":"assistant","ok":true,"output":'
                    '"{\\"reservation_id\\": \\"MZDDS4\\", \\"origin\\": \\"MIA\\", \\"destination\\": \\"LAX\\"}"}'
                ),
            },
        ],
        domain_policy="Cancel only a reservation ID provided by the user.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected is not None
    assert route.selected.name == "respond"
    assert route.chunks[0].candidate is None
    assert route.chunks[0].error == (
        "candidate 'cancel_reservation' uses a tool-output reservation_id that does not match the user request"
    )


def test_parallel_candidate_router_allows_identifiers_from_successful_tool_output() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            if self.calls == 1:
                return [
                    SimpleNamespace(
                        text='{"name":"get_user_details","arguments":{"user_id":"emma_kim_9957"},"confidence":0.8,"evidence":"reservation output includes user_id"}',
                        finish_reason="stop",
                    )
                ]
            return [
                SimpleNamespace(
                    text='{"name":"get_user_details","arguments":{"user_id":"emma_kim_9957"},"confidence":0.9,"evidence":"grounded in tool output"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("get_user_details", "Get airline user", "user_id")],
        messages=[
            {"role": "user", "content": "Please cancel reservation EHGLP3."},
            {
                "role": "user",
                "content": (
                    'Function output:\n{"requestor":"assistant","ok":true,"output":'
                    '"{\\"reservation_id\\": \\"EHGLP3\\", \\"user_id\\": \\"emma_kim_9957\\"}"}'
                ),
            },
        ],
        domain_policy="Use exact IDs from tool outputs.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected == CandidateToolCall(
        name="get_user_details",
        arguments={"user_id": "emma_kim_9957"},
        confidence=0.9,
        evidence="grounded in tool output",
    )


def test_parallel_candidate_router_ignores_ids_echoed_by_failed_tool_output() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text='{"name":"get_reservation_details","arguments":{"reservation_id":"ZFA04Y"},"confidence":0.9,"evidence":"failed output echoed id"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("get_reservation_details", "Get reservation", "reservation_id")],
        messages=[
            {"role": "user", "content": "I need compensation for a canceled business flight."},
            {
                "role": "user",
                "content": (
                    '<tool_response>\n{"requestor":"assistant","ok":false,'
                    '"output":"Error: Reservation ZFA04Y not found"}\n</tool_response>'
                ),
            },
        ],
        domain_policy="Do not reuse failed lookup echoes as source IDs.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected is not None
    assert route.selected.name == "respond"
    assert route.chunks[0].candidate is None
    assert route.chunks[0].error is not None
    assert "reservation_id='ZFA04Y' (reservation)" in route.chunks[0].error


def test_parallel_candidate_router_allows_cancel_with_reservation_id_from_tool_output() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            return [
                SimpleNamespace(
                    text='{"name":"cancel_reservation","arguments":{"reservation_id":"XXDC1M"},"confidence":0.9,"evidence":"matching reservation came from successful tool output"}',
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("cancel_reservation", "Cancel reservation", "reservation_id")],
        messages=[
            {"role": "user", "content": "Please cancel my ATL to JFK flight on May 17."},
            {
                "role": "user",
                "content": (
                    'Function output:\n{"requestor":"assistant","ok":true,"output":'
                    '"{\\"reservation_id\\": \\"XXDC1M\\", \\"origin\\": \\"ATL\\", '
                    '\\"destination\\": \\"JFK\\", \\"flights\\": [{\\"date\\": \\"2024-05-17\\"}]}"}'
                ),
            },
        ],
        domain_policy="Use exact IDs from successful tool outputs.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected == CandidateToolCall(
        name="cancel_reservation",
        arguments={"reservation_id": "XXDC1M"},
        confidence=0.9,
        evidence="matching reservation came from successful tool output",
    )


def test_parallel_candidate_router_normalizes_date_of_birth_alias() -> None:
    class _Engine:
        def generate(self, prompts, **kwargs):  # noqa: ANN001, ARG002
            return [
                SimpleNamespace(
                    text=(
                        '{"name":"update_reservation_passengers",'
                        '"arguments":{"reservation_id":"3RK2T9","passengers":[{"first_name":"Anya",'
                        '"last_name":"Garcia","date_of_birth":"1992-11-12"}]},'
                        '"confidence":0.8,"evidence":"passenger update"}'
                    ),
                    finish_reason="stop",
                )
            ]

    route = route_parallel_candidate_tool_call(
        tools=[_tool("update_reservation_passengers", "Update passengers", "reservation_id", "passengers")],
        messages=[{"role": "user", "content": "Change the passenger name on reservation 3RK2T9."}],
        domain_policy="Use schema keys.",
        domain="airline",
        facts_text=None,
        engine=_Engine(),
        sampling=SimpleNamespace(),
        config=ParallelCandidateRouterConfig(chunk_tools=1),
    )

    assert route.selected is not None
    assert route.selected.arguments["passengers"] == [
        {"first_name": "Anya", "last_name": "Garcia", "dob": "1992-11-12"}
    ]
