from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_TOOL_ROUTER_MAX_TOOLS = 12
DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT = 16
DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS = 6000
DEFAULT_TOOL_ROUTER_CONTEXT_CHARS = 5000
DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS = 240

_LATIN_TERM_RE = re.compile(r"[a-z0-9_]{2,}")
_CJK_SPAN_RE = re.compile("[\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff]{2,}")
_RESERVATION_ID_RE = re.compile(r"\b[A-Z0-9]{6}\b")
_RETAIL_ORDER_ID_RE = re.compile(r"#?[A-Z]\d{7,}\b", re.IGNORECASE)
_USER_ID_RE = re.compile(r"\b[a-z][a-z0-9]*_[a-z][a-z0-9]*_\d+\b", re.IGNORECASE)
_FLIGHT_NUMBER_RE = re.compile(r"\b[A-Z]{2,4}\d{2,4}\b")


@dataclass(frozen=True, slots=True)
class ToolRouterConfig:
    enabled: bool = True
    max_tools: int = DEFAULT_TOOL_ROUTER_MAX_TOOLS
    trigger_tool_count: int = DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT
    trigger_catalog_chars: int = DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS
    context_chars: int = DEFAULT_TOOL_ROUTER_CONTEXT_CHARS
    description_chars: int = DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS
    fallback_to_all_on_empty: bool = True
    enable_domain_hints: bool = True


@dataclass(frozen=True, slots=True)
class ToolRouteResult:
    selected_tools: list[Any]
    selected_names: tuple[str, ...]
    total_tool_count: int
    catalog_chars: int
    routed: bool
    reason: str
    lexical_names: tuple[str, ...] = ()
    heuristic_names: tuple[str, ...] = ()

    def trace_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mode": "lexical",
            "routed": self.routed,
            "reason": self.reason,
            "selected_names": list(self.selected_names),
            "total_tool_count": int(self.total_tool_count),
            "catalog_chars": int(self.catalog_chars),
        }
        if self.lexical_names:
            payload["lexical_names"] = list(self.lexical_names)
        if self.heuristic_names:
            payload["heuristic_names"] = list(self.heuristic_names)
        return payload


def route_tools(
    tools: Sequence[Any],
    messages: Sequence[Mapping[str, Any]],
    *,
    config: ToolRouterConfig | None = None,
    control_tool_names: Sequence[str] = (),
) -> ToolRouteResult:
    context = render_tool_routing_context(messages, max_chars=(config or ToolRouterConfig()).context_chars)
    return route_tools_for_context(tools, context, config=config, control_tool_names=control_tool_names)


def route_tools_for_context(
    tools: Sequence[Any],
    context: str,
    *,
    config: ToolRouterConfig | None = None,
    control_tool_names: Sequence[str] = (),
) -> ToolRouteResult:
    cfg = config or ToolRouterConfig()
    all_tools = list(tools)
    total_count = len(all_tools)
    catalog_chars = tool_catalog_chars(all_tools)
    control_names = _name_set(control_tool_names)

    if not cfg.enabled:
        return _route_result(
            all_tools,
            all_tools,
            total_tool_count=total_count,
            catalog_chars=catalog_chars,
            routed=False,
            reason="disabled",
        )

    should_route = (
        total_count > max(1, int(cfg.max_tools))
        or total_count >= max(1, int(cfg.trigger_tool_count))
        or catalog_chars >= max(1, int(cfg.trigger_catalog_chars))
    )
    if not should_route:
        return _route_result(
            all_tools,
            all_tools,
            total_tool_count=total_count,
            catalog_chars=catalog_chars,
            routed=False,
            reason="below_trigger",
        )

    heuristic_names = tuple(
        _heuristic_tool_names(
            all_tools,
            context,
            max_tools=max(1, int(cfg.max_tools)),
            control_names=control_names,
        )
        if cfg.enable_domain_hints
        else ()
    )
    lexical_names = tuple(
        _dedupe_names(
            [
                *heuristic_names,
                *_lexical_tool_names(
                    all_tools,
                    context,
                    max_tools=max(1, int(cfg.max_tools)),
                    control_names=control_names,
                    description_chars=max(40, int(cfg.description_chars)),
                ),
            ]
        )
    )
    selected = _tools_by_ranked_names(
        all_tools,
        lexical_names,
        max_tools=max(1, int(cfg.max_tools)),
        control_names=control_names,
        fallback_to_all_on_empty=cfg.fallback_to_all_on_empty,
    )
    return _route_result(
        all_tools,
        selected,
        total_tool_count=total_count,
        catalog_chars=catalog_chars,
        routed=True,
        reason="lexical" if lexical_names else "lexical_empty_full_fallback",
        lexical_names=lexical_names,
        heuristic_names=heuristic_names,
    )


def render_tool_routing_context(messages: Sequence[Mapping[str, Any]], *, max_chars: int) -> str:
    budget = max(1, int(max_chars))
    rows: list[str] = []
    for message in messages[-12:]:
        role = str(message.get("role") or "user").strip().lower() or "user"
        content = normalize_text(str(message.get("content") or ""))
        if not content:
            continue
        rows.append(f"{role}: {truncate_text(content, min(1600, budget))}")
    rendered = "\n\n".join(rows)
    if len(rendered) <= budget:
        return rendered
    return rendered[-budget:]


def tool_catalog_chars(tools: Sequence[Any]) -> int:
    return len(json.dumps([normalize_tool_schema(tool) for tool in tools], ensure_ascii=False, sort_keys=True))


def tool_name(tool: Any) -> str:
    schema = normalize_tool_schema(tool)
    return str(schema.get("name") or "").strip()


def normalize_tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "openai_schema", None)
    if isinstance(schema, Mapping):
        return _normalize_schema_mapping(schema)
    if isinstance(tool, Mapping):
        return _normalize_schema_mapping(tool)
    name = str(getattr(tool, "name", "") or "").strip()
    description = str(getattr(tool, "description", "") or tool)
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": {}}}


def summarize_tool(tool: Any, *, description_chars: int = DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS) -> dict[str, Any]:
    schema = normalize_tool_schema(tool)
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = parameters.get("properties")
    property_names = sorted(str(key) for key in properties.keys()) if isinstance(properties, Mapping) else []
    required = parameters.get("required")
    required_names = [str(item) for item in required] if isinstance(required, Sequence) and not isinstance(required, str) else []
    enum_hints: dict[str, list[Any]] = {}
    if isinstance(properties, Mapping):
        for key, value in properties.items():
            if (
                isinstance(value, Mapping)
                and isinstance(value.get("enum"), Sequence)
                and not isinstance(value.get("enum"), str)
            ):
                enum_hints[str(key)] = list(value.get("enum") or [])[:12]
    row: dict[str, Any] = {
        "name": str(schema.get("name") or "").strip(),
        "description": truncate_text(normalize_text(str(schema.get("description") or "")), description_chars),
        "required": required_names,
        "properties": property_names,
    }
    if enum_hints:
        row["enum_hints"] = enum_hints
    return row


def normalize_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def truncate_text(text: str, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    normalized = normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    if limit <= 3:
        return normalized[:limit]
    return normalized[: limit - 3] + "..."


def _normalize_schema_mapping(schema: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(schema)
    function_schema = raw.get("function")
    if isinstance(function_schema, Mapping):
        raw = dict(function_schema)
    parameters = raw.get("parameters", raw.get("arguments", {}))
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}}
    return {
        "name": str(raw.get("name") or "").strip(),
        "description": str(raw.get("description") or ""),
        "parameters": dict(parameters),
    }


def _lexical_tool_names(
    tools: Sequence[Any],
    context: str,
    *,
    max_tools: int,
    control_names: set[str],
    description_chars: int,
) -> list[str]:
    terms = _query_terms(context)
    scored: list[tuple[float, int, str]] = []
    for index, tool in enumerate(tools):
        name = tool_name(tool)
        if not name or name in control_names:
            continue
        text = json.dumps(summarize_tool(tool, description_chars=description_chars), ensure_ascii=False).lower()
        score = _score_text(text, terms)
        if score > 0.0:
            scored.append((score, index, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _score, _index, name in scored[: max(1, int(max_tools))]]


def _heuristic_tool_names(
    tools: Sequence[Any],
    context: str,
    *,
    max_tools: int,
    control_names: set[str],
) -> list[str]:
    available = {tool_name(tool) for tool in tools if tool_name(tool)}
    lowered = str(context or "").lower()
    names: list[str] = []

    def add(name: str) -> None:
        if name in available and name not in control_names and name not in names:
            names.append(name)

    has_retail_catalog = any(
        name in available
        for name in (
            "get_order_details",
            "exchange_delivered_order_items",
            "return_delivered_order_items",
            "list_all_product_types",
        )
    )
    if has_retail_catalog:
        has_order_ref = bool(_RETAIL_ORDER_ID_RE.search(str(context or ""))) or any(
            term in lowered
            for term in (
                "order",
                "order id",
                "order number",
                "delivered",
                "pending order",
                "exchange",
                "return",
                "refund",
                "cancel",
            )
        )
        has_product_ref = any(
            term in lowered
            for term in (
                "product",
                "item",
                "inventory",
                "available",
                "option",
                "store",
                "t-shirt",
                "shirt",
                "headphone",
                "keyboard",
                "thermostat",
                "watch",
                "cleaner",
            )
        )
        wants_exchange = any(term in lowered for term in ("exchange", "swap", "replace"))
        wants_return = any(term in lowered for term in ("return", "refund"))
        wants_cancel = any(term in lowered for term in ("cancel", "cancellation"))

        if has_order_ref:
            add("get_order_details")
        if has_product_ref or wants_exchange or wants_return:
            add("list_all_product_types")
            add("get_product_details")
            add("get_item_details")
        if any(term in lowered for term in ("user", "account", "email", "zip", "address")):
            add("get_user_details")
            add("find_user_id_by_email")
            add("find_user_id_by_name_zip")
        if wants_exchange:
            add("exchange_delivered_order_items")
            add("modify_pending_order_items")
        if wants_return:
            add("return_delivered_order_items")
        if wants_cancel:
            add("cancel_pending_order")
        if any(term in lowered for term in ("payment", "card", "credit")):
            add("modify_pending_order_payment")
        if "address" in lowered or "shipping" in lowered:
            add("modify_pending_order_address")
            add("modify_user_address")
        if any(term in lowered for term in ("human", "agent", "representative", "escalate")):
            add("transfer_to_human_agents")

    has_reservation_id = bool(_RESERVATION_ID_RE.search(str(context or ""))) or any(
        term in lowered
        for term in (
            "reservation_id",
            "reservation id",
            "reservation code",
            "booking id",
            "confirmation",
        )
    )
    has_user_id = bool(_USER_ID_RE.search(str(context or ""))) or "user_id" in lowered or "user id" in lowered
    has_flight_number = bool(_FLIGHT_NUMBER_RE.search(str(context or ""))) or "flight_number" in lowered

    if has_reservation_id:
        add("get_reservation_details")
    if has_user_id or any(term in lowered for term in ("profile", "account", "gift card", "certificate")):
        add("get_user_details")
    if any(term in lowered for term in ("cancel", "cancellation", "refund")):
        add("cancel_reservation")
    if any(term in lowered for term in ("change", "reschedule", "move", "upgrade", "cabin", "business class")):
        add("update_reservation_flights")
        add("search_direct_flight")
        add("search_onestop_flight")
    if any(term in lowered for term in ("book", "new reservation", "reserve", "ticket")):
        add("book_reservation")
        add("search_direct_flight")
        add("search_onestop_flight")
    if any(term in lowered for term in ("baggage", "baggages", "bag ", "bags", "suitcase", "luggage")):
        add("update_reservation_baggages")
    if any(term in lowered for term in ("passenger", "passengers", "date of birth", "dob")):
        add("update_reservation_passengers")
    if any(term in lowered for term in ("compensation", "voucher", "certificate", "gesture")):
        add("send_certificate")
    if has_flight_number or any(term in lowered for term in ("delayed", "delay", "flight status", "cancelled flight")):
        add("get_flight_status")
    if any(term in lowered for term in ("airport", "iata", "city code")):
        add("list_all_airports")

    has_complexfuncbench_travel_catalog = any(
        name in available
        for name in (
            "Search_Attraction_Location",
            "Search_Attractions",
            "Search_Hotel_Destination",
            "Search_Hotels",
            "Search_Car_Location",
            "Search_Car_Rentals",
            "Taxi_Search_Location",
            "Search_Taxi",
            "Get_Popular_Attraction_Near_By",
        )
    )
    if has_complexfuncbench_travel_catalog:
        if any(term in lowered for term in ("hotel", "accommodation", "stay", "room", "check-in", "check in")):
            add("Search_Hotel_Destination")
            add("Search_Hotels")
        if any(
            term in lowered
            for term in (
                "attraction",
                "attractions",
                "activity",
                "activities",
                "things to do",
                "tourist",
                "sightseeing",
                "visit",
                "museum",
                "landmark",
                "tour",
                "top-rated",
                "popular",
                "recommend",
                "fun places",
                "nearby",
                "near by",
            )
        ):
            add("Search_Attraction_Location")
            add("Search_Attractions")
            add("Get_Popular_Attraction_Near_By")
        if "search_hotels_by_coordinates" in lowered or "hotels_by_coordinates" in lowered:
            add("Get_Popular_Attraction_Near_By")
        if any(
            term in lowered
            for term in (
                "taxi",
                "cab",
                "pick me up",
                "pickup",
                "take me to",
                "airport transfer",
            )
        ):
            add("Taxi_Search_Location")
            add("Search_Taxi")
        if any(term in lowered for term in ("car", "rental", "rent a car", "pick up", "pickup", "drop off")):
            add("Search_Car_Location")
            add("Search_Car_Rentals")
        if any(term in lowered for term in ("flight", "airport", "airline", "depart", "arrival")):
            add("Search_Flights")
            add("Search_Flights_Multi_Stops")
            add("Get_Min_Price")
            add("Get_Min_Price_Multi_Stops")

    return names[: max(1, int(max_tools))]


def _tools_by_ranked_names(
    tools: Sequence[Any],
    ranked_names: Sequence[str],
    *,
    max_tools: int,
    control_names: set[str],
    fallback_to_all_on_empty: bool,
) -> list[Any]:
    by_name = {tool_name(tool): tool for tool in tools if tool_name(tool)}
    selected_names: list[str] = []
    for name in ranked_names:
        normalized = str(name or "").strip()
        if not normalized or normalized in control_names or normalized not in by_name:
            continue
        if normalized not in selected_names:
            selected_names.append(normalized)
        if len(selected_names) >= max(1, int(max_tools)):
            break
    if not selected_names and fallback_to_all_on_empty:
        return list(tools)
    selected = [by_name[name] for name in selected_names]
    selected.extend(tool for tool in tools if tool_name(tool) in control_names and tool not in selected)
    return selected


def _route_result(
    all_tools: Sequence[Any],
    selected_tools: Sequence[Any],
    *,
    total_tool_count: int,
    catalog_chars: int,
    routed: bool,
    reason: str,
    lexical_names: Sequence[str] = (),
    heuristic_names: Sequence[str] = (),
) -> ToolRouteResult:
    selected = list(selected_tools)
    if not selected and all_tools:
        selected = list(all_tools)
    return ToolRouteResult(
        selected_tools=selected,
        selected_names=tuple(tool_name(tool) for tool in selected if tool_name(tool)),
        total_tool_count=int(total_tool_count),
        catalog_chars=int(catalog_chars),
        routed=bool(routed),
        reason=reason,
        lexical_names=tuple(lexical_names),
        heuristic_names=tuple(heuristic_names),
    )


def _query_terms(text: str) -> tuple[str, ...]:
    lowered = str(text or "").lower()
    terms = set(_LATIN_TERM_RE.findall(lowered))
    for span in _CJK_SPAN_RE.findall(str(text or "")):
        if len(span) <= 8:
            terms.add(span)
        for size in (2, 3, 4):
            if len(span) < size:
                continue
            for index in range(0, len(span) - size + 1):
                terms.add(span[index : index + size])
    return tuple(sorted(terms, key=lambda item: (-len(item), item)))


def _score_text(text: str, terms: Sequence[str]) -> float:
    lowered = str(text or "").lower()
    score = 0.0
    for term in terms:
        hits = lowered.count(term.lower())
        if hits:
            score += min(hits, 3) * max(1.0, len(term) / 2.0)
    return score


def _name_set(names: Sequence[str]) -> set[str]:
    return {str(name).strip() for name in names if str(name).strip()}


def _dedupe_names(names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        normalized = str(name or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


__all__ = [
    "DEFAULT_TOOL_ROUTER_CONTEXT_CHARS",
    "DEFAULT_TOOL_ROUTER_DESCRIPTION_CHARS",
    "DEFAULT_TOOL_ROUTER_MAX_TOOLS",
    "DEFAULT_TOOL_ROUTER_TRIGGER_CATALOG_CHARS",
    "DEFAULT_TOOL_ROUTER_TRIGGER_TOOL_COUNT",
    "ToolRouteResult",
    "ToolRouterConfig",
    "normalize_text",
    "normalize_tool_schema",
    "render_tool_routing_context",
    "route_tools",
    "route_tools_for_context",
    "summarize_tool",
    "tool_catalog_chars",
    "tool_name",
    "truncate_text",
]
