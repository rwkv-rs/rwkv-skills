from __future__ import annotations

import ast
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .context_budget import (
    DEFAULT_HISTORY_MAX_CHARS,
    DEFAULT_TOOL_ERROR_MAX_CHARS,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    DEFAULT_TOOL_SCHEMA_MAX_CHARS,
    normalize_rwkv_text,
    trim_message_history,
    truncate_text,
)
from .tau_bench import TauDecision, TauToolCall

BFCL_V3_MAX_TOOL_SCHEMA_CHARS = DEFAULT_TOOL_SCHEMA_MAX_CHARS
BFCL_V3_MAX_RESULT_CHARS = DEFAULT_TOOL_RESULT_MAX_CHARS
BFCL_V3_MAX_ERROR_CHARS = DEFAULT_TOOL_ERROR_MAX_CHARS
BFCL_V3_MAX_HISTORY_CHARS = DEFAULT_HISTORY_MAX_CHARS
BFCL_V3_MAX_STATE_CHARS = 4000
BFCL_V3_MAX_COT_CHARS = 4000
BFCL_V3_MAX_HANDOFF_CHARS = 600
BFCL_V3_MAX_COT_SUMMARY_CHARS = 600
BFCL_ADDITIONAL_FUNCTION_PROMPT = "I have updated some more functions you can choose from. What about now?"
BFCL_COT_STOP_SUFFIX = "</think>"
BFCL_DECISION_STOP_SUFFIXES = (
    "\n```",
    "```",
    "\nUser:",
    "\nSystem:",
    "\nAssistant:",
)
BFCL_ROUTER_LABELS = ("TOOL", "ASK", "HANDOFF")
_BFCL_FORBIDDEN_OUTPUT_PATTERNS = (
    "tool_calls",
    "tool_call_id",
    "**Tool Call:**",
    "### Tool Output",
    "<assistant>",
    "</assistant>",
    "<user_input>",
    "</user_input>",
)

_BFCL_OFFICIAL_ROOT_ENV_VARS = ("RWKV_BFCL_OFFICIAL_ROOT", "BFCL_OFFICIAL_ROOT")
_BFCL_POSSIBLE_ANSWER_ROOT_ENV_VARS = (
    "RWKV_BFCL_POSSIBLE_ANSWER_ROOT",
    "BFCL_POSSIBLE_ANSWER_ROOT",
)
_BFCL_FUNC_DOC_ROOT_ENV_VARS = ("RWKV_BFCL_FUNC_DOC_ROOT", "BFCL_FUNC_DOC_ROOT")
_BFCL_MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


@dataclass(frozen=True, slots=True)
class BfclToolCallExpectation:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    state_updates: dict[str, Any] = field(default_factory=dict)
    optional: bool = False


@dataclass(frozen=True, slots=True)
class BfclTurn:
    messages: tuple[dict[str, str], ...] = ()
    ground_truth: tuple[str, ...] = ()
    tool_additions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class BfclTaskRecord:
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...] = ()
    turns: tuple[BfclTurn, ...] = ()
    involved_classes: tuple[str, ...] = ()
    expected_tool_calls: tuple[BfclToolCallExpectation, ...] = ()
    expected_final_answers: tuple[str, ...] = ()
    expected_state: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BfclRuntimeState:
    current_state: dict[str, Any] = field(default_factory=dict)
    next_tool_index: int = 0
    current_turn_index: int = 0
    completed_required_steps: int = 0
    unexpected_tool_calls: int = 0
    decoded_turn_outputs: list[list[list[str]]] = field(default_factory=list)
    official_model_name: str | None = None
    executed_tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BfclToolExecutionResult:
    success: bool
    result: Any = None
    error: str | None = None
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    matched_expectation: bool = False


@dataclass(frozen=True, slots=True)
class BfclEvaluation:
    reward: float
    is_passed: bool
    fail_reason: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _OfficialSupportAssets:
    tools: tuple[dict[str, Any], ...] = ()
    ground_truth_turns: tuple[tuple[str, ...], ...] = ()
    holdout_tools_by_turn: dict[int, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    possible_answer_path: str | None = None
    func_doc_root: str | None = None
    official_root: str | None = None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _OfficialRuntime:
    execute_multi_turn_func_call: Any
    multi_turn_checker: Any
    multi_turn_irrelevance_checker: Any | None = None


def load_bfcl_v3_rows_from_source(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    items = _read_source_items(source)
    return [normalize_bfcl_v3_source_row(item, index=index, source_path=source) for index, item in enumerate(items)]


def normalize_bfcl_v3_source_row(
    item: Any,
    *,
    index: int,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"BFCL V3 source row #{index} is not an object")

    if _looks_like_official_bfcl_v3_row(item):
        return _normalize_official_bfcl_v3_row(item, index=index, source_path=source_path)

    task_id = str(item.get("task_id") or item.get("id") or f"bfcl_v3_{index:04d}")
    instruction = _first_nonempty_str(
        item.get("instruction"),
        item.get("question"),
        item.get("prompt"),
        item.get("user_request"),
        item.get("user_input"),
    )
    if not instruction:
        raise ValueError(f"BFCL V3 source row {task_id!r} is missing instruction/question/prompt")

    tools = tuple(_normalize_tool_schema(tool) for tool in _coerce_list(item.get("tools") or item.get("functions")))
    raw_expectations = (
        item.get("expected_tool_calls")
        or item.get("tool_outcomes")
        or item.get("tool_calls")
        or item.get("steps")
        or _ground_truth_tool_calls(item.get("ground_truth"))
    )
    expected_tool_calls = tuple(_normalize_expectation(step) for step in _coerce_list(raw_expectations))
    expected_final_answers = _normalize_answers(
        item.get("expected_final_answers")
        or item.get("expected_final_answer")
        or item.get("expected_answer")
        or item.get("answer")
        or _ground_truth_answer(item.get("ground_truth"))
    )
    expected_state = _normalize_state_mapping(
        item.get("expected_state")
        or item.get("final_state")
        or item.get("state")
        or _ground_truth_state(item.get("ground_truth"))
    )
    initial_state = _normalize_state_mapping(item.get("initial_state") or item.get("state_before"))

    metadata = {
        key: value
        for key, value in item.items()
        if key
        not in {
            "task_id",
            "id",
            "instruction",
            "question",
            "prompt",
            "user_request",
            "user_input",
            "tools",
            "functions",
            "expected_tool_calls",
            "tool_outcomes",
            "tool_calls",
            "steps",
            "expected_final_answers",
            "expected_final_answer",
            "expected_answer",
            "answer",
            "expected_state",
            "final_state",
            "state",
            "initial_state",
            "state_before",
            "ground_truth",
        }
    }
    if source_path is not None:
        metadata.setdefault("source_path", str(Path(source_path)))
        metadata["manifest_path"] = str(Path(source_path))

    return {
        "task_id": task_id,
        "instruction": instruction,
        "tools": list(tools),
        "expected_tool_calls": [_expectation_to_dict(step) for step in expected_tool_calls],
        "expected_final_answers": list(expected_final_answers),
        "expected_state": expected_state,
        "initial_state": initial_state,
        "metadata": metadata,
    }


def load_bfcl_v3_manifest_records(path: str | Path) -> list[BfclTaskRecord]:
    rows: list[BfclTaskRecord] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            normalized = normalize_bfcl_v3_source_row(payload, index=index, source_path=target)
            rows.append(
                BfclTaskRecord(
                    task_id=str(normalized["task_id"]),
                    instruction=str(normalized["instruction"]),
                    tools=tuple(dict(item) for item in normalized.get("tools", [])),
                    turns=tuple(_normalize_turn(item) for item in normalized.get("turns", [])),
                    involved_classes=tuple(str(item).strip() for item in normalized.get("involved_classes", []) if str(item).strip()),
                    expected_tool_calls=tuple(
                        _normalize_expectation(item) for item in normalized.get("expected_tool_calls", [])
                    ),
                    expected_final_answers=tuple(str(item) for item in normalized.get("expected_final_answers", [])),
                    expected_state=_normalize_state_mapping(normalized.get("expected_state")),
                    initial_state=_normalize_state_mapping(normalized.get("initial_state")),
                    metadata=dict(normalized.get("metadata", {}) or {}),
                )
            )
    return rows


def normalize_bfcl_rwkv_text(text: str) -> str:
    return normalize_rwkv_text(text)


def build_bfcl_system_block(system_prompt: str) -> str:
    return f"System: {normalize_bfcl_rwkv_text(system_prompt)}"


def build_bfcl_assistant_json_prefix() -> str:
    return "Assistant: ```json\n"


def build_bfcl_assistant_json_block(json_text: str) -> str:
    return f"{build_bfcl_assistant_json_prefix()}{normalize_bfcl_decision_output(json_text)}\n```"


def build_bfcl_user_block(
    user_request: str,
    *,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_tool_result: Mapping[str, Any] | None = None,
) -> str:
    parts = [f"Request:\n{normalize_bfcl_rwkv_text(user_request)}"]
    if previous_tool_result is not None:
        parts.append(f"Function output:\n{_render_bfcl_json(previous_tool_result)}")
    if current_state_snapshot:
        parts.append(f"Current structured state snapshot:\n{render_bfcl_state(current_state_snapshot)}")
    user_body = normalize_bfcl_rwkv_text("\n".join(parts))
    return f"User: {user_body}"


def extract_bfcl_cot_hidden_summary(
    text: str,
    *,
    max_chars: int = BFCL_V3_MAX_COT_SUMMARY_CHARS,
) -> str:
    normalized = normalize_bfcl_rwkv_text(text)
    match = re.match(r"(?s)^<think>\s*(.*?)\s*</think>\s*(.*)$", normalized)
    if match:
        summary = normalize_bfcl_rwkv_text(match.group(1) or match.group(2))
    else:
        summary = normalized
    if not summary:
        return ""
    return truncate_text(summary, max_chars)


def render_bfcl_state_delta(
    current_state: Mapping[str, Any],
    *,
    previous_state: Mapping[str, Any] | None = None,
) -> str:
    if previous_state is None:
        return render_bfcl_state(current_state)

    delta: dict[str, Any] = {}
    all_keys = set(current_state.keys()) | set(previous_state.keys())
    for key in sorted(all_keys):
        if key not in current_state:
            delta[key] = {"_removed": True}
            continue
        if key not in previous_state or previous_state[key] != current_state[key]:
            delta[key] = current_state[key]
    if not delta:
        return ""
    return render_bfcl_state(delta)


def render_bfcl_recent_tool_window(
    executed_tool_calls: Sequence[Mapping[str, Any]],
    *,
    max_calls: int = 1,
) -> str:
    if not executed_tool_calls:
        return ""

    window: list[dict[str, Any]] = []
    for item in list(executed_tool_calls)[-max(1, int(max_calls)) :]:
        arguments = item.get("arguments")
        rendered: dict[str, Any] = {
            "name": str(item.get("name") or "").strip(),
            "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
            "success": bool(item.get("success", False)),
        }
        if "result" in item:
            rendered["result"] = _bounded_bfcl_tool_payload_value(
                item.get("result"),
                max_chars=BFCL_V3_MAX_RESULT_CHARS,
            )
        if item.get("error"):
            rendered["error"] = truncate_text(str(item.get("error") or ""), BFCL_V3_MAX_ERROR_CHARS)
        window.append(rendered)
    if len(window) == 1:
        return _render_bfcl_json(window[0])
    return _render_bfcl_json(window)


def _build_bfcl_action_user_block(
    user_request: str,
    *,
    cot_hidden_summary: str | None = None,
    recent_tool_window: Sequence[Mapping[str, Any]] | None = None,
    previous_tool_result: Mapping[str, Any] | None = None,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_state_snapshot: Mapping[str, Any] | None = None,
) -> str:
    parts = [f"Request:\n{normalize_bfcl_rwkv_text(user_request)}"]
    if cot_hidden_summary:
        parts.append(f"Private reasoning summary:\n{normalize_bfcl_rwkv_text(cot_hidden_summary)}")
    if recent_tool_window:
        rendered_window = render_bfcl_recent_tool_window(recent_tool_window)
        if rendered_window:
            parts.append(f"Recent tool call window:\n{rendered_window}")
    if previous_tool_result is not None:
        parts.append(f"Function output:\n{_render_bfcl_json(previous_tool_result)}")
    if current_state_snapshot is not None or previous_state_snapshot is not None:
        rendered_state = render_bfcl_state_delta(
            current_state_snapshot or {},
            previous_state=previous_state_snapshot,
        )
        if rendered_state:
            label = (
                "Current structured state summary"
                if previous_state_snapshot is None
                else "Current structured state delta"
            )
            parts.append(f"{label}:\n{rendered_state}")
    user_body = normalize_bfcl_rwkv_text("\n".join(parts))
    return f"User: {user_body}"


def build_bfcl_tool_result_message(
    previous_tool_result: Mapping[str, Any],
    *,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_state_snapshot: Mapping[str, Any] | None = None,
) -> str:
    parts = [f"Function output:\n{_render_bfcl_json(previous_tool_result)}"]
    if current_state_snapshot is not None or previous_state_snapshot is not None:
        rendered_state = render_bfcl_state_delta(
            current_state_snapshot or {},
            previous_state=previous_state_snapshot,
        )
        if rendered_state:
            label = (
                "Current structured state summary"
                if previous_state_snapshot is None
                else "Current structured state delta"
            )
            parts.append(f"{label}:\n{rendered_state}")
    user_body = normalize_bfcl_rwkv_text("\n".join(parts))
    return f"User: {user_body}"


def build_bfcl_rwkv_prompt(
    system_prompt: str,
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    history_max_chars: int,
) -> str:
    bounded_messages = trim_message_history(prompt_messages, max_chars=history_max_chars)
    parts = [build_bfcl_system_block(system_prompt)]
    for message in bounded_messages:
        role = str(message.get("role") or "").strip().lower()
        content = normalize_bfcl_rwkv_text(str(message.get("content") or ""))
        if not content:
            continue
        if role == "assistant":
            parts.append(build_bfcl_assistant_json_block(content))
            continue
        if content.startswith("User: "):
            parts.append(content)
            continue
        parts.append(build_bfcl_user_block(content))
    parts.append(build_bfcl_assistant_json_prefix())
    return "\n".join(parts)


def build_bfcl_cot_prompt(
    system_prompt: str,
    *,
    user_request: str,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_tool_result: Mapping[str, Any] | None = None,
) -> str:
    return "\n".join(
        [
            build_bfcl_system_block(system_prompt),
            build_bfcl_user_block(
                user_request,
                current_state_snapshot=current_state_snapshot,
                previous_tool_result=previous_tool_result,
            ),
            "Assistant: <think>\n",
        ]
    )


def build_bfcl_router_prompt(
    system_prompt: str,
    *,
    user_request: str,
    cot_hidden_summary: str | None = None,
    recent_tool_window: Sequence[Mapping[str, Any]] | None = None,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_state_snapshot: Mapping[str, Any] | None = None,
    previous_tool_result: Mapping[str, Any] | None = None,
) -> str:
    router_system_prompt = normalize_bfcl_rwkv_text(
        "\n".join(
            [
                system_prompt,
                "Private reasoning is already complete.",
                "Choose the next action type only.",
                "Output exactly one label: TOOL, ASK, or HANDOFF.",
                "TOOL means the next assistant turn should call exactly one tool.",
                "ASK means required information is missing and the assistant should ask a brief clarification question.",
                "HANDOFF means no tool should be called and the assistant should respond in plain language.",
                "Do not output anything except one of those labels.",
            ]
        )
    )
    return "\n".join(
        [
            build_bfcl_system_block(router_system_prompt),
            _build_bfcl_action_user_block(
                user_request,
                cot_hidden_summary=cot_hidden_summary,
                recent_tool_window=recent_tool_window,
                current_state_snapshot=current_state_snapshot,
                previous_state_snapshot=previous_state_snapshot,
                previous_tool_result=previous_tool_result,
            ),
            "Assistant:",
        ]
    )


def build_bfcl_tool_prompt(
    system_prompt: str,
    *,
    user_request: str,
    cot_hidden_summary: str | None = None,
    recent_tool_window: Sequence[Mapping[str, Any]] | None = None,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_state_snapshot: Mapping[str, Any] | None = None,
    previous_tool_result: Mapping[str, Any] | None = None,
) -> str:
    tool_system_prompt = normalize_bfcl_rwkv_text(
        "\n".join(
            [
                system_prompt,
                "Private reasoning is already complete.",
                "The router selected TOOL.",
                "Return only a JSON function call.",
                'The JSON shape is {"name":"tool_name","arguments":{...}}.',
                "Do not output <think> tags, markdown, or natural-language commentary.",
            ]
        )
    )
    return "\n".join(
        [
            build_bfcl_system_block(tool_system_prompt),
            _build_bfcl_action_user_block(
                user_request,
                cot_hidden_summary=cot_hidden_summary,
                recent_tool_window=recent_tool_window,
                current_state_snapshot=current_state_snapshot,
                previous_state_snapshot=previous_state_snapshot,
                previous_tool_result=previous_tool_result,
            ),
            build_bfcl_assistant_json_prefix(),
        ]
    )


def build_bfcl_ask_prompt(
    system_prompt: str,
    *,
    user_request: str,
    cot_hidden_summary: str | None = None,
    recent_tool_window: Sequence[Mapping[str, Any]] | None = None,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_state_snapshot: Mapping[str, Any] | None = None,
    previous_tool_result: Mapping[str, Any] | None = None,
) -> str:
    ask_system_prompt = normalize_bfcl_rwkv_text(
        "\n".join(
            [
                system_prompt,
                "Private reasoning is already complete.",
                "The router selected ASK.",
                "Return only a JSON function call.",
                'Use {"name":"ask_user","arguments":{"question":"..."}} with one brief clarification question.',
                "Do not call any other tool.",
                "Do not output <think> tags, markdown, or natural-language commentary.",
            ]
        )
    )
    return "\n".join(
        [
            build_bfcl_system_block(ask_system_prompt),
            _build_bfcl_action_user_block(
                user_request,
                cot_hidden_summary=cot_hidden_summary,
                recent_tool_window=recent_tool_window,
                current_state_snapshot=current_state_snapshot,
                previous_state_snapshot=previous_state_snapshot,
                previous_tool_result=previous_tool_result,
            ),
            build_bfcl_assistant_json_prefix(),
        ]
    )


def build_bfcl_handoff_prompt(
    system_prompt: str,
    *,
    user_request: str,
    cot_hidden_summary: str | None = None,
    recent_tool_window: Sequence[Mapping[str, Any]] | None = None,
    current_state_snapshot: Mapping[str, Any] | None = None,
    previous_state_snapshot: Mapping[str, Any] | None = None,
    previous_tool_result: Mapping[str, Any] | None = None,
) -> str:
    handoff_system_prompt = normalize_bfcl_rwkv_text(
        "\n".join(
            [
                system_prompt,
                "Private reasoning is already complete.",
                "The router selected HANDOFF.",
                "Return only a JSON function call.",
                'Use {"name":"final_answer","arguments":{"answer":"..."}} with the short direct response.',
                "Do not call any other tool.",
                "Do not output <think> tags, markdown, or natural-language commentary.",
            ]
        )
    )
    return "\n".join(
        [
            build_bfcl_system_block(handoff_system_prompt),
            _build_bfcl_action_user_block(
                user_request,
                cot_hidden_summary=cot_hidden_summary,
                recent_tool_window=recent_tool_window,
                current_state_snapshot=current_state_snapshot,
                previous_state_snapshot=previous_state_snapshot,
                previous_tool_result=previous_tool_result,
            ),
            build_bfcl_assistant_json_prefix(),
        ]
    )


def build_bfcl_system_prompt(tools: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Tools:",
        render_bfcl_tool_catalog(_bfcl_tools_with_control_functions(tools)),
        "Return only a JSON function call.",
        'The JSON shape is {"name":"tool_name","arguments":{...}}.',
        "Use exactly one listed tool name.",
        "Use ask_user only when required information is missing.",
        "Use final_answer only when no environment tool should be called for this turn.",
        "Do not invent tool names, arguments, tool results, or state transitions.",
    ]
    return normalize_bfcl_rwkv_text("\n".join(lines))


def start_bfcl_runtime(record: BfclTaskRecord) -> BfclRuntimeState:
    return BfclRuntimeState(current_state=dict(record.initial_state))


def has_bfcl_official_turns(record: BfclTaskRecord) -> bool:
    return bool(record.turns)


def render_bfcl_assistant_tool_message(tool_call: TauToolCall) -> str:
    payload = {
        "name": tool_call.name,
        "arguments": dict(tool_call.arguments),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_bfcl_tool_result_payload(
    tool_call: TauToolCall,
    *,
    ok: bool,
    output: Any = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": bool(ok),
        "tool": tool_call.name,
    }
    if ok:
        payload["result"] = _bounded_bfcl_tool_payload_value(output, max_chars=BFCL_V3_MAX_RESULT_CHARS)
    else:
        payload["error"] = truncate_text(str(error or "unknown tool error"), BFCL_V3_MAX_ERROR_CHARS)
    return payload


def parse_bfcl_assistant_output(response: str) -> TauDecision:
    normalized = normalize_bfcl_decision_output(response)
    if not normalized:
        raise ValueError("model returned empty response")

    lowered = normalized.lower()
    for pattern in _BFCL_FORBIDDEN_OUTPUT_PATTERNS:
        if pattern.lower() in lowered:
            raise ValueError(f"forbidden BFCL output pattern detected: {pattern}")
    if "<think>" in lowered or "</think>" in lowered:
        raise ValueError("decision output must not contain <think> tags")
    if not (normalized.startswith("{") and normalized.endswith("}")):
        raise ValueError("BFCL decision output must be a JSON function call object")

    payload = json.loads(normalized)
    if not isinstance(payload, Mapping):
        raise ValueError("BFCL tool call payload must be a JSON object")
    allowed_keys = {"name", "arguments"}
    if set(payload.keys()) != allowed_keys:
        raise ValueError("BFCL tool call JSON must contain exactly name and arguments")

    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("BFCL tool call missing name")
    arguments = payload.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("BFCL tool call arguments must be a JSON object")
    if name == "final_answer":
        final_answer = normalize_bfcl_rwkv_text(str(arguments.get("answer") or ""))
        if len(final_answer) > BFCL_V3_MAX_HANDOFF_CHARS:
            raise ValueError("BFCL final_answer exceeded max length")
        return TauDecision(is_tool_call=False, final_answer=final_answer)
    if name == "ask_user":
        question = normalize_bfcl_rwkv_text(str(arguments.get("question") or ""))
        if len(question) > BFCL_V3_MAX_HANDOFF_CHARS:
            raise ValueError("BFCL ask_user question exceeded max length")
        return TauDecision(is_tool_call=False, final_answer=question)
    return TauDecision(is_tool_call=True, tool_call=TauToolCall(name=name, arguments=dict(arguments), requestor="assistant"))


def parse_bfcl_router_output(response: str) -> str:
    normalized = normalize_bfcl_rwkv_text(response)
    if not normalized:
        raise ValueError("router returned empty response")
    if "<think>" in normalized.lower() or "</think>" in normalized.lower():
        raise ValueError("router output must not contain <think> tags")
    label = normalized.upper()
    if label not in BFCL_ROUTER_LABELS:
        raise ValueError(f"router output must be one of {', '.join(BFCL_ROUTER_LABELS)}")
    return label


def interpret_bfcl_assistant_output(
    response: str,
    *,
    tools: Sequence[Mapping[str, Any]],
) -> TauDecision:
    decision = parse_bfcl_assistant_output(response)
    if decision.is_tool_call and decision.tool_call is not None:
        tool = _lookup_bfcl_tool(tools, decision.tool_call.name)
        if tool is None:
            raise ValueError(f"unknown BFCL tool name: {decision.tool_call.name}")
        _validate_bfcl_tool_arguments(decision.tool_call, tool)
    return decision


def decode_bfcl_exec_response(
    response: str,
    *,
    tools: Sequence[Mapping[str, Any]],
) -> tuple[list[TauToolCall], str]:
    decision = interpret_bfcl_assistant_output(response, tools=tools)
    if decision.is_tool_call and decision.tool_call is not None:
        return [decision.tool_call], ""
    return [], decision.final_answer.strip()


def render_bfcl_official_call(tool_call: TauToolCall) -> str:
    arguments = ", ".join(
        f"{key}={_render_bfcl_python_literal(value)}" for key, value in tool_call.arguments.items()
    )
    return f"{tool_call.name}({arguments})" if arguments else f"{tool_call.name}()"


def execute_bfcl_official_tool_call(
    record: BfclTaskRecord,
    runtime_state: BfclRuntimeState,
    tool_call: TauToolCall,
) -> BfclToolExecutionResult:
    official_root = _resolve_official_root_for_record(record)
    if official_root is None:
        raise ValueError(
            "BFCL official backend code not found; set RWKV_BFCL_OFFICIAL_ROOT or point support dirs into an official BFCL checkout"
        )

    runtime = _load_bfcl_official_runtime(str(official_root))
    model_name = runtime_state.official_model_name or "rwkv_bfcl"
    call_string = render_bfcl_official_call(tool_call)
    execution_results, involved_instances = runtime.execute_multi_turn_func_call(
        func_call_list=[call_string],
        initial_config=dict(record.initial_state),
        involved_classes=list(record.involved_classes),
        model_name=model_name,
        test_entry_id=record.task_id,
        long_context=_is_official_long_context_task(record),
        is_evaL_run=False,
    )
    raw_result = execution_results[0] if execution_results else ""
    ok, output, error = _decode_official_execution_result(raw_result)
    runtime_state.current_state = _snapshot_official_instances(involved_instances)
    runtime_state.executed_tool_calls.append(
        {
            "name": tool_call.name,
            "arguments": dict(tool_call.arguments),
            "raw_result": raw_result,
            "success": ok,
            "error": error,
        }
    )
    return BfclToolExecutionResult(
        success=ok,
        result=output,
        error=error,
        state_snapshot=dict(runtime_state.current_state),
        matched_expectation=not str(raw_result).startswith("Error during execution:"),
    )


def apply_bfcl_tool_call(
    record: BfclTaskRecord,
    runtime_state: BfclRuntimeState,
    tool_call: TauToolCall,
) -> BfclToolExecutionResult:
    resolved = _resolve_expected_call(record, runtime_state, tool_call)
    if resolved is None:
        runtime_state.unexpected_tool_calls += 1
        runtime_state.executed_tool_calls.append(
            {
                "name": tool_call.name,
                "arguments": dict(tool_call.arguments),
                "success": False,
                "error": "unexpected extra tool call",
            }
        )
        return BfclToolExecutionResult(
            success=False,
            error="unexpected extra tool call",
            state_snapshot=dict(runtime_state.current_state),
            matched_expectation=False,
        )

    match_index, expectation, mismatch = resolved
    if expectation is None:
        runtime_state.unexpected_tool_calls += 1
        error = mismatch or "unexpected tool call"
        runtime_state.executed_tool_calls.append(
            {
                "name": tool_call.name,
                "arguments": dict(tool_call.arguments),
                "success": False,
                "error": error,
            }
        )
        return BfclToolExecutionResult(
            success=False,
            error=error,
            state_snapshot=dict(runtime_state.current_state),
            matched_expectation=False,
        )

    runtime_state.next_tool_index = match_index + 1
    if not expectation.optional:
        runtime_state.completed_required_steps += 1
    runtime_state.current_state.update(expectation.state_updates)
    execution = {
        "name": tool_call.name,
        "arguments": dict(tool_call.arguments),
        "success": expectation.error is None,
        "result": expectation.result,
        "error": expectation.error,
        "state_snapshot": dict(runtime_state.current_state),
        "matched_expectation": True,
    }
    runtime_state.executed_tool_calls.append(execution)
    return BfclToolExecutionResult(
        success=expectation.error is None,
        result=expectation.result,
        error=expectation.error,
        state_snapshot=dict(runtime_state.current_state),
        matched_expectation=True,
    )


def evaluate_bfcl_v3_episode(
    record: BfclTaskRecord,
    runtime_state: BfclRuntimeState,
    final_answer: str,
    *,
    termination_reason: str,
    error: str | None = None,
) -> BfclEvaluation:
    if has_bfcl_official_turns(record):
        return _evaluate_bfcl_v3_official_episode(
            record,
            runtime_state,
            final_answer,
            termination_reason=termination_reason,
            error=error,
        )

    dataset_issue = _dataset_validation_issue(record)
    required_step_count = sum(1 for step in record.expected_tool_calls if not step.optional)
    steps_ok = (
        runtime_state.completed_required_steps >= required_step_count
        and runtime_state.unexpected_tool_calls == 0
    )
    state_ok = _state_matches(runtime_state.current_state, record.expected_state)
    answer_ok = (
        True
        if not record.expected_final_answers
        else _normalize_answer_text(final_answer) in {_normalize_answer_text(item) for item in record.expected_final_answers}
    )
    dataset_ok = dataset_issue is None

    if dataset_issue is not None:
        failure_bits: list[str] = [dataset_issue]
        if error:
            failure_bits.append(str(error))
        if termination_reason != "agent_stop":
            failure_bits.append(f"termination_reason={termination_reason}")
        return BfclEvaluation(
            reward=0.0,
            is_passed=False,
            fail_reason="; ".join(failure_bits),
            details={
                "dataset_ok": False,
                "dataset_issue": dataset_issue,
                "tool_sequence_ok": steps_ok,
                "state_ok": state_ok,
                "answer_ok": answer_ok,
                "completed_required_steps": runtime_state.completed_required_steps,
                "required_step_count": required_step_count,
                "unexpected_tool_calls": runtime_state.unexpected_tool_calls,
                "current_state": dict(runtime_state.current_state),
                "expected_state": dict(record.expected_state),
            },
        )

    criteria: list[tuple[str, bool]] = []
    if record.expected_tool_calls:
        criteria.append(("tool_sequence_ok", steps_ok))
    if record.expected_state:
        criteria.append(("state_ok", state_ok))
    if record.expected_final_answers:
        criteria.append(("answer_ok", answer_ok))
    if not criteria:
        criteria.append(("no_unexpected_tool_calls", runtime_state.unexpected_tool_calls == 0))

    reward = sum(1.0 for _name, ok in criteria if ok) / len(criteria)
    is_passed = all(ok for _name, ok in criteria) and not error and termination_reason == "agent_stop"

    failure_bits: list[str] = []
    if error:
        failure_bits.append(str(error))
    if termination_reason != "agent_stop":
        failure_bits.append(f"termination_reason={termination_reason}")
    for name, ok in criteria:
        if not ok:
            failure_bits.append(name)
    fail_reason = "; ".join(failure_bits)

    details = {
        "dataset_ok": dataset_ok,
        "dataset_issue": dataset_issue or "",
        "tool_sequence_ok": steps_ok,
        "state_ok": state_ok,
        "answer_ok": answer_ok,
        "completed_required_steps": runtime_state.completed_required_steps,
        "required_step_count": required_step_count,
        "unexpected_tool_calls": runtime_state.unexpected_tool_calls,
        "current_state": dict(runtime_state.current_state),
        "expected_state": dict(record.expected_state),
    }
    return BfclEvaluation(
        reward=float(reward),
        is_passed=bool(is_passed),
        fail_reason=fail_reason,
        details=details,
    )


def _evaluate_bfcl_v3_official_episode(
    record: BfclTaskRecord,
    runtime_state: BfclRuntimeState,
    final_answer: str,
    *,
    termination_reason: str,
    error: str | None = None,
) -> BfclEvaluation:
    dataset_issue = _dataset_validation_issue(record)
    expected_turns = [list(turn.ground_truth) for turn in record.turns]
    decoded_turns = runtime_state.decoded_turn_outputs
    if dataset_issue is not None:
        fail_bits = [dataset_issue]
        if error:
            fail_bits.append(str(error))
        if termination_reason != "agent_stop":
            fail_bits.append(f"termination_reason={termination_reason}")
        return BfclEvaluation(
            reward=0.0,
            is_passed=False,
            fail_reason="; ".join(fail_bits),
            details={
                "dataset_ok": False,
                "dataset_issue": dataset_issue,
                "tool_sequence_ok": False,
                "state_ok": False,
                "answer_ok": False,
                "completed_turns": len(decoded_turns),
                "required_turn_count": len(expected_turns),
                "current_state": dict(runtime_state.current_state),
                "expected_state": {},
                "official_checker_error_type": "",
            },
        )

    checker_result: dict[str, Any] = {"valid": False}
    checker_error_type = ""
    checker_error_message = ""
    if not error and termination_reason == "agent_stop":
        official_root = _resolve_official_root_for_record(record)
        if official_root is None:
            checker_error_message = (
                "BFCL official backend code not found; set RWKV_BFCL_OFFICIAL_ROOT or point support dirs into an official BFCL checkout"
            )
        else:
            runtime = _load_bfcl_official_runtime(str(official_root))
            irrelevance_checker = getattr(runtime, "multi_turn_irrelevance_checker", None)
            if irrelevance_checker is not None:
                irrelevance_result = irrelevance_checker(decoded_turns, expected_turns)
                if not bool(irrelevance_result.get("valid", False)):
                    checker_result = irrelevance_result
                else:
                    checker_result = runtime.multi_turn_checker(
                        decoded_turns,
                        expected_turns,
                        {
                            "id": record.task_id,
                            "initial_config": dict(record.initial_state),
                            "involved_classes": list(record.involved_classes),
                        },
                        "multi_turn",
                        runtime_state.official_model_name or "rwkv_bfcl",
                    )
            else:
                checker_result = runtime.multi_turn_checker(
                    decoded_turns,
                    expected_turns,
                    {
                        "id": record.task_id,
                        "initial_config": dict(record.initial_state),
                        "involved_classes": list(record.involved_classes),
                    },
                    "multi_turn",
                    runtime_state.official_model_name or "rwkv_bfcl",
                )
            checker_error_type = str(checker_result.get("error_type") or "")
            checker_error_message = _stringify_checker_error(checker_result)

    valid = bool(checker_result.get("valid", False)) and not error and termination_reason == "agent_stop"
    state_ok = valid or checker_error_type != "multi_turn:instance_state_mismatch"
    tool_sequence_ok = valid or checker_error_type not in {
        "multi_turn:empty_turn_model_response",
        "multi_turn:execution_response_mismatch",
        "multi_turn:irrelevance_error:decoder_success",
    }
    details = {
        "dataset_ok": True,
        "dataset_issue": "",
        "tool_sequence_ok": tool_sequence_ok,
        "state_ok": state_ok,
        "answer_ok": True,
        "completed_turns": len(decoded_turns),
        "required_turn_count": len(expected_turns),
        "current_state": dict(runtime_state.current_state),
        "expected_state": {},
        "official_checker_error_type": checker_error_type,
        "official_checker_error_message": checker_error_message,
        "official_checker_details": dict(checker_result.get("details", {}) or {}),
    }
    failure_bits: list[str] = []
    if error:
        failure_bits.append(str(error))
    if termination_reason != "agent_stop":
        failure_bits.append(f"termination_reason={termination_reason}")
    if checker_error_message and not valid:
        failure_bits.append(checker_error_type or checker_error_message)

    return BfclEvaluation(
        reward=1.0 if valid else 0.0,
        is_passed=valid,
        fail_reason="; ".join(bit for bit in failure_bits if bit),
        details=details,
    )


def build_bfcl_ref_answer(record: BfclTaskRecord) -> str:
    if record.turns:
        payload = {
            "ground_truth_turns": [list(turn.ground_truth) for turn in record.turns],
            "tool_additions_by_turn": {
                str(index): [tool.get("name") for tool in turn.tool_additions]
                for index, turn in enumerate(record.turns)
                if turn.tool_additions
            },
            "involved_classes": list(record.involved_classes),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    payload = {
        "expected_final_answers": list(record.expected_final_answers),
        "expected_state": dict(record.expected_state),
        "required_tool_calls": [
            {
                "name": step.name,
                "arguments": dict(step.arguments),
            }
            for step in record.expected_tool_calls
            if not step.optional
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def render_bfcl_state(state: Mapping[str, Any]) -> str:
    if not state:
        return ""
    rendered = _render_bfcl_json(state)
    if len(rendered) <= BFCL_V3_MAX_STATE_CHARS:
        return rendered
    return _render_bfcl_json(
        {
            "_truncated": True,
            "preview": truncate_text(rendered, BFCL_V3_MAX_STATE_CHARS // 2),
        }
    )


def _bounded_bfcl_tool_payload_value(value: Any, *, max_chars: int) -> Any:
    rendered = _render_bfcl_json(value)
    if len(rendered) <= max_chars:
        return value
    return {
        "_truncated": True,
        "preview": truncate_text(rendered, max(32, max_chars // 2)),
    }


def normalize_bfcl_decision_output(response: str) -> str:
    normalized = normalize_bfcl_rwkv_text(response)
    if normalized.startswith("```"):
        lines = normalized.split("\n")
        if lines:
            first = lines[0].strip().lower()
            if first in {"```", "```json", "```javascript", "```js"}:
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                normalized = normalize_bfcl_rwkv_text("\n".join(lines))
    if normalized.endswith("```"):
        normalized = normalize_bfcl_rwkv_text(normalized[: -len("```")])
    return normalized


def render_bfcl_tool_catalog(tools: Sequence[Mapping[str, Any]]) -> str:
    if not tools:
        return "[]"
    rendered_tools: list[dict[str, Any]] = []
    for tool in tools:
        name = str(tool.get("name") or "").strip() or "unknown_tool"
        description = normalize_bfcl_rwkv_text(
            truncate_text(str(tool.get("description") or "").strip() or "No description available", 400)
        )
        schema = tool.get("parameters")
        if not isinstance(schema, Mapping):
            schema = {"type": "object", "properties": {}, "required": []}
        arguments: Any = {}
        raw_properties = schema.get("properties")
        if isinstance(raw_properties, Mapping):
            arguments = dict(raw_properties)
        else:
            arguments = dict(schema)
        rendered = _render_bfcl_json(arguments)
        if len(rendered) > BFCL_V3_MAX_TOOL_SCHEMA_CHARS:
            arguments = {
                "_truncated": True,
                "preview": truncate_text(rendered, BFCL_V3_MAX_TOOL_SCHEMA_CHARS // 2),
            }
        rendered_tools.append(
            {
                "name": name,
                "description": description,
                "arguments": arguments,
            }
        )
    return json.dumps(
        sorted(rendered_tools, key=lambda item: str(item.get("name") or "")),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def _bfcl_tools_with_control_functions(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rendered = [dict(tool) for tool in tools]
    rendered.extend(
        [
            {
                "name": "ask_user",
                "description": "Ask the user one brief clarification question when required information is missing.",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "final_answer",
                "description": "Return the final short answer when no environment tool should be called for this turn.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        ]
    )
    return rendered


def _read_source_items(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_items(path)

    raw_text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Official BFCL V3 downloads are line-delimited JSON files even when
        # they use a ".json" suffix. Treat "Extra data" as a JSONL signal.
        if "Extra data" not in str(exc):
            raise
        return _read_jsonl_items(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("test", "data", "records", "examples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError(f"Unsupported BFCL V3 source payload: {path}")


def _read_jsonl_items(path: Path) -> list[Any]:
    items: list[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if raw:
                items.append(json.loads(raw))
    return items


def _normalize_official_bfcl_v3_row(
    item: Mapping[str, Any],
    *,
    index: int,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    metadata_in = item.get("metadata")
    metadata_source = dict(metadata_in) if isinstance(metadata_in, Mapping) else {}
    source_hint = _resolve_bfcl_source_hint(item, metadata=metadata_source, source_path=source_path)

    task_id = str(item.get("task_id") or item.get("id") or metadata_source.get("task_id") or f"bfcl_v3_{index:04d}")
    tool_paths = _coerce_str_list(item.get("path") or metadata_source.get("path"))
    involved_classes = _coerce_str_list(
        item.get("involved_classes") or metadata_source.get("involved_classes") or item.get("classes")
    )
    if not involved_classes and tool_paths:
        involved_classes = _derive_involved_classes_from_paths(tool_paths)
    question_turns = _extract_official_question_turns(item, metadata=metadata_source)
    explicit_turns = tuple(_normalize_turn(turn) for turn in _coerce_list(item.get("turns")))
    instruction = _render_official_question_turns(question_turns)
    if not instruction:
        instruction = _first_nonempty_str(
            item.get("instruction"),
            item.get("prompt"),
            item.get("user_request"),
            item.get("user_input"),
        )
    if not instruction:
        raise ValueError(f"BFCL V3 source row {task_id!r} is missing official question/instruction content")

    support = _load_official_support_assets(
        task_id=task_id,
        source_hint=source_hint,
        involved_classes=involved_classes,
        missed_function=item.get("missed_function") or metadata_source.get("missed_function"),
    )

    raw_tools = _coerce_list(item.get("tools") or item.get("functions"))
    tools = tuple(_normalize_tool_schema(tool) for tool in raw_tools)
    if not tools:
        tools = support.tools
    if not tools and tool_paths:
        tools = tuple(_build_official_tool_schema(name) for name in tool_paths)

    raw_expectations = (
        item.get("expected_tool_calls")
        or item.get("tool_outcomes")
        or item.get("tool_calls")
        or item.get("steps")
        or _ground_truth_tool_calls(item.get("ground_truth"))
    )
    expected_tool_calls = tuple(_normalize_expectation(step) for step in _coerce_list(raw_expectations))
    if not expected_tool_calls and explicit_turns:
        expected_tool_calls = _flatten_ground_truth_expectations(explicit_turns, tools=tools)
    if not expected_tool_calls and support.ground_truth_turns:
        expected_tool_calls = _flatten_ground_truth_expectations(
            _build_official_turns(question_turns, support.ground_truth_turns, support.holdout_tools_by_turn),
            tools=tools,
        )
    if not expected_tool_calls and tool_paths:
        expected_tool_calls = tuple(BfclToolCallExpectation(name=name) for name in tool_paths)

    expected_final_answers = _normalize_answers(
        item.get("expected_final_answers")
        or item.get("expected_final_answer")
        or item.get("expected_answer")
        or item.get("answer")
        or _ground_truth_answer(item.get("ground_truth"))
    )
    expected_state = _normalize_state_mapping(
        item.get("expected_state")
        or item.get("final_state")
        or item.get("state")
        or _ground_truth_state(item.get("ground_truth"))
    )
    initial_config = item.get("initial_config") or metadata_source.get("initial_config")
    initial_state = _normalize_state_mapping(
        item.get("initial_state") or item.get("state_before") or initial_config
    )

    turns = explicit_turns or _build_official_turns(question_turns, support.ground_truth_turns, support.holdout_tools_by_turn)

    metadata = {
        **metadata_source,
        "source_format": "official_bfcl_v3_multi_turn",
        "path": tool_paths,
        "involved_classes": involved_classes,
        "question_turn_count": len(question_turns),
        "normalization_mode": "official_support_assets" if support.ground_truth_turns else "path_only_expectations",
    }
    if initial_config is not None:
        metadata["initial_config"] = initial_config
    if support.possible_answer_path is not None:
        metadata["possible_answer_path"] = support.possible_answer_path
    if support.func_doc_root is not None:
        metadata["func_doc_root"] = support.func_doc_root
    if support.official_root is not None:
        metadata["official_root"] = support.official_root
    if support.issues:
        metadata["normalization_error"] = "; ".join(support.issues)
    elif tool_paths and not raw_tools and not support.ground_truth_turns:
        metadata["normalization_warning"] = (
            "official BFCL row loaded without possible_answer ground truth; scoring will fail closed until support assets are available"
        )
    if source_hint is not None:
        metadata.setdefault("source_path", str(source_hint))
    if source_path is not None:
        metadata["manifest_path"] = str(Path(source_path))

    return {
        "task_id": task_id,
        "instruction": instruction,
        "tools": list(tools),
        "turns": [_turn_to_dict(turn) for turn in turns],
        "involved_classes": list(involved_classes),
        "expected_tool_calls": [_expectation_to_dict(step) for step in expected_tool_calls],
        "expected_final_answers": list(expected_final_answers),
        "expected_state": expected_state,
        "initial_state": initial_state,
        "metadata": metadata,
    }


def _resolve_bfcl_source_hint(
    item: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    source_path: str | Path | None,
) -> Path | None:
    for candidate in (
        metadata.get("source_path"),
        item.get("source_path"),
        item.get("raw_source_path"),
        source_path,
    ):
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path.resolve()
        return path
    return None


def _load_official_support_assets(
    *,
    task_id: str,
    source_hint: Path | None,
    involved_classes: Sequence[str],
    missed_function: Any,
) -> _OfficialSupportAssets:
    possible_answer_root = _resolve_possible_answer_root(source_hint)
    func_doc_root = _resolve_func_doc_root(source_hint)
    official_root = _resolve_official_root(source_hint, possible_answer_root=possible_answer_root, func_doc_root=func_doc_root)

    issues: list[str] = []
    ground_truth_turns: tuple[tuple[str, ...], ...] = ()
    possible_answer_path: Path | None = None
    if possible_answer_root is None:
        issues.append("missing BFCL possible_answer support directory")
    elif source_hint is None:
        issues.append("missing BFCL raw source path for possible_answer lookup")
    else:
        possible_answer_path = possible_answer_root / source_hint.name
        if not possible_answer_path.is_file():
            issues.append(f"missing BFCL possible_answer file: {possible_answer_path}")
        else:
            lookup = _load_possible_answer_lookup(str(possible_answer_path))
            ground_truth_turns = lookup.get(task_id, ())
            if not ground_truth_turns:
                issues.append(f"missing BFCL ground truth entry for task_id={task_id}")

    all_tools: tuple[dict[str, Any], ...] = ()
    if func_doc_root is None:
        issues.append("missing BFCL multi_turn_func_doc support directory")
    else:
        all_tools, missing_classes = _load_func_docs_for_classes(func_doc_root, involved_classes)
        for class_name in missing_classes:
            issues.append(f"missing BFCL function docs for class={class_name}")

    holdout_name_map = _normalize_holdout_function_map(missed_function)
    all_holdout_names = {name for names in holdout_name_map.values() for name in names}
    initial_tools = tuple(tool for tool in all_tools if str(tool.get("name") or "") not in all_holdout_names)
    tool_lookup = {str(tool.get("name") or ""): tool for tool in all_tools}
    holdout_tools_by_turn = {
        turn_index: tuple(tool_lookup[name] for name in names if name in tool_lookup)
        for turn_index, names in holdout_name_map.items()
    }

    return _OfficialSupportAssets(
        tools=initial_tools,
        ground_truth_turns=ground_truth_turns,
        holdout_tools_by_turn=holdout_tools_by_turn,
        possible_answer_path=str(possible_answer_path) if possible_answer_path is not None else None,
        func_doc_root=str(func_doc_root) if func_doc_root is not None else None,
        official_root=str(official_root) if official_root is not None else None,
        issues=tuple(dict.fromkeys(issue for issue in issues if issue)),
    )


@lru_cache(maxsize=None)
def _load_possible_answer_lookup(path_str: str) -> dict[str, tuple[tuple[str, ...], ...]]:
    path = Path(path_str)
    lookup: dict[str, tuple[tuple[str, ...], ...]] = {}
    for item in _read_source_items(path):
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or "").strip()
        if not task_id:
            continue
        raw_turns = item.get("ground_truth")
        if not isinstance(raw_turns, Sequence) or isinstance(raw_turns, (str, bytes, bytearray)):
            continue
        turns: list[tuple[str, ...]] = []
        for turn in raw_turns:
            calls = [str(entry).strip() for entry in _coerce_list(turn) if str(entry).strip()]
            turns.append(tuple(calls))
        lookup[task_id] = tuple(turns)
    return lookup


@lru_cache(maxsize=None)
def _load_func_doc_file(path_str: str) -> tuple[dict[str, Any], ...]:
    path = Path(path_str)
    return tuple(_normalize_tool_schema(item) for item in _read_source_items(path))


def _load_func_docs_for_classes(
    func_doc_root: Path,
    involved_classes: Sequence[str],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    tools: list[dict[str, Any]] = []
    missing: list[str] = []
    for class_name in involved_classes:
        filename = _BFCL_MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(class_name)
        if not filename:
            missing.append(str(class_name))
            continue
        path = func_doc_root / filename
        if not path.is_file():
            missing.append(str(class_name))
            continue
        tools.extend(dict(item) for item in _load_func_doc_file(str(path)))
    return _dedupe_tools_by_name(tools), tuple(missing)


def _normalize_holdout_function_map(raw: Any) -> dict[int, tuple[str, ...]]:
    if not isinstance(raw, Mapping):
        return {}
    normalized: dict[int, tuple[str, ...]] = {}
    for key, value in raw.items():
        try:
            turn_index = int(key)
        except (TypeError, ValueError):
            continue
        names = tuple(str(item).strip() for item in _coerce_list(value) if str(item).strip())
        if names:
            normalized[turn_index] = names
    return normalized


def _resolve_possible_answer_root(source_hint: Path | None) -> Path | None:
    env_path = _resolve_env_path(_BFCL_POSSIBLE_ANSWER_ROOT_ENV_VARS)
    if env_path is not None:
        return env_path
    for base in _support_search_roots(source_hint):
        for candidate in (
            base / "possible_answer",
            base / "possible_answer_v3",
            base / "bfcl_support" / "possible_answer",
            base / "bfcl_support" / "possible_answer_v3",
        ):
            if candidate.is_dir():
                return candidate.resolve()
    return None


def _resolve_func_doc_root(source_hint: Path | None) -> Path | None:
    env_path = _resolve_env_path(_BFCL_FUNC_DOC_ROOT_ENV_VARS)
    if env_path is not None:
        return env_path
    for base in _support_search_roots(source_hint):
        for candidate in (
            base / "multi_turn_func_doc",
            base / "bfcl_support" / "multi_turn_func_doc",
        ):
            if candidate.is_dir():
                return candidate.resolve()
    return None


def _resolve_official_root(
    source_hint: Path | None,
    *,
    possible_answer_root: Path | None,
    func_doc_root: Path | None,
) -> Path | None:
    env_path = _resolve_env_path(_BFCL_OFFICIAL_ROOT_ENV_VARS)
    if env_path is not None:
        return env_path
    for candidate in (possible_answer_root, func_doc_root, source_hint):
        resolved = _candidate_official_root(candidate)
        if resolved is not None:
            return resolved
    return None


def _resolve_env_path(names: Sequence[str]) -> Path | None:
    for name in names:
        raw = str(os.environ.get(name, "")).strip()
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path.resolve()
            return path
    return None


def _support_search_roots(source_hint: Path | None) -> tuple[Path, ...]:
    if source_hint is None:
        return ()
    candidates = [source_hint.parent]
    candidates.extend(parent for parent in source_hint.parents[:4])
    deduped: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in deduped:
            deduped.append(resolved)
    return tuple(deduped)


def _candidate_official_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    current = path.resolve()
    candidates = [current]
    candidates.extend(current.parents[:4])
    for candidate in candidates:
        if (candidate / "bfcl_eval").is_dir():
            return candidate
        if candidate.name == "bfcl_eval" and candidate.parent.is_dir():
            return candidate.parent
    return None


def _build_official_turns(
    question_turns: Sequence[Sequence[Mapping[str, str]]],
    ground_truth_turns: Sequence[Sequence[str]],
    holdout_tools_by_turn: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[BfclTurn, ...]:
    turn_count = max(len(question_turns), len(ground_truth_turns), max(holdout_tools_by_turn.keys(), default=-1) + 1)
    turns: list[BfclTurn] = []
    for turn_index in range(turn_count):
        messages = tuple(dict(item) for item in (question_turns[turn_index] if turn_index < len(question_turns) else ()))
        ground_truth = tuple(
            str(entry).strip()
            for entry in (ground_truth_turns[turn_index] if turn_index < len(ground_truth_turns) else ())
            if str(entry).strip()
        )
        tool_additions = tuple(
            dict(item) for item in holdout_tools_by_turn.get(turn_index, ()) if isinstance(item, Mapping)
        )
        turns.append(BfclTurn(messages=messages, ground_truth=ground_truth, tool_additions=tool_additions))
    return tuple(turns)


def _normalize_turn(raw: Any) -> BfclTurn:
    if not isinstance(raw, Mapping):
        return BfclTurn()
    messages = tuple(_normalize_turn_messages(raw.get("messages") or raw.get("prompt_messages") or raw.get("question")))
    ground_truth = tuple(str(item).strip() for item in _coerce_list(raw.get("ground_truth")) if str(item).strip())
    tool_additions = tuple(
        _normalize_tool_schema(item) for item in _coerce_list(raw.get("tool_additions") or raw.get("additional_tools"))
    )
    return BfclTurn(messages=messages, ground_truth=ground_truth, tool_additions=tool_additions)


def _normalize_turn_messages(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, Mapping):
        entries = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        entries = list(raw)
    else:
        text = str(raw or "").strip()
        return [{"role": "user", "content": text}] if text else []
    messages: list[dict[str, str]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            content = str(entry.get("content") or entry.get("text") or "").strip()
            if not content:
                continue
            role = str(entry.get("role") or "user").strip().lower() or "user"
            messages.append({"role": role, "content": content})
            continue
        text = str(entry or "").strip()
        if text:
            messages.append({"role": "user", "content": text})
    return messages


def _turn_to_dict(turn: BfclTurn) -> dict[str, Any]:
    return {
        "messages": [dict(item) for item in turn.messages],
        "ground_truth": list(turn.ground_truth),
        "tool_additions": [dict(item) for item in turn.tool_additions],
    }


def _flatten_ground_truth_expectations(
    turns: Sequence[BfclTurn],
    *,
    tools: Sequence[Mapping[str, Any]],
) -> tuple[BfclToolCallExpectation, ...]:
    tool_lookup = {str(tool.get("name") or ""): tool for tool in tools}
    expectations: list[BfclToolCallExpectation] = []
    for turn in turns:
        for call_string in turn.ground_truth:
            name, arguments = _parse_bfcl_call_string(call_string, tool_lookup=tool_lookup)
            expectations.append(BfclToolCallExpectation(name=name, arguments=arguments))
    return tuple(expectations)


def _ground_truth_tool_calls(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("tool_calls", "expected_tool_calls", "steps"):
            if isinstance(value.get(key), list):
                return value.get(key)
    return None


def _ground_truth_answer(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("answer") or value.get("expected_answer") or value.get("final_answer")
    return None


def _ground_truth_state(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("state") or value.get("expected_state") or value.get("final_state")
    return None


def _normalize_tool_schema(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {
            "name": "unknown_tool",
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    parameters = source.get("parameters") or source.get("input_schema") or source.get("arguments_schema")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    return {
        "name": str(source.get("name") or raw.get("name") or "unknown_tool"),
        "description": str(source.get("description") or raw.get("description") or ""),
        "parameters": _normalize_bfcl_json_schema(parameters),
    }


def _normalize_expectation(raw: Any) -> BfclToolCallExpectation:
    if not isinstance(raw, Mapping):
        return BfclToolCallExpectation(name="unknown_tool")

    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else None
    source = function or raw
    arguments = source.get("arguments") or source.get("parameters") or raw.get("arguments") or raw.get("parameters")
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            arguments = parsed if isinstance(parsed, Mapping) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}

    state_updates = raw.get("state_updates") or raw.get("updates") or raw.get("state")
    return BfclToolCallExpectation(
        name=str(source.get("name") or raw.get("tool_name") or raw.get("function_name") or "unknown_tool"),
        arguments=dict(arguments),
        result=raw.get("result") if "result" in raw else raw.get("tool_result", raw.get("response")),
        error=str(raw.get("error")) if raw.get("error") is not None else None,
        state_updates=_normalize_state_mapping(state_updates),
        optional=bool(raw.get("optional", False)),
    )


def _normalize_answers(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return tuple(items)
    return (str(raw).strip(),) if str(raw).strip() else ()


def _normalize_state_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _coerce_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    return []


def _coerce_str_list(raw: Any) -> list[str]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = [str(item).strip() for item in raw if str(item).strip()]
        return items
    text = str(raw or "").strip()
    return [text] if text else []


def _derive_involved_classes_from_paths(tool_paths: Sequence[str]) -> list[str]:
    classes: list[str] = []
    for path in tool_paths:
        namespace, _, _method = str(path).partition(".")
        if namespace and namespace not in classes:
            classes.append(namespace)
    return classes


def _looks_like_official_bfcl_v3_row(item: Mapping[str, Any]) -> bool:
    metadata = item.get("metadata")
    if isinstance(item.get("turns"), Sequence) and not isinstance(item.get("turns"), (str, bytes, bytearray)):
        return True
    if _coerce_str_list(item.get("path")):
        return True
    if isinstance(item.get("initial_config"), Mapping):
        return True
    if _coerce_str_list(item.get("involved_classes")):
        return True
    if isinstance(item.get("question"), Sequence) and not isinstance(item.get("question"), (str, bytes, bytearray)):
        return True
    if isinstance(metadata, Mapping):
        if str(metadata.get("source_format") or "").strip() == "official_bfcl_v3_multi_turn":
            return True
        if _coerce_str_list(metadata.get("path")):
            return True
        if isinstance(metadata.get("initial_config"), Mapping):
            return True
    return _looks_like_marshaled_bfcl_question(str(item.get("instruction") or ""))


def _looks_like_marshaled_bfcl_question(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped.startswith("["):
        return False
    return "'role'" in stripped and "'content'" in stripped


def _extract_official_question_turns(
    item: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> list[list[dict[str, str]]]:
    raw_question = item.get("question")
    if raw_question is None:
        raw_question = _parse_marshaled_bfcl_question(item.get("instruction"))
    turns = _normalize_question_turns(raw_question)
    if turns:
        return turns
    return _normalize_question_turns(_parse_marshaled_bfcl_question(metadata.get("instruction")))


def _parse_marshaled_bfcl_question(raw: Any) -> Any:
    text = str(raw or "").strip()
    if not _looks_like_marshaled_bfcl_question(text):
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _normalize_question_turns(raw: Any) -> list[list[dict[str, str]]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    turns: list[list[dict[str, str]]] = []
    for turn in raw:
        entries = [turn] if isinstance(turn, Mapping) or isinstance(turn, str) else _coerce_list(turn)
        messages: list[dict[str, str]] = []
        for entry in entries:
            if isinstance(entry, Mapping):
                content = str(entry.get("content") or entry.get("text") or "").strip()
                if not content:
                    continue
                role = str(entry.get("role") or "user").strip().lower() or "user"
                messages.append({"role": role, "content": content})
                continue
            text = str(entry or "").strip()
            if text:
                messages.append({"role": "user", "content": text})
        if messages:
            turns.append(messages)
    return turns


def _render_official_question_turns(turns: Sequence[Sequence[Mapping[str, str]]]) -> str:
    if not turns:
        return ""
    parts = ["Multi-turn requests:"]
    for index, turn in enumerate(turns, start=1):
        parts.append(f"Turn {index}:")
        for message in turn:
            role = str(message.get("role") or "user").strip().lower() or "user"
            content = str(message.get("content") or "").strip()
            if content:
                parts.append(f"{role.title()}: {content}")
        if index < len(turns):
            parts.append("")
    return "\n".join(parts).strip()


def _build_official_tool_schema(name: str) -> dict[str, Any]:
    namespace, _, method = name.partition(".")
    label = method or namespace or "unknown_tool"
    description = (
        f"Official BFCL V3 tool placeholder for {label}."
        if not namespace or not method
        else f"Official BFCL V3 tool placeholder for {namespace}.{method}."
    )
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": True},
    }


def _dataset_validation_issue(record: BfclTaskRecord) -> str | None:
    metadata_issue = str(record.metadata.get("normalization_error") or "").strip()
    if metadata_issue:
        return metadata_issue
    if record.turns:
        if not record.tools:
            return "missing BFCL tool catalog for official multi-turn task"
        if not any(turn.ground_truth for turn in record.turns):
            return "missing BFCL possible_answer ground truth for official multi-turn task"
        if not record.involved_classes:
            return "missing BFCL involved_classes for official multi-turn task"
        return None
    if record.expected_tool_calls and not record.tools:
        return "missing BFCL tool catalog for expected tool sequence"
    if not record.expected_tool_calls and not record.expected_state and not record.expected_final_answers:
        return "missing BFCL supervision targets"
    return None


def collect_bfcl_dataset_issues(records: Sequence[BfclTaskRecord]) -> tuple[str, ...]:
    issues: list[str] = []
    for record in records:
        issue = _dataset_validation_issue(record)
        if issue:
            issues.append(f"{record.task_id}: {issue}")
    return tuple(dict.fromkeys(issues))


def _arguments_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return dict(expected) == dict(actual)


def _resolve_expected_call(
    record: BfclTaskRecord,
    runtime_state: BfclRuntimeState,
    tool_call: TauToolCall,
) -> tuple[int, BfclToolCallExpectation | None, str | None] | None:
    start = runtime_state.next_tool_index
    if start >= len(record.expected_tool_calls):
        return None

    for index in range(start, len(record.expected_tool_calls)):
        expectation = record.expected_tool_calls[index]
        mismatch = _expectation_mismatch(expectation, tool_call)
        if mismatch is None:
            return index, expectation, None
        if not expectation.optional:
            return index, None, mismatch
    return None


def _expectation_mismatch(expectation: BfclToolCallExpectation, tool_call: TauToolCall) -> str | None:
    if tool_call.name.strip() != expectation.name.strip():
        return f"unexpected tool name: got={tool_call.name!r}, expected={expectation.name!r}"
    if expectation.arguments and not _arguments_match(expectation.arguments, tool_call.arguments):
        return (
            "unexpected tool arguments: "
            f"got={json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True)}, "
            f"expected={json.dumps(expectation.arguments, ensure_ascii=False, sort_keys=True)}"
        )
    return None


def _state_matches(current: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if not expected:
        return True
    for key, expected_value in expected.items():
        if key not in current or current[key] != expected_value:
            return False
    return True


def _normalize_answer_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def render_bfcl_turn_request(messages: Sequence[Mapping[str, str]]) -> str:
    if not messages:
        return BFCL_ADDITIONAL_FUNCTION_PROMPT
    parts: list[str] = []
    for message in messages:
        content = normalize_bfcl_rwkv_text(str(message.get("content") or ""))
        if not content:
            continue
        role = str(message.get("role") or "user").strip().lower()
        if role == "user":
            parts.append(content)
        else:
            parts.append(f"{role.title()}: {content}")
    return normalize_bfcl_rwkv_text("\n".join(parts))


def _render_bfcl_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def _strip_bfcl_outer_think_block(text: str) -> str:
    normalized = normalize_bfcl_rwkv_text(text)
    if not normalized:
        return ""
    match = re.match(r"(?s)^<think>\s*.*?\s*</think>\s*(.*)$", normalized)
    if match:
        return normalize_bfcl_rwkv_text(match.group(1))
    return normalized


def _dedupe_tools_by_name(tools: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    deduped: dict[str, dict[str, Any]] = {}
    for tool in tools:
        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        deduped[name] = dict(tool)
    return tuple(deduped.values())


def _parse_bfcl_call_string(
    call_string: str,
    *,
    tool_lookup: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    text = str(call_string or "").strip()
    if not text:
        return "unknown_tool", {}
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        name = text.split("(", 1)[0].strip() or "unknown_tool"
        return name, {}
    if not isinstance(node, ast.Call):
        return text, {}
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    else:
        name = text.split("(", 1)[0].strip() or "unknown_tool"
    arguments: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            continue
        try:
            arguments[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            continue
    if node.args:
        param_names = _tool_parameter_names(tool_lookup.get(name))
        for index, arg_node in enumerate(node.args):
            key = param_names[index] if index < len(param_names) else f"arg{index}"
            try:
                arguments[key] = ast.literal_eval(arg_node)
            except Exception:
                continue
    return name, arguments


def _tool_parameter_names(tool: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(tool, Mapping):
        return []
    parameters = tool.get("parameters")
    if not isinstance(parameters, Mapping):
        return []
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        return []
    return [str(name) for name in properties.keys()]


def _lookup_bfcl_tool(
    tools: Sequence[Mapping[str, Any]],
    name: str,
) -> Mapping[str, Any] | None:
    target = str(name or "").strip()
    if not target:
        return None
    for tool in tools:
        if str(tool.get("name") or "").strip() == target:
            return tool
    return None


def _validate_bfcl_tool_arguments(tool_call: TauToolCall, tool: Mapping[str, Any]) -> None:
    schema = tool.get("parameters")
    if not isinstance(schema, Mapping):
        return
    normalized_schema = _normalize_bfcl_json_schema(schema)
    try:
        jsonschema.validate(instance=dict(tool_call.arguments), schema=normalized_schema)
    except jsonschema.exceptions.ValidationError as exc:
        raise ValueError(f"invalid arguments for BFCL tool {tool_call.name}: {exc.message}") from exc
    except jsonschema.exceptions.SchemaError:
        # Be defensive against malformed upstream BFCL schemas.
        return


def _normalize_bfcl_json_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "type":
                normalized[key] = _normalize_bfcl_json_type(item)
                continue
            if key == "properties" and isinstance(item, Mapping):
                normalized[key] = {str(prop): _normalize_bfcl_json_schema(schema) for prop, schema in item.items()}
                continue
            if key == "items":
                normalized[key] = _normalize_bfcl_json_schema(item)
                continue
            if key in {"oneOf", "anyOf", "allOf"} and isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                normalized[key] = [_normalize_bfcl_json_schema(schema) for schema in item]
                continue
            normalized[key] = _normalize_bfcl_json_schema(item)
        if "properties" in normalized and "type" not in normalized:
            normalized["type"] = "object"
        if normalized.get("type") == "object":
            normalized.setdefault("required", [])
        return normalized
    if isinstance(value, list):
        return [_normalize_bfcl_json_schema(item) for item in value]
    return value


def _normalize_bfcl_json_type(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_bfcl_json_type(item) for item in value]
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    mapping = {
        "dict": "object",
        "map": "object",
        "float": "number",
        "double": "number",
        "int": "integer",
        "integer": "integer",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "tuple": "array",
        "str": "string",
    }
    return mapping.get(lowered, lowered or value)


def _render_bfcl_python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_render_bfcl_python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        rendered = ", ".join(_render_bfcl_python_literal(item) for item in value)
        return f"({rendered}{',' if len(value) == 1 else ''})"
    if isinstance(value, Mapping):
        inner = ", ".join(
            f"{_render_bfcl_python_literal(key)}: {_render_bfcl_python_literal(item)}"
            for key, item in value.items()
        )
        return "{" + inner + "}"
    return repr(value)


@lru_cache(maxsize=None)
def _load_bfcl_official_runtime(root_str: str) -> _OfficialRuntime:
    root = Path(root_str).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"BFCL official root does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    checker_module = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    utils_module = importlib.import_module("bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    return _OfficialRuntime(
        execute_multi_turn_func_call=utils_module.execute_multi_turn_func_call,
        multi_turn_checker=checker_module.multi_turn_checker,
        multi_turn_irrelevance_checker=getattr(checker_module, "multi_turn_irrelevance_checker", None),
    )


def _resolve_official_root_for_record(record: BfclTaskRecord) -> Path | None:
    root_value = str(record.metadata.get("official_root") or "").strip()
    if root_value:
        return Path(root_value).expanduser()
    source_hint = _resolve_bfcl_source_hint(record.metadata, metadata=record.metadata, source_path=record.metadata.get("source_path"))
    return _resolve_official_root(
        source_hint,
        possible_answer_root=_resolve_possible_answer_root(source_hint),
        func_doc_root=_resolve_func_doc_root(source_hint),
    )


def _is_official_long_context_task(record: BfclTaskRecord) -> bool:
    task_id = record.task_id.lower()
    return "long_context" in task_id or "composite" in task_id


def _decode_official_execution_result(raw_result: Any) -> tuple[bool, Any, str | None]:
    text = str(raw_result or "")
    if text.startswith("Error during execution:"):
        return False, None, text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return True, raw_result, None
    if isinstance(parsed, Mapping) and parsed.get("error") is not None:
        return False, dict(parsed), str(parsed.get("error"))
    return True, parsed, None


def _snapshot_official_instances(instances: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for class_name, instance in instances.items():
        if not hasattr(instance, "__dict__"):
            snapshot[str(class_name)] = str(instance)
            continue
        state: dict[str, Any] = {}
        for key, value in vars(instance).items():
            if key.startswith("_"):
                continue
            state[str(key)] = _to_json_safe(value)
        snapshot[str(class_name)] = state
    return snapshot


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    return str(value)


def _stringify_checker_error(checker_result: Mapping[str, Any]) -> str:
    error_message = checker_result.get("error_message")
    if isinstance(error_message, Sequence) and not isinstance(error_message, (str, bytes, bytearray)):
        return "; ".join(str(item) for item in error_message if str(item))
    return str(error_message or "").strip()


def _first_nonempty_str(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _expectation_to_dict(step: BfclToolCallExpectation) -> dict[str, Any]:
    return {
        "name": step.name,
        "arguments": dict(step.arguments),
        "result": step.result,
        "error": step.error,
        "state_updates": dict(step.state_updates),
        "optional": bool(step.optional),
    }


__all__ = [
    "BFCL_ADDITIONAL_FUNCTION_PROMPT",
    "BFCL_V3_MAX_ERROR_CHARS",
    "BFCL_V3_MAX_HISTORY_CHARS",
    "BFCL_V3_MAX_RESULT_CHARS",
    "BFCL_V3_MAX_COT_SUMMARY_CHARS",
    "BFCL_V3_MAX_STATE_CHARS",
    "BFCL_V3_MAX_TOOL_SCHEMA_CHARS",
    "BFCL_ROUTER_LABELS",
    "BfclEvaluation",
    "BfclRuntimeState",
    "BfclTaskRecord",
    "BfclTurn",
    "BfclToolCallExpectation",
    "BfclToolExecutionResult",
    "apply_bfcl_tool_call",
    "build_bfcl_rwkv_prompt",
    "build_bfcl_ref_answer",
    "build_bfcl_router_prompt",
    "build_bfcl_tool_prompt",
    "build_bfcl_ask_prompt",
    "build_bfcl_handoff_prompt",
    "build_bfcl_system_block",
    "build_bfcl_system_prompt",
    "build_bfcl_tool_result_payload",
    "build_bfcl_user_block",
    "decode_bfcl_exec_response",
    "extract_bfcl_cot_hidden_summary",
    "execute_bfcl_official_tool_call",
    "evaluate_bfcl_v3_episode",
    "has_bfcl_official_turns",
    "interpret_bfcl_assistant_output",
    "load_bfcl_v3_manifest_records",
    "load_bfcl_v3_rows_from_source",
    "parse_bfcl_router_output",
    "normalize_bfcl_rwkv_text",
    "normalize_bfcl_v3_source_row",
    "parse_bfcl_assistant_output",
    "render_bfcl_assistant_tool_message",
    "render_bfcl_official_call",
    "render_bfcl_recent_tool_window",
    "render_bfcl_state",
    "render_bfcl_state_delta",
    "render_bfcl_tool_catalog",
    "render_bfcl_turn_request",
    "start_bfcl_runtime",
]
