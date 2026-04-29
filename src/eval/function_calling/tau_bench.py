from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context_budget import normalize_rwkv_text as _normalize_rwkv_text


@dataclass(frozen=True, slots=True)
class TauManifestRecord:
    task_id: str
    domain: str
    instruction: str
    task: dict[str, Any]
    benchmark_version: str
    index: int = 0


@dataclass(frozen=True, slots=True)
class TauToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    requestor: str = "assistant"


@dataclass(frozen=True, slots=True)
class TauDecision:
    is_tool_call: bool
    tool_call: TauToolCall | None = None
    final_answer: str = ""


def load_tau_manifest_records(path: str | Path) -> list[TauManifestRecord]:
    rows: list[TauManifestRecord] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            task = payload.get("task")
            if not isinstance(task, dict):
                task = {}
            rows.append(
                TauManifestRecord(
                    task_id=str(payload.get("task_id") or ""),
                    domain=str(payload.get("domain") or ""),
                    instruction=str(payload.get("instruction") or ""),
                    task=task,
                    benchmark_version=str(payload.get("benchmark_version") or ""),
                    index=int(payload.get("index") or 0),
                )
            )
    return rows


def render_tau_user_prompt(task_payload: Mapping[str, Any]) -> str:
    ticket = str(task_payload.get("ticket") or "").strip()
    if ticket:
        return ticket

    parts: list[str] = []
    user_scenario = task_payload.get("user_scenario")
    if isinstance(user_scenario, Mapping):
        persona = str(user_scenario.get("persona") or "").strip()
        if persona:
            parts.append(f"Persona:\n\t{persona}")

        instructions = task_payload.get("user_scenario", {}).get("instructions")
        rendered_instructions = _render_user_instructions(instructions)
        if rendered_instructions:
            parts.append(f"Instructions:\n\t{rendered_instructions}")

    description = task_payload.get("description")
    if isinstance(description, Mapping):
        purpose = str(description.get("purpose") or "").strip()
        notes = str(description.get("notes") or "").strip()
        if purpose:
            parts.append(f"Purpose:\n{purpose}")
        if notes:
            parts.append(f"Notes:\n{notes}")

    text = _normalize_rwkv_text("\n".join(part for part in parts if part))
    if text:
        return text
    return ""


def build_tau_system_prompt(
    policy: str,
    *,
    assistant_tools: Sequence[Mapping[str, Any]],
    user_tools: Sequence[Mapping[str, Any]],
) -> str:
    tools = [
        _function_call_tool_schema(item, requestor="assistant")
        for item in assistant_tools
    ]
    tools.extend(_function_call_tool_schema(item, requestor="user") for item in user_tools)
    tools.append(
        {
            "name": "final_answer",
            "description": "Use this when the task is complete.",
            "arguments": {
                "answer": {"type": "string"},
            },
        }
    )
    lines = [
        "Tools:",
        json.dumps(tools, ensure_ascii=False, indent=2, sort_keys=True),
        "Return only a JSON function call.",
        'The JSON shape is {"name":"tool_name","arguments":{...}}.',
        "Use exactly one listed tool name.",
        "Use final_answer only when the task is complete.",
        "Policy:",
        _normalize_rwkv_text(policy),
    ]
    return _normalize_rwkv_text("\n".join(lines))


def build_expected_context(system_prompt: str, messages: Sequence[Mapping[str, str]]) -> str:
    parts = [f"System: {_normalize_rwkv_text(system_prompt)}"]
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = _normalize_rwkv_text(str(message.get("content") or ""))
        if not content:
            continue
        if role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant: <think><|completions_of_cot|>")
    return "\n\n".join(parts)


def build_turn_completion_prompt(cot_context: str, cot: str) -> str:
    return (
        cot_context.replace("<|completions_of_cot|>", cot)
        + "</think>\nReturn only a JSON function call.\n"
    )


def parse_tool_call_or_final_answer(response: str) -> TauDecision:
    trimmed = response.strip()
    if not trimmed:
        raise ValueError("model returned empty response")

    if not (trimmed.startswith("{") and trimmed.endswith("}")):
        raise ValueError(f"model response must be a JSON function call object: {trimmed}")

    payload = json.loads(trimmed)
    if not isinstance(payload, dict):
        raise ValueError("tool call payload must be a JSON object")
    if set(payload.keys()) != {"name", "arguments"}:
        raise ValueError("tool call JSON must contain exactly name and arguments")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError(f"tool call missing name: {trimmed}")
    requestor = "assistant"
    if "." in name:
        prefix, unprefixed = name.split(".", 1)
        if prefix in {"assistant", "user"} and unprefixed.strip():
            requestor = prefix
            name = unprefixed.strip()
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "final_answer":
        return TauDecision(is_tool_call=False, final_answer=str(arguments.get("answer") or "").strip())
    return TauDecision(
        is_tool_call=True,
        tool_call=TauToolCall(name=name, arguments=dict(arguments), requestor=requestor),
    )


def render_tool_result(tool_call: TauToolCall, *, ok: bool, output: Any = None, error: str | None = None) -> str:
    payload: dict[str, Any] = {
        "requestor": tool_call.requestor,
        "name": tool_call.name,
        "ok": bool(ok),
    }
    if ok:
        payload["output"] = output
    else:
        payload["error"] = str(error or "unknown tool error")
    return "Function output:\n" + json.dumps(payload, ensure_ascii=False)


def render_assistant_tool_message(cot: str, tool_call: TauToolCall) -> str:
    del cot
    payload = {
        "name": _prefixed_tool_name(tool_call),
        "arguments": tool_call.arguments,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def load_rwkv_rs_tau_bench_rows(
    *,
    datasets_root: str | Path,
    domain: str,
) -> list[dict[str, Any]]:
    base = Path(datasets_root).expanduser().resolve() / "tau_bench" / domain
    tasks_path = base / "tasks.json"
    split_path = base / "split_tasks.json"
    if not tasks_path.is_file():
        raise FileNotFoundError(f"missing tau_bench tasks file: {tasks_path}")
    if not split_path.is_file():
        raise FileNotFoundError(f"missing tau_bench split file: {split_path}")

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError(f"invalid tau_bench task payload: {tasks_path}")
    base_ids = {str(item) for item in (split_payload.get("base") or [])}

    rows: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "")
        if task_id not in base_ids:
            continue
        rows.append(
            {
                "task_id": task_id,
                "domain": domain,
                "index": len(rows),
                "instruction": render_tau_user_prompt(task),
                "task": task,
                "benchmark_version": "tau_bench",
            }
        )
    return rows


def _render_user_instructions(instructions: Any) -> str:
    if isinstance(instructions, str):
        return instructions.strip()
    if not isinstance(instructions, Mapping):
        return ""

    parts: list[str] = []
    domain = str(instructions.get("domain") or "").strip()
    reason_for_call = str(instructions.get("reason_for_call") or "").strip()
    known_info = str(instructions.get("known_info") or "").strip()
    unknown_info = str(instructions.get("unknown_info") or "").strip()
    task_instructions = str(instructions.get("task_instructions") or "").strip()
    if domain:
        parts.append(f"Domain: {domain}")
    if reason_for_call:
        parts.append(f"Reason for call:\n\t{reason_for_call}")
    if known_info:
        parts.append(f"Known info:\n\t{known_info}")
    if unknown_info:
        parts.append(f"Unknown info:\n\t{unknown_info}")
    if task_instructions:
        parts.append(f"Task instructions:\n\t{task_instructions}")
    return "\n".join(parts).strip()


def _render_tool_schema(schema: Mapping[str, Any], *, requestor: str) -> str:
    name = str(schema.get("name") or "").strip() or "unknown_tool"
    description = str(schema.get("description") or "").strip()
    parameters = schema.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    return "\n".join(
        [
            f"### `{requestor}.{name}`",
            f"**Description:** {_normalize_rwkv_text(description or 'No description available')}",
            "**Parameters:**",
            json.dumps(parameters, ensure_ascii=False, indent=2, sort_keys=True),
        ]
    )


def _function_call_tool_schema(schema: Mapping[str, Any], *, requestor: str) -> dict[str, Any]:
    name = str(schema.get("name") or "").strip() or "unknown_tool"
    description = str(schema.get("description") or "").strip()
    parameters = schema.get("parameters")
    arguments: Any = {}
    if isinstance(parameters, Mapping):
        raw_properties = parameters.get("properties")
        if isinstance(raw_properties, Mapping):
            arguments = dict(raw_properties)
        else:
            arguments = dict(parameters)
    return {
        "name": f"{requestor}.{name}",
        "description": _normalize_rwkv_text(description or "No description available"),
        "arguments": arguments,
    }


def _prefixed_tool_name(tool_call: TauToolCall) -> str:
    requestor = tool_call.requestor.strip().lower() or "assistant"
    if requestor in {"assistant", "user"}:
        return f"{requestor}.{tool_call.name}"
    return tool_call.name


__all__ = [
    "TauDecision",
    "TauManifestRecord",
    "TauToolCall",
    "build_expected_context",
    "build_tau_system_prompt",
    "build_turn_completion_prompt",
    "load_rwkv_rs_tau_bench_rows",
    "load_tau_manifest_records",
    "parse_tool_call_or_final_answer",
    "render_assistant_tool_message",
    "render_tau_user_prompt",
    "render_tool_result",
]
