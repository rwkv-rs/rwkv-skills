from __future__ import annotations

"""Prepare answer-key or tool-call rows as RWKVC agent JSON-call datasets."""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.eval.benchmark_sources import AGENT_TOOL_CALL_BENCHMARK_SOURCES
from src.eval.datasets.data_prepper.prepper_registry import FUNCTION_CALLING_REGISTRY
from src.eval.datasets.data_prepper.source_files import (
    load_hf_rows,
    per_dataset_source_env,
    read_source_rows,
    resolve_source_path,
)
from src.eval.datasets.runtime import MaterializingDatasetSpec
from src.eval.scheduler.config import REPO_ROOT
from src.eval.tasks.function_calling.simple_tool_call import (
    SimpleToolCallRecord,
    normalize_simple_tool_call_manifest_row,
)

_REQUIRED_FIELDS = ("task_id", "instruction", "tools", "expected_tool_calls")
_SOURCE_ROOT_ENV = "RWKV_AGENT_TOOL_CALL_SOURCE_ROOT"
_LEGACY_SOURCE_ROOT_ENV = "RWKV_AGENT_BENCHMARK_SOURCE_ROOT"

_AGENT_TOOL_CALL_DATASETS: dict[str, dict[str, str | None]] = {
    item.benchmark_name: {
        "source": item.source_url,
        "source_kind": item.source_kind,
    }
    for item in AGENT_TOOL_CALL_BENCHMARK_SOURCES
}

_QUESTION_KEYS = (
    "instruction",
    "question",
    "prompt",
    "problem",
    "query",
    "task",
    "input",
    "text",
)
_ANSWER_KEYS = (
    "expected_tool_calls",
    "expected_calls",
    "tool_calls",
    "gold_tool_calls",
    "answer",
    "expected_answer",
    "reference_answer",
    "reference_solution",
    "solution",
    "target",
    "final_answer",
    "output",
)
_CONTEXT_KEYS = ("context", "source_context", "document", "documents", "passage", "passages")
_TOOL_KEYS = ("tools", "functions", "tool_schemas", "available_tools")
_OPTION_KEYS = ("answer_options", "accepted_answers", "aliases", "alternative_answers")


class AgentToolCallDatasetSpec(MaterializingDatasetSpec):
    def __init__(self, output_root: Path, split: str, *, name: str) -> None:
        if name not in _AGENT_TOOL_CALL_DATASETS:
            raise ValueError(f"unknown agent tool-call dataset alias: {name}")
        super().__init__(
            name,
            output_root,
            split,
            required_fields=_REQUIRED_FIELDS,
            source_kind="rwkvc_agent_tool_call_source",
        )
        self._source_path: Path | None = None

    def source_path(self) -> Path:
        if self._source_path is None:
            self._source_path = _resolve_source_path(self.name, self.split)
        return self._source_path

    def download(self) -> None:
        path = self.source_path()
        if not path.exists():
            if _dataset_info(self.name).get("source_kind") == "hf_dataset":
                return None
            env_name = _per_dataset_source_env(self.name)
            raise FileNotFoundError(
                f"missing source for {self.name}:{self.split}: {path}. "
                f"Set {env_name}=<json/jsonl file or dir> or {_SOURCE_ROOT_ENV}=<root>."
            )

    def load_records(self) -> Iterable[dict[str, Any]]:
        path = self.source_path()
        info = _dataset_info(self.name)
        if path.exists():
            rows = _read_source_rows(path)
            source_label = str(path)
        elif info.get("source_kind") == "hf_dataset":
            rows = _load_hf_rows(str(info.get("source") or ""), self.split)
            source_label = str(info.get("source") or "")
        else:
            raise FileNotFoundError(path)
        return [
            _record_to_payload(
                normalize_simple_tool_call_manifest_row(
                    _normalize_source_row(row, dataset_name=self.name, index=index, source_path=source_label),
                    index=index,
                    source_path=source_label,
                )
            )
            for index, row in enumerate(rows)
        ]

    def manifest_extra(self) -> dict[str, Any]:
        info = _dataset_info(self.name)
        return {
            "source_path": str(self.source_path()),
            "source": info["source"],
            "source_kind": info["source_kind"],
            "input_contract": "JSONL rows may be explicit tool-call manifests or question/answer rows.",
        }


def _resolve_source_path(dataset_name: str, split: str) -> Path:
    return resolve_source_path(
        dataset_name,
        split,
        env_prefix="RWKV_AGENT_TOOL_CALL_SOURCE",
        root_envs=(_SOURCE_ROOT_ENV, _LEGACY_SOURCE_ROOT_ENV),
        default_root=REPO_ROOT / "data" / "agent_tool_call_sources",
    )


_read_source_rows = read_source_rows
_load_hf_rows = load_hf_rows


def _normalize_source_row(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    index: int,
    source_path: str | Path,
) -> dict[str, Any]:
    if _has_explicit_manifest_shape(row):
        payload = dict(row)
        payload.setdefault("task_id", _task_id(row, dataset_name, index))
        payload.setdefault("metadata", {})
        if isinstance(payload["metadata"], Mapping):
            payload["metadata"] = {
                **dict(payload["metadata"]),
                "source_benchmark": dataset_name,
                "source_path": str(source_path),
            }
        return payload

    instruction = _extract_instruction(row)
    tools = _extract_tools(row)
    expected_tool_calls = _extract_expected_tool_calls(row)
    metadata = _metadata_from_row(row, dataset_name=dataset_name, source_path=source_path)
    return {
        "task_id": _task_id(row, dataset_name, index),
        "instruction": instruction,
        "tools": tools,
        "expected_tool_calls": expected_tool_calls,
        "metadata": metadata,
    }


def _has_explicit_manifest_shape(row: Mapping[str, Any]) -> bool:
    return bool(row.get("instruction") or row.get("question")) and bool(row.get("tools")) and bool(
        row.get("expected_tool_calls")
    )


def _extract_instruction(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        rendered = _render_messages(messages)
        if rendered:
            return rendered

    question = _first_present(row, _QUESTION_KEYS)
    if question is None:
        raise ValueError(f"agent tool-call row missing instruction/question: {row}")
    question_text = _stringify(question)
    context_text = _context_text(row)
    if context_text:
        return f"Context:\n{context_text}\n\nQuestion:\n{question_text}"
    return question_text


def _render_messages(messages: Sequence[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = str(message.get("role") or "user").strip().title() or "User"
            content = str(message.get("content") or message.get("text") or "").strip()
            if content:
                parts.append(f"{role}: {content}")
        elif str(message or "").strip():
            parts.append(str(message).strip())
    return "\n".join(parts).strip()


def _context_text(row: Mapping[str, Any]) -> str:
    raw = _first_present(row, _CONTEXT_KEYS)
    if raw in (None, ""):
        return ""
    return _stringify(raw)


def _extract_tools(row: Mapping[str, Any]) -> list[Any]:
    raw = _first_present(row, _TOOL_KEYS)
    if isinstance(raw, list) and raw:
        return raw
    if isinstance(raw, Mapping):
        return [raw]
    return [_final_answer_tool()]


def _extract_expected_tool_calls(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit = _first_present(row, ("expected_tool_calls", "expected_calls", "tool_calls", "gold_tool_calls"))
    if isinstance(explicit, list):
        return [dict(item) for item in explicit if isinstance(item, Mapping)]
    if isinstance(explicit, Mapping):
        return [dict(explicit)]

    answer = _first_present(row, _ANSWER_KEYS)
    if answer is None:
        raise ValueError(f"agent tool-call row missing answer/expected_tool_calls: {row}")
    parsed = _maybe_parse_json(answer)
    if isinstance(parsed, Mapping) and _looks_like_tool_call(parsed):
        return [dict(parsed)]
    if isinstance(parsed, list) and parsed and all(isinstance(item, Mapping) for item in parsed):
        return [dict(item) for item in parsed]
    options = _answer_options(row, answer)
    return [
        {
            "name": "final_answer",
            "arguments": {"answer": options[0]},
            "argument_options": {"answer": options},
        }
    ]


def _answer_options(row: Mapping[str, Any], answer: Any) -> list[str]:
    raw = _first_present(row, _OPTION_KEYS)
    if isinstance(raw, list) and raw:
        return [_stringify(item) for item in raw]
    if isinstance(answer, list) and answer and not all(isinstance(item, Mapping) for item in answer):
        return [_stringify(item) for item in answer]
    return [_stringify(answer)]


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    candidate = _strip_json_fence(value.strip())
    if not candidate:
        return value
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return value


def _strip_json_fence(text: str) -> str:
    if "```" not in text:
        return text
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _looks_like_tool_call(value: Mapping[str, Any]) -> bool:
    return bool(value.get("name") or value.get("tool_name") or value.get("function_name"))


def _metadata_from_row(row: Mapping[str, Any], *, dataset_name: str, source_path: str | Path) -> dict[str, Any]:
    used = {
        "task_id",
        "id",
        "messages",
        *_QUESTION_KEYS,
        *_ANSWER_KEYS,
        *_CONTEXT_KEYS,
        *_TOOL_KEYS,
        *_OPTION_KEYS,
    }
    metadata = {key: value for key, value in row.items() if key not in used}
    metadata.setdefault("source_format", "rwkvc_agent_tool_call")
    metadata["source_benchmark"] = dataset_name
    metadata["source_path"] = str(source_path)
    context = _context_text(row)
    if context:
        metadata.setdefault("context", context)
    return metadata


def _task_id(row: Mapping[str, Any], dataset_name: str, index: int) -> str:
    raw = row.get("task_id") or row.get("id") or row.get("instance_id") or row.get("problem_id")
    if raw not in (None, ""):
        return str(raw)
    return f"{dataset_name}__{index:05d}"


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _final_answer_tool() -> dict[str, Any]:
    return {
        "name": "final_answer",
        "description": "Submit the final answer for the benchmark item.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    }


def _record_to_payload(record: SimpleToolCallRecord) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "instruction": record.instruction,
        "tools": [dict(tool) for tool in record.tools],
        "expected_tool_calls": [
            {
                "name": item.name,
                "arguments": dict(item.arguments),
                "argument_options": {key: list(value) for key, value in item.argument_options.items()},
            }
            for item in record.expected_tool_calls
        ],
        "metadata": dict(record.metadata),
    }


def _per_dataset_source_env(dataset_name: str) -> str:
    return per_dataset_source_env("RWKV_AGENT_TOOL_CALL_SOURCE", dataset_name)


def _dataset_info(dataset_name: str) -> dict[str, str | None]:
    return _AGENT_TOOL_CALL_DATASETS[dataset_name]


def _register(name: str):
    @FUNCTION_CALLING_REGISTRY.register_spec(name)
    def _prepare(output_root: Path, split: str = "test") -> AgentToolCallDatasetSpec:
        return AgentToolCallDatasetSpec(output_root, split, name=name)

    return _prepare


for _dataset_name in _AGENT_TOOL_CALL_DATASETS:
    _register(_dataset_name)


__all__ = [
    "AgentToolCallDatasetSpec",
]
