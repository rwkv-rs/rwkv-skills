from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from src.eval.datasets.data_struct.function_call import (
    AgentTaskDataset,
    AgentTaskRecord,
    FunctionCallTaskDataset,
    FunctionCallTaskRecord,
)

from .base import JsonlDatasetLoader

TASK_ID_KEYS: tuple[str, ...] = (
    "task_id",
    "id",
    "uid",
    "key",
)
INSTRUCTION_KEYS: tuple[str, ...] = (
    "instruction",
    "question",
    "prompt",
    "task",
    "query",
    "problem",
)
ANSWER_KEYS: tuple[str, ...] = (
    "expected_call",
    "expected_tool_call",
    "target_call",
    "tool_call",
    "function_call",
    "answer",
)
MESSAGES_KEYS: tuple[str, ...] = (
    "messages",
    "conversation",
    "turns",
)


class JsonlFunctionCallTaskLoader(
    JsonlDatasetLoader[FunctionCallTaskRecord, FunctionCallTaskDataset]
):
    """Load canonical function-call tasks from JSONL."""

    dataset_cls = FunctionCallTaskDataset

    def _extract_str(
        self,
        payload: dict,
        keys: Sequence[str],
        field_name: str,
        *,
        required: bool = True,
    ) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
        if not required:
            return None
        raise ValueError(f"{self.path}: {field_name} 字段缺失或类型错误, payload={payload}")

    def _extract_dict(self, payload: dict, keys: Sequence[str]) -> dict[str, Any]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, dict):
                return dict(value)
        return {}

    def _extract_list_of_dict(self, payload: dict, keys: Sequence[str]) -> list[dict[str, Any]]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                items: list[dict[str, Any]] = []
                for item in value:
                    if isinstance(item, dict):
                        items.append(dict(item))
                    else:
                        items.append({"value": item})
                return items
        return []

    def _extract_messages(self, payload: dict) -> list[dict[str, Any]]:
        messages = self._extract_list_of_dict(payload, MESSAGES_KEYS)
        if messages:
            return messages
        instruction = self._extract_str(payload, INSTRUCTION_KEYS, "instruction", required=False)
        if instruction:
            return [{"role": "user", "content": instruction}]
        raise ValueError(f"{self.path}: messages 或 instruction 字段缺失, payload={payload}")

    def _extract_expected_call(self, payload: dict) -> dict[str, Any]:
        for key in ANSWER_KEYS:
            value = payload.get(key)
            if isinstance(value, dict):
                return self._normalize_call(value)
            if isinstance(value, str) and value.strip():
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.path}: {key} 必须是 JSON function call 对象, payload={payload}"
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"{self.path}: {key} 必须是 JSON object, payload={payload}"
                    )
                return self._normalize_call(parsed)
        raise ValueError(f"{self.path}: expected_call 字段缺失, payload={payload}")

    def _normalize_call(self, call: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.path}: expected_call.name 字段缺失或类型错误, call={call}")
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError(f"{self.path}: expected_call.arguments 必须是 object, call={call}")
        return {"name": name.strip(), "arguments": dict(arguments)}

    @staticmethod
    def _coerce_positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float) and value.is_integer():
            number = int(value)
            return number if number > 0 else None
        return None

    @staticmethod
    def _coerce_positive_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
            return number if number > 0 else None
        return None

    def _parse_record(self, payload: dict) -> FunctionCallTaskRecord:
        task_id = self._extract_str(payload, TASK_ID_KEYS, "task_id")
        instruction = self._extract_str(payload, INSTRUCTION_KEYS, "instruction", required=False) or ""
        messages = self._extract_messages(payload)
        expected_call = self._extract_expected_call(payload)

        env = self._extract_dict(payload, ("env", "environment", "env_spec"))
        scorer = self._extract_dict(payload, ("scorer", "scorer_spec", "evaluation"))
        tools = self._extract_list_of_dict(payload, ("tools", "tools_spec"))
        attachments = self._extract_list_of_dict(payload, ("attachments", "files"))
        max_steps = self._coerce_positive_int(payload.get("max_steps"))
        time_limit_s = self._coerce_positive_float(payload.get("time_limit_s"))
        base_metadata = payload.get("metadata")
        merged_metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}

        if not env:
            env = {"type": "json_function_call"}
        if not scorer:
            scorer = {"type": "json_function_call_exact"}

        used_keys = {
            key
            for key in (
                *TASK_ID_KEYS,
                *INSTRUCTION_KEYS,
                *ANSWER_KEYS,
                "env",
                "environment",
                "env_spec",
                "scorer",
                "scorer_spec",
                "evaluation",
                "tools",
                "tools_spec",
                *MESSAGES_KEYS,
                "attachments",
                "files",
                "max_steps",
                "time_limit_s",
                "metadata",
            )
            if key in payload
        }
        metadata = dict(merged_metadata)
        metadata.update({key: value for key, value in payload.items() if key not in used_keys})

        return FunctionCallTaskRecord(
            task_id=str(task_id),
            instruction=instruction,
            messages=messages,
            expected_call=expected_call,
            env=env,
            scorer=scorer,
            tools=tools,
            attachments=attachments,
            max_steps=max_steps,
            time_limit_s=time_limit_s,
            metadata=metadata,
        )


JsonlAgentTaskLoader = JsonlFunctionCallTaskLoader


__all__ = [
    "JsonlFunctionCallTaskLoader",
    "JsonlAgentTaskLoader",
]
