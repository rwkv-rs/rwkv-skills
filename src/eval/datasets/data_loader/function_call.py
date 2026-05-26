from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.eval.datasets.data_struct.function_call import (
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
MESSAGES_KEYS: tuple[str, ...] = (
    "messages",
    "conversation",
    "turns",
)


class JsonlFunctionCallTaskLoader(
    JsonlDatasetLoader[FunctionCallTaskRecord, FunctionCallTaskDataset]
):
    """Load simple tool-call tasks from JSONL."""

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

    def _extract_expected_tool_calls(self, payload: dict) -> list[dict[str, Any]]:
        value = payload.get("expected_tool_calls")
        if not isinstance(value, list):
            raise ValueError(f"{self.path}: expected_tool_calls 字段缺失或类型错误, payload={payload}")
        calls: list[dict[str, Any]] = []
        for call in value:
            if not isinstance(call, dict):
                raise ValueError(f"{self.path}: expected_tool_calls 必须只包含 object, call={call}")
            calls.append(dict(call))
        if not calls:
            raise ValueError(f"{self.path}: expected_tool_calls 不能为空")
        return calls

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
        expected_tool_calls = self._extract_expected_tool_calls(payload)

        env = self._extract_dict(payload, ("env", "environment", "env_spec"))
        scorer = self._extract_dict(payload, ("scorer", "scorer_spec", "evaluation"))
        tools = self._extract_list_of_dict(payload, ("tools", "tools_spec"))
        attachments = self._extract_list_of_dict(payload, ("attachments", "files"))
        max_steps = self._coerce_positive_int(payload.get("max_steps"))
        time_limit_s = self._coerce_positive_float(payload.get("time_limit_s"))
        base_metadata = payload.get("metadata")
        merged_metadata = dict(base_metadata) if isinstance(base_metadata, dict) else {}

        if not env:
            env = {"type": "simple_tool_call"}
        if str(env.get("type") or "") != "simple_tool_call":
            raise ValueError(f"{self.path}: function_call 仅支持 simple_tool_call env, payload={payload}")

        if not scorer:
            scorer = {
                "type": "simple_tool_call",
                "expected_tool_calls": expected_tool_calls,
            }
        scorer_type = str(scorer.get("type") or "simple_tool_call")
        if scorer_type not in {
            "simple_tool_call",
            "bfcl_exec",
            "toolalpaca_official",
            "apibank_official",
            "complexfuncbench_subset",
        }:
            raise ValueError(
                f"{self.path}: function_call 仅支持 simple_tool_call/bfcl_exec/toolalpaca_official/"
                f"apibank_official/complexfuncbench_subset scorer, "
                f"payload={payload}"
            )

        used_keys = {
            key
            for key in (
                *TASK_ID_KEYS,
                *INSTRUCTION_KEYS,
                "expected_tool_calls",
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
            expected_tool_calls=expected_tool_calls,
            env=env,
            scorer=scorer,
            tools=tools,
            attachments=attachments,
            max_steps=max_steps,
            time_limit_s=time_limit_s,
            metadata=metadata,
        )


__all__ = [
    "JsonlFunctionCallTaskLoader",
]
