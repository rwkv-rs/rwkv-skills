from __future__ import annotations

import argparse
import ast
import copy
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from src.eval.function_calling.simple_tool_call import (
    SimpleToolCallEvaluation,
    SimpleToolCallRecord,
    _run_simple_tool_call,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext
    from src.eval.function_calling.runner_common import ResolvedFunctionCallingRun

_API_BANK_SANDBOX_CACHE: dict[str, "ApiBankSandbox"] = {}
_API_BANK_ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "CancelTimedSwitch": {"device_id": "name"},
    "TimedSwitch": {"device_id": "name"},
}


@dataclass(frozen=True, slots=True)
class ApiBankCallResult:
    success: bool
    result: Any = None
    error: str | None = None


def load_api_bank_rows_from_source(source_dir: str | Path, *, dataset_name: str, level: int) -> list[dict[str, Any]]:
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"API-Bank source directory not found: {root}")

    rows: list[dict[str, Any]] = []
    for file_path in sorted(root.glob("*.jsonl")):
        file_level = _api_bank_level_from_name(file_path.name)
        if file_level != level:
            continue
        history = _read_jsonl(file_path)
        api_names = sorted(
            {
                str(item.get("api_name") or "").strip()
                for item in history
                if isinstance(item, Mapping) and item.get("role") == "API"
            }
        )
        for turn_index, item in enumerate(history):
            if not isinstance(item, Mapping) or item.get("role") != "API":
                continue
            api_name = str(item.get("api_name") or "").strip()
            if not api_name:
                continue
            param_dict = item.get("param_dict")
            if not isinstance(param_dict, Mapping):
                param_dict = {}
            rows.append(
                {
                    "task_id": f"{dataset_name}__{file_path.stem}_{turn_index:03d}",
                    "instruction": _render_api_bank_history(history[:turn_index]),
                    "tools": [_api_bank_tool_schema(root.parent.parent if root.name.startswith("level-") else root, name, param_dict) for name in api_names],
                    "expected_tool_calls": [
                        {
                            "name": api_name,
                            "arguments": dict(param_dict),
                            "argument_options": {key: [value] for key, value in dict(param_dict).items()},
                        }
                    ],
                    "metadata": {
                        "source_format": "official_api_bank",
                        "source_path": str(file_path),
                        "source_dir": str(root),
                        "level": level,
                        "turn_index": turn_index,
                        "api_name": api_name,
                        "expected_result": item.get("result"),
                    },
                }
            )
    return rows


def evaluate_api_bank_calls(
    record: SimpleToolCallRecord,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
    sandbox: "ApiBankSandbox | None" = None,
) -> SimpleToolCallEvaluation:
    expected = record.expected_tool_calls[0] if record.expected_tool_calls else None
    expected_name = expected.name if expected is not None else ""
    expected_result = _normalize_api_bank_expected_result(expected_name, record.metadata.get("expected_result"))
    details: dict[str, Any] = {
        "expected_tool_calls": [
            {"name": expected.name, "arguments": dict(expected.arguments)} for expected in record.expected_tool_calls
        ],
        "decoded_tool_calls": [{"name": str(item.get("name") or ""), "arguments": dict(item.get("arguments") or {})} for item in decoded_calls],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return SimpleToolCallEvaluation(0.0, False, parse_error, details)
    if not expected or not decoded_calls:
        return SimpleToolCallEvaluation(0.0, False, "missing_call", details)
    actual = decoded_calls[0]
    actual_name = str(actual.get("name") or "").strip()
    actual_args = actual.get("arguments")
    if not isinstance(actual_args, Mapping):
        actual_args = {}
    if actual_name != expected_name:
        return SimpleToolCallEvaluation(0.0, False, f"api_name_mismatch:{actual_name}!={expected_name}", details)

    source_dir = _api_bank_root_from_metadata(record.metadata)
    sandbox = sandbox or ApiBankSandbox(source_dir)
    call_result = sandbox.api_call(actual_name, dict(actual_args))
    details["execution_result"] = _api_bank_result_payload(call_result)
    details["expected_result"] = expected_result
    if not call_result.success:
        return SimpleToolCallEvaluation(0.0, False, call_result.error or "api_execution_failed", details)
    try:
        ok = sandbox.check_api_call_correctness(
            actual_name,
            copy.deepcopy(call_result.result),
            copy.deepcopy(expected_result),
        )
    except Exception as exc:  # noqa: BLE001
        details["check_error"] = str(exc)
        ok = False
    return SimpleToolCallEvaluation(1.0 if ok else 0.0, bool(ok), "" if ok else "api_result_mismatch", details)


class ApiBankSandbox:
    def __init__(self, source_root: str | Path) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self._api_classes: dict[str, type] | None = None
        self._tools: dict[str, Any] = {}
        self._init_databases: dict[str, Any] | None = None

    def api_call(self, api_name: str, arguments: Mapping[str, Any]) -> ApiBankCallResult:
        try:
            tool = self.init_tool(api_name)
            api_info = self._api_info(api_name)
            normalized_arguments = _normalize_api_bank_arguments(api_name, arguments)
            processed = {
                key: self._coerce_arg(value, api_info.get("input_parameters", {}).get(key, {}).get("type"))
                for key, value in normalized_arguments.items()
            }
            return ApiBankCallResult(True, tool.call(**processed))
        except Exception as exc:  # noqa: BLE001
            return ApiBankCallResult(False, error=str(exc))

    def check_api_call_correctness(self, api_name: str, actual: Any, expected: Any) -> bool:
        return bool(self.init_tool(api_name).check_api_call_correctness(actual, expected))

    def get_api_description(self, api_name: str) -> dict[str, Any] | None:
        try:
            info = dict(self._api_info(api_name))
        except Exception:
            return None
        info.pop("class", None)
        info.pop("init_database", None)
        return info

    def init_tool(self, api_name: str) -> Any:
        if api_name in self._tools:
            return self._tools[api_name]
        info = self._api_info(api_name)
        args: list[Any] = []
        if "init_database" in info:
            args.append(info["init_database"])
        if api_name != "CheckToken" and "token" in info.get("input_parameters", {}) and "CheckToken" in self._api_classes_by_name():
            args.append(self.init_tool("CheckToken"))
        tool = info["class"](*args)
        self._tools[api_name] = tool
        return tool

    def _api_info(self, api_name: str) -> dict[str, Any]:
        cls = self._api_classes_by_name().get(api_name)
        if cls is None:
            raise ValueError(f"invalid API-Bank tool name: {api_name}")
        info: dict[str, Any] = {
            "name": api_name,
            "class": cls,
            "description": getattr(cls, "description", ""),
            "input_parameters": getattr(cls, "input_parameters", {}),
            "output_parameters": getattr(cls, "output_parameters", {}),
        }
        database_name = getattr(cls, "database_name", None)
        init_databases = self._load_init_databases()
        if database_name in init_databases:
            info["init_database"] = init_databases[database_name]
        return info

    def _api_classes_by_name(self) -> dict[str, type]:
        if self._api_classes is not None:
            return self._api_classes
        classes: dict[str, type] = {}
        with _temporary_api_bank_import_path(self.source_root):
            api_base = importlib.import_module("apis.api").API
            apis_dir = self.source_root / "apis"
            for file_path in sorted(apis_dir.glob("*.py")):
                if file_path.name in {"__init__.py", "api.py", "tool_search.py"}:
                    continue
                try:
                    module = importlib.import_module(f"apis.{file_path.stem}")
                except Exception:
                    continue
                for value in vars(module).values():
                    if isinstance(value, type) and issubclass(value, api_base) and value is not api_base:
                        classes[value.__name__] = value
        self._api_classes = classes
        return classes

    def _load_init_databases(self) -> dict[str, Any]:
        if self._init_databases is not None:
            return self._init_databases
        databases: dict[str, Any] = {}
        db_dir = self.source_root / "init_database"
        if db_dir.is_dir():
            for file_path in db_dir.glob("*.json"):
                databases[file_path.stem] = json.loads(file_path.read_text(encoding="utf-8"))
        self._init_databases = databases
        return databases

    @staticmethod
    def _coerce_arg(value: Any, arg_type: Any) -> Any:
        if arg_type == "int":
            return int(value)
        if arg_type == "float":
            return float(value)
        if arg_type == "bool":
            return value if isinstance(value, bool) else str(value) == "True"
        if str(arg_type) in {"list", "list(str)"}:
            return _coerce_api_bank_list_arg(value)
        return value


def _normalize_api_bank_arguments(api_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    aliases = _API_BANK_ARGUMENT_ALIASES.get(str(api_name), {})
    normalized: dict[str, Any] = {}
    for key, value in dict(arguments).items():
        normalized[aliases.get(str(key), str(key))] = value
    return normalized


def _normalize_api_bank_expected_result(api_name: str, expected: Any) -> Any:
    if not isinstance(expected, Mapping):
        return expected
    normalized = copy.deepcopy(dict(expected))
    input_payload = normalized.get("input")
    if isinstance(input_payload, Mapping):
        normalized["input"] = _normalize_api_bank_arguments(api_name, input_payload)
    return normalized


def _coerce_api_bank_list_arg(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return value


def _run_api_bank(
    args: argparse.Namespace,
    run: "ResolvedFunctionCallingRun",
    *,
    run_context: "RunContext | None" = None,
) -> int:
    return _run_simple_tool_call(
        args,
        run,
        default_job_name="function_api_bank",
        evaluator=evaluate_api_bank_calls,
        run_context=run_context,
    )


def _api_bank_level_from_name(file_name: str) -> int | None:
    if "level-1" in file_name:
        return 1
    if "level-2" in file_name:
        return 2
    return None


def _render_api_bank_history(history: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for item in history:
        role = str(item.get("role") or "").strip()
        if role == "User":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            lines.append(f"User: {text}" if text else "User:")
        elif role == "AI":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            lines.append(f"Assistant: {text}" if text else "Assistant:")
        elif role == "API":
            args = ", ".join(
                f"{key}={_official_arg_repr(value)}" for key, value in dict(item.get("param_dict") or {}).items()
            )
            lines.append(f"API: [{item.get('api_name')}({args})] Response: {item.get('result')}")
    return "\n".join(lines).strip()


def _api_bank_tool_schema(source_root: Path, api_name: str, fallback_args: Mapping[str, Any]) -> dict[str, Any]:
    description = _api_bank_sandbox_for(source_root).get_api_description(api_name)
    if not description:
        return {
            "name": api_name,
            "description": f"API-Bank tool {api_name}",
            "parameters": {
                "type": "object",
                "properties": {str(key): {"type": _json_type(value)} for key, value in fallback_args.items()},
                "required": [str(key) for key in fallback_args],
            },
        }
    parameters = description.get("input_parameters")
    properties: dict[str, Any] = {}
    required: list[str] = []
    if isinstance(parameters, Mapping):
        for key, spec in parameters.items():
            spec = spec if isinstance(spec, Mapping) else {}
            properties[str(key)] = {
                "type": _api_bank_json_type(spec.get("type")),
                "description": str(spec.get("description") or ""),
            }
            required.append(str(key))
    return {
        "name": api_name,
        "description": str(description.get("description") or ""),
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _api_bank_sandbox_for(source_root: Path) -> ApiBankSandbox:
    key = str(source_root.expanduser().resolve())
    sandbox = _API_BANK_SANDBOX_CACHE.get(key)
    if sandbox is None:
        sandbox = ApiBankSandbox(source_root)
        _API_BANK_SANDBOX_CACHE[key] = sandbox
    return sandbox


def _api_bank_root_from_metadata(metadata: Mapping[str, Any]) -> Path:
    source_dir = Path(str(metadata.get("source_dir") or "")).expanduser()
    if source_dir.name.startswith("level-"):
        return source_dir.parent.parent.resolve()
    return source_dir.resolve()


def _api_bank_result_payload(result: ApiBankCallResult) -> dict[str, Any]:
    return {"success": result.success, "result": result.result, "error": result.error}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _official_arg_repr(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    return repr(value)


def _api_bank_json_type(value: Any) -> str:
    return {
        "int": "integer",
        "float": "number",
        "str": "string",
        "list": "array",
        "list(str)": "array",
        "bool": "boolean",
    }.get(str(value), "string")


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "string"


@contextmanager
def _temporary_api_bank_import_path(source_root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    root_text = str(source_root)
    sys.path.insert(0, root_text)
    os.chdir(source_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(root_text)
        except ValueError:
            pass


__all__ = [
    "ApiBankCallResult",
    "ApiBankSandbox",
    "evaluate_api_bank_calls",
    "load_api_bank_rows_from_source",
    "_run_api_bank",
]
