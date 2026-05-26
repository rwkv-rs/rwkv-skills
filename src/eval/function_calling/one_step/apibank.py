from __future__ import annotations

"""API-Bank Level-1 adapter boundary.

The official project is kept external and should be called through this module
when API-Bank Level-1 scoring is wired.
"""

import json
import os
import ast
import re
import sys
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.function_calling.common.action import ToolAction
from src.eval.function_calling.one_step.simple_tool_call import SimpleToolCallEvaluation

DEFAULT_OFFICIAL_APIBANK_ROOT = Path("/tmp/ref-DAMO-ConvAI/api-bank")
DEFAULT_OFFICIAL_APIBANK_ROOT_CANDIDATES = (
    Path("references/API-Bank"),
    Path("../API-Bank"),
    DEFAULT_OFFICIAL_APIBANK_ROOT,
)
OFFICIAL_APIBANK_SOURCE = "DAMO-ConvAI/API-Bank"
_APIBANK_ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "CancelTimedSwitch": {"device_id": "name"},
    "TimedSwitch": {"device_id": "name"},
}


@dataclass(frozen=True, slots=True)
class ApiBankOneStepAdapterConfig:
    official_root: Path = DEFAULT_OFFICIAL_APIBANK_ROOT


def official_apibank_root() -> Path:
    override = (
        os.environ.get("API_BANK_SOURCE_ROOT")
        or os.environ.get("RWKV_API_BANK_SOURCE_ROOT")
        or os.environ.get("RWKV_APIBANK_SOURCE_ROOT")
        or os.environ.get("APIBANK_SOURCE_ROOT")
        or os.environ.get("APIBANK_OFFICIAL_ROOT")
    )
    if override:
        return Path(override).expanduser().resolve()
    for candidate in DEFAULT_OFFICIAL_APIBANK_ROOT_CANDIDATES:
        resolved = candidate.expanduser().resolve()
        if (resolved / "evaluator.py").exists() or (resolved / "apis").is_dir():
            return resolved
    return DEFAULT_OFFICIAL_APIBANK_ROOT.expanduser().resolve()


def require_official_apibank_root(root: str | Path | None = None) -> Path:
    resolved = Path(root) if root is not None else official_apibank_root()
    if not (resolved / "evaluator.py").exists() and not (resolved / "apis").is_dir():
        raise FileNotFoundError(f"API-Bank official evaluator not found under {resolved}")
    return resolved


def load_api_bank_rows_from_source(source_dir: str | Path, *, dataset_name: str, level: int) -> list[dict[str, Any]]:
    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"API-Bank source directory not found: {root}")
    official_root = _api_bank_root_from_source_dir(root)
    return _load_apibank_rows(root, official_root=official_root, dataset_name=dataset_name, level=level)


def apibank_action_text(action: ToolAction) -> str:
    args = ", ".join(f"{key}={_official_arg_repr(value)}" for key, value in action.arguments.items())
    return f"[{action.name}({args})]"


def apibank_actions_text(actions: Sequence[ToolAction]) -> str:
    return "\n".join(apibank_action_text(action) for action in actions)


def load_apibank_level1_rows_from_source_dir(
    samples_dir: str | Path,
    *,
    official_root: str | Path | None = None,
    dataset_name: str = "apibank_l1",
) -> list[dict[str, Any]]:
    root = require_official_apibank_root(official_root)
    source_dir = Path(samples_dir)
    return _load_apibank_rows(source_dir, official_root=root, dataset_name=dataset_name, level=1)


def _load_apibank_rows(
    source_dir: Path,
    *,
    official_root: Path,
    dataset_name: str,
    level: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.jsonl")):
        if _api_bank_level_from_name(path.name) != int(level):
            continue
        history = _read_apibank_history(path)
        api_names = sorted({str(item.get("api_name")) for item in history if item.get("role") == "API"})
        tools = [
            _api_description_to_tool_schema(_api_description_from_source(official_root, api_name))
            for api_name in api_names
        ]
        for item_index, item in enumerate(history):
            if item.get("role") != "API":
                continue
            api_name = str(item.get("api_name") or "")
            arguments = dict(item.get("param_dict") if isinstance(item.get("param_dict"), Mapping) else {})
            rows.append(
                {
                    "task_id": f"{dataset_name}__{path.stem}__{item_index:03d}",
                    "instruction": _render_apibank_history(history[:item_index]),
                    "tools": tools,
                    "expected_tool_calls": [
                        {
                            "name": api_name,
                            "arguments": arguments,
                            "argument_options": {key: [value] for key, value in arguments.items()},
                        }
                    ],
                    "scorer": {"type": "apibank_official", "level": int(level)},
                    "metadata": {
                        "source_format": "official_api_bank",
                        "apibank_level": int(level),
                        "apibank_official_source": OFFICIAL_APIBANK_SOURCE,
                        "apibank_official_root": str(official_root),
                        "apibank_source_path": str(path),
                        "source_dir": str(source_dir),
                        "source_path": str(path),
                        "level": int(level),
                        "turn_index": item_index,
                        "api_name": api_name,
                        "expected_result": item.get("result"),
                        "apibank_ground_truth_result": item.get("result"),
                    },
                }
            )
    return rows


def evaluate_apibank_official_calls(
    record,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    parse_error: str | None = None,
) -> SimpleToolCallEvaluation:
    expected = list(record.expected_tool_calls or [])
    details: dict[str, Any] = {
        "official_apibank_source": OFFICIAL_APIBANK_SOURCE,
        "expected_tool_calls": expected,
        "decoded_tool_calls": [
            {
                "name": str(item.get("name") or ""),
                "arguments": dict(item.get("arguments") or {}),
            }
            for item in decoded_calls
        ],
        "parse_error": parse_error or "",
    }
    if parse_error:
        return SimpleToolCallEvaluation(0.0, False, parse_error, details)
    if len(expected) != 1 or len(decoded_calls) != 1:
        return SimpleToolCallEvaluation(0.0, False, "apibank_official:call_count_mismatch", details)
    actual = decoded_calls[0]
    expected_call = expected[0]
    api_name = str(actual.get("name") or "")
    if api_name != str(expected_call.get("name") or ""):
        return SimpleToolCallEvaluation(0.0, False, "apibank_official:name_mismatch", details)
    metadata = record.metadata or {}
    root = require_official_apibank_root(metadata.get("apibank_official_root"))
    tool_manager = _official_tool_manager(root, api_names=[api_name])
    arguments = actual.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    try:
        result = tool_manager.api_call(api_name, **_normalize_apibank_arguments(api_name, dict(arguments)))
        api = tool_manager.init_tool(api_name)
        ground_truth = _normalize_apibank_expected_result(
            api_name,
            metadata.get("apibank_ground_truth_result", metadata.get("expected_result")),
        )
        correct = api.check_api_call_correctness(result, ground_truth)
    except Exception as exc:  # noqa: BLE001
        details["exception"] = repr(exc)
        return SimpleToolCallEvaluation(0.0, False, "apibank_official:execution_failed", details)
    details["actual_result"] = result
    passed = bool(correct)
    return SimpleToolCallEvaluation(
        1.0 if passed else 0.0,
        passed,
        "" if passed else "apibank_official:result_mismatch",
        details,
    )


def _api_bank_level_from_name(file_name: str) -> int | None:
    if "level-1" in file_name:
        return 1
    if "level-2" in file_name:
        return 2
    return None


def _api_bank_root_from_source_dir(source_dir: Path) -> Path:
    parts = set(source_dir.parts)
    if source_dir.name.startswith("level-") and "lv1-lv2-samples" in parts:
        return source_dir.parent.parent.resolve()
    if (source_dir / "apis").is_dir():
        return source_dir.resolve()
    return official_apibank_root()


def _normalize_apibank_arguments(api_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    aliases = _APIBANK_ARGUMENT_ALIASES.get(str(api_name), {})
    normalized: dict[str, Any] = {}
    for key, value in dict(arguments).items():
        normalized[aliases.get(str(key), str(key))] = value
    return normalized


def _normalize_apibank_expected_result(api_name: str, expected: Any) -> Any:
    if not isinstance(expected, Mapping):
        return expected
    normalized = copy.deepcopy(dict(expected))
    input_payload = normalized.get("input")
    if isinstance(input_payload, Mapping):
        normalized["input"] = _normalize_apibank_arguments(api_name, input_payload)
    return normalized


def _official_arg_repr(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if value is None:
        return "None"
    return repr(value)


def _read_apibank_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(dict(json.loads(line)))
    return rows


def _render_apibank_history(history: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for item in history:
        role = str(item.get("role") or "")
        if role == "User":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            rendered.append(f"User: {text}" if text else "User:")
        elif role == "AI":
            text = str(item.get("text") or "").lstrip().rstrip(" ")
            rendered.append(f"Assistant: {text}" if text else "Assistant:")
        elif role == "API":
            action = ToolAction(
                name=str(item.get("api_name") or ""),
                arguments=dict(item.get("param_dict") if isinstance(item.get("param_dict"), Mapping) else {}),
            )
            result = item.get("result")
            rendered.append(f"API: {apibank_action_text(action)} Response: {result}")
    return "\n".join(rendered).strip()


def _api_description_to_tool_schema(
    description: str | Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(description) if isinstance(description, str) else dict(description)
    input_parameters = payload.get("input_parameters")
    if not isinstance(input_parameters, Mapping):
        input_parameters = {}
    properties = {
        str(name): {
            "type": _apibank_type_to_json_schema(param.get("type") if isinstance(param, Mapping) else None),
            "description": str(param.get("description") or "") if isinstance(param, Mapping) else "",
        }
        for name, param in input_parameters.items()
    }
    return {
        "name": str(payload.get("name") or "unknown_tool"),
        "description": str(payload.get("description") or ""),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
        },
    }


def _apibank_type_to_json_schema(value: Any) -> str:
    raw = str(value or "str").lower()
    if raw in {"int", "integer"}:
        return "integer"
    if raw in {"float", "number"}:
        return "number"
    if raw in {"bool", "boolean"}:
        return "boolean"
    if raw.startswith("list"):
        return "array"
    return "string"


def _official_tool_manager(root: Path, *, api_names: Sequence[str] | None = None):
    return _ApiBankToolManager(root, api_names=api_names)


class _ApiBankToolManager:
    def __init__(self, root: Path, *, api_names: Sequence[str] | None = None) -> None:
        self.root = root
        with _official_import_context(root):
            from apis import API  # type: ignore

            init_databases: dict[str, Any] = {}
            init_database_dir = root / "init_database"
            if init_database_dir.exists():
                for path in init_database_dir.glob("*.json"):
                    init_databases[path.stem] = json.loads(path.read_text(encoding="utf-8"))

            apis: list[dict[str, Any]] = []
            if api_names is None:
                api_paths = sorted((root / "apis").glob("*.py"))
            else:
                names = {str(name) for name in api_names}
                names.add("CheckToken")
                api_paths = sorted(
                    path for name in names for path in (root / "apis").glob(f"{_camel_to_snake(name)}.py")
                )
            for path in api_paths:
                if path.name in {"__init__.py", "api.py", "tool_search.py"}:
                    continue
                module = importlib.import_module(f"apis.{path.stem}")
                classes = [getattr(module, name) for name in dir(module) if isinstance(getattr(module, name), type)]
                for cls in classes:
                    if not issubclass(cls, API) or cls is API:
                        continue
                    info = {
                        "name": cls.__name__,
                        "class": cls,
                        "description": cls.description,
                        "input_parameters": cls.input_parameters,
                        "output_parameters": cls.output_parameters,
                    }
                    database_name = getattr(cls, "database_name", None)
                    if database_name in init_databases:
                        info["init_database"] = init_databases[database_name]
                    apis.append(info)

        self.apis = apis
        self.inited_tools: dict[str, Any] = {}
        self.token_checker = self.init_tool("CheckToken") if "CheckToken" in self.list_all_apis() else None

    def list_all_apis(self) -> list[str]:
        return [api["name"] for api in self.apis]

    def get_api_by_name(self, name: str) -> dict[str, Any]:
        for api in self.apis:
            if api["name"] == name:
                return api
        raise ValueError(f"invalid API-Bank tool name: {name}")

    def get_api_description(self, name: str) -> str:
        api_info = dict(self.get_api_by_name(name))
        api_info.pop("class", None)
        api_info.pop("init_database", None)
        return json.dumps(api_info, ensure_ascii=False)

    def init_tool(self, tool_name: str, *args: Any, **kwargs: Any):
        if tool_name in self.inited_tools:
            return self.inited_tools[tool_name]
        api_info = self.get_api_by_name(tool_name)
        api_class = api_info["class"]
        init_args: list[Any] = []
        if "init_database" in api_info:
            init_args.append(api_info["init_database"])
        if (
            tool_name != "CheckToken"
            and "token" in api_info.get("input_parameters", {})
            and self.token_checker is not None
        ):
            init_args.append(self.token_checker)
        tool = api_class(*init_args, *args, **kwargs)
        self.inited_tools[tool_name] = tool
        return tool

    def api_call(self, tool_name: str, **kwargs: Any):
        api_info = self.get_api_by_name(tool_name)
        input_parameters = api_info.get("input_parameters") or {}
        processed_parameters: dict[str, Any] = {}
        for input_key, input_value in kwargs.items():
            if input_key not in input_parameters:
                raise ValueError(f"invalid parameter name: {input_key}")
            required_type = str(input_parameters[input_key].get("type") or "str")
            if required_type == "int":
                processed_parameters[input_key] = int(input_value)
            elif required_type == "float":
                processed_parameters[input_key] = float(input_value)
            elif required_type == "bool":
                processed_parameters[input_key] = (
                    input_value if isinstance(input_value, bool) else input_value == "True"
                )
            elif required_type.startswith("list"):
                if isinstance(input_value, str):
                    try:
                        parsed = ast.literal_eval(input_value)
                    except Exception:
                        parsed = input_value
                    processed_parameters[input_key] = parsed if isinstance(parsed, list) else input_value
                else:
                    processed_parameters[input_key] = input_value
            else:
                processed_parameters[input_key] = input_value
        tool = self.init_tool(tool_name)
        return tool.call(**processed_parameters)


def _api_description_from_source(root: Path, api_name: str) -> dict[str, Any]:
    path = root / "apis" / f"{_camel_to_snake(api_name)}.py"
    if not path.exists():
        raise FileNotFoundError(f"API-Bank API source not found for {api_name}: {path}")
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != api_name:
            continue
        fields: dict[str, Any] = {"name": api_name}
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "description",
                    "input_parameters",
                    "output_parameters",
                }:
                    fields[target.id] = ast.literal_eval(stmt.value)
        fields.setdefault("description", "")
        fields.setdefault("input_parameters", {})
        fields.setdefault("output_parameters", {})
        return fields
    raise ValueError(f"API-Bank API class {api_name} not found in {path}")


def _camel_to_snake(value: str) -> str:
    first = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", value)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", first).lower()


@contextmanager
def _official_import_context(root: Path):
    old_cwd = Path.cwd()
    root_text = str(root)
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True
    try:
        os.chdir(root)
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass


__all__ = [
    "ApiBankOneStepAdapterConfig",
    "DEFAULT_OFFICIAL_APIBANK_ROOT",
    "OFFICIAL_APIBANK_SOURCE",
    "apibank_action_text",
    "apibank_actions_text",
    "evaluate_apibank_official_calls",
    "load_api_bank_rows_from_source",
    "load_apibank_level1_rows_from_source_dir",
    "official_apibank_root",
    "require_official_apibank_root",
]
