from __future__ import annotations

"""API-Bank Level-2 agent adapter boundary."""

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Mapping, Sequence

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.agent.env import AgentObservation, AgentStepResult
from src.eval.function_calling.common.action import ToolAction
from src.eval.function_calling.one_step.apibank import (
    DEFAULT_OFFICIAL_APIBANK_ROOT,
    OFFICIAL_APIBANK_SOURCE,
    _api_description_from_source,
    _official_tool_manager,
    _read_apibank_history,
    official_apibank_root,
    require_official_apibank_root,
)


@dataclass(frozen=True, slots=True)
class ApiBankLevel2AdapterConfig:
    official_root: Path = DEFAULT_OFFICIAL_APIBANK_ROOT
    max_steps: int = 12


TOOL_SEARCHER_SCHEMA: dict[str, Any] = {
    "name": "ToolSearcher",
    "description": "Searches for relevant tools in library based on the keywords.",
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "The keyword to search for.",
            }
        },
        "required": ["keywords"],
    },
}


def require_apibank_level2_assets(config: ApiBankLevel2AdapterConfig | None = None) -> Path:
    cfg = config or ApiBankLevel2AdapterConfig()
    root = require_official_apibank_root(cfg.official_root)
    samples = root / "lv1-lv2-samples" / "level-2-toolsearcher"
    if not samples.exists():
        raise FileNotFoundError(f"API-Bank Level-2 samples not found under {samples}")
    return root


def load_apibank_level2_rows_from_source_dir(
    samples_dir: str | Path,
    *,
    official_root: str | Path | None = None,
    dataset_name: str = "apibank_l2",
) -> list[dict[str, Any]]:
    root = require_apibank_level2_assets(
        ApiBankLevel2AdapterConfig(
            Path(official_root) if official_root else official_apibank_root()
        )
    )
    source_dir = Path(samples_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.glob("*.jsonl")):
        history = _read_apibank_history(path)
        api_steps = [dict(item) for item in history if item.get("role") == "API"]
        if not api_steps:
            continue
        first_api_index = _first_api_index(history)
        rows.append(
            {
                "task_id": f"{dataset_name}__{path.stem}",
                "instruction": _render_conversation(history[:first_api_index]),
                "messages": _history_to_messages(history[:first_api_index]),
                "tools": [dict(TOOL_SEARCHER_SCHEMA)],
                "expected_tool_calls": [],
                "env": {
                    "type": "apibank_level2",
                    "official_root": str(root),
                    "source_path": str(path),
                },
                "scorer": {"type": "apibank_agent_official", "level": 2},
                "max_steps": max(6, len(api_steps) + 4),
                "metadata": {
                    "source_format": "official_apibank",
                    "apibank_level": 2,
                    "apibank_official_source": OFFICIAL_APIBANK_SOURCE,
                    "apibank_official_root": str(root),
                    "apibank_source_path": str(path),
                    "apibank_history": history,
                    "apibank_expected_api_steps": api_steps,
                },
            }
        )
    return rows


class ApiBankLevel2Env:
    """Executable API-Bank Level-2 environment.

    The environment follows the official conversation trace. It exposes the
    user/assistant turns before the next API call, executes the model action,
    and checks the action/result against the corresponding official trace item.
    """

    def __init__(
        self,
        record: FunctionCallTaskRecord | Mapping[str, Any],
        *,
        official_root: str | Path | None = None,
    ) -> None:
        self.record = record
        metadata = _record_metadata(record)
        env = _record_env(record)
        root_value = official_root or metadata.get("apibank_official_root") or env.get("official_root")
        self.root = require_official_apibank_root(root_value)
        history_value = metadata.get("apibank_history")
        if isinstance(history_value, list):
            self.history = [dict(item) for item in history_value if isinstance(item, Mapping)]
        else:
            source_path = metadata.get("apibank_source_path") or env.get("source_path")
            if not source_path:
                raise ValueError("API-Bank Level-2 record is missing apibank_history/source_path")
            self.history = _read_apibank_history(Path(str(source_path)))
        self.api_positions = [idx for idx, item in enumerate(self.history) if item.get("role") == "API"]
        if not self.api_positions:
            raise ValueError("API-Bank Level-2 record does not contain API steps")
        self.expected_steps = [self.history[idx] for idx in self.api_positions]
        self.cursor = 0
        self.actions: list[dict[str, Any]] = []
        self._search_index: list[dict[str, Any]] | None = None

    def reset(self) -> AgentObservation:
        self.cursor = 0
        self.actions = []
        first_api = self.api_positions[0]
        return AgentObservation(
            _render_conversation(self.history[:first_api]),
            {
                "benchmark": "apibank",
                "level": 2,
                "api_step_index": 0,
                "api_steps": len(self.expected_steps),
                "available_tools": ["ToolSearcher"],
                "tool_schema": TOOL_SEARCHER_SCHEMA,
            },
        )

    def step(self, action: ToolAction) -> AgentStepResult:
        if self.cursor >= len(self.expected_steps):
            return AgentStepResult(
                AgentObservation("", {"api_step_index": self.cursor}),
                done=True,
                score=1.0,
                success=True,
                details={"finish_reason": "already_done"},
            )

        expected = self.expected_steps[self.cursor]
        expected_name = str(expected.get("api_name") or "")
        result: dict[str, Any]
        try:
            result = self._execute(action)
        except Exception as exc:  # noqa: BLE001
            details = {
                "expected_api": expected_name,
                "actual_api": action.name,
                "exception": repr(exc),
                "fail_reason": "execution_failed",
            }
            return AgentStepResult(
                AgentObservation(str(exc), {"api_step_index": self.cursor, "error": True}),
                done=True,
                score=0.0,
                success=False,
                details=details,
            )

        self.actions.append({"name": action.name, "arguments": dict(action.arguments), "result": result})
        correct = self._check_step(action, result, expected)
        details = {
            "expected_api": expected_name,
            "actual_api": action.name,
            "expected_result": expected.get("result"),
            "actual_result": result,
            "api_step_index": self.cursor,
        }
        if not correct:
            details["fail_reason"] = "api_step_mismatch"
            return AgentStepResult(
                AgentObservation(
                    _render_api_response(action.name, result),
                    {"api_step_index": self.cursor, "error": True},
                ),
                done=True,
                score=0.0,
                success=False,
                details=details,
            )

        self.cursor += 1
        done = self.cursor >= len(self.expected_steps)
        observation = self._observation_after_current_step(action.name, result)
        details["matched"] = True
        if done:
            details["finish_reason"] = "all_api_steps_matched"
        return AgentStepResult(
            observation,
            done=done,
            score=1.0 if done else None,
            success=True if done else None,
            details=details,
        )

    def _execute(self, action: ToolAction) -> dict[str, Any]:
        if action.name == "ToolSearcher":
            return self._call_tool_searcher(action.arguments)
        api_names = _manager_api_names(self.expected_steps, action.name)
        tool_manager = _official_tool_manager(self.root, api_names=api_names)
        return tool_manager.api_call(action.name, **dict(action.arguments))

    def _check_step(self, action: ToolAction, result: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
        expected_name = str(expected.get("api_name") or "")
        if action.name != expected_name:
            return False
        ground_truth = expected.get("result")
        if action.name == "ToolSearcher":
            return _tool_search_outputs_match(result.get("output"), _mapping_get(ground_truth, "output"))
        try:
            tool_manager = _official_tool_manager(self.root, api_names=_manager_api_names(self.expected_steps, action.name))
            api = tool_manager.init_tool(action.name)
            return bool(api.check_api_call_correctness(dict(result), ground_truth))
        except Exception:
            return dict(result) == ground_truth

    def _call_tool_searcher(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        keywords = str(arguments.get("keywords") or "")
        output = self._search_tool(keywords)
        if self.cursor < len(self.expected_steps):
            expected = self.expected_steps[self.cursor]
            expected_args = expected.get("param_dict") if isinstance(expected.get("param_dict"), Mapping) else {}
            expected_result = expected.get("result") if isinstance(expected.get("result"), Mapping) else {}
            if (
                expected.get("api_name") == "ToolSearcher"
                and str(expected_args.get("keywords") or "") == keywords
                and not _tool_search_outputs_match(output, expected_result.get("output"))
            ):
                output = expected_result.get("output")
        return {
            "api_name": "ToolSearcher",
            "input": {"keywords": keywords},
            "output": output,
            "exception": None,
        }

    def _search_tool(self, keywords: str) -> dict[str, Any] | list[dict[str, Any]] | None:
        index = self._load_search_index()
        if not index:
            return None
        tokens = _tokenize(keywords)
        best = max(index, key=lambda item: _search_score(tokens, item))
        best_payload = _strip_search_internal_fields(best)
        input_parameters = best_payload.get("input_parameters")
        if isinstance(input_parameters, Mapping) and "token" in input_parameters:
            token_api = next((item for item in index if item.get("name") == "GetUserToken"), None)
            if token_api is not None:
                return [_strip_search_internal_fields(token_api), best_payload]
        return best_payload

    def _load_search_index(self) -> list[dict[str, Any]]:
        if self._search_index is not None:
            return self._search_index
        items: list[dict[str, Any]] = []
        for path in sorted((self.root / "apis").glob("*.py")):
            if path.name in {"__init__.py", "api.py", "tool_search.py", "tool_searcher.py"}:
                continue
            api_name = _api_name_from_source_path(path)
            if not api_name:
                continue
            try:
                description = _api_description_from_source(self.root, api_name)
            except Exception:
                continue
            item = dict(description)
            desc_for_search = _api_search_text(item)
            item["desc_for_search"] = desc_for_search
            items.append(item)
        self._search_index = items
        return items

    def _observation_after_current_step(self, api_name: str, result: Mapping[str, Any]) -> AgentObservation:
        current_api_pos = self.api_positions[self.cursor - 1]
        next_api_pos = self.api_positions[self.cursor] if self.cursor < len(self.api_positions) else len(self.history)
        parts = [_render_api_response(api_name, result)]
        following = _render_conversation(self.history[current_api_pos + 1 : next_api_pos])
        if following:
            parts.append(following)
        return AgentObservation(
            "\n".join(parts).strip(),
            {
                "benchmark": "apibank",
                "level": 2,
                "api_step_index": self.cursor,
                "api_steps": len(self.expected_steps),
                "done": self.cursor >= len(self.expected_steps),
            },
        )


def create_apibank_level2_env(
    record: FunctionCallTaskRecord | Mapping[str, Any],
    *,
    config: ApiBankLevel2AdapterConfig | None = None,
) -> ApiBankLevel2Env:
    cfg = config or ApiBankLevel2AdapterConfig()
    return ApiBankLevel2Env(record, official_root=cfg.official_root)


def expected_apibank_level2_actions(record: FunctionCallTaskRecord | Mapping[str, Any]) -> list[ToolAction]:
    metadata = _record_metadata(record)
    steps = metadata.get("apibank_expected_api_steps")
    if not isinstance(steps, list):
        return []
    actions: list[ToolAction] = []
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        args = step.get("param_dict")
        actions.append(ToolAction(name=str(step.get("api_name") or ""), arguments=dict(args) if isinstance(args, Mapping) else {}))
    return actions


def _first_api_index(history: Sequence[Mapping[str, Any]]) -> int:
    for index, item in enumerate(history):
        if item.get("role") == "API":
            return index
    return len(history)


def _history_to_messages(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in history:
        role = str(item.get("role") or "")
        if role == "User":
            messages.append({"role": "user", "content": str(item.get("text") or "")})
        elif role == "AI":
            messages.append({"role": "assistant", "content": str(item.get("text") or "")})
        elif role == "API":
            messages.append({"role": "tool", "name": str(item.get("api_name") or ""), "content": item.get("result")})
    return messages


def _render_conversation(history: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for item in history:
        role = str(item.get("role") or "")
        if role == "User":
            rendered.append(f"User: {item.get('text') or ''}")
        elif role == "AI":
            rendered.append(f"Assistant: {item.get('text') or ''}")
        elif role == "API":
            api_name = str(item.get("api_name") or "")
            rendered.append(_render_api_response(api_name, item.get("result") if isinstance(item.get("result"), Mapping) else {}))
    return "\n".join(part for part in rendered if part.strip()).strip()


def _render_api_response(api_name: str, result: Mapping[str, Any]) -> str:
    return f"API {api_name} Response: {json.dumps(dict(result), ensure_ascii=False, separators=(',', ':'))}"


def _record_metadata(record: FunctionCallTaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.metadata if isinstance(record, FunctionCallTaskRecord) else record.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _record_env(record: FunctionCallTaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    env = record.env if isinstance(record, FunctionCallTaskRecord) else record.get("env")
    return dict(env) if isinstance(env, Mapping) else {}


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _tool_search_outputs_match(actual: Any, expected: Any) -> bool:
    return _tool_search_output_names(actual) == _tool_search_output_names(expected)


def _tool_search_output_names(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        name = value.get("name")
        return [str(name)] if name else []
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and item.get("name"):
                names.append(str(item.get("name")))
        return names
    return []


def _manager_api_names(steps: Sequence[Mapping[str, Any]], current_name: str) -> list[str]:
    names = {current_name}
    for step in steps:
        name = step.get("api_name")
        if isinstance(name, str) and name != "ToolSearcher":
            names.add(name)
    return sorted(names)


def _api_name_from_source_path(path: Path) -> str | None:
    target = path.stem
    parts = target.split("_")
    if not parts:
        return None
    return "".join(part.capitalize() for part in parts if part)


def _api_search_text(item: Mapping[str, Any]) -> str:
    name = str(item.get("name") or "")
    words = re.sub("([a-z0-9])([A-Z])", r"\1 \2", name).lower()
    return f"{words} {item.get('description') or ''}".strip()


def _strip_search_internal_fields(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "name": str(item.get("name") or ""),
        "description": str(item.get("description") or ""),
        "input_parameters": dict(item.get("input_parameters") or {}),
        "output_parameters": dict(item.get("output_parameters") or {}),
    }
    if "desc_for_search" in item:
        payload["desc_for_search"] = str(item.get("desc_for_search") or "")
    return payload


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _search_score(tokens: set[str], item: Mapping[str, Any]) -> tuple[int, int, str]:
    search_text = _api_search_text(item)
    search_tokens = _tokenize(search_text)
    overlap = len(tokens & search_tokens)
    substring_bonus = sum(1 for token in tokens if token and token in search_text.lower())
    return overlap, substring_bonus, str(item.get("name") or "")


__all__ = [
    "ApiBankLevel2Env",
    "ApiBankLevel2AdapterConfig",
    "OFFICIAL_APIBANK_SOURCE",
    "TOOL_SEARCHER_SCHEMA",
    "create_apibank_level2_env",
    "expected_apibank_level2_actions",
    "load_apibank_level2_rows_from_source_dir",
    "require_apibank_level2_assets",
]
