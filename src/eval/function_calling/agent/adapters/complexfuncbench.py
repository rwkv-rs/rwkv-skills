from __future__ import annotations

"""Official ComplexFuncBench agent adapter.

This adapter keeps ComplexFuncBench in the multi-step agent pipeline and
delegates turn comparison and sandbox observations to the official
ComplexFuncBench repository.
"""

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.function_calling.agent.env import AgentObservation, AgentStepResult
from src.eval.function_calling.common.action import ToolAction

OFFICIAL_COMPLEXFUNC_SOURCE = "zai-org/ComplexFuncBench"
DEFAULT_COMPLEXFUNC_MAX_ROWS = 0
COMPLEXFUNCBENCH_TASK_PREFIX = "complexfuncbench_subset__"

COMPLEXFUNCBENCH_FINAL_SCHEMA: dict[str, Any] = {
    "name": "final_answer",
    "description": "Finish the ComplexFuncBench task with the final natural language response.",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "Final response to the user after all required function calls.",
            }
        },
        "required": ["answer"],
    },
}

_REQUIRED_OFFICIAL_FILES: tuple[str, ...] = (
    "runner/base_runner.py",
    "runner/response_runner.py",
    "utils/compare_method.py",
    "utils/rapidapi.py",
    "utils/tool_info.json",
    "utils/exact_match_values.json",
    "models/gpt.py",
    "prompts/compare.py",
    "prompts/response.py",
    "prompts/prompts.py",
)


@dataclass(frozen=True, slots=True)
class ComplexFuncBenchOfficialMetrics:
    success_rate: float
    call_accuracy: float
    completeness: float
    correctness: float
    response_eval_samples: int


class _NullLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        return None


@dataclass(slots=True)
class OfficialComplexFuncBenchSandbox:
    official_root: Path
    enable_response_eval: bool = True
    logger: Any = None

    def __post_init__(self) -> None:
        self.official_root = require_complexfuncbench_official_root(self.official_root)
        if self.logger is None:
            self.logger = _NullLogger()

    def create_model_runner(self) -> Any:
        with _official_import_context(self.official_root):
            module = importlib.import_module("runner.base_runner")
            return module.ModelRunner(SimpleNamespace(), self.logger)

    def run_response_eval(self, official_row: Mapping[str, Any], final_response: str) -> dict[str, Any] | None:
        if not self.enable_response_eval:
            return None
        with _official_import_context(self.official_root):
            module = importlib.import_module("runner.response_runner")
            runner = module.RespEvalRunner(SimpleNamespace(), self.logger)
            return runner.run(dict(official_row), final_response)


class ComplexFuncBenchOfficialEnv:
    """Multi-step environment backed by official ComplexFuncBench comparison.

    Model outputs may contain one JSON tool-call object or a JSON array of
    function calls for the current assistant turn. When all official function
    calls have matched, the model should call the synthetic `final_answer` tool
    with the final user-facing response.
    """

    def __init__(
        self,
        record: FunctionCallTaskRecord | Mapping[str, Any],
        *,
        sandbox: OfficialComplexFuncBenchSandbox | None = None,
    ) -> None:
        self.record = record
        metadata = _record_metadata(record)
        env = _record_env(record)
        self.official_root = Path(
            str(
                env.get("official_root")
                or metadata.get("complexfuncbench_official_root")
                or metadata.get("official_root")
                or "."
            )
        ).expanduser()
        self.response_eval_enabled = bool(env.get("response_eval", True))
        self.functions = _list_of_dicts(metadata.get("complexfuncbench_functions") or _record_tools(record))
        self.functions = [
            item for item in self.functions
            if str(item.get("name") or "") != COMPLEXFUNCBENCH_FINAL_SCHEMA["name"]
        ]
        self.conversations = _list_of_dicts(metadata.get("complexfuncbench_conversations"))
        if not self.conversations:
            raise ValueError("ComplexFuncBench record is missing official conversations")
        if not self.functions:
            raise ValueError("ComplexFuncBench record is missing official functions")
        self.official_id = str(metadata.get("official_id") or _record_task_id(record))
        self.official_row = {
            "id": self.official_id,
            "functions": copy.deepcopy(self.functions),
            "conversations": copy.deepcopy(self.conversations),
        }
        self.sandbox = sandbox or OfficialComplexFuncBenchSandbox(
            self.official_root,
            enable_response_eval=self.response_eval_enabled,
        )
        self.runner: Any | None = None
        self.messages: list[dict[str, Any]] = []
        self.done = False
        self.error_message: Any = None
        self.final_response = ""
        self.last_resp_eval: dict[str, Any] | None = None

    def reset(self) -> AgentObservation:
        self.runner = self.sandbox.create_model_runner()
        self.runner.CompareClass.add_free_function(self.conversations)
        self.runner.init_golden(self.conversations)
        self.messages = [{"role": "user", "content": self._initial_query()}]
        self.done = False
        self.error_message = None
        self.final_response = ""
        self.last_resp_eval = None
        return AgentObservation(
            self._initial_query(),
            {
                "benchmark": "complexfuncbench",
                "official_id": self.official_id,
                "api_step_index": 0,
                "api_steps": len(self.runner.fc_chain),
                "available_tools": [tool["name"] for tool in self._tool_schemas()],
                "allows_parallel_tool_calls": True,
                "requires_final_answer": True,
            },
        )

    def step(self, action: ToolAction) -> AgentStepResult:
        return self.step_many([action])

    def step_many(self, actions: Sequence[ToolAction]) -> AgentStepResult:
        if self.runner is None:
            raise RuntimeError("ComplexFuncBench env must be reset before step")
        if self.done:
            return AgentStepResult(
                AgentObservation("ComplexFuncBench task is already done.", {"done": True}),
                done=True,
                score=1.0 if self._is_success_message(self.error_message) else 0.0,
                success=self._is_success_message(self.error_message),
                details=self._run_details("already_done"),
            )
        if not actions:
            return self._finish_with_error({"error_type": "empty_action", "content": "No tool call was provided."})

        if any(action.name == COMPLEXFUNCBENCH_FINAL_SCHEMA["name"] for action in actions):
            if len(actions) != 1:
                return self._finish_with_error(
                    {
                        "error_type": "invalid_final_answer_turn",
                        "content": "final_answer must be the only action in its turn.",
                    }
                )
            return self._final_answer(actions[0])

        function_calls = [{"name": action.name, "arguments": dict(action.arguments)} for action in actions]
        self.messages.append({"role": "assistant", "function_call": copy.deepcopy(function_calls)})
        try:
            error_message, success_map, success_matched, format_error = (
                self.runner.CompareClass.compare_turn_prediction(
                    self.functions,
                    self.messages[:-1],
                    copy.deepcopy(function_calls),
                    self.runner.golden_fcs,
                    self.runner.golden_obs,
                )
            )
        except Exception as exc:  # noqa: BLE001 - official sandbox failures are benchmark failures.
            return self._finish_with_error(
                {
                    "error_type": "official_sandbox_error",
                    "content": repr(exc),
                }
            )

        self.error_message = error_message
        if len(success_map) == 0 and format_error == {}:
            return self._finish_with_error(error_message or {"error_type": "func_error", "content": "No official call matched."})

        self.runner.correct_count += len(success_map)
        observations: list[Any] = []
        for index, _call in enumerate(function_calls):
            if index in success_map:
                observations.append(success_map[index])
            elif index in format_error:
                observations.append(format_error[index])
            else:
                observations.append(copy.deepcopy(self.runner.unexpect_call_resp))

        self.runner.process_matches(success_matched)
        self.messages.append({"role": "observation", "content": copy.deepcopy(observations)})
        content = _json_dumps(observations)
        if not self.runner.golden_fcs and self.runner.turn_id >= len(self.runner.fc_chain):
            content += "\nAll required official function calls matched. Call final_answer with your final response."
        return AgentStepResult(
            AgentObservation(
                f"Official sandbox observation: {content}",
                {
                    "benchmark": "complexfuncbench",
                    "official_id": self.official_id,
                    "api_step_index": min(self.runner.turn_id, len(self.runner.fc_chain)),
                    "api_steps": len(self.runner.fc_chain),
                    "matched_call_count": len(success_map),
                },
            ),
            done=False,
            details={
                "finish_reason": "official_observation",
                "matched_call_count": len(success_map),
                "format_error": format_error,
                "error_message": error_message,
                **self._run_details("official_observation"),
            },
        )

    def _final_answer(self, action: ToolAction) -> AgentStepResult:
        answer = str(
            action.arguments.get("answer")
            or action.arguments.get("response")
            or action.arguments.get("content")
            or ""
        ).strip()
        self.final_response = answer
        self.messages.append({"role": "assistant", "content": answer})
        error_info = self.error_message if self.error_message else None
        try:
            messages, message, success_turn, correct_count = self.runner.return_result(self.messages, error_info)
        except Exception as exc:  # noqa: BLE001
            return self._finish_with_error({"error_type": "official_return_result_error", "content": repr(exc)})
        self.messages = [dict(item) for item in messages if isinstance(item, Mapping)]
        self.error_message = message
        if correct_count is not None:
            self.runner.correct_count = correct_count
        success = self._is_success_message(message)
        resp_eval = None
        if answer:
            try:
                resp_eval = self.sandbox.run_response_eval(self.official_row, answer)
            except Exception as exc:  # noqa: BLE001
                resp_eval = {
                    "complete": {"score": -1, "reason": f"Official response eval failed: {exc!r}"},
                    "correct": {"score": -1, "reason": f"Official response eval failed: {exc!r}"},
                }
        self.last_resp_eval = resp_eval
        self.done = True
        details = {
            "finish_reason": "final_answer",
            "message": message,
            "count_dict": self._count_dict(success_turn_num=success_turn),
            "resp_eval": resp_eval,
            "final_response": answer,
            **self._run_details("final_answer"),
        }
        return AgentStepResult(
            AgentObservation("Final response recorded.", {"benchmark": "complexfuncbench", "done": True}),
            done=True,
            score=1.0 if success else 0.0,
            success=success,
            details=details,
        )

    def _finish_with_error(self, message: Any) -> AgentStepResult:
        self.done = True
        self.error_message = message
        success_turn = self.runner.get_success_turn(self.runner.golden_fcs, self.runner.fc_chain) if self.runner else 0
        details = {
            "finish_reason": "official_error",
            "message": message,
            "count_dict": self._count_dict(success_turn_num=success_turn),
            **self._run_details("official_error"),
        }
        return AgentStepResult(
            AgentObservation(_message_text(message), {"benchmark": "complexfuncbench", "error": True}),
            done=True,
            score=0.0,
            success=False,
            details=details,
        )

    def _initial_query(self) -> str:
        first = self.conversations[0] if self.conversations else {}
        return str(first.get("content") or first.get("text") or _record_instruction(self.record) or "")

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(tool) for tool in self.functions] + [copy.deepcopy(COMPLEXFUNCBENCH_FINAL_SCHEMA)]

    def _count_dict(self, *, success_turn_num: int | None = None) -> dict[str, Any]:
        if self.runner is None:
            return {
                "success_turn_num": 0,
                "total_turn_num": 0,
                "correct_call_num": 0,
                "total_call_num": 0,
                "real_turn_num": 0,
            }
        total_call_num = sum(len(turn) for turn in self.runner.fc_chain)
        real_turn_num = sum(1 for turn in self.messages if "function_call" in turn)
        return {
            "success_turn_num": int(success_turn_num if success_turn_num is not None else self.runner.turn_id),
            "total_turn_num": len(self.runner.fc_chain),
            "correct_call_num": int(self.runner.correct_count),
            "total_call_num": total_call_num,
            "real_turn_num": real_turn_num,
        }

    def _run_details(self, finish_reason: str) -> dict[str, Any]:
        count_dict = self._count_dict()
        call_accuracy = (
            count_dict["correct_call_num"] / count_dict["total_call_num"]
            if count_dict["total_call_num"]
            else 0.0
        )
        return {
            "official_complexfuncbench_source": OFFICIAL_COMPLEXFUNC_SOURCE,
            "official_id": self.official_id,
            "finish_reason": finish_reason,
            "count_dict": count_dict,
            "call_accuracy": call_accuracy,
            "resp_eval": self.last_resp_eval,
        }

    @staticmethod
    def _is_success_message(message: Any) -> bool:
        return message == "Success."


def create_complexfuncbench_official_env(
    record: FunctionCallTaskRecord | Mapping[str, Any],
) -> ComplexFuncBenchOfficialEnv:
    return ComplexFuncBenchOfficialEnv(record)


def load_complexfuncbench_rows_from_source(
    path: str | Path,
    *,
    official_root: str | Path,
    dataset_name: str = "complexfuncbench_subset",
    max_rows: int = DEFAULT_COMPLEXFUNC_MAX_ROWS,
) -> list[dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    root = Path(official_root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(_read_json_or_jsonl_items(source_path)):
        if not isinstance(item, Mapping):
            continue
        converted = _convert_official_row(
            item,
            index=index,
            dataset_name=dataset_name,
            source_path=source_path,
            official_root=root,
        )
        if converted is None:
            continue
        rows.append(converted)
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def summarize_complexfuncbench_official_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> ComplexFuncBenchOfficialMetrics:
    total = 0
    success_count = 0
    correct_calls = 0
    total_calls = 0
    complete_score = 0.0
    correct_score = 0.0
    complete_samples = 0
    correct_samples = 0

    for payload in payloads:
        details = _agent_final_env_details(payload)
        if not details:
            continue
        count_dict = details.get("count_dict")
        if not isinstance(count_dict, Mapping):
            continue
        total += 1
        if details.get("message") == "Success." or payload.get("success") is True:
            success_count += 1
        correct_calls += _int_value(count_dict.get("correct_call_num"))
        total_calls += _int_value(count_dict.get("total_call_num"))
        resp_eval = details.get("resp_eval")
        if isinstance(resp_eval, Mapping):
            complete = resp_eval.get("complete")
            if isinstance(complete, Mapping) and complete.get("score") in {0, 1, 2}:
                complete_score += float(complete["score"])
                complete_samples += 1
            correct = resp_eval.get("correct")
            if isinstance(correct, Mapping) and correct.get("score") in {0, 1, 2}:
                correct_score += float(correct["score"])
                correct_samples += 1

    return ComplexFuncBenchOfficialMetrics(
        success_rate=success_count / total if total else 0.0,
        call_accuracy=correct_calls / total_calls if total_calls else 0.0,
        completeness=complete_score / complete_samples if complete_samples else 0.0,
        correctness=correct_score / correct_samples if correct_samples else 0.0,
        response_eval_samples=min(complete_samples, correct_samples),
    )


def require_complexfuncbench_official_root(root: str | Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    missing = [relative for relative in _REQUIRED_OFFICIAL_FILES if not (resolved / relative).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"ComplexFuncBench official sandbox is incomplete at {resolved}; missing: {missing_text}"
        )
    return resolved


def _convert_official_row(
    item: Mapping[str, Any],
    *,
    index: int,
    dataset_name: str,
    source_path: Path,
    official_root: Path,
) -> dict[str, Any] | None:
    conversations = _list_of_dicts(item.get("conversations"))
    functions = _list_of_dicts(item.get("functions"))
    fc_chain, _obs_chain = _official_chains(conversations)
    if not conversations or not functions or not fc_chain:
        return None
    official_id = str(item.get("id") or item.get("task_id") or index)
    tools = [copy.deepcopy(tool) for tool in functions] + [copy.deepcopy(COMPLEXFUNCBENCH_FINAL_SCHEMA)]
    return {
        "task_id": f"{dataset_name}__{official_id}",
        "instruction": str(conversations[0].get("content") or conversations[0].get("text") or ""),
        "messages": [{"role": "user", "content": str(conversations[0].get("content") or "")}],
        "tools": tools,
        "expected_tool_calls": [],
        "env": {
            "type": "complexfuncbench_official",
            "official_root": str(official_root),
            "source_path": str(source_path),
            "response_eval": True,
        },
        "scorer": {"type": "complexfuncbench_official"},
        "max_steps": max(4, len(fc_chain) + 3),
        "metadata": {
            "source_format": "official_complexfuncbench",
            "official_source": OFFICIAL_COMPLEXFUNC_SOURCE,
            "official_id": official_id,
            "complexfuncbench_official_root": str(official_root),
            "complexfuncbench_source_path": str(source_path),
            "complexfuncbench_functions": functions,
            "complexfuncbench_conversations": conversations,
            "complexfuncbench_total_turn_num": len(fc_chain),
            "complexfuncbench_total_call_num": sum(len(turn) for turn in fc_chain),
            "category": item.get("category") or item.get("type") or "",
        },
    }


def _official_chains(
    conversations: Sequence[Mapping[str, Any]],
) -> tuple[list[list[dict[str, Any]]], list[Any]]:
    fc_chain: list[list[dict[str, Any]]] = []
    obs_chain: list[Any] = []
    for turn in conversations:
        if "function_call" in turn:
            calls = _normalize_tool_calls(turn.get("function_call"))
            if calls:
                fc_chain.append(calls)
        elif str(turn.get("role") or "").lower() == "observation":
            obs_chain.append(turn.get("content"))
    return fc_chain, obs_chain


def _normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in _coerce_list(raw):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or item.get("tool_name") or item.get("function_name") or "").strip()
        arguments = item.get("arguments")
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                arguments = parsed if isinstance(parsed, Mapping) else {}
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        if name:
            calls.append({"name": name, "arguments": dict(arguments)})
    return calls


def _read_json_or_jsonl_items(path: Path) -> list[Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if "Extra data" not in str(exc):
            raise
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return [payload]
    raise ValueError(f"unsupported JSON payload: {path}")


@contextmanager
def _official_import_context(root: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    root_text = str(root)
    inserted = False
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
        inserted = True
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass


def _agent_final_env_details(payload: Mapping[str, Any]) -> dict[str, Any]:
    details = payload.get("agent_details")
    if isinstance(details, Mapping):
        final = details.get("final_env_details")
        if isinstance(final, Mapping):
            return dict(final)
    context = payload.get("context")
    if isinstance(context, Mapping):
        details = context.get("agent_details")
        if isinstance(details, Mapping):
            final = details.get("final_env_details")
            if isinstance(final, Mapping):
                return dict(final)
    return {}


def _record_metadata(record: FunctionCallTaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.metadata if isinstance(record, FunctionCallTaskRecord) else record.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _record_env(record: FunctionCallTaskRecord | Mapping[str, Any]) -> dict[str, Any]:
    env = record.env if isinstance(record, FunctionCallTaskRecord) else record.get("env")
    return dict(env) if isinstance(env, Mapping) else {}


def _record_tools(record: FunctionCallTaskRecord | Mapping[str, Any]) -> list[dict[str, Any]]:
    tools = record.tools if isinstance(record, FunctionCallTaskRecord) else record.get("tools")
    return _list_of_dicts(tools)


def _record_task_id(record: FunctionCallTaskRecord | Mapping[str, Any]) -> str:
    return str(record.task_id if isinstance(record, FunctionCallTaskRecord) else record.get("task_id") or "")


def _record_instruction(record: FunctionCallTaskRecord | Mapping[str, Any]) -> str:
    return str(record.instruction if isinstance(record, FunctionCallTaskRecord) else record.get("instruction") or "")


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    return []


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    return _json_dumps(message)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


__all__ = [
    "COMPLEXFUNCBENCH_FINAL_SCHEMA",
    "COMPLEXFUNCBENCH_TASK_PREFIX",
    "DEFAULT_COMPLEXFUNC_MAX_ROWS",
    "OFFICIAL_COMPLEXFUNC_SOURCE",
    "ComplexFuncBenchOfficialEnv",
    "ComplexFuncBenchOfficialMetrics",
    "OfficialComplexFuncBenchSandbox",
    "create_complexfuncbench_official_env",
    "load_complexfuncbench_rows_from_source",
    "require_complexfuncbench_official_root",
    "summarize_complexfuncbench_official_payloads",
]
