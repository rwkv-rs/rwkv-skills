from __future__ import annotations

"""ComplexFuncBench official multi-turn function-calling runner."""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import types
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    assistant_json_prefix,
    coerce_json_function_call_payloads,
    extract_json_call_value_text,
)
from src.eval.function_calling.tool_router import ToolRouteResult, route_tools_for_prompt, tool_routing_config_from_args
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext
    from src.eval.function_calling.tool_router import ToolRoutingConfig

OFFICIAL_COMPLEXFUNC_SOURCE = "zai-org/ComplexFuncBench"
DEFAULT_COMPLEXFUNC_MAX_ROWS = 0
COMPLEXFUNCBENCH_TASK_PREFIX = "complexfuncbench_official__"

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

COMPLEXFUNCBENCH_STOP_SUFFIXES: tuple[str, ...] = (
    *JSON_CALL_STOP_SUFFIXES,
    "\nTool call:",
    "\nEnvironment:",
    "\nCurrent observation:",
    "\nTrajectory:",
    "\nStep:",
)

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

_OFFICIAL_IMPORT_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class ComplexFuncBenchRecord:
    task_id: str
    instruction: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    expected_tool_calls: list[dict[str, Any]]
    env: dict[str, Any]
    scorer: dict[str, Any]
    max_steps: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ToolAction":
        arguments = payload.get("arguments")
        return cls(
            name=str(payload.get("name") or ""),
            arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
        )


@dataclass(frozen=True, slots=True)
class AgentObservation:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    observation: AgentObservation
    done: bool = False
    score: float | None = None
    success: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComplexFuncBenchOfficialMetrics:
    success_rate: float
    call_accuracy: float
    completeness: float
    correctness: float
    response_eval_samples: int


@dataclass(frozen=True, slots=True)
class ComplexFuncBenchEpisodeResult:
    stages: list[StageRecord]
    completions: list[str]
    events: list[dict[str, Any]]
    tool_routes: list[dict[str, Any]]
    format_bridges: list[dict[str, Any]]
    success: bool
    score: float | None
    details: dict[str, Any]
    final_answer: str

    @property
    def final_response(self) -> str:
        return self.final_answer

    @property
    def count_dict(self) -> dict[str, Any]:
        final_details = self.details.get("final_env_details")
        if isinstance(final_details, Mapping):
            count_dict = final_details.get("count_dict")
            if isinstance(count_dict, Mapping):
                return dict(count_dict)
        return {}

    @property
    def call_accuracy(self) -> float:
        final_details = self.details.get("final_env_details")
        if isinstance(final_details, Mapping):
            try:
                return float(final_details.get("call_accuracy", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        count_dict = self.count_dict
        total = _int_value(count_dict.get("total_call_num"))
        return _int_value(count_dict.get("correct_call_num")) / total if total else 0.0


ComplexFuncBenchCall = ToolAction
ComplexFuncBenchMetrics = ComplexFuncBenchOfficialMetrics


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
    offline_compare: bool = False
    openai_connection: dict[str, Any] | None = None
    logger: Any = None

    def __post_init__(self) -> None:
        self.official_root = require_complexfuncbench_official_root(self.official_root)
        if self.openai_connection is None:
            self.openai_connection = _resolve_complexfuncbench_openai_connection()
        if self.logger is None:
            self.logger = _NullLogger()

    def create_model_runner(self) -> Any:
        with _official_import_context(self.official_root, openai_connection=self.openai_connection):
            module = importlib.import_module("runner.base_runner")
            _patch_official_gpt_model_aliases(self.openai_connection)
            runner = module.ModelRunner(SimpleNamespace(), self.logger)
            if self.offline_compare:
                _patch_official_compare_for_offline(runner)
            return runner

    def run_response_eval(self, official_row: Mapping[str, Any], final_response: str) -> dict[str, Any] | None:
        if not self.enable_response_eval:
            return None
        with _official_import_context(self.official_root, openai_connection=self.openai_connection):
            module = importlib.import_module("runner.response_runner")
            _patch_official_gpt_model_aliases(self.openai_connection)
            runner = module.RespEvalRunner(SimpleNamespace(), self.logger)
            return runner.run(dict(official_row), final_response)


class ComplexFuncBenchOfficialEnv:
    def __init__(
        self,
        record: ComplexFuncBenchRecord | Mapping[str, Any],
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
        self.offline_compare = bool(env.get("offline_compare", False))
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
            offline_compare=self.offline_compare,
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

        function_calls = _actions_to_official_calls(actions)
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
        except Exception as exc:  # noqa: BLE001
            return self._finish_with_error({"error_type": "official_sandbox_error", "content": repr(exc)})

        self.error_message = error_message
        if len(success_map) == 0 and format_error == {}:
            return self._finish_with_error(
                error_message or {"error_type": "func_error", "content": "No official call matched."}
            )

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
                "internal_tool_actions": _tool_action_payloads(actions),
                "official_sandbox_calls": copy.deepcopy(function_calls),
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
            "official_response_eval_input": {
                "official_id": self.official_id,
                "final_response": answer,
            },
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

    def failure_details(self, *, finish_reason: str, message: Any) -> dict[str, Any]:
        self.error_message = message
        return {
            "finish_reason": finish_reason,
            "message": message,
            **self._run_details(finish_reason),
        }

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


class ComplexFuncBenchLocalEnv:
    """Portable ComplexFuncBench evaluator backed by stored golden turns."""

    def __init__(self, record: ComplexFuncBenchRecord | Mapping[str, Any]) -> None:
        self.record = record
        metadata = _record_metadata(record)
        self.functions = _list_of_dicts(metadata.get("complexfuncbench_functions") or _record_tools(record))
        self.functions = [
            item for item in self.functions
            if str(item.get("name") or "") != COMPLEXFUNCBENCH_FINAL_SCHEMA["name"]
        ]
        self.conversations = _list_of_dicts(metadata.get("complexfuncbench_conversations"))
        self.fc_chain, self.obs_chain = _official_chains(self.conversations)
        if not self.conversations:
            raise ValueError("ComplexFuncBench record is missing official conversations")
        if not self.functions:
            raise ValueError("ComplexFuncBench record is missing official functions")
        if not self.fc_chain:
            raise ValueError("ComplexFuncBench record is missing official function-call turns")
        self.official_id = str(metadata.get("official_id") or _record_task_id(record))
        self.messages: list[dict[str, Any]] = []
        self.turn_id = 0
        self.correct_count = 0
        self.done = False
        self.error_message: Any = None
        self.final_response = ""

    def reset(self) -> AgentObservation:
        self.messages = [{"role": "user", "content": self._initial_query()}]
        self.turn_id = 0
        self.correct_count = 0
        self.done = False
        self.error_message = None
        self.final_response = ""
        return AgentObservation(
            self._initial_query(),
            {
                "benchmark": "complexfuncbench",
                "runtime": "local_golden_conversation",
                "official_id": self.official_id,
                "api_step_index": 0,
                "api_steps": len(self.fc_chain),
                "available_tools": [tool["name"] for tool in self._tool_schemas()],
                "allows_parallel_tool_calls": True,
                "requires_final_answer": True,
            },
        )

    def step_many(self, actions: Sequence[ToolAction]) -> AgentStepResult:
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
        if self.turn_id >= len(self.fc_chain):
            return self._finish_with_error(
                {"error_type": "unexpected_extra_call", "content": "All official calls were already matched."}
            )

        expected = self.fc_chain[self.turn_id]
        if not _function_call_turns_equal(function_calls, expected):
            return self._finish_with_error(
                {
                    "error_type": "func_error",
                    "content": "Predicted calls did not match the next official turn.",
                    "predicted": function_calls,
                    "expected": expected,
                }
            )

        self.correct_count += len(expected)
        observation = self.obs_chain[self.turn_id] if self.turn_id < len(self.obs_chain) else []
        self.turn_id += 1
        self.messages.append({"role": "observation", "content": copy.deepcopy(observation)})
        content = _json_dumps(observation)
        if self.turn_id >= len(self.fc_chain):
            content += "\nAll required official function calls matched. Call final_answer with your final response."
        return AgentStepResult(
            AgentObservation(
                f"Official sandbox observation: {content}",
                {
                    "benchmark": "complexfuncbench",
                    "runtime": "local_golden_conversation",
                    "official_id": self.official_id,
                    "api_step_index": min(self.turn_id, len(self.fc_chain)),
                    "api_steps": len(self.fc_chain),
                    "matched_call_count": len(expected),
                },
            ),
            done=False,
            details={
                "finish_reason": "local_observation",
                "matched_call_count": len(expected),
                **self._run_details("local_observation"),
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
        if self.turn_id >= len(self.fc_chain) and answer:
            self.error_message = "Success."
            success = True
        else:
            self.error_message = {"error_type": "stop_early", "content": "Final answer before all calls matched."}
            success = False
        self.done = True
        details = {
            "finish_reason": "final_answer",
            "message": self.error_message,
            "count_dict": self._count_dict(success_turn_num=self.turn_id),
            "resp_eval": None,
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
        details = {
            "finish_reason": "local_error",
            "message": message,
            "count_dict": self._count_dict(success_turn_num=self.turn_id),
            **self._run_details("local_error"),
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
        total_call_num = sum(len(turn) for turn in self.fc_chain)
        real_turn_num = sum(1 for turn in self.messages if "function_call" in turn)
        return {
            "success_turn_num": int(success_turn_num if success_turn_num is not None else self.turn_id),
            "total_turn_num": len(self.fc_chain),
            "correct_call_num": int(self.correct_count),
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
            "runtime": "local_golden_conversation",
            "finish_reason": finish_reason,
            "count_dict": count_dict,
            "call_accuracy": call_accuracy,
            "resp_eval": None,
        }

    @staticmethod
    def _is_success_message(message: Any) -> bool:
        return message == "Success."


def create_complexfuncbench_env(record: ComplexFuncBenchRecord | Mapping[str, Any]) -> ComplexFuncBenchOfficialEnv:
    env = _record_env(record)
    env_type = str(env.get("type") or "").strip().lower()
    official_root = str(env.get("official_root") or "").strip()
    if env_type == "complexfuncbench_official" and official_root:
        return ComplexFuncBenchOfficialEnv(record)
    task_id = _record_task_id(record)
    raise ValueError(
        "ComplexFuncBench must run with the official sandbox. "
        f"Record {task_id or '<unknown>'} has env.type={env_type!r} and no usable official_root. "
        "Refresh data/complexfuncbench_official with RWKV_COMPLEXFUNC_OFFICIAL_ROOT "
        "or RWKV_COMPLEXFUNCBENCH_OFFICIAL_ROOT pointing at the official ComplexFuncBench repository."
    )


def load_complexfuncbench_manifest_records(path: str | Path, sample_limit: int | None = None) -> list[ComplexFuncBenchRecord]:
    target = Path(path)
    records: list[ComplexFuncBenchRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{target}: ComplexFuncBench row must be a JSON object")
            records.append(_record_from_payload(payload, index=index))
            if sample_limit is not None and sample_limit > 0 and len(records) >= sample_limit:
                break
    return records


def load_complexfuncbench_rows_from_source(
    path: str | Path,
    *,
    official_root: str | Path | None = None,
    dataset_name: str = "complexfuncbench_official",
    max_rows: int = DEFAULT_COMPLEXFUNC_MAX_ROWS,
    response_eval: bool = True,
) -> list[dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    if official_root is None:
        raise ValueError(
            "complexfuncbench_official requires RWKV_COMPLEXFUNC_OFFICIAL_ROOT or "
            "RWKV_COMPLEXFUNCBENCH_OFFICIAL_ROOT so the official sandbox and response evaluation can run."
        )
    root = require_complexfuncbench_official_root(official_root)
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
            response_eval=response_eval,
        )
        if converted is None:
            continue
        rows.append(converted)
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def build_complexfuncbench_prompt(
    record: ComplexFuncBenchRecord,
    events: Sequence[Mapping[str, Any]],
    observation: AgentObservation,
    step: int,
    *,
    tool_route: ToolRouteResult | None = None,
) -> str:
    selected_tools = tool_route.selected_tools if tool_route is not None else record.tools
    tools_json = json.dumps(selected_tools or [], ensure_ascii=False, indent=2)
    trajectory = _render_agent_trajectory(events)
    current = str(observation.content or "")
    route_note = ""
    if tool_route is not None:
        route_note = (
            "Tool router trace:\n"
            + json.dumps(tool_route.trace_payload(), ensure_ascii=False, separators=(",", ":"))
            + "\n\n"
        )
    return (
        "You are running the official ComplexFuncBench function-calling task.\n"
        "Return only JSON and no extra text.\n"
        'For one call, use {"name":"ToolName","arguments":{"arg":"value"}}.\n'
        "For multiple calls in the same assistant turn, return a JSON array of those objects.\n"
        "After all official sandbox observations indicate the required calls are complete, "
        'call {"name":"final_answer","arguments":{"answer":"..."}}.\n'
        "Available tools:\n"
        f"{tools_json}\n\n"
        f"{route_note}"
        f"Trajectory:\n{trajectory}\n\n"
        f"Current observation:\n{current}\n\n"
        f"Step: {step}\n\n"
        + assistant_json_prefix()
    )


def build_complexfuncbench_system_prompt(
    tools: Sequence[Mapping[str, Any]],
    *,
    total_tool_count: int | None = None,
) -> str:
    route_note = ""
    if total_tool_count is not None and int(total_tool_count) != len(tools):
        route_note = f"\nOnly {len(tools)} of {int(total_tool_count)} tools are visible this turn."
    return _normalize_prompt_text(
        "You are running the official ComplexFuncBench function-calling task.\n"
        "Return only JSON and no extra text.\n"
        'For one call, use {"name":"ToolName","arguments":{"arg":"value"}}.\n'
        "For multiple calls in the same assistant turn, return a JSON array of those objects.\n"
        f"{route_note}\n"
        "Available tools:\n"
        + json.dumps(list(tools), ensure_ascii=False, indent=2)
    )


def parse_complexfuncbench_tool_calls(response: str) -> list[ToolAction]:
    candidate = extract_json_call_value_text(str(response or ""))
    payload = json.loads(candidate)
    if payload == []:
        return []
    calls = coerce_json_function_call_payloads(payload, context_label="ComplexFuncBench tool-call selection")
    return [ToolAction.from_mapping(call) for call in calls]


def parse_complexfuncbench_calls(response: str) -> list[ToolAction]:
    return parse_complexfuncbench_tool_calls(response)


def build_complexfuncbench_format_bridge(
    response: str,
    actions: Sequence[ToolAction],
    *,
    parse_error: str | None = None,
) -> dict[str, Any]:
    """Describe the exact conversion from RWKV text to official sandbox call dicts."""

    raw_output = str(response or "")
    json_payload_text = ""
    json_payload: Any = None
    detected_parse_error: str | None = None
    try:
        json_payload_text = extract_json_call_value_text(raw_output)
        json_payload = json.loads(json_payload_text)
    except Exception as exc:  # noqa: BLE001 - this is diagnostic payload, not control flow.
        detected_parse_error = str(exc)
    internal_actions = _tool_action_payloads(actions)
    official_calls = _actions_to_official_calls(actions)
    return {
        "format_version": 1,
        "rwkv_output_format": "json_tool_call_object_or_array",
        "rwkv_raw_output": raw_output,
        "json_payload_text": json_payload_text,
        "json_payload": json_payload,
        "internal_format": "ToolAction(name, arguments)",
        "internal_tool_actions": internal_actions,
        "official_sandbox_format": "list[dict(name, arguments)]",
        "official_sandbox_calls": official_calls,
        "parse_error": parse_error or detected_parse_error,
    }


def normalize_complexfuncbench_source_row(
    item: Mapping[str, Any],
    *,
    index: int,
    dataset_name: str = "complexfuncbench_official",
    official_root: str | Path | None = None,
    response_eval: bool = True,
) -> dict[str, Any] | None:
    if official_root is None:
        raise ValueError("normalize_complexfuncbench_source_row requires an official ComplexFuncBench root")
    return _convert_official_row(
        item,
        index=index,
        dataset_name=dataset_name,
        source_path=None,
        official_root=require_complexfuncbench_official_root(official_root),
        response_eval=response_eval,
    )


def summarize_complexfuncbench_official_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> ComplexFuncBenchOfficialMetrics:
    total = len(payloads)
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
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                total_calls += _int_value(metadata.get("complexfuncbench_total_call_num"))
            continue
        count_dict = details.get("count_dict")
        if not isinstance(count_dict, Mapping):
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                total_calls += _int_value(metadata.get("complexfuncbench_total_call_num"))
            continue
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


def summarize_complexfuncbench_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> ComplexFuncBenchOfficialMetrics:
    return summarize_complexfuncbench_official_payloads(payloads)


def require_complexfuncbench_official_root(root: str | Path) -> Path:
    resolved = Path(root).expanduser().resolve()
    missing = [relative for relative in _REQUIRED_OFFICIAL_FILES if not (resolved / relative).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"ComplexFuncBench official sandbox is incomplete at {resolved}; missing: {missing_text}"
        )
    return resolved


def _run_complexfuncbench(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_complexfuncbench_manifest_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        run.dataset_slug,
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]
    if getattr(args, "complexfuncbench_disable_response_eval", False):
        raise ValueError("ComplexFuncBench must use official response evaluation; remove --complexfuncbench-disable-response-eval")
    if getattr(args, "complexfuncbench_offline_compare", False):
        raise ValueError("ComplexFuncBench must use the official sandbox comparison; remove --complexfuncbench-offline-compare")
    if not records:
        raise ValueError("ComplexFuncBench manifest is empty")
    _validate_complexfuncbench_official_records(records)

    plan = _resolve_function_calling_plan(
        run.dataset_slug,
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    tool_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    if tool_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    tool_sampling = clamp_function_calling_sampling(tool_sampling, max(1, int(args.decision_max_tokens or 1024)))
    tool_sampling = replace(tool_sampling, stop_tokens=())
    tool_routing_config = tool_routing_config_from_args(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, tool_sampling)]),
        tool_routing_config=tool_routing_config,
    )
    batch_size = max(1, int(args.batch_size or 1))
    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        prompts: list[str] = []
        for index, record in repeated:
            observation = AgentObservation(record.instruction, {"probe": True})
            route = _route_complexfuncbench_tools(
                record,
                [],
                observation,
                config=tool_routing_config,
                engine=run.engine,
                sampling=tool_sampling,
                prompt_seed=sample_repeat_seed(index, 0, stage=1),
                progress_desc=f"ComplexFuncBench router probe {index}",
            )
            prompts.append(build_complexfuncbench_prompt(record, [], observation, 0, tool_route=route))
        run.engine.generate(
            prompts,
            sampling=tool_sampling,
            batch_size=len(prompts),
            progress_desc="ComplexFuncBench-Probe",
            prompt_stop_suffixes=[list(COMPLEXFUNCBENCH_STOP_SUFFIXES) for _ in prompts],
            prompt_seeds=[sample_repeat_seed(index, 0, stage=1) for index, _record in repeated],
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    job_name = _resolve_job_name("function_complexfuncbench", run_context=run_context)
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 32),
        run_context=run_context,
    )
    runtime = ctx.runtime
    writer = ctx.writer
    flush_partial = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_complexfuncbench_completion_to_eval_payload,
        runner_name="function_complexfuncbench",
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=flush_partial,
        ):
            try:
                pending = list(build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys))
                max_attempt_workers = batch_size if run.engine.__class__.__name__ == "RemoteInferenceBackend" else 1
                with ThreadPoolExecutor(max_workers=max(1, int(max_attempt_workers))) as executor:
                    futures = {
                        executor.submit(
                            _run_complexfuncbench_attempt,
                            record=record,
                            engine=run.engine,
                            sampling=tool_sampling,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            max_steps=max(1, int(record.max_steps or args.max_steps or 8)),
                            tool_routing_config=tool_routing_config,
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sampling_payload=sampling_payload,
                        ): key
                        for key, record in pending
                    }
                    for future in as_completed(futures):
                        writer.enqueue(future.result())
            except BaseException:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_complexfuncbench_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: _complexfuncbench_score_payload(
                completions_payloads,
                metrics=metrics,
                run=run,
                model_name=run.model_name,
                job_name=job_name,
                plan=plan,
                tool_routing_config=tool_routing_config,
            ),
        )
    except BaseException as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"function_complexfuncbench done: {len(completions_payloads)} samples")
    return 0


def _run_complexfuncbench_attempt(
    *,
    record: ComplexFuncBenchRecord,
    engine: Any,
    sampling: Any,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    max_steps: int,
    tool_routing_config: "ToolRoutingConfig",
    benchmark_name: str,
    dataset_split: str,
    sampling_payload: Sequence[tuple[int, Mapping[str, Any]]],
) -> dict[str, object]:
    episode = _run_complexfuncbench_episode(
        record,
        engine=engine,
        sampling=sampling,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        max_steps=max_steps,
        tool_routing_config=tool_routing_config,
    )
    return _complexfuncbench_completion_payload(
        episode,
        record=record,
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        sampling_payload=sampling_payload,
    )


def _validate_complexfuncbench_official_records(records: Sequence[ComplexFuncBenchRecord]) -> None:
    bad_env: list[str] = []
    disabled_response_eval: list[str] = []
    for record in records:
        env = dict(record.env)
        env_type = str(env.get("type") or "").strip().lower()
        official_root = str(env.get("official_root") or "").strip()
        if env_type != "complexfuncbench_official" or not official_root:
            bad_env.append(record.task_id)
            continue
        if env.get("response_eval") is False:
            disabled_response_eval.append(record.task_id)
    if bad_env:
        preview = ", ".join(bad_env[:5])
        suffix = "..." if len(bad_env) > 5 else ""
        raise ValueError(
            "ComplexFuncBench manifest contains non-official sandbox rows "
            f"({preview}{suffix}). Refresh data/complexfuncbench_official with "
            "RWKV_COMPLEXFUNC_OFFICIAL_ROOT or RWKV_COMPLEXFUNCBENCH_OFFICIAL_ROOT."
        )
    if disabled_response_eval:
        preview = ", ".join(disabled_response_eval[:5])
        suffix = "..." if len(disabled_response_eval) > 5 else ""
        raise ValueError(
            "ComplexFuncBench manifest has official response evaluation disabled "
            f"({preview}{suffix}). Reprepare the dataset with RWKV_COMPLEXFUNCBENCH_RESPONSE_EVAL=1."
        )


def _with_complexfuncbench_response_eval(
    record: ComplexFuncBenchRecord,
    *,
    enabled: bool,
) -> ComplexFuncBenchRecord:
    env = dict(record.env)
    if str(env.get("type") or "").strip().lower() == "complexfuncbench_official":
        env["response_eval"] = bool(enabled)
    return replace(record, env=env)


def _with_complexfuncbench_offline_compare(
    record: ComplexFuncBenchRecord,
    *,
    enabled: bool,
) -> ComplexFuncBenchRecord:
    env = dict(record.env)
    if str(env.get("type") or "").strip().lower() == "complexfuncbench_official":
        env["offline_compare"] = bool(enabled)
    return replace(record, env=env)


def _run_complexfuncbench_episode(
    record: ComplexFuncBenchRecord,
    *,
    engine: Any,
    sampling: Any,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    max_steps: int,
    tool_routing_config: "ToolRoutingConfig",
) -> ComplexFuncBenchEpisodeResult:
    env = create_complexfuncbench_env(record)
    observation = env.reset()
    events: list[dict[str, Any]] = [_observation_event(observation, step=0)]
    stages: list[StageRecord] = []
    completions: list[str] = []
    tool_routes: list[dict[str, Any]] = []
    format_bridges: list[dict[str, Any]] = []
    invalid_action_count = 0
    parse_error_count = 0
    finish_reason = "max_steps"
    final_score: float | None = None
    final_success: bool | None = None
    final_env_details: dict[str, Any] = {}

    for step in range(max(0, max_steps)):
        seed = sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=step + 1)
        route = _route_complexfuncbench_tools(
            record,
            events,
            observation,
            config=tool_routing_config,
            engine=engine,
            sampling=sampling,
            prompt_seed=seed,
            progress_desc=f"ComplexFuncBench router {sample_index}",
        )
        tool_routes.append(route.trace_payload())
        prompt = build_complexfuncbench_prompt(record, events, observation, step, tool_route=route)
        output = engine.generate(
            [prompt],
            sampling=sampling,
            batch_size=1,
            progress_desc=f"ComplexFuncBench sample {sample_index}",
            prompt_stop_suffixes=[list(COMPLEXFUNCBENCH_STOP_SUFFIXES)],
            prompt_seeds=[seed],
            show_progress=False,
        )[0]
        completion = _trim_stop_suffixes(str(getattr(output, "text", "") or ""), COMPLEXFUNCBENCH_STOP_SUFFIXES).strip()
        stop_reason = str(getattr(output, "finish_reason", "") or "")
        stages.append(StageRecord(prompt=prompt, completion=completion, stop_reason=stop_reason))
        completions.append(completion)
        events.append(
            {
                "type": "model_output",
                "role": "assistant",
                "content": completion,
                "metadata": {"step": step, "tool_route": route.trace_payload()},
            }
        )

        try:
            if _looks_like_template_leak(completion):
                raise ValueError("decision stage leaked internal template/control tokens")
            actions = parse_complexfuncbench_tool_calls(completion)
        except Exception as exc:  # noqa: BLE001
            bridge = build_complexfuncbench_format_bridge(completion, [], parse_error=str(exc))
            format_bridges.append(bridge)
            parse_error_count += 1
            invalid_action_count += 1
            finish_reason = "parse_error"
            final_score = 0.0
            final_success = False
            final_env_details = _complexfuncbench_env_failure_details(
                env,
                finish_reason="parse_error",
                message={"error_type": "parse_error", "content": str(exc)},
            )
            events.append(
                {
                    "type": "format_bridge",
                    "role": "adapter",
                    "content": "Failed to convert RWKV JSON output into official ComplexFuncBench sandbox calls.",
                    "metadata": {"step": step, **bridge},
                }
            )
            events.append(
                {
                    "type": "error",
                    "role": "parser",
                    "content": str(exc),
                    "metadata": {"step": step, "kind": "parse_error"},
                }
            )
            break
        if not actions:
            invalid_action_count += 1
            finish_reason = "empty_action"
            final_score = 0.0
            final_success = False
            final_env_details = _complexfuncbench_env_failure_details(
                env,
                finish_reason="empty_action",
                message={"error_type": "empty_action", "content": "model output did not contain a tool action"},
            )
            events.append(
                {
                    "type": "error",
                    "role": "parser",
                    "content": "model output did not contain a tool action",
                    "metadata": {"step": step, "kind": "empty_action"},
                }
            )
            break

        bridge = build_complexfuncbench_format_bridge(completion, actions)
        format_bridges.append(bridge)
        events.append(
            {
                "type": "format_bridge",
                "role": "adapter",
                "content": "Converted RWKV JSON output to internal ToolAction objects and official sandbox call dicts.",
                "metadata": {"step": step, **bridge},
            }
        )

        for call_index, action in enumerate(actions):
            events.append(
                {
                    "type": "action",
                    "role": "assistant",
                    "name": action.name,
                    "arguments": dict(action.arguments),
                    "metadata": {"step": step, "call_index": call_index},
                }
            )

        step_result = env.step_many(actions)
        observation = step_result.observation
        final_score = step_result.score if step_result.score is not None else final_score
        final_success = step_result.success if step_result.success is not None else final_success
        if step_result.details:
            final_env_details = dict(step_result.details)
        events.append(
            {
                "type": "env_result",
                "role": "environment",
                "content": observation.content,
                "metadata": {
                    "step": step,
                    "call_count": len(actions),
                    "done": step_result.done,
                    "score": step_result.score,
                    "success": step_result.success,
                    "details": dict(step_result.details),
                    "observation_metadata": dict(observation.metadata),
                },
            }
        )
        if step_result.done:
            finish_reason = "done"
            break

    success = _coerce_success(final_success, final_score, finish_reason)
    details = {
        "finish_reason": finish_reason,
        "steps": sum(1 for event in events if event.get("type") == "model_output"),
        "invalid_action_count": invalid_action_count,
        "parse_error_count": parse_error_count,
        "timeout": False,
    }
    if final_env_details:
        details["final_env_details"] = final_env_details
    events.append(
        {
            "type": "final_score",
            "role": "scorer",
            "metadata": {"success": success, "score": final_score, **details},
        }
    )
    return ComplexFuncBenchEpisodeResult(
        stages=stages,
        completions=completions,
        events=events,
        tool_routes=tool_routes,
        format_bridges=format_bridges,
        success=success,
        score=final_score,
        details=details,
        final_answer=_final_response_from_details(details, completions),
    )


def run_complexfuncbench_local_episode(
    record: ComplexFuncBenchRecord,
    *,
    engine: Any,
    sampling: Any,
    tool_routing_config: "ToolRoutingConfig",
    history_max_chars: int | None = None,
    prompt_seed_base: int | None = None,
) -> ComplexFuncBenchEpisodeResult:
    del history_max_chars, prompt_seed_base
    return _run_complexfuncbench_episode(
        record,
        engine=engine,
        sampling=sampling,
        sample_index=0,
        repeat_index=0,
        pass_index=0,
        max_steps=max(1, int(record.max_steps or 8)),
        tool_routing_config=tool_routing_config,
    )


def _route_complexfuncbench_tools(
    record: ComplexFuncBenchRecord,
    events: Sequence[Mapping[str, Any]],
    observation: AgentObservation,
    *,
    config: "ToolRoutingConfig",
    engine: Any | None,
    sampling: Any | None,
    prompt_seed: int | None,
    progress_desc: str,
) -> ToolRouteResult:
    messages = _routing_messages(events, observation)
    return route_tools_for_prompt(
        record.tools,
        messages,
        config=config,
        engine=engine,
        sampling=sampling,
        control_tool_names=(COMPLEXFUNCBENCH_FINAL_SCHEMA["name"],),
        progress_desc=progress_desc,
        prompt_seed=prompt_seed,
    )


def _complexfuncbench_completion_payload(
    episode: ComplexFuncBenchEpisodeResult,
    *,
    record: ComplexFuncBenchRecord,
    benchmark_name: str,
    dataset_split: str,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    sampling_payload: dict[str, Any],
) -> dict[str, Any]:
    payload = SampleRecord(
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        stages=list(episode.stages),
        sampling_config=sampling_payload,
    ).as_payload()
    payload["_stage"] = "answer"
    payload["agent_result"] = {
        "task_id": record.task_id,
        "reward": float(episode.score or 0.0),
        "num_turns": int(episode.details.get("steps") or 0),
        "cost": 0.0,
        "is_passed": bool(episode.success),
        "error": None if episode.success else str(episode.details.get("finish_reason") or ""),
    }
    payload["agent_info"] = {
        "cot_mode": CoTMode.COT.value,
        "final_answer": episode.final_answer,
        "official_id": record.metadata.get("official_id") or record.task_id,
    }
    payload["agent_trace"] = [
        {
            "tool_route": route,
            "completion": episode.completions[index] if index < len(episode.completions) else "",
            "format_bridge": episode.format_bridges[index] if index < len(episode.format_bridges) else {},
        }
        for index, route in enumerate(episode.tool_routes)
    ]
    payload["format_bridges"] = list(episode.format_bridges)
    payload["events"] = list(episode.events)
    payload["success"] = bool(episode.success)
    payload["official_score"] = episode.score
    payload["agent_details"] = dict(episode.details)
    payload["final_answer"] = episode.final_answer
    payload["task_id"] = record.task_id
    payload["domain"] = "function_call"
    payload["instruction"] = record.instruction
    payload["metadata"] = dict(record.metadata)
    final_env_details = episode.details.get("final_env_details")
    if isinstance(final_env_details, Mapping):
        payload["complexfuncbench_official_result"] = {
            key: final_env_details.get(key)
            for key in ("message", "count_dict", "resp_eval", "call_accuracy")
            if key in final_env_details
        }
    return payload


def _complexfuncbench_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    success = bool(payload.get("success"))
    details = payload.get("agent_details")
    if isinstance(details, Mapping):
        final_details = details.get("final_env_details")
        if isinstance(final_details, Mapping) and final_details.get("message") == "Success.":
            success = True
    reason = "" if success else str((details or {}).get("finish_reason") if isinstance(details, Mapping) else "")
    metadata = payload.get("metadata")
    ref_answer = ""
    if isinstance(metadata, Mapping):
        ref_answer = str(metadata.get("official_id") or "")
    return make_eval_payload(
        payload,
        is_passed=success,
        fail_reason=reason,
        answer=str(payload.get("final_answer") or ""),
        ref_answer=ref_answer,
    )


def _complexfuncbench_score_payload(
    completions_payloads: Sequence[dict[str, object]],
    *,
    metrics: dict[str, float],
    run: ResolvedFunctionCallingRun,
    model_name: str,
    job_name: str,
    plan: Any,
    tool_routing_config: "ToolRoutingConfig",
) -> dict[str, Any]:
    complex_metrics = summarize_complexfuncbench_official_payloads(completions_payloads)
    merged = dict(metrics)
    merged["avg@1"] = complex_metrics.success_rate
    merged["success_rate"] = complex_metrics.success_rate
    merged["official_score"] = complex_metrics.success_rate
    merged["call_accuracy"] = complex_metrics.call_accuracy
    merged["completeness"] = complex_metrics.completeness
    merged["correctness"] = complex_metrics.correctness
    merged["response_eval_samples"] = float(complex_metrics.response_eval_samples)
    return make_score_payload(
        run.dataset_slug,
        is_cot=True,
        model_name=model_name,
        metrics=merged,
        samples=len(completions_payloads),
        problems=plan.sample_size,
        task=job_name,
        task_details={
            **build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
            "complexfuncbench_official": {
                "call_accuracy": complex_metrics.call_accuracy,
                "response_eval_samples": complex_metrics.response_eval_samples,
            },
            "tool_router": {
                "mode": tool_routing_config.mode,
                "max_tools": tool_routing_config.max_tools,
            },
        },
        extra={"cot_mode": CoTMode.COT.value},
    )


def _convert_official_row(
    item: Mapping[str, Any],
    *,
    index: int,
    dataset_name: str,
    source_path: Path | None,
    official_root: Path | None,
    response_eval: bool,
) -> dict[str, Any] | None:
    conversations = _list_of_dicts(item.get("conversations"))
    functions = _list_of_dicts(item.get("functions"))
    fc_chain, _obs_chain = _official_chains(conversations)
    if not conversations or not functions or not fc_chain:
        return None
    official_id = str(item.get("id") or item.get("task_id") or index)
    tools = [copy.deepcopy(tool) for tool in functions] + [copy.deepcopy(COMPLEXFUNCBENCH_FINAL_SCHEMA)]
    if official_root is None:
        raise ValueError("ComplexFuncBench official rows require an official sandbox root")
    official_root = require_complexfuncbench_official_root(official_root)
    env: dict[str, Any] = {
        "type": "complexfuncbench_official",
        "official_root": str(official_root),
        "response_eval": bool(response_eval),
    }
    scorer: dict[str, Any] = {"type": "complexfuncbench_official"}
    metadata: dict[str, Any] = {
        "source_format": "official_complexfuncbench",
        "official_source": OFFICIAL_COMPLEXFUNC_SOURCE,
        "official_id": official_id,
        "complexfuncbench_functions": functions,
        "complexfuncbench_conversations": conversations,
        "complexfuncbench_total_turn_num": len(fc_chain),
        "complexfuncbench_total_call_num": sum(len(turn) for turn in fc_chain),
        "category": item.get("category") or item.get("type") or "",
    }
    metadata["complexfuncbench_official_root"] = str(official_root)
    return {
        "task_id": f"{dataset_name}__{official_id}",
        "instruction": str(conversations[0].get("content") or conversations[0].get("text") or ""),
        "messages": [{"role": "user", "content": str(conversations[0].get("content") or "")}],
        "tools": tools,
        "expected_tool_calls": [],
        "env": env,
        "scorer": scorer,
        "max_steps": max(4, len(fc_chain) + 3),
        "metadata": metadata,
    }


def _record_from_payload(payload: Mapping[str, Any], *, index: int) -> ComplexFuncBenchRecord:
    task_id = str(payload.get("task_id") or payload.get("id") or f"complexfuncbench_{index:05d}")
    return ComplexFuncBenchRecord(
        task_id=task_id,
        instruction=str(payload.get("instruction") or ""),
        messages=_list_of_dicts(payload.get("messages")),
        tools=_list_of_dicts(payload.get("tools") or payload.get("tools_spec")),
        expected_tool_calls=_list_of_dicts(payload.get("expected_tool_calls")),
        env=_dict_value(payload.get("env") or payload.get("environment") or payload.get("env_spec")),
        scorer=_dict_value(payload.get("scorer") or payload.get("scorer_spec") or payload.get("evaluation")),
        max_steps=_positive_int(payload.get("max_steps")),
        metadata=_dict_value(payload.get("metadata")),
    )


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


def _function_call_turns_equal(predicted: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]) -> bool:
    if len(predicted) != len(expected):
        return False
    return [_canonical_function_call(item) for item in predicted] == [
        _canonical_function_call(item) for item in expected
    ]


def _canonical_function_call(item: Mapping[str, Any]) -> tuple[str, str]:
    arguments = item.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    return (
        str(item.get("name") or "").strip(),
        json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str),
    )


def _tool_action_payloads(actions: Sequence[ToolAction]) -> list[dict[str, Any]]:
    return [{"name": action.name, "arguments": dict(action.arguments)} for action in actions]


def _actions_to_official_calls(actions: Sequence[ToolAction]) -> list[dict[str, Any]]:
    return [{"name": action.name, "arguments": dict(action.arguments)} for action in actions]


def _complexfuncbench_env_failure_details(
    env: Any,
    *,
    finish_reason: str,
    message: Any,
) -> dict[str, Any]:
    failure_details = getattr(env, "failure_details", None)
    if callable(failure_details):
        details = failure_details(finish_reason=finish_reason, message=message)
        return dict(details) if isinstance(details, Mapping) else {}
    return {"finish_reason": finish_reason, "message": message}


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


def _render_agent_trajectory(events: Sequence[Mapping[str, Any]]) -> str:
    rendered: list[str] = []
    for event in events:
        event_type = str(event.get("type") or "")
        content = str(event.get("content") or "")
        if event_type == "observation":
            rendered.append(f"Environment: {content}")
        elif event_type == "model_output":
            rendered.append(f"Assistant action: {content}")
        elif event_type == "action":
            rendered.append(
                "Tool call: "
                + json.dumps(
                    {"name": event.get("name"), "arguments": event.get("arguments") or {}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        elif event_type == "env_result":
            rendered.append(f"Environment: {content}")
        elif event_type == "error":
            rendered.append(f"Error: {content}")
    return "\n".join(part for part in rendered if part.strip()).strip()


def _routing_messages(events: Sequence[Mapping[str, Any]], observation: AgentObservation) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type in {"observation", "env_result"}:
            content = str(event.get("content") or "")
            if content:
                messages.append({"role": "user", "content": content})
        elif event_type in {"model_output", "action"}:
            content = str(event.get("content") or "")
            if not content and event_type == "action":
                content = json.dumps(
                    {"name": event.get("name"), "arguments": event.get("arguments") or {}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            if content:
                messages.append({"role": "assistant", "content": content})
    if observation.content:
        messages.append({"role": "user", "content": observation.content})
    return messages


def _observation_event(observation: AgentObservation, *, step: int) -> dict[str, Any]:
    return {
        "type": "observation",
        "role": "environment",
        "content": observation.content,
        "metadata": {"step": step, **dict(observation.metadata)},
    }


def _coerce_success(success: bool | None, score: float | None, finish_reason: str) -> bool:
    if success is not None:
        return bool(success)
    if finish_reason != "done":
        return False
    if score is None:
        return True
    return float(score) > 0.0


def _final_response_from_details(details: Mapping[str, Any], completions: Sequence[str]) -> str:
    final_details = details.get("final_env_details")
    if isinstance(final_details, Mapping):
        final_response = final_details.get("final_response")
        if isinstance(final_response, str):
            return final_response
    return str(completions[-1] if completions else "")


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


def _record_metadata(record: ComplexFuncBenchRecord | Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.metadata if isinstance(record, ComplexFuncBenchRecord) else record.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _record_env(record: ComplexFuncBenchRecord | Mapping[str, Any]) -> dict[str, Any]:
    env = record.env if isinstance(record, ComplexFuncBenchRecord) else record.get("env")
    return dict(env) if isinstance(env, Mapping) else {}


def _record_tools(record: ComplexFuncBenchRecord | Mapping[str, Any]) -> list[dict[str, Any]]:
    tools = record.tools if isinstance(record, ComplexFuncBenchRecord) else record.get("tools")
    return _list_of_dicts(tools)


def _record_task_id(record: ComplexFuncBenchRecord | Mapping[str, Any]) -> str:
    return str(record.task_id if isinstance(record, ComplexFuncBenchRecord) else record.get("task_id") or "")


def _record_instruction(record: ComplexFuncBenchRecord | Mapping[str, Any]) -> str:
    return str(record.instruction if isinstance(record, ComplexFuncBenchRecord) else record.get("instruction") or "")


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    return []


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


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


def _patch_official_compare_for_offline(runner: Any) -> None:
    compare = getattr(runner, "CompareClass", None)
    if compare is None:
        return

    def _disabled_response_based(_predict: Any, _golden: Any) -> bool:
        return False

    def _disabled_llm_based(*_args: Any, **_kwargs: Any) -> bool:
        return False

    compare.response_based = _disabled_response_based
    compare.llm_based = _disabled_llm_based


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _normalize_prompt_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text or "").replace("\r\n", "\n").split("\n")).strip()


def _trim_stop_suffixes(text: str, stop_suffixes: tuple[str, ...]) -> str:
    earliest: int | None = None
    for suffix in stop_suffixes:
        start = len(suffix) if suffix == "```" and text.startswith("```") else 0
        index = text.find(suffix, start)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index
    if earliest is None:
        return text
    return text[:earliest]


def _resolve_complexfuncbench_openai_connection() -> dict[str, Any]:
    raw_connection = (
        os.environ.get("RWKV_COMPLEXFUNCBENCH_OPENAI_CONN")
        or os.environ.get("RWKV_COMPLEXFUNC_OPENAI_CONN")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_RESPONSE_EVAL_API")
        or os.environ.get("RWKV_COMPLEXFUNC_RESPONSE_EVAL_API")
    )
    payload: Mapping[str, Any] = {}
    if raw_connection:
        parsed = json.loads(raw_connection)
        if not isinstance(parsed, Mapping):
            raise ValueError("ComplexFuncBench OpenAI connection env must be a JSON object")
        payload = parsed

    api_key = (
        payload.get("key")
        or payload.get("api_key")
        or payload.get("OPENAI_API_KEY")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_OPENAI_API_KEY")
        or os.environ.get("RWKV_COMPLEXFUNC_OPENAI_API_KEY")
    )
    base_url = (
        payload.get("url")
        or payload.get("base_url")
        or payload.get("api_base")
        or payload.get("OPENAI_BASE_URL")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_OPENAI_BASE_URL")
        or os.environ.get("RWKV_COMPLEXFUNC_OPENAI_BASE_URL")
    )

    model_map: dict[str, str] = {}
    raw_model_map = payload.get("model_map") or os.environ.get("RWKV_COMPLEXFUNCBENCH_OPENAI_MODEL_MAP")
    if raw_model_map:
        parsed_model_map = json.loads(raw_model_map) if isinstance(raw_model_map, str) else raw_model_map
        if not isinstance(parsed_model_map, Mapping):
            raise ValueError("ComplexFuncBench OpenAI model map must be a JSON object")
        model_map.update({str(key): str(value) for key, value in parsed_model_map.items() if value})
    response_eval_model = (
        payload.get("response_eval_model")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_RESPONSE_EVAL_MODEL")
        or os.environ.get("RWKV_COMPLEXFUNC_RESPONSE_EVAL_MODEL")
    )
    if response_eval_model:
        model_map["gpt-4o-2024-08-06"] = str(response_eval_model)
    compare_model = (
        payload.get("compare_model")
        or os.environ.get("RWKV_COMPLEXFUNCBENCH_COMPARE_MODEL")
        or os.environ.get("RWKV_COMPLEXFUNC_COMPARE_MODEL")
    )
    if compare_model:
        model_map["gpt-4o-2024-05-13"] = str(compare_model)

    connection: dict[str, Any] = {}
    if api_key:
        connection["api_key"] = str(api_key)
    if base_url:
        connection["base_url"] = _normalize_openai_base_url(str(base_url))
    if model_map:
        connection["model_map"] = model_map
    return connection


def _normalize_openai_base_url(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return text
    if "://" not in text:
        return text
    scheme, remainder = text.split("://", 1)
    host_and_path = remainder.split("/", 1)
    if len(host_and_path) == 1:
        return f"{scheme}://{host_and_path[0]}/v1"
    host, path = host_and_path
    if not path:
        return f"{scheme}://{host}/v1"
    return f"{scheme}://{host}/{path.rstrip('/')}"


def _complexfuncbench_openai_model_map(openai_connection: Mapping[str, Any] | None = None) -> dict[str, str]:
    connection = dict(openai_connection or _resolve_complexfuncbench_openai_connection())
    model_map = connection.get("model_map")
    if not isinstance(model_map, Mapping):
        return {}
    return {str(key): str(value) for key, value in model_map.items() if value}


def _patch_official_gpt_model_aliases(openai_connection: Mapping[str, Any] | None = None) -> None:
    model_map = _complexfuncbench_openai_model_map(openai_connection)
    if not model_map:
        return
    for module_name in ("models.gpt", "utils.compare_method", "runner.response_runner"):
        module = sys.modules.get(module_name)
        if module is None or not hasattr(module, "GPTModel"):
            continue
        original = getattr(module, "_rwkv_complexfuncbench_original_gpt_model", None)
        if original is None:
            original = getattr(module, "GPTModel")
            setattr(module, "_rwkv_complexfuncbench_original_gpt_model", original)

        class ModelAliasGPT(original):  # type: ignore[misc, valid-type]
            def __init__(self, model_name: str) -> None:
                original_model_name = str(model_name)
                mapped_model_name = model_map.get(original_model_name, original_model_name)
                self.rwkv_complexfuncbench_original_model_name = original_model_name
                super().__init__(mapped_model_name)

        setattr(module, "GPTModel", ModelAliasGPT)


@contextmanager
def _official_import_context(root: Path, *, openai_connection: Mapping[str, Any] | None = None) -> Iterator[None]:
    with _OFFICIAL_IMPORT_LOCK:
        old_cwd = Path.cwd()
        root_text = str(root)
        inserted = False
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
            inserted = True
        _ensure_flag_embedding_module()
        _ensure_scipy_optimize_module()
        old_openai_api_key = os.environ.get("OPENAI_API_KEY")
        old_openai_base_url = os.environ.get("OPENAI_BASE_URL")
        connection = dict(openai_connection or _resolve_complexfuncbench_openai_connection())
        connection_api_key = str(connection.get("api_key") or "").strip()
        connection_base_url = str(connection.get("base_url") or "").strip()
        if connection_api_key:
            os.environ["OPENAI_API_KEY"] = connection_api_key
        elif not old_openai_api_key:
            os.environ["OPENAI_API_KEY"] = "complexfuncbench-offline-dummy-key"
        if connection_base_url:
            os.environ["OPENAI_BASE_URL"] = connection_base_url
        os.chdir(root)
        try:
            yield
        finally:
            os.chdir(old_cwd)
            if old_openai_api_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old_openai_api_key
            if old_openai_base_url is None:
                os.environ.pop("OPENAI_BASE_URL", None)
            else:
                os.environ["OPENAI_BASE_URL"] = old_openai_base_url
            if inserted:
                try:
                    sys.path.remove(root_text)
                except ValueError:
                    pass


def _ensure_flag_embedding_module() -> None:
    if "FlagEmbedding" in sys.modules:
        return
    try:
        importlib.import_module("FlagEmbedding")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "FlagEmbedding":
            raise
    module = types.ModuleType("FlagEmbedding")

    class FlagModel:  # noqa: D401 - mirrors the optional external package API.
        """Small deterministic fallback for ComplexFuncBench offline exact-match runs."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def encode(self, texts: Sequence[str]) -> Any:
            import hashlib
            import re

            import numpy as np

            vectors: list[Any] = []
            for text in texts:
                vector = np.zeros(128, dtype=np.float32)
                tokens = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
                for token in tokens:
                    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                    index = int.from_bytes(digest[:4], "little") % vector.shape[0]
                    vector[index] += 1.0
                norm = float(np.linalg.norm(vector))
                if norm > 0:
                    vector /= norm
                vectors.append(vector)
            if not vectors:
                return np.zeros((0, 128), dtype=np.float32)
            return np.vstack(vectors)

    module.FlagModel = FlagModel
    sys.modules["FlagEmbedding"] = module


def _ensure_scipy_optimize_module() -> None:
    if "scipy.optimize" in sys.modules:
        return
    try:
        importlib.import_module("scipy.optimize")
        return
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
    scipy_module = sys.modules.get("scipy") or types.ModuleType("scipy")
    optimize_module = types.ModuleType("scipy.optimize")

    def linear_sum_assignment(cost_matrix: Any) -> tuple[Any, Any]:
        import numpy as np

        matrix = np.asarray(cost_matrix)
        if matrix.ndim != 2:
            raise ValueError("expected a 2-D cost matrix")
        row_count, col_count = matrix.shape
        pairs: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        candidates: list[tuple[float, int, int]] = []
        for row in range(row_count):
            for col in range(col_count):
                candidates.append((float(matrix[row, col]), row, col))
        for _score, row, col in sorted(candidates, key=lambda item: item[0]):
            if row in used_rows or col in used_cols:
                continue
            used_rows.add(row)
            used_cols.add(col)
            pairs.append((row, col))
            if len(pairs) >= min(row_count, col_count):
                break
        row_ind = np.asarray([row for row, _col in pairs], dtype=np.int64)
        col_ind = np.asarray([col for _row, col in pairs], dtype=np.int64)
        return row_ind, col_ind

    optimize_module.linear_sum_assignment = linear_sum_assignment
    scipy_module.optimize = optimize_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.optimize"] = optimize_module


__all__ = [
    "COMPLEXFUNCBENCH_FINAL_SCHEMA",
    "COMPLEXFUNCBENCH_TASK_PREFIX",
    "DEFAULT_COMPLEXFUNC_MAX_ROWS",
    "OFFICIAL_COMPLEXFUNC_SOURCE",
    "AgentObservation",
    "AgentStepResult",
    "ComplexFuncBenchCall",
    "ComplexFuncBenchLocalEnv",
    "ComplexFuncBenchOfficialEnv",
    "ComplexFuncBenchOfficialMetrics",
    "ComplexFuncBenchMetrics",
    "ComplexFuncBenchEpisodeResult",
    "ComplexFuncBenchRecord",
    "OfficialComplexFuncBenchSandbox",
    "ToolAction",
    "build_complexfuncbench_format_bridge",
    "build_complexfuncbench_system_prompt",
    "build_complexfuncbench_prompt",
    "create_complexfuncbench_env",
    "load_complexfuncbench_manifest_records",
    "load_complexfuncbench_rows_from_source",
    "normalize_complexfuncbench_source_row",
    "parse_complexfuncbench_calls",
    "parse_complexfuncbench_tool_calls",
    "require_complexfuncbench_official_root",
    "run_complexfuncbench_local_episode",
    "summarize_complexfuncbench_payloads",
    "summarize_complexfuncbench_official_payloads",
]
