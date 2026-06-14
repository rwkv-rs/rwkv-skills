from __future__ import annotations

import json
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.env_config import resolve_judge_model_config
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
)
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_rwkv_json_call_prompt,
    coerce_json_function_call_payload,
    extract_json_call_value_text,
    normalize_function_prompt_style,
    render_function_output_user_block,
    render_json_function_call,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage
from src.eval.scheduler.config import REPO_ROOT

from .context_budget import (
    DEFAULT_HISTORY_MAX_CHARS,
    DEFAULT_TOOL_ERROR_MAX_CHARS,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    DEFAULT_TOOL_SCHEMA_MAX_CHARS,
    normalize_rwkv_text,
    trim_history,
    truncate_text,
)

if TYPE_CHECKING:
    import argparse

    from src.eval.evaluating.contracts import RunContext

MCP_BENCH_PASS_THRESHOLD = 7.0
MCP_BENCH_MAX_TOOL_SCHEMA_CHARS = DEFAULT_TOOL_SCHEMA_MAX_CHARS
MCP_BENCH_MAX_RESULT_CHARS = DEFAULT_TOOL_RESULT_MAX_CHARS
MCP_BENCH_MAX_ERROR_CHARS = DEFAULT_TOOL_ERROR_MAX_CHARS
MCP_BENCH_MAX_HISTORY_CHARS = DEFAULT_HISTORY_MAX_CHARS
MCP_BENCH_TASK_FILES: tuple[str, ...] = (
    "mcpbench_tasks_single_runner_format.json",
    "mcpbench_tasks_multi_2server_runner_format.json",
    "mcpbench_tasks_multi_3server_runner_format.json",
)


@dataclass(frozen=True, slots=True)
class McpBenchTaskSpec:
    task_id: str
    task_description: str
    fuzzy_description: str = ""
    dependency_analysis: str = ""
    distraction_servers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class McpBenchItem:
    task_file: str
    server_name: str
    combination_name: str
    combination_type: str
    servers: tuple[str, ...]
    task: McpBenchTaskSpec
    runtime_root: str | None = None


@dataclass(frozen=True, slots=True)
class PlannedToolCall:
    server: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return f"{self.server}:{self.tool}"


@dataclass(frozen=True, slots=True)
class PlanningDecision:
    reasoning: str
    should_continue: bool
    tool_calls: tuple[PlannedToolCall, ...]
    final_answer: str = ""


@dataclass(frozen=True, slots=True)
class McpBenchExecutionResult:
    tool: str
    server: str
    parameters: dict[str, Any]
    round_num: int
    planned_layer: int | None
    success: bool
    result: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class McpBenchEvaluation:
    task_completion_score: float = 0.0
    tool_selection_score: float = 0.0
    planning_effectiveness_and_efficiency_score: float = 0.0
    task_fulfillment: float = 0.0
    grounding: float = 0.0
    tool_appropriateness: float = 0.0
    parameter_accuracy: float = 0.0
    dependency_awareness: float = 0.0
    parallelism_and_efficiency: float = 0.0
    input_schema_compliance: float | None = None
    valid_tool_name_rate: float | None = None
    execution_success_rate: float | None = None
    planning_json_compliance: float | None = None


@dataclass(frozen=True, slots=True)
class McpBenchPreflightReport:
    ok: bool
    runtime_root: str
    worker_script: str
    checked_servers: tuple[str, ...]
    errors: tuple[str, ...] = ()


class McpBenchWorkerClient:
    def __init__(self, *, runtime_root: str | Path, worker_script: str | Path) -> None:
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.worker_script = Path(worker_script).expanduser().resolve()
        python_bin = self.runtime_root / ".venv" / "bin" / "python"
        if not python_bin.is_file():
            raise FileNotFoundError(f"missing MCP-Bench runtime python: {python_bin}")
        self._proc = subprocess.Popen(
            [str(python_bin), str(self.worker_script), "--runtime-root", str(self.runtime_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_lines: deque[str] = deque(maxlen=200)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, name="McpBenchWorkerStderr", daemon=True)
        self._stderr_thread.start()
        self._closed = False

    def open_task(self, item: McpBenchItem) -> dict[str, Any]:
        response = self._request(
            "open_task",
            {
                "servers": list(item.servers),
            },
        )
        available_tools = response.get("available_tools")
        if not isinstance(available_tools, dict):
            raise RuntimeError("worker returned invalid available_tools payload")
        return available_tools

    def call_tool(self, full_tool_name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        response = self._request(
            "call_tool",
            {"tool_name": full_tool_name, "arguments": dict(arguments)},
        )
        if not isinstance(response, dict):
            raise RuntimeError("worker returned invalid tool response")
        return response

    def evaluate(self, request: Mapping[str, Any]) -> McpBenchEvaluation:
        response = self._request("evaluate", {"request": dict(request)})
        if not isinstance(response, dict):
            raise RuntimeError("worker returned invalid evaluation payload")
        return McpBenchEvaluation(
            task_completion_score=float(response.get("task_completion_score", 0.0)),
            tool_selection_score=float(response.get("tool_selection_score", 0.0)),
            planning_effectiveness_and_efficiency_score=float(
                response.get("planning_effectiveness_and_efficiency_score", 0.0)
            ),
            task_fulfillment=float(response.get("task_fulfillment", 0.0)),
            grounding=float(response.get("grounding", 0.0)),
            tool_appropriateness=float(response.get("tool_appropriateness", 0.0)),
            parameter_accuracy=float(response.get("parameter_accuracy", 0.0)),
            dependency_awareness=float(response.get("dependency_awareness", 0.0)),
            parallelism_and_efficiency=float(response.get("parallelism_and_efficiency", 0.0)),
            input_schema_compliance=_float_or_none(response.get("input_schema_compliance")),
            valid_tool_name_rate=_float_or_none(response.get("valid_tool_name_rate")),
            execution_success_rate=_float_or_none(response.get("execution_success_rate")),
            planning_json_compliance=_float_or_none(response.get("planning_json_compliance")),
        )

    def close_task(self) -> None:
        if self._closed:
            return
        try:
            self._request("close_task", {})
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None:
                try:
                    self._request("shutdown", {})
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=5.0)

    def _request(self, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("worker client already closed")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        wire = json.dumps({"action": action, "payload": dict(payload)}, ensure_ascii=False)
        try:
            self._proc.stdin.write(wire + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._worker_failure_message("worker stdin closed")) from exc

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(self._worker_failure_message("worker stdout closed"))
            raw = line.strip()
            if not raw:
                continue
            try:
                response = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(response, dict):
                continue
            if not response.get("ok", False):
                error = str(response.get("error") or "unknown worker error")
                raise RuntimeError(self._worker_failure_message(error))
            data = response.get("data")
            if not isinstance(data, dict):
                return {}
            return data

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            self._stderr_lines.append(line.rstrip("\n"))

    def _worker_failure_message(self, message: str) -> str:
        stderr = "\n".join(self._stderr_lines)
        if stderr:
            return f"{message}\nworker stderr:\n{stderr}"
        return message


def load_mcp_bench_task_items(
    tasks_root: str | Path,
    runtime_root: str | Path,
    *,
    file_names: Sequence[str] | None = None,
) -> list[McpBenchItem]:
    root = Path(tasks_root)
    runtime = Path(runtime_root)
    selected_file_names = tuple(file_names or MCP_BENCH_TASK_FILES)
    items: list[McpBenchItem] = []
    for file_name in selected_file_names:
        payload = json.loads((root / file_name).read_text(encoding="utf-8"))
        for group in payload.get("server_tasks", []):
            if not isinstance(group, dict):
                continue
            for task_payload in group.get("tasks", []) or []:
                if not isinstance(task_payload, dict):
                    continue
                items.append(
                    McpBenchItem(
                        task_file=file_name,
                        server_name=str(group.get("server_name") or ""),
                        combination_name=str(group.get("combination_name") or ""),
                        combination_type=str(group.get("combination_type") or ""),
                        servers=tuple(str(item) for item in (group.get("servers") or [])),
                        task=McpBenchTaskSpec(
                            task_id=str(task_payload.get("task_id") or ""),
                            task_description=str(task_payload.get("task_description") or ""),
                            fuzzy_description=str(task_payload.get("fuzzy_description") or ""),
                            dependency_analysis=str(task_payload.get("dependency_analysis") or ""),
                            distraction_servers=tuple(
                                str(item) for item in (task_payload.get("distraction_servers") or [])
                            ),
                        ),
                        runtime_root=str(runtime),
                    )
                )
    return items


def load_mcp_bench_manifest_records(path: str | Path) -> list[McpBenchItem]:
    items: list[McpBenchItem] = []
    target = Path(path)
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            task_payload = payload.get("task") or {}
            if not isinstance(task_payload, dict):
                task_payload = {}
            items.append(
                McpBenchItem(
                    task_file=str(payload.get("task_file") or ""),
                    server_name=str(payload.get("server_name") or ""),
                    combination_name=str(payload.get("combination_name") or ""),
                    combination_type=str(payload.get("combination_type") or ""),
                    servers=tuple(str(item) for item in (payload.get("servers") or [])),
                    task=McpBenchTaskSpec(
                        task_id=str(task_payload.get("task_id") or ""),
                        task_description=str(task_payload.get("task_description") or ""),
                        fuzzy_description=str(task_payload.get("fuzzy_description") or ""),
                        dependency_analysis=str(task_payload.get("dependency_analysis") or ""),
                        distraction_servers=tuple(
                            str(item) for item in (task_payload.get("distraction_servers") or [])
                        ),
                    ),
                    runtime_root=str(payload.get("runtime_root") or ""),
                )
            )
    return items


def preflight_mcp_bench_runtime(
    items: Sequence[McpBenchItem],
    *,
    runtime_root: str | Path,
    worker_script: str | Path,
    open_first_task: bool = True,
    raise_on_error: bool = True,
) -> McpBenchPreflightReport:
    runtime = Path(runtime_root).expanduser().resolve()
    worker = Path(worker_script).expanduser().resolve()
    errors: list[str] = []
    if not items:
        errors.append("mcp_bench_no_items")
    python_bin = runtime / ".venv" / "bin" / "python"
    commands_path = runtime / "mcp_servers" / "commands.json"
    if not python_bin.is_file():
        errors.append(f"missing_runtime_python:{python_bin}")
    if not worker.is_file():
        errors.append(f"missing_worker_script:{worker}")
    commands: Mapping[str, Any] = {}
    if not commands_path.is_file():
        errors.append(f"missing_commands_json:{commands_path}")
    else:
        try:
            payload = json.loads(commands_path.read_text(encoding="utf-8"))
            commands = payload if isinstance(payload, Mapping) else {}
            if not commands:
                errors.append(f"invalid_commands_json:{commands_path}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid_commands_json:{commands_path}:{exc}")

    checked_servers = tuple(sorted({server for item in items for server in item.servers if server}))
    for server_name in checked_servers:
        raw = commands.get(server_name) if isinstance(commands, Mapping) else None
        if not isinstance(raw, Mapping):
            errors.append(f"missing_server_config:{server_name}")
            continue
        if not str(raw.get("cmd") or "").strip():
            errors.append(f"missing_server_command:{server_name}")
        cwd = _resolve_mcp_server_cwd(runtime, str(raw.get("cwd") or ""))
        if not cwd.exists():
            errors.append(f"missing_server_cwd:{server_name}:{cwd}")

    if not errors and open_first_task and items:
        client = McpBenchWorkerClient(runtime_root=runtime, worker_script=worker)
        try:
            client.open_task(items[0])
            client.close_task()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"worker_open_task_failed:{exc}")
        finally:
            client.close()

    report = McpBenchPreflightReport(
        ok=not errors,
        runtime_root=str(runtime),
        worker_script=str(worker),
        checked_servers=checked_servers,
        errors=tuple(errors),
    )
    if errors and raise_on_error:
        raise RuntimeError("MCP-Bench runtime preflight failed: " + "; ".join(errors))
    return report


def _resolve_mcp_server_cwd(runtime_root: Path, raw_cwd: str) -> Path:
    if not raw_cwd:
        return runtime_root
    if raw_cwd.startswith("../"):
        return (runtime_root / "mcp_servers" / raw_cwd[3:]).resolve()
    return (runtime_root / raw_cwd).resolve()


def presented_task(item: McpBenchItem) -> str:
    fuzzy = item.task.fuzzy_description.strip()
    return fuzzy or item.task.task_description.strip()


def build_planning_json_call_prompt(
    item: McpBenchItem,
    available_tools: Mapping[str, Mapping[str, Any]],
    prompt_messages: Sequence[Mapping[str, object]],
    *,
    history_max_chars: int,
) -> str:
    system_prompt = normalize_rwkv_text(
        "\n".join(
            [
                "Tools:",
                render_tool_catalog(available_tools),
                "Output JSON schema:",
                _render_mcp_output_schema(),
                "Return exactly one JSON value that validates against the schema.",
                "Use final_answer when no more MCP tool calls are needed.",
                "Do not invent tool names, arguments, or tool results.",
                "Return no prose, no markdown, and no extra text outside the JSON value.",
            ]
        )
    )
    if not prompt_messages:
        prompt_messages = ({"role": "user", "content": build_mcp_task_user_message(item)},)
    return build_rwkv_json_call_prompt(system_prompt, prompt_messages, history_max_chars=history_max_chars)


def _render_mcp_output_schema() -> str:
    schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"},
        },
    }
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=False)


def build_mcp_task_user_message(item: McpBenchItem) -> str:
    return normalize_rwkv_text(
        "\n".join(
            [
                "Task:",
                presented_task(item),
            ]
        )
    )


def build_final_answer_prompt(item: McpBenchItem, accumulated_information: str) -> str:
    history = (
        trim_history(accumulated_information, MCP_BENCH_MAX_HISTORY_CHARS)
        if accumulated_information.strip()
        else "No tool evidence was gathered."
    )
    system_prompt = normalize_rwkv_text(
        "\n".join(
            [
                "You are the final answer synthesizer for an MCP benchmark agent.",
                "Use only the gathered evidence below. Do not invent missing facts.",
                "Return only the final answer requested by the task.",
                "If the task requires JSON, return valid JSON and nothing else.",
            ]
        )
    )
    user_prompt = normalize_rwkv_text(
        "\n".join(
            [
                "Task:",
                presented_task(item),
                "Function output history:",
                history,
            ]
        )
    )
    return "\n".join(
        [
            f"System: {system_prompt}",
            f"User:{user_prompt}",
            "Assistant:",
        ]
    )


def append_round_summary(
    accumulated_information: str,
    round_num: int,
    reasoning: str,
    executions: Sequence[McpBenchExecutionResult],
) -> str:
    lines = [accumulated_information, "", f"--- Summary of Round {round_num} ---"]
    if reasoning.strip():
        lines.append(f"Planner reasoning: {reasoning.strip()}")
    for execution in executions:
        params = json.dumps(execution.parameters, ensure_ascii=False) if execution.parameters else "{}"
        if execution.success:
            rendered = truncate_text(execution.result or "", MCP_BENCH_MAX_RESULT_CHARS)
            lines.append(
                f"Tool `{execution.tool}` with Parameter {params} on {execution.server} succeeded. Result: {rendered}"
            )
        else:
            rendered = truncate_text(execution.error or "", MCP_BENCH_MAX_ERROR_CHARS)
            lines.append(
                f"Tool `{execution.tool}` with Parameter {params} on {execution.server} failed. Error: {rendered}"
            )
    updated = "\n".join(part for part in lines if part is not None)
    return trim_history(updated, MCP_BENCH_MAX_HISTORY_CHARS)


def render_trace(steps: Sequence[dict[str, Any]]) -> str:
    parts: list[str] = []
    for step in steps:
        round_num = int(step.get("round_num", 0))
        parts.append(f"[Round {round_num}]")
        cot = str(step.get("cot") or "").strip()
        if cot:
            parts.append(f"<think>{cot}</think>")
        decision = step.get("decision")
        if decision is not None:
            parts.append("Decision:")
            parts.append(json.dumps(decision, ensure_ascii=False, indent=2))
        for execution in step.get("executions", []) or []:
            if not isinstance(execution, dict):
                continue
            parts.append(
                f"- {execution.get('tool')} [{execution.get('server')}] success={execution.get('success')}"
            )
        parts.append("")
    return "\n".join(parts).strip()


def parse_planning_decision(response: str) -> PlanningDecision:
    candidate = extract_json_candidate(response)
    try:
        payload = coerce_json_function_call_payload(json.loads(candidate), context_label="planning")
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse planning json: {exc}; json={candidate}") from exc
    name = str(payload["name"]).strip()
    if not name:
        raise ValueError("planning JSON missing name")
    arguments = payload["arguments"]
    if name == "final_answer":
        return PlanningDecision(
            reasoning="",
            should_continue=False,
            tool_calls=(),
            final_answer=str(arguments.get("answer") or "").strip(),
        )
    server = ""
    tool = name
    if ":" in tool:
        server, tool = [part.strip() for part in tool.split(":", 1)]

    return PlanningDecision(
        reasoning="",
        should_continue=True,
        tool_calls=(PlannedToolCall(server=server, tool=tool, arguments=dict(arguments)),),
    )


def normalize_planned_tool_call(
    call: PlannedToolCall,
    available_tools: Mapping[str, Mapping[str, Any]],
) -> PlannedToolCall:
    server = call.server.strip()
    tool = call.tool.strip()
    if not server and ":" in tool:
        server, tool = [part.strip() for part in tool.split(":", 1)]
    full_name = f"{server}:{tool}" if server else tool
    if server and full_name in available_tools:
        return PlannedToolCall(server=server, tool=tool, arguments=dict(call.arguments))
    if not server:
        matches = sorted(name for name in available_tools if name.endswith(f":{tool}"))
        if len(matches) == 1:
            matched_server, matched_tool = matches[0].split(":", 1)
            return PlannedToolCall(server=matched_server, tool=matched_tool, arguments=dict(call.arguments))
    raise ValueError(
        f"planned tool `{full_name}` was not found in available tools"
    )


def extract_json_candidate(response: str) -> str:
    if not response.strip():
        raise ValueError("model returned empty planning response")
    return extract_json_call_value_text(response)


def collapse_mcp_bench_pass(evaluation: McpBenchEvaluation) -> bool:
    return (
        evaluation.task_completion_score >= MCP_BENCH_PASS_THRESHOLD
        and evaluation.tool_selection_score >= MCP_BENCH_PASS_THRESHOLD
        and evaluation.planning_effectiveness_and_efficiency_score >= MCP_BENCH_PASS_THRESHOLD
    )


def summarize_mcp_bench_evaluation(evaluation: McpBenchEvaluation) -> str:
    return (
        f"task_completion_score={evaluation.task_completion_score:.2f}, "
        f"tool_selection_score={evaluation.tool_selection_score:.2f}, "
        f"planning_effectiveness_and_efficiency_score={evaluation.planning_effectiveness_and_efficiency_score:.2f}, "
        f"valid_tool_name_rate={_fmt_optional(evaluation.valid_tool_name_rate)}, "
        f"execution_success_rate={_fmt_optional(evaluation.execution_success_rate)}, "
        f"planning_json_compliance={_fmt_optional(evaluation.planning_json_compliance)}"
    )


def render_tool_catalog(available_tools: Mapping[str, Mapping[str, Any]]) -> str:
    rendered: list[dict[str, Any]] = []
    for tool in available_tools.values():
        server = str(tool.get("server") or "")
        name = str(tool.get("name") or "").strip()
        schema = tool.get("input_schema")
        arguments: Any = {}
        if isinstance(schema, Mapping):
            properties = schema.get("properties")
            arguments = dict(properties) if isinstance(properties, Mapping) else dict(schema)
        rendered.append(
            {
                "name": f"{server}:{name}" if server else name,
                "description": truncate_text(
                    str(tool.get("description") or "").strip() or "No description available",
                    400,
                ),
                "arguments": arguments,
            }
        )
    if not any(str(item.get("name") or "") == "final_answer" for item in rendered):
        rendered.append(
            {
                "name": "final_answer",
                "description": "Use this when the task is complete.",
                "arguments": {"answer": {"type": "string"}},
            }
        )
    return json.dumps(sorted(rendered, key=lambda item: str(item.get("name") or "")), ensure_ascii=False, indent=2)


def render_schema_summary(schema: Any) -> str:
    if schema is None:
        return ""
    return truncate_text(json.dumps(schema, ensure_ascii=False), MCP_BENCH_MAX_TOOL_SCHEMA_CHARS)


def _fmt_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_mcp_bench_ref_answer(item: McpBenchItem) -> str:
    return (
        f"task_id={item.task.task_id}\n"
        f"task_file={item.task_file}\n"
        f"server_name={item.server_name}\n"
        f"servers={', '.join(item.servers)}\n"
        f"combination_type={item.combination_type}\n"
        "runtime_origin=local_rwkv_rs_snapshot\n"
        "evaluator=official_mcp_bench_evaluator_phase2"
    )


def _mcp_bench_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("fail_reason") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=str(agent_info.get("final_answer") or ""),
        ref_answer=str(agent_info.get("ref_answer") or ""),
    )


def _execution_to_dict(entry: McpBenchExecutionResult) -> dict[str, object]:
    return {
        "tool": entry.tool,
        "server": entry.server,
        "parameters": dict(entry.parameters),
        "round_num": int(entry.round_num),
        "planned_layer": entry.planned_layer,
        "success": bool(entry.success),
        "result": entry.result,
        "error": entry.error,
    }


def _run_mcp_bench(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    items = load_mcp_bench_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        items = items[: int(args.max_samples)]
    if not items:
        raise ValueError("MCP-Bench manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(items), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    base_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="cot",
        fallback_templates="instruction_following_default",
    )
    if base_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    normalize_function_prompt_style(getattr(args, "prompt_style", None))
    decision_sampling = clamp_function_calling_sampling(base_sampling, args.decision_max_tokens or 2048)
    final_sampling = base_sampling.clamp(args.final_max_tokens)

    runtime_root = Path(items[0].runtime_root or "").expanduser().resolve()
    worker_script = REPO_ROOT / "src" / "eval" / "function_calling" / "mcp_bench_worker.py"
    if not bool(getattr(args, "skip_runtime_preflight", False)):
        preflight_mcp_bench_runtime(
            items,
            runtime_root=runtime_root,
            worker_script=worker_script,
            open_first_task=True,
        )
    if args.probe_only:
        worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
        try:
            available_tools = worker.open_task(items[0])
            prompt = build_planning_json_call_prompt(
                items[0],
                available_tools,
                ({"role": "user", "content": build_mcp_task_user_message(items[0])},),
                history_max_chars=int(args.history_max_chars),
            )
            run.engine.generate(
                [prompt],
                sampling=decision_sampling,
                batch_size=1,
                progress_desc="MCPBench-Probe",
                prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
            )
            worker.close_task()
        finally:
            worker.close()
        print("probe-only run completed: 1 task")
        return 0

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("MCP-Bench requires JUDGE_MODEL + JUDGE_API_KEY")

    job_name = _resolve_job_name("function_mcp_bench", run_context=run_context)
    sampling_payload = normalize_sampling_config_by_stage([(1, decision_sampling), (2, final_sampling)])
    ctx = prepare_function_calling_run(
        dataset_slug=str(run.dataset_slug),
        model_name=run.model_name,
        job_name=job_name,
        attempt_keys=attempt_keys,
        expected_attempt_count=plan_attempt_count(plan, max_pass_k=1),
        sampling_payload=sampling_payload,
        avg_k=plan.avg_k,
        effective_sample_count=plan.effective_sample_count,
        db_write_queue=int(args.db_write_queue or 8),
        run_context=run_context,
        judger_model_name=judge_cfg.model_name,
    )
    worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
    runtime = ctx.runtime
    writer = ctx.writer
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_mcp_bench_completion_to_eval_payload,
        runner_name="mcp_bench",
    )

    def _handle_runtime_interrupt(signame: str) -> None:
        try:
            worker.close()
        finally:
            _flush_partial_eval(signame)

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=_handle_runtime_interrupt,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, items, skip_keys=ctx.skip_keys)
                max_rounds = max(1, int(args.max_rounds))
                for key, item in pending:
                    sample_index = key.sample_index
                    repeat_index = key.repeat_index
                    stages: list[StageRecord] = []
                    steps: list[dict[str, object]] = []
                    accumulated_information = ""
                    execution_results: list[McpBenchExecutionResult] = []
                    final_answer = ""
                    fail_reason = ""
                    evaluation_summary = ""
                    is_passed = False
                    try:
                        available_tools = worker.open_task(item)
                        prompt_messages: list[dict[str, str]] = [
                            {"role": "user", "content": build_mcp_task_user_message(item)}
                        ]
                        total_planned_tools = 0
                        valid_planned_tools = 0
                        executed_rounds = 0
                        for round_num in range(1, max_rounds + 1):
                            decision_prompt = build_planning_json_call_prompt(
                                item,
                                available_tools,
                                prompt_messages,
                                history_max_chars=int(args.history_max_chars),
                            )
                            decision_output = run.engine.generate(
                                [decision_prompt],
                                sampling=decision_sampling,
                                batch_size=1,
                                progress_desc="MCPBench-Decision",
                                prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                                prompt_seeds=[
                                    sample_repeat_seed(
                                        sample_index,
                                        repeat_index,
                                        pass_index=key.pass_index,
                                        stage=round_num,
                                    )
                                ],
                            )[0]
                            stages.append(
                                StageRecord(
                                    prompt=decision_prompt,
                                    completion=decision_output.text,
                                    stop_reason=decision_output.finish_reason,
                                )
                            )
                            try:
                                decision = parse_planning_decision(decision_output.text)
                            except Exception as exc:
                                fail_reason = str(exc)
                                steps.append(
                                    {
                                        "round_num": round_num,
                                        "cot": "",
                                        "decision": {"raw": decision_output.text, "parse_error": fail_reason},
                                        "executions": [],
                                    }
                                )
                                break

                            round_executions: list[McpBenchExecutionResult] = []
                            if decision.should_continue:
                                for raw_call in decision.tool_calls:
                                    prompt_messages.append(
                                        {
                                            "role": "assistant",
                                            "content": render_json_function_call(
                                                raw_call.full_name if raw_call.server.strip() else raw_call.tool,
                                                raw_call.arguments,
                                            ),
                                        }
                                    )
                            else:
                                final_answer = decision.final_answer.strip()
                                prompt_messages.append(
                                    {
                                        "role": "assistant",
                                        "content": render_json_function_call(
                                            "final_answer",
                                            {"answer": final_answer},
                                        ),
                                    }
                                )
                            if decision.should_continue:
                                for planned_layer, raw_call in enumerate(decision.tool_calls):
                                    total_planned_tools += 1
                                    try:
                                        normalized = normalize_planned_tool_call(raw_call, available_tools)
                                        valid_planned_tools += 1
                                        tool_response = worker.call_tool(normalized.full_name, normalized.arguments)
                                        success = bool(tool_response.get("success", False))
                                        round_executions.append(
                                            McpBenchExecutionResult(
                                                tool=normalized.full_name,
                                                server=normalized.server,
                                                parameters=dict(normalized.arguments),
                                                round_num=round_num,
                                                planned_layer=planned_layer,
                                                success=success,
                                                result=str(tool_response.get("result") or "") or None,
                                                error=str(tool_response.get("error") or "") or None,
                                            )
                                        )
                                    except Exception as exc:
                                        tool_name = raw_call.full_name if raw_call.server.strip() else raw_call.tool
                                        round_executions.append(
                                            McpBenchExecutionResult(
                                                tool=tool_name,
                                                server=raw_call.server.strip() or "unknown",
                                                parameters=dict(raw_call.arguments),
                                                round_num=round_num,
                                                planned_layer=planned_layer,
                                                success=False,
                                                error=str(exc),
                                            )
                                        )
                            for execution in round_executions:
                                prompt_messages.append(
                                    {
                                        "role": "user",
                                        "content": render_function_output_user_block(_execution_to_dict(execution)),
                                    }
                                )
                            steps.append(
                                {
                                    "round_num": round_num,
                                    "cot": "",
                                    "decision": {
                                        "reasoning": decision.reasoning,
                                        "should_continue": decision.should_continue,
                                        "final_answer": decision.final_answer,
                                        "tool_calls": [
                                            {
                                                "server": call.server,
                                                "tool": call.tool,
                                                "arguments": dict(call.arguments),
                                            }
                                            for call in decision.tool_calls
                                        ],
                                    },
                                    "executions": [_execution_to_dict(entry) for entry in round_executions],
                                }
                            )
                            if round_executions:
                                accumulated_information = append_round_summary(
                                    accumulated_information,
                                    round_num,
                                    decision.reasoning,
                                    round_executions,
                                )
                                execution_results.extend(round_executions)
                                executed_rounds = round_num
                            if not decision.should_continue or not round_executions:
                                break

                        if not fail_reason and not final_answer:
                            final_prompt = build_final_answer_prompt(item, accumulated_information)
                            final_output = run.engine.generate(
                                [final_prompt],
                                sampling=final_sampling,
                                batch_size=1,
                                progress_desc="MCPBench-Final",
                                prompt_seeds=[
                                    sample_repeat_seed(
                                        sample_index,
                                        repeat_index,
                                        pass_index=key.pass_index,
                                        stage=max_rounds * 3,
                                    )
                                ],
                            )[0]
                            stages.append(
                                StageRecord(
                                    prompt=final_prompt,
                                    completion=final_output.text,
                                    stop_reason=final_output.finish_reason,
                                )
                            )
                            final_answer = final_output.text.strip()
                            planning_json_compliance = (
                                valid_planned_tools / total_planned_tools if total_planned_tools > 0 else 1.0
                            )
                            evaluation = worker.evaluate(
                                {
                                    "judge_config": {
                                        "api_key": judge_cfg.api_key,
                                        "base_url": judge_cfg.base_url or "",
                                        "model": judge_cfg.model_name,
                                    },
                                    "task": presented_task(item),
                                    "final_solution": final_answer,
                                    "total_rounds": executed_rounds,
                                    "available_tools": available_tools,
                                    "planning_json_compliance": planning_json_compliance,
                                    "accumulated_information": accumulated_information,
                                    "concrete_task_description": (
                                        item.task.task_description if item.task.fuzzy_description.strip() else ""
                                    ),
                                    "dependency_analysis": item.task.dependency_analysis,
                                    "execution_results": [_execution_to_dict(entry) for entry in execution_results],
                                }
                            )
                            is_passed = collapse_mcp_bench_pass(evaluation)
                            evaluation_summary = summarize_mcp_bench_evaluation(evaluation)
                            if not is_passed:
                                fail_reason = evaluation_summary
                        if final_answer and not fail_reason and not evaluation_summary:
                            planning_json_compliance = (
                                valid_planned_tools / total_planned_tools if total_planned_tools > 0 else 1.0
                            )
                            evaluation = worker.evaluate(
                                {
                                    "judge_config": {
                                        "api_key": judge_cfg.api_key,
                                        "base_url": judge_cfg.base_url or "",
                                        "model": judge_cfg.model_name,
                                    },
                                    "task": presented_task(item),
                                    "final_solution": final_answer,
                                    "total_rounds": executed_rounds,
                                    "available_tools": available_tools,
                                    "planning_json_compliance": planning_json_compliance,
                                    "accumulated_information": accumulated_information,
                                    "concrete_task_description": (
                                        item.task.task_description if item.task.fuzzy_description.strip() else ""
                                    ),
                                    "dependency_analysis": item.task.dependency_analysis,
                                    "execution_results": [_execution_to_dict(entry) for entry in execution_results],
                                }
                            )
                            is_passed = collapse_mcp_bench_pass(evaluation)
                            evaluation_summary = summarize_mcp_bench_evaluation(evaluation)
                            if not is_passed:
                                fail_reason = evaluation_summary
                        if not final_answer and not fail_reason:
                            fail_reason = "mcp_bench produced no final answer"
                        ref_answer = build_mcp_bench_ref_answer(item)
                        payload = SampleRecord(
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sample_index=sample_index,
                            repeat_index=repeat_index,
                            pass_index=key.pass_index,
                            stages=stages,
                            sampling_config=sampling_payload,
                        ).as_payload()
                        payload["agent_result"] = {
                            "reward": 1.0 if is_passed else 0.0,
                            "num_turns": len(steps),
                            "cost": 0.0,
                            "is_passed": is_passed,
                            "error": fail_reason or None,
                        }
                        payload["agent_info"] = {
                            "final_answer": final_answer,
                            "ref_answer": ref_answer,
                            "fail_reason": fail_reason,
                            "evaluation_summary": evaluation_summary,
                            "cot_mode": CoTMode.COT.value,
                            "execution_count": len(execution_results),
                        }
                        payload["agent_trace"] = steps
                        payload["task_id"] = item.task.task_id
                        payload["domain"] = "function_call"
                        payload["instruction"] = presented_task(item)
                        writer.enqueue(payload)
                    except Exception as exc:
                        ref_answer = build_mcp_bench_ref_answer(item)
                        payload = SampleRecord(
                            benchmark_name=run.benchmark_name,
                            dataset_split=run.dataset_split,
                            sample_index=sample_index,
                            repeat_index=repeat_index,
                            pass_index=key.pass_index,
                            stages=stages,
                            sampling_config=sampling_payload,
                        ).as_payload()
                        payload["agent_result"] = {
                            "reward": 0.0,
                            "num_turns": len(steps),
                            "cost": 0.0,
                            "is_passed": False,
                            "error": str(exc),
                        }
                        payload["agent_info"] = {
                            "final_answer": final_answer,
                            "ref_answer": ref_answer,
                            "fail_reason": str(exc),
                            "evaluation_summary": "",
                            "cot_mode": CoTMode.COT.value,
                            "execution_count": len(execution_results),
                        }
                        payload["agent_trace"] = steps
                        payload["task_id"] = item.task.task_id
                        payload["domain"] = "function_call"
                        payload["instruction"] = presented_task(item)
                        writer.enqueue(payload)
                    finally:
                        try:
                            worker.close_task()
                        except Exception:
                            pass
            except Exception:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _handle_runtime_interrupt("exception"),
                )
                raise
    finally:
        worker.close()

    try:
        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_mcp_bench_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=len(items),
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={
                    "sampling_config": sampling_payload,
                    "judger_model_name": judge_cfg.model_name,
                    "cot_mode": CoTMode.COT.value,
                },
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"mcp_bench done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0
