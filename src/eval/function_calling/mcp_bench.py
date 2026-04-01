from __future__ import annotations

import json
import re
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


MCP_BENCH_PASS_THRESHOLD = 7.0
MCP_BENCH_MAX_TOOL_SCHEMA_CHARS = 1200
MCP_BENCH_MAX_RESULT_CHARS = 4000
MCP_BENCH_MAX_ERROR_CHARS = 1000
MCP_BENCH_MAX_HISTORY_CHARS = 24000


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
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None:
                try:
                    self._request("shutdown", {})
                except Exception:
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


def load_mcp_bench_task_items(tasks_root: str | Path, runtime_root: str | Path) -> list[McpBenchItem]:
    root = Path(tasks_root)
    runtime = Path(runtime_root)
    file_names = (
        "mcpbench_tasks_single_runner_format.json",
        "mcpbench_tasks_multi_2server_runner_format.json",
        "mcpbench_tasks_multi_3server_runner_format.json",
    )
    items: list[McpBenchItem] = []
    for file_name in file_names:
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


def presented_task(item: McpBenchItem) -> str:
    fuzzy = item.task.fuzzy_description.strip()
    return fuzzy or item.task.task_description.strip()


def build_context_summary(item: McpBenchItem) -> str:
    return (
        "benchmark=mcp_bench\n"
        "phase=2_python_bridge_completion_loop\n"
        f"task_file={item.task_file}\n"
        f"server_name={item.server_name}\n"
        f"servers={', '.join(item.servers)}\n"
        f"combination_name={item.combination_name}\n"
        f"combination_type={item.combination_type}\n"
        f"task_id={item.task.task_id}\n\n"
        "task_presented_to_agent:\n"
        f"{presented_task(item)}\n\n"
        "concrete_task_reference:\n"
        f"{item.task.task_description.strip()}\n\n"
        "dependency_analysis_reference:\n"
        f"{item.task.dependency_analysis.strip()}\n"
    )


def build_planning_context(
    item: McpBenchItem,
    available_tools: Mapping[str, Mapping[str, Any]],
    accumulated_information: str,
) -> str:
    history = (
        trim_history(accumulated_information, MCP_BENCH_MAX_HISTORY_CHARS)
        if accumulated_information.strip()
        else "No previous tool results."
    )
    return (
        "System: You are a tool-using MCP benchmark agent operating in a real multi-server environment.\n"
        "You must decide round by round which MCP tools to call.\n"
        "Only plan tool calls that can be executed with information already available at the start of this round.\n"
        "If multiple tool calls are independent, you may include all of them in the same round.\n"
        "Do not invent tool names, parameters, or results.\n"
        "When the task has enough evidence, stop planning and let the final answer be synthesized from gathered evidence.\n\n"
        "User: TASK PRESENTED TO AGENT:\n"
        f"{presented_task(item)}\n\n"
        "AVAILABLE TOOLS:\n"
        f"{render_tool_catalog(available_tools)}\n\n"
        "EXECUTION HISTORY:\n"
        f"{history}\n\n"
        "Return reasoning privately inside <think> and then output a strict JSON planning decision.\n"
        "The JSON schema is:\n"
        "{\n"
        '  "reasoning": "brief planning rationale",\n'
        '  "should_continue": true,\n'
        '  "tool_calls": [\n'
        '    {"server": "Exact Server Name", "tool": "exact_tool_name", "arguments": {}}\n'
        "  ]\n"
        "}\n"
        'If no more tools are needed, set "should_continue" to false and "tool_calls" to [].\n\n'
        "Assistant: <think><|completions_of_cot|>"
    )


def build_planning_decision_prompt(cot_context: str, cot: str) -> str:
    return cot_context.replace("<|completions_of_cot|>", cot) + "</think>\nReturn ONLY the JSON planning decision object and nothing else.\n"


def build_final_answer_prompt(item: McpBenchItem, accumulated_information: str) -> str:
    history = (
        trim_history(accumulated_information, MCP_BENCH_MAX_HISTORY_CHARS)
        if accumulated_information.strip()
        else "No tool evidence was gathered."
    )
    return (
        "System: You are the final answer synthesizer for an MCP benchmark agent.\n"
        "Use only the gathered evidence below. Do not invent missing facts.\n"
        "Return only the final answer requested by the task.\n"
        "If the task requires JSON, return valid JSON and nothing else.\n\n"
        "User: TASK PRESENTED TO AGENT:\n"
        f"{presented_task(item)}\n\n"
        "GATHERED EVIDENCE:\n"
        f"{history}\n\n"
        "Assistant:"
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
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse planning json: {exc}; json={candidate}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"planning response is not a JSON object: {candidate}")

    raw_calls = payload.get("tool_calls")
    if not isinstance(raw_calls, list) or (not raw_calls and isinstance(payload.get("planned_tools"), list)):
        raw_calls = payload.get("planned_tools") or []

    tool_calls: list[PlannedToolCall] = []
    for item in raw_calls:
        if not isinstance(item, dict):
            continue
        server = str(item.get("server") or "").strip()
        tool = str(item.get("tool") or "").strip()
        arguments = item.get("arguments")
        if not isinstance(arguments, dict) or not arguments:
            parameters = item.get("parameters")
            if isinstance(parameters, dict):
                arguments = parameters
        if not isinstance(arguments, dict):
            arguments = {}
        if not server and ":" in tool:
            server, tool = [part.strip() for part in tool.split(":", 1)]
        tool_calls.append(PlannedToolCall(server=server, tool=tool, arguments=dict(arguments)))

    return PlanningDecision(
        reasoning=str(payload.get("reasoning") or "").strip(),
        should_continue=bool(payload.get("should_continue", False)),
        tool_calls=tuple(tool_calls),
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
    trimmed = response.strip()
    if not trimmed:
        raise ValueError("model returned empty planning response")
    match = re.search(r"```json\s*(\{.*?\})\s*```", trimmed, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start >= 0 and end > start:
        return trimmed[start : end + 1].strip()
    raise ValueError(f"model response did not contain json: {trimmed}")


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
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for tool in available_tools.values():
        server = str(tool.get("server") or "")
        grouped.setdefault(server, []).append(tool)
    lines: list[str] = []
    for server in sorted(grouped):
        lines.append(f"[{server}]")
        for tool in sorted(grouped[server], key=lambda item: str(item.get("name") or "")):
            description = str(tool.get("description") or "").strip() or "No description available"
            lines.append(f"- {tool.get('name')}: {truncate_text(description, 400)}")
            schema = render_schema_summary(tool.get("input_schema"))
            if schema:
                lines.append(f"  schema: {schema}")
        lines.append("")
    return "\n".join(lines).strip()


def render_schema_summary(schema: Any) -> str:
    if schema is None:
        return ""
    return truncate_text(json.dumps(schema, ensure_ascii=False), MCP_BENCH_MAX_TOOL_SCHEMA_CHARS)


def trim_history(history: str, max_chars: int) -> str:
    if len(history) <= max_chars:
        return history
    keep_tail = max(0, max_chars - 64)
    return "[Earlier execution history truncated]\n\n" + history[-keep_tail:]


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


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
