from __future__ import annotations

"""Generic multi-turn agent-loop benchmark runner.

The model acts in the RWKV trained format — tools listed in the system prompt
with ``Return only a JSON function call.``, one JSON call per turn primed by
``Assistant: ```json``, and tool results fed back as
``User: Function output:\\n<json>``. Executors map calls onto the benchmark's
real environment and the benchmark's OFFICIAL verifier grades the episode
(see agent_loop_executors.py / agent_loop_verifiers.py).
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.concurrent_runner import run_episodes
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage
from src.eval.tasks.function_calling.agent_loop_executors import (
    AgentLoopExecutor,
    DEFAULT_MAX_OUTPUT_CHARS,
    ExecutorSpec,
    ManifestReplayExecutor,
    McpWorkerExecutor,
    ShellSandboxExecutor,
    WebSearchExecutor,
    step_outcome_to_function_output,
)
from src.eval.tasks.function_calling.agent_loop_verifiers import (
    AgentLoopVerdict,
    VerifierSpec,
    build_agent_loop_verifier,
    preflight_agent_loop_runtime,
)
from src.eval.tasks.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_messages_for_long_context, infer_query_from_messages
from src.eval.tasks.function_calling.context_budget import normalize_rwkv_text, truncate_text
from src.eval.tasks.function_calling.final_answer import FINAL_ANSWER_TOOL_NAME, final_answer_tool_schema
from src.eval.experiments.parallel_candidate_router.router import (
    ParallelCandidateRouterConfig,
    route_parallel_candidate_tool_call,
)
from src.eval.tasks.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_function_calling_sample_limit,
    _resolve_job_name,
)
from src.eval.tasks.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    assistant_json_prefix,
    build_rwkv_json_call_prompt,
    coerce_json_function_call_payloads,
    extract_json_call_value_text,
    render_function_output_user_block,
    render_json_function_call,
)
from src.eval.tasks.function_calling.simple_tool_call import _render_tool_catalog

if TYPE_CHECKING:
    import argparse

    from src.eval.evaluating.contracts import RunContext

DEFAULT_AGENT_LOOP_MAX_STEPS = 20
DEFAULT_AGENT_LOOP_MAX_TOOL_ERRORS = 5
_AUTO_CANDIDATE_ROUTER_MIN_TOOLS = 8
_AGENT_LOOP_CANDIDATE_ROUTER_POLICY = (
    "Agent-loop policy: choose exactly one next JSON function call for the official executor. "
    "Use only listed tool names. Use final_answer only when the task is complete. "
    "For shell-sandbox repo tasks, inspect the workspace, make required edits, and run relevant tests or verification commands before final_answer. "
    "For web-search tasks, gather search evidence before final_answer. "
    "Do not invent tool names, file paths, IDs, URLs, or tool outputs."
)
_FACTS_METADATA_KEYS = (
    "facts_text",
    "facts",
    "context",
    "source_context",
    "document",
    "documents",
    "policy",
    "test_commands",
    "test_files",
    "test_case_count",
    "test_command",
    "official_task_id",
    "display_title",
    "repo",
    "language",
    "category",
    "difficulty",
    "parser_name",
    "base_commit_hash",
    "docker_image",
)


@dataclass(frozen=True, slots=True)
class AgentLoopRecord:
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    executor: ExecutorSpec
    verifier: VerifierSpec
    system_extra: str = ""
    expected_tool_calls: tuple[dict[str, Any], ...] = ()
    recorded_tool_outputs: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def load_agent_loop_records(path: str | Path) -> list[AgentLoopRecord]:
    records: list[AgentLoopRecord] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{index + 1}: agent-loop rows must be objects")
            records.append(agent_loop_record_from_row(row, index=index))
    if not records:
        raise ValueError(f"agent-loop dataset is empty: {path}")
    return records


def agent_loop_record_from_row(row: Mapping[str, Any], *, index: int = 0) -> AgentLoopRecord:
    executor_raw = row.get("executor")
    verifier_raw = row.get("verifier")
    if not isinstance(executor_raw, Mapping) or not str(executor_raw.get("kind") or ""):
        raise ValueError(f"agent-loop row missing executor spec: {row.get('task_id')!r}")
    if not isinstance(verifier_raw, Mapping) or not str(verifier_raw.get("kind") or ""):
        raise ValueError(f"agent-loop row missing verifier spec: {row.get('task_id')!r}")
    return AgentLoopRecord(
        task_id=str(row.get("task_id") or f"agent_loop__{index:05d}"),
        instruction=str(row.get("instruction") or ""),
        tools=tuple(dict(tool) for tool in row.get("tools") or () if isinstance(tool, Mapping)),
        executor=ExecutorSpec(
            kind=str(executor_raw.get("kind")),
            config=dict(executor_raw.get("config") or {}),
        ),
        verifier=VerifierSpec(
            kind=str(verifier_raw.get("kind")),
            config=dict(verifier_raw.get("config") or {}),
        ),
        system_extra=str(row.get("system_extra") or ""),
        expected_tool_calls=tuple(dict(item) for item in row.get("expected_tool_calls") or () if isinstance(item, Mapping)),
        recorded_tool_outputs=tuple(
            dict(item) for item in row.get("recorded_tool_outputs") or () if isinstance(item, Mapping)
        ),
        metadata=dict(row.get("metadata") or {}),
    )


def build_agent_loop_executor(record: AgentLoopRecord, args: "argparse.Namespace") -> AgentLoopExecutor:
    kind = record.executor.kind
    config = record.executor.config
    if kind == "manifest_replay":
        return ManifestReplayExecutor(
            recorded_tool_outputs=record.recorded_tool_outputs,
            match=str(config.get("match") or "by_name"),
        )
    if kind == "shell_sandbox":
        return ShellSandboxExecutor(
            backend=str(config.get("backend") or "subprocess"),
            image=(str(config.get("image")) if config.get("image") else None) or (str(record.metadata.get("docker_image")) if record.metadata.get("docker_image") else None),
            dockerfile_context=(str(config.get("dockerfile_context")) if config.get("dockerfile_context") else None)
            or (str(record.metadata.get("dockerfile_context")) if record.metadata.get("dockerfile_context") else None),
            dockerfile_path=(str(config.get("dockerfile_path")) if config.get("dockerfile_path") else None)
            or (str(record.metadata.get("dockerfile_path")) if record.metadata.get("dockerfile_path") else None),
            docker_compose_file=(str(config.get("docker_compose_file")) if config.get("docker_compose_file") else None)
            or (str(record.metadata.get("docker_compose_file")) if record.metadata.get("docker_compose_file") else None),
            docker_copy_paths=tuple(
                dict(item) for item in config.get("docker_copy_paths") or () if isinstance(item, Mapping)
            ),
            workspace_archive=(str(config.get("workspace_archive")) if config.get("workspace_archive") else None),
            setup_commands=tuple(str(item) for item in config.get("setup_commands") or ()),
            command_timeout_s=float(
                config.get("command_timeout_s") or getattr(args, "agent_loop_command_timeout_s", None) or 60.0
            ),
            max_output_chars=int(
                config.get("max_output_chars") or getattr(args, "agent_loop_max_output_chars", None) or DEFAULT_MAX_OUTPUT_CHARS
            ),
            workspace_root=getattr(args, "agent_loop_workspace_root", None),
            container_workdir=str(config.get("container_workdir") or "/app"),
        )
    if kind == "mcp_worker":
        return McpWorkerExecutor(
            runtime_root=str(config.get("runtime_root") or ""),
            worker_script=(str(config.get("worker_script")) if config.get("worker_script") else None),
            servers=tuple(str(item) for item in config.get("servers") or ()),
        )
    if kind == "web_search":
        return WebSearchExecutor(
            max_output_chars=int(
                config.get("max_output_chars") or getattr(args, "agent_loop_max_output_chars", None) or DEFAULT_MAX_OUTPUT_CHARS
            ),
        )
    raise ValueError(f"unknown agent-loop executor kind: {kind!r}")


def build_agent_loop_system_prompt(record: AgentLoopRecord, tools: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Tools:",
        _render_tool_catalog(tuple(tools)),
        "Return only a JSON function call.",
        'The JSON shape is {"name":"tool_name","arguments":{...}}.',
        "Use only listed tool names.",
        'Call {"name":"final_answer","arguments":{"answer":"..."}} when the task is complete.',
        "Policy:",
        _agent_loop_runtime_policy(record),
    ]
    facts_text = _agent_loop_record_facts_text(record, max_chars=1600)
    if facts_text:
        lines.extend(["Known benchmark facts:", facts_text])
    if record.system_extra:
        lines.append(record.system_extra)
    return normalize_rwkv_text("\n".join(lines))


def build_agent_loop_prompt(
    record: AgentLoopRecord,
    tools: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, object]],
    *,
    history_max_chars: int,
) -> str:
    return build_rwkv_json_call_prompt(
        build_agent_loop_system_prompt(record, tools),
        messages,
        history_max_chars=history_max_chars,
        assistant_prefix=assistant_json_prefix(prefill_object=False),
        single_user_turn=False,
    )


def _active_tools(record: AgentLoopRecord, executor_tools: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    tools: list[dict[str, Any]] = [dict(tool) for tool in record.tools]
    known = {str(tool.get("name") or "") for tool in tools}
    for tool in executor_tools:
        name = str(tool.get("name") or "")
        if name and name not in known:
            tools.append(dict(tool))
            known.add(name)
    if FINAL_ANSWER_TOOL_NAME not in known:
        tools.append(final_answer_tool_schema())
    return tuple(tools)


def _decode_agent_loop_calls(completion: str) -> list[dict[str, Any]]:
    if _looks_like_template_leak(completion):
        raise ValueError("decision stage leaked internal template/control tokens")
    try:
        candidate = extract_json_call_value_text(completion)
        payload = json.loads(candidate)
        calls = coerce_json_function_call_payloads(payload, context_label="agent-loop decision")
    except ValueError:
        recovered = _recover_truncated_final_answer_call(completion)
        if recovered is None:
            raise
        calls = [recovered]
    return [{"name": str(call["name"]), "arguments": dict(call.get("arguments") or {})} for call in calls]


def _recover_truncated_final_answer_call(completion: str) -> dict[str, Any] | None:
    text = normalize_rwkv_text(completion)
    if not re.search(r'"name"\s*:\s*"final_answer"', text):
        return None
    answer: str | None = None
    object_match = re.search(r'"arguments"\s*:\s*\{\s*"answer"\s*:\s*"(?P<answer>.*)\Z', text, flags=re.S)
    if object_match:
        answer = _decode_partial_json_string_fragment(object_match.group("answer"))
    else:
        string_match = re.search(r'"arguments"\s*:\s*"(?P<arguments>.*)\Z', text, flags=re.S)
        if not string_match:
            return None
        arguments_text = _decode_partial_json_string_fragment(string_match.group("arguments"))
        try:
            parsed_arguments = json.loads(arguments_text)
        except json.JSONDecodeError:
            parsed_arguments = None
        if isinstance(parsed_arguments, Mapping):
            raw_answer = parsed_arguments.get("answer")
            if raw_answer is not None:
                answer = str(raw_answer)
        if answer is None:
            answer_match = re.search(r'"answer"\s*:\s*"(?P<answer>.*)\Z', arguments_text, flags=re.S)
            if answer_match:
                answer = _decode_partial_json_string_fragment(answer_match.group("answer"))
    if not answer:
        return None
    return {"name": FINAL_ANSWER_TOOL_NAME, "arguments": {"answer": answer.rstrip()}}


def _decode_partial_json_string_fragment(fragment: str) -> str:
    chars: list[str] = []
    index = 0
    while index < len(fragment):
        ch = fragment[index]
        if ch == '"':
            break
        if ch != "\\":
            chars.append(ch)
            index += 1
            continue
        index += 1
        if index >= len(fragment):
            break
        esc = fragment[index]
        if esc == "u" and index + 4 < len(fragment):
            hex_text = fragment[index + 1 : index + 5]
            try:
                chars.append(chr(int(hex_text, 16)))
                index += 5
                continue
            except ValueError:
                chars.append(esc)
                index += 1
                continue
        chars.append(
            {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }.get(esc, esc)
        )
        index += 1
    return "".join(chars)


def _agent_loop_long_doc_config(args: Any) -> LongDocEvidenceConfig:
    mode = str(getattr(args, "long_doc_mode", "lexical") or "lexical").strip().lower()
    enabled = mode != "off"
    if mode == "off":
        mode = "lexical"
    return LongDocEvidenceConfig(
        enabled=enabled,
        mode=mode,  # type: ignore[arg-type]
        max_chunk_chars=max(1, int(getattr(args, "long_doc_max_chars", 1000) or 1000)),
        overlap_lines=max(0, int(getattr(args, "long_doc_overlap_lines", 3) or 0)),
        min_long_text_chars=max(1, int(getattr(args, "long_doc_min_chars", 6000) or 6000)),
        max_evidence_chunks=max(1, int(getattr(args, "long_doc_max_evidence_chunks", 4) or 4)),
        max_evidence_chars=max(1, int(getattr(args, "long_doc_max_evidence_chars", 6000) or 6000)),
    )


def _agent_loop_candidate_router_mode(args: Any) -> str:
    mode = str(getattr(args, "candidate_router_mode", None) or "auto").strip().lower()
    if mode not in {"off", "auto", "parallel"}:
        raise ValueError(f"unsupported candidate_router_mode={mode!r}; expected off, auto, or parallel")
    return mode


def _agent_loop_candidate_router_config_from_args(args: Any) -> ParallelCandidateRouterConfig | None:
    mode = _agent_loop_candidate_router_mode(args)
    if mode == "off":
        return None
    defaults = ParallelCandidateRouterConfig()
    tool_schema_mode = str(
        getattr(args, "candidate_router_tool_schema_mode", defaults.tool_schema_mode) or defaults.tool_schema_mode
    )
    if tool_schema_mode not in {"minimal", "compact", "full"}:
        tool_schema_mode = defaults.tool_schema_mode
    return ParallelCandidateRouterConfig(
        chunk_tools=_positive_int(getattr(args, "candidate_router_chunk_tools", None), defaults.chunk_tools),
        batch_size=_positive_int(getattr(args, "candidate_router_batch_size", None), defaults.batch_size),
        context_chars=_positive_int(getattr(args, "candidate_router_context_chars", None), defaults.context_chars),
        prompt_max_chars=_positive_int(
            getattr(args, "candidate_router_prompt_max_chars", None),
            defaults.prompt_max_chars,
        ),
        candidate_max_tokens=_positive_int(
            getattr(args, "candidate_router_candidate_max_tokens", None),
            defaults.candidate_max_tokens,
        ),
        aggregate_max_tokens=_positive_int(
            getattr(args, "candidate_router_aggregate_max_tokens", None),
            defaults.aggregate_max_tokens,
        ),
        max_candidates=_positive_int(getattr(args, "candidate_router_max_candidates", None), defaults.max_candidates),
        tool_schema_mode=tool_schema_mode,
        include_respond=False,
        fallback_to_highest_confidence=True,
        evidence_chars=_positive_int(getattr(args, "candidate_router_evidence_chars", None), defaults.evidence_chars),
        policy_chars=_positive_int(getattr(args, "candidate_router_policy_chars", None), defaults.policy_chars),
        ground_identifier_arguments=not bool(getattr(args, "disable_candidate_router_grounding", False)),
    )


def _positive_int(raw: object, default: int) -> int:
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)


def _agent_loop_tool_schema_chars(tools: Sequence[Mapping[str, Any]]) -> int:
    return len(json.dumps(list(tools), ensure_ascii=False, sort_keys=True))


def _agent_loop_message_chars(messages: Sequence[Mapping[str, object]]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages)


def _should_use_agent_loop_candidate_router(
    *,
    mode: str,
    config: ParallelCandidateRouterConfig | None,
    tools: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, object]],
    candidate_context_chars: int = 0,
    raw_message_chars: int = 0,
    long_doc_compacted: bool = False,
) -> bool:
    if config is None or not tools:
        return False
    if mode == "parallel":
        return True
    if mode != "auto":
        return False
    if len(tools) >= max(_AUTO_CANDIDATE_ROUTER_MIN_TOOLS, int(config.chunk_tools) * 2):
        return True
    if long_doc_compacted:
        return True
    if int(raw_message_chars) > int(config.context_chars):
        return True
    if _agent_loop_message_chars(messages) > int(config.context_chars):
        return True
    if int(candidate_context_chars) > int(config.context_chars):
        return True
    if _agent_loop_tool_schema_chars(tools) > max(1, int(config.prompt_max_chars) // 2):
        return True
    return False


def _compact_agent_loop_messages(
    messages: Sequence[Mapping[str, object]],
    *,
    config: LongDocEvidenceConfig,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    query = infer_query_from_messages(
        messages,
        skip_longer_than=max(1, int(config.min_long_text_chars)),
    )
    compaction = compact_messages_for_long_context(messages, query=query, config=config)
    trace = {
        "enabled": bool(config.enabled),
        "mode": str(config.mode),
        "query_chars": len(query),
        "compacted_message_count": int(compaction.compacted_message_count),
        "selected_chunk_ids": {
            str(index): list(chunk_ids)
            for index, chunk_ids in compaction.selected_chunk_ids.items()
        },
    }
    return compaction.messages, trace


def _agent_loop_runtime_policy(record: AgentLoopRecord) -> str:
    lines = [_AGENT_LOOP_CANDIDATE_ROUTER_POLICY]
    domain = _agent_loop_domain(record).strip().lower()
    executor_kind = str(record.executor.kind or "").strip().lower()
    verifier_kind = str(record.verifier.kind or "").strip().lower()
    if executor_kind == "shell_sandbox":
        lines.append(
            "Shell workflow: first inspect the current workspace with bash/read_file, then create or edit files with write_file or shell commands, then run focused verification before final_answer."
        )
        lines.append("Never call final_answer as a substitute for implementing files or checking the environment.")
    if domain == "nl2repo":
        lines.append(
            "NL2Repo workflow: build the complete requested project in the empty workspace. Use metadata test_commands/test_files as verifier hints, but do not assume tests passed until commands have run or the workspace is complete."
        )
    elif domain == "deepswe":
        lines.append(
            "DeepSWE workflow: modify the checked-out repository in /app, preserve the requested base task, and use /pre_artifacts.sh plus /tests/test.sh as the official verification path when available."
        )
    elif domain == "terminal_bench_2_1":
        lines.append(
            "Terminal-Bench workflow: solve the terminal task inside /app. Inspect the task files and run the task's own validation or smoke checks before final_answer."
        )
    elif domain == "widesearch":
        lines.append(
            "WideSearch workflow: use web_search for broad discovery, fetch_url only for URLs returned by search or visible evidence, and answer with facts supported by gathered search results."
        )
        lines.append("When a Markdown table is requested, include each unique entity only once and do not repeat rows to fill space.")
    elif domain == "deepsearchqa":
        lines.append("DeepSearchQA workflow: call available tools before final_answer; do not answer from memory alone.")
    elif verifier_kind == "llm_rubric_judge":
        lines.append("Rubric-judge workflow: gather enough supporting evidence to satisfy the rubric before final_answer.")
    return normalize_rwkv_text(" ".join(lines))


def _agent_loop_record_facts_text(record: AgentLoopRecord, *, max_chars: int = 4000) -> str | None:
    rows: list[str] = [
        f"benchmark={_agent_loop_domain(record)}",
        f"executor={record.executor.kind}",
        f"verifier={record.verifier.kind}",
    ]
    for key in _FACTS_METADATA_KEYS:
        raw = record.metadata.get(key)
        if raw in (None, ""):
            continue
        rows.append(f"{key}={_compact_fact_value(raw)}")
    if record.verifier.config:
        for key in ("official_task_id", "test_command", "test_timeout_s", "pass_threshold"):
            raw = record.verifier.config.get(key)
            if raw in (None, ""):
                continue
            rows.append(f"verifier.{key}={_compact_fact_value(raw)}")
    if record.executor.config:
        for key in ("backend", "image", "container_workdir", "command_timeout_s", "max_output_chars"):
            raw = record.executor.config.get(key)
            if raw in (None, ""):
                continue
            rows.append(f"executor.{key}={_compact_fact_value(raw)}")
    text = normalize_rwkv_text("\n".join(rows))
    return truncate_text(text, max_chars) if text else None


def _compact_fact_value(value: Any) -> str:
    if isinstance(value, str):
        rendered = normalize_rwkv_text(value)
    else:
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return truncate_text(rendered, 900)


def _agent_loop_candidate_facts_text(record: AgentLoopRecord) -> str | None:
    return _agent_loop_record_facts_text(record, max_chars=4000)


def _agent_loop_candidate_context_chars(record: AgentLoopRecord) -> int:
    facts_text = _agent_loop_record_facts_text(record, max_chars=1_000_000)
    raw_max_chars = 0
    for key in _FACTS_METADATA_KEYS:
        raw = record.metadata.get(key)
        if raw in (None, ""):
            continue
        if isinstance(raw, str):
            rendered = raw
        else:
            rendered = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        raw_max_chars = max(raw_max_chars, len(rendered))
    return max(raw_max_chars, len(facts_text or ""))


def _agent_loop_domain(record: AgentLoopRecord) -> str:
    return str(
        record.metadata.get("source_benchmark")
        or record.metadata.get("benchmark")
        or record.metadata.get("dataset")
        or record.executor.kind
        or "agent_loop"
    )


def _run_candidate_routed_agent_loop_decision(
    *,
    record: AgentLoopRecord,
    engine: Any,
    tool_sampling: Any,
    tools: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, object]],
    config: ParallelCandidateRouterConfig,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    step: int,
    progress_prefix: str,
) -> tuple[StageRecord, list[dict[str, Any]], dict[str, Any], str | None]:
    route = route_parallel_candidate_tool_call(
        tools=tools,
        messages=messages,
        domain_policy=_agent_loop_runtime_policy(record),
        domain=_agent_loop_domain(record),
        facts_text=_agent_loop_candidate_facts_text(record),
        engine=engine,
        sampling=tool_sampling,
        config=config,
        progress_desc=f"{progress_prefix}-CandidateRouter sample {sample_index} step {step}",
        prompt_seed=sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=50_000 + step),
    )
    selected = route.selected
    if selected is None:
        calls: list[dict[str, Any]] = []
        decision_text = ""
        parse_error = str(route.aggregate_error or "candidate router did not select a tool call")
    else:
        calls = [{"name": selected.name, "arguments": dict(selected.arguments)}]
        decision_text = json.dumps(calls[0], ensure_ascii=False, sort_keys=True)
        parse_error = None
    completion = route.aggregate_completion or decision_text
    finish_reason = route.aggregate_finish_reason or ("candidate_router_empty" if selected is None else "stop")
    trace_payload = {
        "kind": "decision",
        "step": step,
        "decision_io": "parallel_candidate",
        "decoded_calls": calls,
        "parse_error": parse_error or "",
        "candidate_router": route.trace_payload(include_prompts=True),
    }
    return StageRecord(prompt=route.aggregate_prompt, completion=completion, stop_reason=finish_reason), calls, trace_payload, parse_error


def run_agent_loop_episode(
    *,
    record: AgentLoopRecord,
    engine: Any,
    tool_sampling: Any,
    executor: AgentLoopExecutor,
    verifier: Any,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    max_steps: int,
    max_tool_errors: int,
    history_max_chars: int,
    max_output_chars: int,
    long_doc_config: LongDocEvidenceConfig | None = None,
    candidate_router_mode: str = "off",
    candidate_router_config: ParallelCandidateRouterConfig | None = None,
    progress_prefix: str = "AgentLoop",
) -> dict[str, Any]:
    stages: list[StageRecord] = []
    trace: list[dict[str, Any]] = []
    final_answer = ""
    termination_reason = "max_steps"
    error: str | None = None
    tool_errors = 0

    executor_tools = executor.open()
    tools = _active_tools(record, executor_tools)
    messages: list[dict[str, object]] = [{"role": "user", "content": record.instruction}]
    candidate_context_chars = _agent_loop_candidate_context_chars(record)

    for step in range(1, max_steps + 1):
        raw_message_chars = _agent_loop_message_chars(messages)
        compacted_messages, long_doc_trace = _compact_agent_loop_messages(
            messages,
            config=long_doc_config or LongDocEvidenceConfig(enabled=False),
        )
        use_candidate_router = _should_use_agent_loop_candidate_router(
            mode=candidate_router_mode,
            config=candidate_router_config,
            tools=tools,
            messages=compacted_messages,
            candidate_context_chars=candidate_context_chars,
            raw_message_chars=raw_message_chars,
            long_doc_compacted=bool(long_doc_trace.get("compacted_message_count")),
        )
        if use_candidate_router and candidate_router_config is not None:
            stage, calls, decision_trace, parse_error = _run_candidate_routed_agent_loop_decision(
                record=record,
                engine=engine,
                tool_sampling=tool_sampling,
                tools=tools,
                messages=compacted_messages,
                config=candidate_router_config,
                sample_index=sample_index,
                repeat_index=repeat_index,
                pass_index=pass_index,
                step=step,
                progress_prefix=progress_prefix,
            )
            decision_trace["long_doc"] = long_doc_trace
            stages.append(stage)
            trace.append(decision_trace)
            if parse_error:
                termination_reason = "parse_error"
                error = parse_error
                break
        else:
            prompt = build_agent_loop_prompt(record, tools, compacted_messages, history_max_chars=history_max_chars)
            output = engine.generate(
                [prompt],
                sampling=tool_sampling,
                batch_size=1,
                progress_desc=f"{progress_prefix} sample {sample_index} step {step}",
                prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                prompt_seeds=[sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=step)],
            )[0]
            stages.append(StageRecord(prompt=prompt, completion=output.text, stop_reason=output.finish_reason))
            try:
                calls = _decode_agent_loop_calls(output.text)
                trace.append(
                    {
                        "kind": "decision",
                        "step": step,
                        "decision_io": "rwkv_json",
                        "decoded_calls": calls,
                        "long_doc": long_doc_trace,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - parse failures terminate the episode
                termination_reason = "parse_error"
                error = str(exc)
                trace.append(
                    {
                        "kind": "parse_error",
                        "step": step,
                        "error": str(exc),
                        "raw": output.text[:2000],
                        "long_doc": long_doc_trace,
                    }
                )
                break

        stop = False
        for call in calls:
            name = call["name"]
            arguments = call["arguments"]
            messages.append({"role": "assistant", "content": render_json_function_call(name, arguments)})
            if name == FINAL_ANSWER_TOOL_NAME:
                final_answer = str(arguments.get("answer") or "")
                termination_reason = "agent_stop"
                trace.append({"kind": "final_answer", "step": step, "answer": final_answer})
                stop = True
                break
            outcome = executor.execute(name, arguments)
            feedback = step_outcome_to_function_output(outcome, max_chars=max_output_chars)
            messages.append({"role": "user", "content": render_function_output_user_block(feedback)})
            trace.append(
                {
                    "kind": "tool_call",
                    "step": step,
                    "name": name,
                    "arguments": dict(arguments),
                    "success": bool(outcome.ok),
                    "output": feedback.get("output"),
                    "error": outcome.error,
                }
            )
            if not outcome.ok:
                tool_errors += 1
                if tool_errors >= max_tool_errors:
                    termination_reason = "too_many_errors"
                    error = f"tool errors reached limit ({max_tool_errors})"
                    stop = True
                    break
        if stop:
            break

    verdict: AgentLoopVerdict
    try:
        verdict = verifier.verify(
            record,
            final_answer=final_answer,
            trace=trace,
            executor_snapshot=executor.snapshot(),
        )
    except Exception as exc:  # noqa: BLE001 - checker failures degrade to failed verdicts
        verdict = AgentLoopVerdict(
            reward=0.0,
            is_passed=False,
            fail_reason=f"checker_error: {exc}",
            details={},
        )

    fail_reason = error or ("" if verdict.is_passed else verdict.fail_reason)
    return {
        "stages": stages,
        "trace": trace,
        "final_answer": final_answer,
        "termination_reason": termination_reason,
        "error": error,
        "verdict": verdict,
        "fail_reason": fail_reason,
        "num_turns": len(stages),
    }


def _agent_loop_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
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


def _ref_answer(record: AgentLoopRecord) -> str:
    if record.expected_tool_calls:
        return json.dumps(list(record.expected_tool_calls), ensure_ascii=False)
    reference = record.verifier.config.get("reference_answer") or record.metadata.get("reference_answer")
    return str(reference or "")


def _run_agent_loop(
    args: "argparse.Namespace",
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_agent_loop_records(run.dataset_path)
    sample_limit = _resolve_function_calling_sample_limit(
        str(run.dataset_slug),
        run.model_name,
        max_samples=args.max_samples,
    )
    if sample_limit is not None:
        records = records[:sample_limit]

    plan = _resolve_function_calling_plan(
        str(run.dataset_slug),
        len(records),
        avg_ks=args.avg_k,
        model_name=run.model_name,
        config_defaults=True,
    )
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    tool_sampling = resolve_sampling_config(
        str(run.dataset_slug),
        run.model_name,
        stage="tool",
        fallback_templates="function_call_default",
    )
    if tool_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    tool_sampling = clamp_function_calling_sampling(tool_sampling, max(1, int(args.decision_max_tokens or 1024)))
    long_doc_config = _agent_loop_long_doc_config(args)
    candidate_router_mode = _agent_loop_candidate_router_mode(args)
    candidate_router_config = _agent_loop_candidate_router_config_from_args(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, tool_sampling)]),
        long_doc_config=long_doc_config,
        candidate_router_config=candidate_router_config,
    )
    sampling_payload["agent_loop_adapter"] = {
        "candidate_router_mode": candidate_router_mode,
        "candidate_router_auto_min_tools": _AUTO_CANDIDATE_ROUTER_MIN_TOOLS,
        "decision_io": "rwkv_json_or_parallel_candidate",
    }
    history_max_chars = max(0, int(args.history_max_chars or 24000))
    max_steps = max(1, int(getattr(args, "max_steps", None) or DEFAULT_AGENT_LOOP_MAX_STEPS))
    max_tool_errors = max(1, int(getattr(args, "max_tool_errors", None) or DEFAULT_AGENT_LOOP_MAX_TOOL_ERRORS))
    max_output_chars = int(getattr(args, "agent_loop_max_output_chars", None) or DEFAULT_MAX_OUTPUT_CHARS)

    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]
    selected_records = [record for _index, record in selected_entries]
    if not bool(getattr(args, "skip_runtime_preflight", False)) and not args.probe_only:
        preflight_agent_loop_runtime(selected_records, args)

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=max(1, int(args.batch_size or 16)))
        prompts = [
            build_agent_loop_prompt(
                record,
                _active_tools(record, ()),
                [{"role": "user", "content": record.instruction}],
                history_max_chars=history_max_chars,
            )
            for _index, record in repeated
        ]
        run.engine.generate(
            prompts,
            sampling=tool_sampling,
            batch_size=len(prompts),
            progress_desc="AgentLoop-Probe",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
            prompt_seeds=[sample_repeat_seed(index, 0, stage=1) for index, _record in repeated],
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    verifiers = {
        kind: build_agent_loop_verifier(kind, args)
        for kind in sorted({record.verifier.kind for record in selected_records})
    }

    job_name = _resolve_job_name("function_agent_loop", run_context=run_context)
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
    )
    runtime = ctx.runtime
    writer = ctx.writer
    flush_partial = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_agent_loop_completion_to_eval_payload,
        runner_name="agent_loop",
    )

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=flush_partial,
        ):
            try:
                pending = build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys)
                sample_workers = max(1, int(getattr(args, "sample_workers", 1) or 1))

                def _run_pending_item(item: tuple[Any, AgentLoopRecord]) -> dict[str, Any]:
                    key, record = item
                    executor = build_agent_loop_executor(record, args)
                    episode_error: str | None = None
                    try:
                        episode = run_agent_loop_episode(
                            record=record,
                            engine=run.engine,
                            tool_sampling=tool_sampling,
                            executor=executor,
                            verifier=verifiers[record.verifier.kind],
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            max_steps=max_steps,
                            max_tool_errors=max_tool_errors,
                            history_max_chars=history_max_chars,
                            max_output_chars=max_output_chars,
                            long_doc_config=long_doc_config,
                            candidate_router_mode=candidate_router_mode,
                            candidate_router_config=candidate_router_config,
                        )
                    except Exception as exc:  # noqa: BLE001 - episode infra failures become failed samples
                        episode_error = str(exc)
                        episode = {
                            "stages": [],
                            "trace": [],
                            "final_answer": "",
                            "termination_reason": "episode_error",
                            "error": episode_error,
                            "verdict": AgentLoopVerdict(0.0, False, episode_error, {}),
                            "fail_reason": episode_error,
                            "num_turns": 0,
                        }
                    finally:
                        try:
                            executor.close()
                        except Exception:
                            pass
                    verdict: AgentLoopVerdict = episode["verdict"]
                    payload = SampleRecord(
                        benchmark_name=run.benchmark_name,
                        dataset_split=run.dataset_split,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        stages=episode["stages"],
                        sampling_config=sampling_payload,
                    ).as_payload()
                    payload["agent_result"] = {
                        "reward": float(verdict.reward),
                        "num_turns": int(episode["num_turns"]),
                        "cost": 0.0,
                        "is_passed": bool(verdict.is_passed),
                        "error": episode["error"] or (None if verdict.is_passed else verdict.fail_reason or None),
                    }
                    payload["agent_info"] = {
                        "final_answer": episode["final_answer"],
                        "ref_answer": _ref_answer(record),
                        "fail_reason": episode["fail_reason"],
                        "termination_reason": episode["termination_reason"],
                        "verifier_kind": record.verifier.kind,
                        "executor_kind": record.executor.kind,
                        "verdict_details": dict(verdict.details),
                        "cot_mode": CoTMode.COT.value,
                    }
                    payload["agent_trace"] = episode["trace"]
                    payload["task_id"] = record.task_id
                    payload["domain"] = "function_call"
                    payload["instruction"] = record.instruction
                    return payload

                run_episodes(
                    pending,
                    _run_pending_item,
                    max_workers=sample_workers,
                    on_result=writer.enqueue,
                    label="agent_loop episode",
                    collect_results=False,
                )
            except Exception:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_agent_loop_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={"cot_mode": CoTMode.COT.value, "max_steps": max_steps},
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"function_agent_loop done: {len(completions_payloads)} samples")
    return 0


__all__ = [
    "AgentLoopRecord",
    "agent_loop_record_from_row",
    "build_agent_loop_executor",
    "build_agent_loop_prompt",
    "build_agent_loop_system_prompt",
    "load_agent_loop_records",
    "run_agent_loop_episode",
    "_run_agent_loop",
]
