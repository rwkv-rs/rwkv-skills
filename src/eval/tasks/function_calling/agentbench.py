from __future__ import annotations

import argparse
import ast
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunSignalGuard
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
from src.eval.field_common import build_plan_task_details
from src.eval.tasks.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.tasks.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS, normalize_rwkv_text
from src.eval.experiments.parallel_candidate_router.router import (
    ParallelCandidateRouterConfig,
    route_parallel_candidate_tool_call,
)
from src.eval.tasks.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.tasks.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_rwkv_json_call_prompt,
)
from src.eval.tasks.function_calling.simple_tool_call import decode_simple_tool_call_response
from src.eval.tasks.function_calling.tool_router import (
    ToolRoutingConfig,
    route_tools_for_prompt,
    tool_catalog_chars,
    tool_routing_config_from_args,
)
from src.eval.long_doc_evidence import LongDocEvidenceConfig, compact_messages_for_long_context
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext


@dataclass(frozen=True, slots=True)
class AgentBenchRecord:
    task_id: str
    task_name: str
    index: int
    metadata: dict[str, Any]


_AGENTBENCH_CANDIDATE_ROUTER_MIN_TOOLS = 4
_MYSQL_MULTIWORD_TABLE_REF = re.compile(
    r"\b(?P<keyword>UPDATE|FROM|JOIN|INTO|DESCRIBE|DESC)\s+"
    r"(?P<name>(?!`)[A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*)+?)"
    r"(?=\s+(?:SET|WHERE|JOIN|ON|VALUES|ORDER|GROUP|LIMIT)\b|$)",
    flags=re.IGNORECASE,
)
_AGENTBENCH_CANDIDATE_ROUTER_POLICY = (
    "AgentBench policy: choose exactly one next JSON function call for the official controller. "
    "Use only listed tool names. Do not invent tool names, IDs, entities, tables, columns, or tool outputs. "
    "For DB tasks, call the SQL execution tool with a complete SQL query; arguments must be a JSON object, not an escaped string. "
    "For SQL identifiers with spaces, punctuation, reserved words, or mixed case, quote the full identifier with MySQL backticks. "
    "For example use `Team Information`, not Team Information, and `Season`, not an unverified column spelling. "
    "For KG tasks, use exploration tools until the target entity is identified, then use final_answer with exactly `Final Answer: #id`."
)


def _agentbench_final_answer_tool() -> dict[str, Any]:
    return {
        "name": "final_answer",
        "description": "Submit the final answer as assistant text content.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    }


def load_agentbench_rows_from_source(
    data_file: str | Path,
    *,
    dataset_name: str,
    task_name: str,
) -> list[dict[str, Any]]:
    path = Path(data_file).expanduser().resolve()
    count = _agentbench_data_count(path)
    return [
        {
            "task_id": f"{dataset_name}__{index:05d}",
            "task_name": task_name,
            "index": index,
            "metadata": {
                "source_format": "official_agentbench_controller",
                "source_path": str(path),
                "task_name": task_name,
            },
        }
        for index in range(count)
    ]


def load_agentbench_manifest_records(path: str | Path) -> list[AgentBenchRecord]:
    target = Path(path)
    records: list[AgentBenchRecord] = []
    with target.open("r", encoding="utf-8") as fh:
        for line_index, line in enumerate(fh):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            records.append(
                AgentBenchRecord(
                    task_id=str(payload.get("task_id") or f"agentbench_{line_index:04d}"),
                    task_name=str(payload.get("task_name") or payload.get("name") or ""),
                    index=int(payload.get("index", line_index)),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    return records


class AgentBenchControllerClient:
    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.timeout_s = max(1.0, float(timeout_s))

    def start_sample(self, task_name: str, index: int) -> tuple[str, dict[str, Any]]:
        data, headers = self._post("start_sample", {"name": task_name, "index": index})
        session_id = headers.get("session_id") or headers.get("Session-Id") or headers.get("Session-ID")
        if not session_id:
            raise RuntimeError("AgentBench controller did not return session_id header")
        return str(session_id), data

    def interact(self, session_id: str, message: Mapping[str, Any]) -> dict[str, Any]:
        data, _headers = self._post("interact", {"messages": [dict(message)]}, headers={"session_id": session_id})
        return data

    def cancel(self, session_id: str) -> None:
        try:
            self._post("cancel", {}, headers={"session_id": session_id})
        except Exception:  # noqa: BLE001
            pass

    def ensure_available(self) -> None:
        req = urllib.request.Request(f"{self.base_url}/list_workers", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout_s, 10.0)) as resp:
                resp.read()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"AgentBench controller is not available at {self.base_url}: {exc}"
            ) from exc

    def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        req_headers = {"content-type": "application/json"}
        req_headers.update(dict(headers or {}))
        req = urllib.request.Request(
            f"{self.base_url}/{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw) if raw.strip() else {}
                if not isinstance(data, dict):
                    raise RuntimeError("AgentBench controller response must be a JSON object")
                return data, resp.headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"AgentBench controller HTTP {exc.code}: {detail}") from exc


def build_agentbench_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    history_max_chars: int,
    allow_final_answer_text: bool,
    prompt_max_chars: int | None = None,
    long_doc_config: LongDocEvidenceConfig | None = None,
) -> str:
    system_messages = [str(item.get("content") or "") for item in messages if str(item.get("role") or "").lower() == "system"]
    dialog_messages = [item for item in messages if str(item.get("role") or "").lower() != "system"]
    if long_doc_config is not None:
        dialog_messages = compact_messages_for_long_context(dialog_messages, config=long_doc_config).messages
    tool_schemas = [_normalize_openai_tool(tool) for tool in tools]
    if allow_final_answer_text:
        tool_schemas.append(
            {
                "name": "final_answer",
                "description": "Submit the final answer as assistant text content.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            }
        )
    system_prompt = normalize_rwkv_text(
        "\n".join(
            [
                *system_messages,
                "Tools:",
                json.dumps(tool_schemas, ensure_ascii=False, indent=2, sort_keys=False),
                "Output JSON schema:",
                _render_agentbench_output_schema(),
                "Return exactly one JSON function call object.",
                "Use only listed tool names.",
                "The `arguments` value must be a JSON object, not an escaped JSON string.",
                "Do not abbreviate, summarize, or replace argument values with ellipses; output complete SQL queries and complete parameter strings.",
                "For DB tasks using MySQL, quote table or column names containing spaces, punctuation, reserved words, or mixed case with backticks.",
                "If a tool response shows a table name such as `Team Information`, reuse that exact full name inside backticks; do not split it into `Team`.",
                "Inspect available tables/schema before writing UPDATE/INSERT/DELETE unless the current conversation already contains the exact table and columns.",
                "For AgentBench KG final answers, use final_answer with the exact content `Final Answer: #id`.",
                "Return no prose, no markdown, and no extra text outside the JSON value.",
            ]
        )
    )
    prompt = build_rwkv_json_call_prompt(system_prompt, dialog_messages, history_max_chars=history_max_chars)
    if prompt_max_chars is None or int(prompt_max_chars) <= 0 or len(prompt) <= int(prompt_max_chars):
        return prompt
    overflow = len(prompt) - int(prompt_max_chars)
    adjusted_history = max(0, int(history_max_chars) - overflow - 512)
    return build_rwkv_json_call_prompt(system_prompt, dialog_messages, history_max_chars=adjusted_history)


def _agentbench_completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_result.get("error") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=json.dumps(payload.get("agent_info") or {}, ensure_ascii=False, sort_keys=True),
        ref_answer="official_agentbench_controller",
    )


def _run_agentbench(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_agentbench_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("AgentBench manifest is empty")

    plan = _resolve_function_calling_plan(run.dataset_slug, len(records), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="tool",
        fallback_templates="instruction_following_default",
    )
    if sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    sampling = clamp_function_calling_sampling(sampling, max(1, int(args.decision_max_tokens or 1024)))
    history_max_chars = max(0, int(args.history_max_chars or DEFAULT_HISTORY_MAX_CHARS))
    prompt_max_chars = _agentbench_prompt_max_chars(args)
    long_doc_config = _agentbench_long_doc_config(args)
    tool_routing_config = tool_routing_config_from_args(args)
    candidate_router_mode = _agentbench_candidate_router_mode(args)
    candidate_router_config = _agentbench_candidate_router_config_from_args(args)
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, sampling)]),
        long_doc_config=long_doc_config,
        tool_routing_config=tool_routing_config,
        candidate_router_config=candidate_router_config,
        prompt_max_chars=prompt_max_chars,
    )
    sampling_payload["agentbench_adapter"] = {
        "candidate_router_mode": candidate_router_mode,
        "candidate_router_auto_min_tools": _AGENTBENCH_CANDIDATE_ROUTER_MIN_TOOLS,
        "decision_io": "rwkv_json_or_parallel_candidate",
    }
    controller_url = _agentbench_controller_url(args)
    controller = AgentBenchControllerClient(controller_url)
    controller.ensure_available()
    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=max(1, int(args.batch_size or 1)))
        prompts: list[str] = []
        sessions: list[str] = []
        try:
            for _sample_index, record in repeated:
                session_id, data = controller.start_sample(record.task_name, record.index)
                sessions.append(session_id)
                tool_route = route_tools_for_prompt(
                    data.get("tools") or [],
                    data.get("messages") or [],
                    config=tool_routing_config,
                    engine=run.engine,
                    sampling=sampling,
                    control_tool_names=("final_answer",) if _is_agentbench_kg(record) else (),
                    progress_desc="AgentBench-ToolRouter-Probe",
                )
                prompts.append(
                    build_agentbench_prompt(
                        data.get("messages") or [],
                        tool_route.selected_tools,
                        history_max_chars=history_max_chars,
                        allow_final_answer_text=_is_agentbench_kg(record),
                        prompt_max_chars=prompt_max_chars,
                        long_doc_config=long_doc_config,
                    )
                )
            run.engine.generate(
                prompts,
                sampling=sampling,
                batch_size=len(prompts),
                progress_desc="AgentBench-Probe",
                prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in prompts],
            )
        finally:
            for session_id in sessions:
                controller.cancel(session_id)
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    job_name = _resolve_job_name("function_agentbench", run_context=run_context)
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
        completion_to_eval=_agentbench_completion_to_eval_payload,
        runner_name=job_name,
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
                for key, record in pending:
                    payload = _run_one_agentbench_attempt(
                        args=args,
                        run=run,
                        controller=controller,
                        record=record,
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        sampling=sampling,
                        sampling_payload=sampling_payload,
                        history_max_chars=history_max_chars,
                        prompt_max_chars=prompt_max_chars,
                        long_doc_config=long_doc_config,
                        tool_routing_config=tool_routing_config,
                        candidate_router_mode=candidate_router_mode,
                        candidate_router_config=candidate_router_config,
                    )
                    writer.enqueue(payload)
            except Exception:  # noqa: BLE001
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: flush_partial("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_agentbench_completion_to_eval_payload,
            model_name=run.model_name,
            avg_k=plan.avg_k,
            timeout_s=float(args.db_close_timeout_s),
            build_score_payload=lambda completions_payloads, _eval_payloads, metrics: make_score_payload(
                run.dataset_slug,
                is_cot=False,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.NO_COT.value),
                extra={"cot_mode": CoTMode.NO_COT.value, "controller_url": controller_url},
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"{job_name} done: {len(completions_payloads)} samples")
    return 0


def _run_one_agentbench_attempt(
    *,
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    controller: AgentBenchControllerClient,
    record: AgentBenchRecord,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    sampling: Any,
    sampling_payload: dict[str, Any],
    history_max_chars: int,
    prompt_max_chars: int,
    long_doc_config: LongDocEvidenceConfig,
    tool_routing_config: ToolRoutingConfig,
    candidate_router_mode: str,
    candidate_router_config: ParallelCandidateRouterConfig | None,
) -> dict[str, Any]:
    session_id = ""
    stages: list[StageRecord] = []
    trace: list[dict[str, Any]] = []
    reward = 0.0
    is_passed = False
    error = ""
    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    try:
        session_id, data = controller.start_sample(record.task_name, record.index)
        messages = [dict(item) for item in data.get("messages") or [] if isinstance(item, Mapping)]
        tools = [dict(item) for item in data.get("tools") or [] if isinstance(item, Mapping)]
        for round_index in range(1, max(1, int(args.max_steps or 20)) + 1):
            decision_completion = ""
            tool_route = route_tools_for_prompt(
                tools,
                messages,
                config=tool_routing_config,
                engine=run.engine,
                sampling=sampling,
                control_tool_names=("final_answer",) if _is_agentbench_kg(record) else (),
                progress_desc=f"AgentBench tool route {sample_index} round {round_index}",
                prompt_seed=sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=10_000 + round_index),
            )
            decision_tools = _agentbench_decision_tools(
                tool_route.selected_tools,
                allow_final_answer=_is_agentbench_kg(record),
            )
            if _should_use_agentbench_candidate_router(
                mode=candidate_router_mode,
                config=candidate_router_config,
                tools=decision_tools,
                messages=messages,
                prompt_max_chars=prompt_max_chars,
            ):
                route = route_parallel_candidate_tool_call(
                    tools=decision_tools,
                    messages=messages,
                    domain_policy=_agentbench_candidate_policy(record),
                    domain=record.task_name,
                    facts_text=_agentbench_candidate_facts(record),
                    engine=run.engine,
                    sampling=sampling,
                    config=candidate_router_config,
                    progress_desc=f"AgentBench candidate route {sample_index} round {round_index}",
                    prompt_seed=sample_repeat_seed(
                        sample_index,
                        repeat_index,
                        pass_index=pass_index,
                        stage=20_000 + round_index,
                    ),
                )
                if route.selected is None:
                    decoded_calls: list[dict[str, Any]] = []
                    completion = route.aggregate_completion
                    stop_reason = route.aggregate_finish_reason or "candidate_router_empty"
                    error = route.aggregate_error or "candidate router did not select a tool call"
                else:
                    decoded_calls = [{"name": route.selected.name, "arguments": dict(route.selected.arguments)}]
                    completion = route.aggregate_completion or json.dumps(decoded_calls[0], ensure_ascii=False, sort_keys=True)
                    stop_reason = route.aggregate_finish_reason or "stop"
                decision_completion = completion
                stages.append(StageRecord(prompt=route.aggregate_prompt, completion=completion, stop_reason=stop_reason))
                trace.append(
                    {
                        "round": round_index,
                        "tool_route": tool_route.trace_payload(),
                        "decision_io": "parallel_candidate",
                        "candidate_router": route.trace_payload(include_prompts=True),
                        "decoded_calls": decoded_calls,
                        "parse_error": error if route.selected is None else "",
                    }
                )
                if route.selected is None:
                    break
                if _is_agentbench_db(record):
                    decoded_calls = _repair_agentbench_db_sql_calls(decoded_calls, messages)
                assistant_message = _agentbench_assistant_message(decoded_calls, round_index)
            else:
                prompt = build_agentbench_prompt(
                    messages,
                    decision_tools,
                    history_max_chars=history_max_chars,
                    allow_final_answer_text=False,
                    prompt_max_chars=prompt_max_chars,
                    long_doc_config=long_doc_config,
                )
                output = run.engine.generate(
                    [prompt],
                    sampling=sampling,
                    batch_size=1,
                    progress_desc=f"AgentBench sample {sample_index} round {round_index}",
                    prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES)],
                    prompt_seeds=[sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=round_index)],
                )[0]
                decision_completion = output.text
                stages.append(StageRecord(prompt=prompt, completion=output.text, stop_reason=output.finish_reason))
                try:
                    if _looks_like_template_leak(output.text):
                        raise ValueError("decision stage leaked internal template/control tokens")
                    decoded_calls = decode_simple_tool_call_response(output.text)
                    if _is_agentbench_db(record):
                        decoded_calls = _repair_agentbench_db_sql_calls(decoded_calls, messages)
                    assistant_message = _agentbench_assistant_message(decoded_calls, round_index)
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                    trace.append(
                        {
                            "round": round_index,
                            "tool_route": tool_route.trace_payload(),
                            "decision_io": "rwkv_json",
                            "completion": output.text,
                            "parse_error": error,
                        }
                    )
                    break
            messages.append(assistant_message)
            response = controller.interact(session_id, assistant_message)
            trace.append(
                {
                    "round": round_index,
                    "tool_route": tool_route.trace_payload(),
                    "completion": decision_completion,
                    "decoded_calls": decoded_calls,
                    "controller_response": response,
                }
            )
            if bool(response.get("finish")):
                reward = float(response.get("reward") or response.get("score") or 0.0)
                is_passed = reward > 0.0
                break
            messages.extend(dict(item) for item in response.get("messages") or [] if isinstance(item, Mapping))
        else:
            error = "max_steps"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        if session_id and not is_passed:
            controller.cancel(session_id)

    payload = SampleRecord(
        benchmark_name=run.benchmark_name,
        dataset_split=run.dataset_split,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        stages=stages,
        sampling_config=sampling_payload,
    ).as_payload()
    payload["agent_result"] = {
        "reward": reward,
        "num_turns": len(stages),
        "cost": 0.0,
        "is_passed": is_passed,
        "error": error or None,
    }
    payload["agent_info"] = {
        "session_id": session_id,
        "task_name": record.task_name,
        "task_index": record.index,
        "reward": reward,
        "cot_mode": CoTMode.NO_COT.value,
    }
    payload["agent_trace"] = trace
    payload["task_id"] = record.task_id
    payload["domain"] = record.task_name
    payload["instruction"] = record.task_name
    payload["metadata"] = dict(record.metadata)
    return payload


def _agentbench_candidate_router_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "candidate_router_mode", None) or "off").strip().lower()
    if mode not in {"off", "auto", "parallel"}:
        raise ValueError(f"unsupported candidate_router_mode={mode!r}; expected off, auto, or parallel")
    return mode


def _agentbench_candidate_router_config_from_args(args: argparse.Namespace) -> ParallelCandidateRouterConfig | None:
    mode = _agentbench_candidate_router_mode(args)
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
        prompt_max_chars=_positive_int(getattr(args, "candidate_router_prompt_max_chars", None), defaults.prompt_max_chars),
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


def _agentbench_decision_tools(
    selected_tools: Sequence[Mapping[str, Any]],
    *,
    allow_final_answer: bool,
) -> list[dict[str, Any]]:
    tools = [dict(tool) for tool in selected_tools]
    if not allow_final_answer:
        return tools
    names = {_agentbench_tool_name(tool) for tool in tools}
    if "final_answer" not in names:
        tools.append(_agentbench_final_answer_tool())
    return tools


def _agentbench_tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
    return str(function.get("name") or "")


def _should_use_agentbench_candidate_router(
    *,
    mode: str,
    config: ParallelCandidateRouterConfig | None,
    tools: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
    prompt_max_chars: int,
) -> bool:
    if config is None or not tools:
        return False
    if mode == "parallel":
        return True
    if mode != "auto":
        return False
    if len(tools) >= max(_AGENTBENCH_CANDIDATE_ROUTER_MIN_TOOLS, int(config.chunk_tools) * 2):
        return True
    if tool_catalog_chars(list(tools)) > max(1, int(prompt_max_chars) // 3):
        return True
    message_chars = sum(len(str(message.get("content") or "")) for message in messages)
    return message_chars > int(config.context_chars)


def _agentbench_candidate_policy(record: AgentBenchRecord) -> str:
    task_name = record.task_name.lower()
    lines = [_AGENTBENCH_CANDIDATE_ROUTER_POLICY]
    if "db" in task_name:
        lines.append(
            "DB task: inspect the schema from the conversation or with SHOW TABLES/DESCRIBE before mutating data. "
            "Use MySQL syntax. Quote every table or column identifier that contains spaces, punctuation, reserved words, or mixed case with backticks, "
            "for example SELECT * FROM `Team Information` and UPDATE `Team Information` SET `Capacity` = '45,000'. "
            "If the controller says a table does not exist because an identifier was split, retry with the full table name in backticks."
        )
    if "kg" in task_name:
        lines.append("KG task: entity ids must come from tool output or user-visible graph context; final answer must be exactly `Final Answer: #id`.")
    return normalize_rwkv_text(" ".join(lines))


def _agentbench_candidate_facts(record: AgentBenchRecord) -> str:
    payload = {
        "task_id": record.task_id,
        "task_name": record.task_name,
        "index": record.index,
        "metadata": record.metadata,
    }
    return normalize_rwkv_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _repair_agentbench_db_sql_calls(
    decoded_calls: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    table_names = _agentbench_db_table_names(messages)
    repaired: list[dict[str, Any]] = []
    for raw_call in decoded_calls:
        call = dict(raw_call)
        if str(call.get("name") or "").strip() != "execute_sql":
            repaired.append(call)
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            repaired.append(call)
            continue
        args = dict(arguments)
        query = str(args.get("query") or "")
        if query:
            args["query"] = _repair_mysql_identifier_quotes(query, table_names)
            call["arguments"] = args
        repaired.append(call)
    return repaired


def _agentbench_db_table_names(messages: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: list[str] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "tool":
            continue
        content = str(message.get("content") or "").strip()
        if not content.startswith("["):
            continue
        try:
            parsed = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, (list, tuple)):
            continue
        parsed_names: list[str] = []
        for row in parsed:
            if isinstance(row, (list, tuple)) and len(row) == 1 and isinstance(row[0], str):
                parsed_names.append(row[0])
        if parsed_names and len(parsed_names) == len(parsed):
            names.extend(parsed_names)
    return tuple(dict.fromkeys(name for name in names if name.strip()))


def _repair_mysql_identifier_quotes(query: str, table_names: Sequence[str]) -> str:
    repaired = _quote_multiword_table_references(query)
    for table_name in table_names:
        if _needs_mysql_identifier_quotes(table_name):
            repaired = _replace_outside_single_quotes(repaired, table_name, f"`{table_name}`")
    return repaired


def _quote_multiword_table_references(query: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group("name").strip()
        if not _needs_mysql_identifier_quotes(name):
            return match.group(0)
        return f"{match.group('keyword')} `{name}`"

    return _MYSQL_MULTIWORD_TABLE_REF.sub(replace, query)


def _replace_outside_single_quotes(text: str, needle: str, replacement: str) -> str:
    if not needle or needle not in text:
        return text
    parts = text.split("'")
    pattern = re.compile(rf"(?<![`A-Za-z0-9_]){re.escape(needle)}(?![`A-Za-z0-9_])")
    for index in range(0, len(parts), 2):
        parts[index] = pattern.sub(replacement, parts[index])
    return "'".join(parts)


def _needs_mysql_identifier_quotes(identifier: str) -> bool:
    parts = [part for part in identifier.strip().split() if part]
    return len(parts) > 1 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier.strip())


def _agentbench_assistant_message(decoded_calls: Sequence[Mapping[str, Any]], round_index: int) -> dict[str, Any]:
    if not decoded_calls:
        raise ValueError("missing tool call")
    first = decoded_calls[0]
    name = str(first.get("name") or "").strip()
    arguments = first.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    if name == "final_answer":
        return {"role": "assistant", "content": str(arguments.get("answer") or "")}
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{round_index}_{index}",
                "type": "function",
                "function": {
                    "name": str(call.get("name") or ""),
                    "arguments": json.dumps(dict(call.get("arguments") or {}), ensure_ascii=False),
                },
            }
            for index, call in enumerate(decoded_calls)
        ],
    }


def _normalize_openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
    return {
        "name": str(function.get("name") or ""),
        "description": str(function.get("description") or ""),
        "parameters": dict(function.get("parameters") or {}),
    }


def _render_agentbench_output_schema() -> str:
    schema = {
        "type": "object",
        "required": ["name", "arguments"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"},
        },
    }
    example = {"name": "execute_sql", "arguments": {"query": "SELECT 1"}}
    return (
        json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=False)
        + "\nExample:\n"
        + json.dumps(example, ensure_ascii=False, separators=(",", ":"))
    )


def _agentbench_data_count(path: Path) -> int:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return sum(1 for line in raw.splitlines() if line.strip())
    payload = json.loads(raw)
    if isinstance(payload, list):
        return len(payload)
    raise ValueError(f"unsupported AgentBench data file format: {path}")


def _agentbench_controller_url(args: argparse.Namespace) -> str:
    return str(
        getattr(args, "agentbench_controller_url", None)
        or os.environ.get("AGENTBENCH_CONTROLLER_URL")
        or os.environ.get("AGENTRL_CONTROLLER_URL")
        or "http://127.0.0.1:5020/api"
    ).rstrip("/")


def _agentbench_prompt_max_chars(args: argparse.Namespace) -> int:
    raw = getattr(args, "prompt_max_chars", None) or os.environ.get("RWKV_AGENTBENCH_PROMPT_MAX_CHARS", "24576")
    try:
        return max(4096, int(raw))
    except (TypeError, ValueError):
        return 24576


def _agentbench_long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
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


def _is_agentbench_kg(record: AgentBenchRecord) -> bool:
    return "kg" in record.task_name.lower()


def _is_agentbench_db(record: AgentBenchRecord) -> bool:
    return "db" in record.task_name.lower()


__all__ = [
    "AgentBenchControllerClient",
    "AgentBenchRecord",
    "build_agentbench_prompt",
    "load_agentbench_manifest_records",
    "load_agentbench_rows_from_source",
    "_run_agentbench",
]
