from __future__ import annotations

import argparse
import json
import os
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
from src.eval.function_calling.common import (
    attach_function_calling_context_metadata,
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.function_calling.context_budget import DEFAULT_HISTORY_MAX_CHARS, normalize_rwkv_text
from src.eval.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _looks_like_template_leak,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    build_rwkv_json_call_prompt,
    render_json_function_call,
)
from src.eval.function_calling.simple_tool_call import decode_simple_tool_call_response
from src.eval.function_calling.tool_router import (
    ToolRoutingConfig,
    route_tools_for_prompt,
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
        except Exception:
            pass

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
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, sampling)]),
        long_doc_config=long_doc_config,
        tool_routing_config=tool_routing_config,
        prompt_max_chars=prompt_max_chars,
    )
    controller_url = _agentbench_controller_url(args)
    controller = AgentBenchControllerClient(controller_url)
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
                    )
                    writer.enqueue(payload)
            except BaseException:
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
                is_cot=True,
                model_name=run.model_name,
                metrics=metrics,
                samples=len(completions_payloads),
                problems=plan.sample_size,
                task=job_name,
                task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
                extra={"cot_mode": CoTMode.COT.value, "controller_url": controller_url},
            ),
        )
    except BaseException as exc:
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
            prompt = build_agentbench_prompt(
                messages,
                tool_route.selected_tools,
                history_max_chars=history_max_chars,
                allow_final_answer_text=_is_agentbench_kg(record),
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
            stages.append(StageRecord(prompt=prompt, completion=output.text, stop_reason=output.finish_reason))
            try:
                if _looks_like_template_leak(output.text):
                    raise ValueError("decision stage leaked internal template/control tokens")
                decoded_calls = decode_simple_tool_call_response(output.text)
                assistant_message = _agentbench_assistant_message(decoded_calls, round_index)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                trace.append(
                    {
                        "round": round_index,
                        "tool_route": tool_route.trace_payload(),
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
                    "completion": output.text,
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
        "cot_mode": CoTMode.COT.value,
    }
    payload["agent_trace"] = trace
    payload["task_id"] = record.task_id
    payload["domain"] = record.task_name
    payload["instruction"] = record.task_name
    payload["metadata"] = dict(record.metadata)
    return payload


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
    return json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=False)


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
        model_max_tokens=max(1, int(getattr(args, "long_doc_model_max_tokens", 96) or 96)),
        model_parallel_batch_size=max(1, int(getattr(args, "long_doc_model_parallel_batch_size", 8) or 8)),
    )


def _is_agentbench_kg(record: AgentBenchRecord) -> bool:
    return "kg" in record.task_name.lower()


__all__ = [
    "AgentBenchControllerClient",
    "AgentBenchRecord",
    "build_agentbench_prompt",
    "load_agentbench_manifest_records",
    "load_agentbench_rows_from_source",
    "_run_agentbench",
]
