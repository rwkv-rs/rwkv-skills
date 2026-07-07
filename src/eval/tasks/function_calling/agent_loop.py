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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
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
    step_outcome_to_function_output,
)
from src.eval.tasks.function_calling.agent_loop_verifiers import (
    AgentLoopVerdict,
    VerifierSpec,
    build_agent_loop_verifier,
    preflight_agent_loop_runtime,
)
from src.eval.tasks.function_calling.common import (
    build_partial_eval_flusher,
    build_pending_attempts,
    clamp_function_calling_sampling,
    finalize_function_calling_run,
    prepare_function_calling_run,
    repeat_probe_entries,
)
from src.eval.tasks.function_calling.context_budget import normalize_rwkv_text
from src.eval.tasks.function_calling.final_answer import FINAL_ANSWER_TOOL_NAME, final_answer_tool_schema
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
            workspace_archive=(str(config.get("workspace_archive")) if config.get("workspace_archive") else None),
            setup_commands=tuple(str(item) for item in config.get("setup_commands") or ()),
            command_timeout_s=float(
                config.get("command_timeout_s") or getattr(args, "agent_loop_command_timeout_s", None) or 60.0
            ),
            max_output_chars=int(
                config.get("max_output_chars") or getattr(args, "agent_loop_max_output_chars", None) or DEFAULT_MAX_OUTPUT_CHARS
            ),
            workspace_root=getattr(args, "agent_loop_workspace_root", None),
        )
    if kind == "mcp_worker":
        return McpWorkerExecutor(
            runtime_root=str(config.get("runtime_root") or ""),
            worker_script=(str(config.get("worker_script")) if config.get("worker_script") else None),
            servers=tuple(str(item) for item in config.get("servers") or ()),
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
    ]
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
    candidate = extract_json_call_value_text(completion)
    payload = json.loads(candidate)
    calls = coerce_json_function_call_payloads(payload, context_label="agent-loop decision")
    return [{"name": str(call["name"]), "arguments": dict(call.get("arguments") or {})} for call in calls]


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

    for step in range(1, max_steps + 1):
        prompt = build_agent_loop_prompt(record, tools, messages, history_max_chars=history_max_chars)
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
        except Exception as exc:  # noqa: BLE001 - parse failures terminate the episode
            termination_reason = "parse_error"
            error = str(exc)
            trace.append({"kind": "parse_error", "step": step, "error": str(exc), "raw": output.text[:2000]})
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
    sampling_payload = normalize_sampling_config_by_stage([(1, tool_sampling)])
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
                for key, record in pending:
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
                    writer.enqueue(payload)
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
