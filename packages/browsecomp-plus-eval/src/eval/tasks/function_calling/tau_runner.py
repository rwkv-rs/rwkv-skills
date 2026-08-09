from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.eval.tasks.agent_bench.envs.tau_v2 import TauV2Env
from src.eval.tasks.agent_bench.tau_official import (
    DEFAULT_TAU_PROMPT_MAX_CHARS,
    RWKVTauOfficialAgent,
    TauOfficialRuntime,
)
from src.eval.tasks.agent_bench.tasks import require_tau_v3_source
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.env_config import apply_openai_env, resolve_judge_model_config, resolve_required_user_model_config
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
from src.eval.tasks.function_calling.runner_common import (
    ResolvedFunctionCallingRun,
    _resolve_function_calling_plan,
    _resolve_job_name,
)
from src.eval.tasks.function_calling.rwkv_prompt import (
    JSON_CALL_STOP_SUFFIXES,
    normalize_function_prompt_style,
)
from src.eval.tasks.function_calling.tau_bench import (
    TauManifestRecord,
    TauToolCall,
    build_tau_system_prompt,
    load_tau_manifest_records,
    render_tau_user_prompt,
)
from src.eval.tasks.function_calling.tool_router import ToolRoutingConfig, tool_routing_config_from_args
from src.eval.long_doc_evidence import LongDocEvidenceConfig
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_TOOL_ERRORS = 10
DEFAULT_TAU_HISTORY_MAX_CHARS = 16000
DEFAULT_TAU_DECISION_MAX_TOKENS = 384

@dataclass(slots=True)
class _ActiveEpisode:
    sample_index: int
    repeat_index: int
    pass_index: int
    record: TauManifestRecord
    runtime_env: TauV2Env
    task: Any
    environment: Any
    system_prompt: str
    prompt_messages: list[dict[str, str]]
    trajectory: list[Any]
    stages: list[StageRecord] = field(default_factory=list)
    tool_calls: list[TauToolCall] = field(default_factory=list)
    turn_count: int = 0
    tool_errors: int = 0
    final_answer: str = ""
    termination_reason: str | None = None
    error: str | None = None


def _trajectory_to_prompt_messages(trajectory: Sequence[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in trajectory:
        role = str(getattr(item, "role", "") or "").strip().lower()
        if role == "assistant":
            tool_calls = getattr(item, "tool_calls", None)
            content = str(getattr(item, "content", "") or "").strip()
            if tool_calls:
                blocks: list[str] = []
                if content:
                    blocks.append(content)
                for tool_call in tool_calls:
                    payload = {
                        "name": _prefixed_tau_tool_name(
                            str(getattr(tool_call, "requestor", "assistant") or "assistant"),
                            str(getattr(tool_call, "name", "") or ""),
                        ),
                        "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
                    }
                    blocks.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                rendered = "\n".join(blocks).strip()
                if rendered:
                    messages.append({"role": "assistant", "content": rendered})
            elif content:
                messages.append({"role": "assistant", "content": content})
        elif role == "user":
            content = str(getattr(item, "content", "") or "").strip()
            if content:
                messages.append({"role": "user", "content": content})
        elif role == "tool":
            content = str(getattr(item, "content", "") or "").strip()
            if content:
                messages.append({"role": "user", "content": content})
    return messages


def _prefixed_tau_tool_name(requestor: str, name: str) -> str:
    requestor = requestor.strip().lower() or "assistant"
    name = name.strip()
    if requestor in {"assistant", "user"} and name and "." not in name:
        return f"{requestor}.{name}"
    return name


def _start_episode(
    *,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    record: TauManifestRecord,
    runtime_env: TauV2Env,
) -> _ActiveEpisode:
    task = runtime_env.load_task(record.task)
    environment = runtime_env.create_environment(solo_mode=False)
    trajectory = runtime_env.apply_initial_state(environment=environment, task=task)
    system_prompt = build_tau_system_prompt(
        runtime_env.system_prompt(environment),
        assistant_tools=runtime_env.tools_schema(environment),
        user_tools=runtime_env.user_tools_schema(environment),
    )
    prompt_messages = _trajectory_to_prompt_messages(trajectory)
    user_prompt = render_tau_user_prompt(record.task).strip() or record.instruction.strip()
    if user_prompt and not any(item.get("role") == "user" for item in prompt_messages):
        prompt_messages.append({"role": "user", "content": user_prompt})
    return _ActiveEpisode(
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        record=record,
        runtime_env=runtime_env,
        task=task,
        environment=environment,
        system_prompt=system_prompt,
        prompt_messages=prompt_messages,
        trajectory=list(trajectory),
    )


def _tool_output_payload(message: Any) -> tuple[bool, Any, str | None]:
    error_flag = bool(getattr(message, "error", False))
    raw = getattr(message, "content", "")
    text = str(raw or "")
    if error_flag:
        return False, None, text
    try:
        return True, json.loads(text), None
    except json.JSONDecodeError:
        return True, text, None


def _trajectory_dump(trajectory: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for message in trajectory:
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            if isinstance(dumped, dict):
                rows.append(dumped)
                continue
        rows.append(
            {
                "role": str(getattr(message, "role", "unknown")),
                "content": str(getattr(message, "content", "")),
            }
        )
    return rows


def _sum_message_costs(trajectory: Sequence[Any]) -> float:
    total = 0.0
    for item in trajectory:
        try:
            total += float(getattr(item, "cost", None))
        except Exception:
            pass
    return total


def _sum_float(items: Sequence[dict[str, Any]], key: str) -> float:
    total = 0.0
    for item in items:
        try:
            total += float(item.get(key) or 0.0)
        except Exception:
            pass
    return total


def _tau_agent_perf_summary(
    *,
    agent: RWKVTauOfficialAgent,
    simulation: Any,
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    steps = [dict(item) for item in list(getattr(agent, "step_timings", []) or [])]
    simulation_duration_s = float(getattr(simulation, "duration", 0.0) or 0.0)
    agent_generation_s = _sum_float(steps, "generation_s")
    agent_prompt_build_s = _sum_float(steps, "prompt_build_s")
    agent_parse_s = _sum_float(steps, "parse_s")
    agent_total_s = _sum_float(steps, "total_s")
    prompt_chars = [int(item.get("prompt_chars") or 0) for item in steps]
    completion_chars = [int(item.get("completion_chars") or 0) for item in steps]
    payload: dict[str, Any] = {
        "simulation_duration_s": simulation_duration_s,
        "agent_turns": len(steps),
        "agent_total_s": agent_total_s,
        "agent_generation_s": agent_generation_s,
        "agent_prompt_build_s": agent_prompt_build_s,
        "agent_parse_s": agent_parse_s,
        "non_agent_simulation_s": max(0.0, simulation_duration_s - agent_total_s),
        "avg_agent_generation_s": agent_generation_s / len(steps) if steps else 0.0,
        "max_prompt_chars": max(prompt_chars, default=0),
        "avg_prompt_chars": (sum(prompt_chars) / len(prompt_chars)) if prompt_chars else 0.0,
        "max_completion_chars": max(completion_chars, default=0),
        "avg_completion_chars": (sum(completion_chars) / len(completion_chars)) if completion_chars else 0.0,
        "steps": steps,
    }
    if timing:
        payload.update(dict(timing))
        if "total_attempt_s" in timing:
            try:
                payload["attempt_overhead_s"] = max(
                    0.0,
                    float(timing["total_attempt_s"])
                    - float(timing.get("orchestrator_run_s") or 0.0)
                    - float(timing.get("evaluation_s") or 0.0),
                )
            except Exception:
                pass
    return payload


def _ref_answer(state: _ActiveEpisode) -> str:
    task_id = str(getattr(state.task, "id", "") or state.record.task_id)
    return f"domain={state.record.domain}\ntask_id={task_id}\nbenchmark_version={state.record.benchmark_version}"


def _tau_completion_payload(
    state: _ActiveEpisode,
    *,
    benchmark_name: str,
    dataset_split: str,
    sampling_payload: dict[str, Any],
) -> dict[str, Any]:
    evaluation = state.runtime_env.evaluate(
        task=state.task,
        trajectory=state.trajectory,
        termination_reason=state.termination_reason or "max_steps",
        solo_mode=False,
    )
    info = dict(evaluation.details)
    info["termination_reason"] = evaluation.termination_reason
    info["tool_errors"] = state.tool_errors
    info["domain"] = state.record.domain
    info["task_id"] = str(getattr(state.task, "id", "") or state.record.task_id)
    info["benchmark_version"] = state.record.benchmark_version
    info["tool_calls"] = [
        {
            "requestor": call.requestor,
            "name": call.name,
            "arguments": dict(call.arguments),
        }
        for call in state.tool_calls
    ]
    info["final_answer"] = state.final_answer
    info["ref_answer"] = _ref_answer(state)

    payload = SampleRecord(
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        sample_index=state.sample_index,
        repeat_index=state.repeat_index,
        pass_index=state.pass_index,
        stages=list(state.stages),
        sampling_config=sampling_payload,
    ).as_payload()
    payload["_stage"] = "answer"
    payload["agent_result"] = {
        "task_id": info["task_id"],
        "domain": state.record.domain,
        "reward": float(evaluation.reward),
        "num_turns": int(state.turn_count),
        "cost": _sum_message_costs(state.trajectory),
        "is_passed": bool(evaluation.is_passed),
        "error": state.error,
    }
    payload["agent_info"] = info
    payload["agent_trace"] = _trajectory_dump(state.trajectory)
    return payload


def _tau_completion_to_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("agent_result")
    if not isinstance(result, dict):
        result = {}
    info = payload.get("agent_info")
    if not isinstance(info, dict):
        info = {}
    passed = bool(result.get("is_passed", False))
    fail_reason = ""
    if not passed:
        fail_reason = str(result.get("error") or info.get("termination_reason") or "tau_bench evaluation failed")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=fail_reason,
        answer=str(info.get("final_answer") or ""),
        ref_answer=str(info.get("ref_answer") or ""),
    )


def _tau_official_completion_payload(
    *,
    record: TauManifestRecord,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    simulation: Any,
    evaluation: Any,
    agent: RWKVTauOfficialAgent,
    benchmark_name: str,
    dataset_split: str,
    sampling_payload: dict[str, Any],
    timing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = str(getattr(simulation, "task_id", "") or record.task_id)
    official_reward = float(getattr(evaluation, "reward", 0.0))
    official_is_passed = bool(getattr(evaluation, "is_passed", False))
    parse_errors = list(agent.parse_errors)
    details = dict(getattr(evaluation, "details", {}) or {})
    details["domain"] = record.domain
    details["task_id"] = task_id
    details["benchmark_version"] = record.benchmark_version
    details["parse_errors"] = parse_errors
    details["official_reward"] = official_reward
    details["official_is_passed"] = official_is_passed
    details["tool_routes"] = list(getattr(agent, "tool_routes", []) or [])
    perf = _tau_agent_perf_summary(agent=agent, simulation=simulation, timing=timing)
    details["perf"] = perf
    details["ref_answer"] = (
        f"domain={record.domain}\n"
        f"task_id={task_id}\n"
        f"benchmark_version={record.benchmark_version}\n"
        "runtime=official_tau_orchestrator"
    )

    payload = SampleRecord(
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        stages=list(agent.stages),
        sampling_config=sampling_payload,
    ).as_payload()
    payload["_stage"] = "answer"
    payload["agent_result"] = {
        "task_id": task_id,
        "domain": record.domain,
        "reward": official_reward,
        "num_turns": len(agent.stages),
        "cost": float(getattr(simulation, "agent_cost", None) or 0.0)
        + float(getattr(simulation, "user_cost", None) or 0.0),
        "is_passed": official_is_passed,
        "error": "; ".join(parse_errors) if parse_errors else None,
    }
    payload["agent_info"] = details
    payload["agent_trace"] = _trajectory_dump(list(getattr(simulation, "messages", []) or []))
    payload["perf"] = perf
    return payload


def _run_tau_official_attempt(
    *,
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    record: TauManifestRecord,
    sample_index: int,
    repeat_index: int,
    pass_index: int,
    runtime_env: TauOfficialRuntime,
    user_model: Any | None,
    judge_model: Any,
    sampling: Any,
    sampling_payload: dict[str, Any],
    history_max_chars: int,
    prompt_max_chars: int,
    long_doc_config: LongDocEvidenceConfig,
    max_steps: int,
    max_tool_errors: int,
    tool_routing_config: ToolRoutingConfig | None = None,
    retail_repeated_read_guard: bool = False,
    retail_tool_use_guard: bool = False,
    retail_progressive_tool_disclosure: bool = False,
) -> dict[str, Any]:
    attempt_started = time.perf_counter()
    timing: dict[str, Any] = {}
    max_runtime_retries = _tau_runtime_retry_count()
    runtime_retry_count = 0
    for runtime_attempt in range(max_runtime_retries + 1):
        task = runtime_env.load_task(record.task)
        environment = runtime_env.create_environment(solo_mode=False)
        agent = RWKVTauOfficialAgent(
            engine=run.engine,
            sampling=sampling,
            tools=environment.get_tools(),
            domain_policy=str(environment.get_policy()),
            domain=record.domain,
            history_max_chars=history_max_chars,
            prompt_max_chars=prompt_max_chars,
            long_doc_config=long_doc_config,
            tool_routing_config=tool_routing_config or ToolRoutingConfig(),
            retail_repeated_read_guard=retail_repeated_read_guard,
            retail_tool_use_guard=retail_tool_use_guard,
            retail_progressive_tool_disclosure=retail_progressive_tool_disclosure,
        )
        user = runtime_env.build_user(task=task, environment=environment, user_model=user_model)
        seed = sample_repeat_seed(sample_index, repeat_index, pass_index=pass_index, stage=1)
        orchestrator = runtime_env.build_orchestrator(
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=max_tool_errors,
            seed=seed,
            validate_communication=True,
        )
        try:
            run_started = time.perf_counter()
            simulation = orchestrator.run()
            timing["orchestrator_run_s"] = time.perf_counter() - run_started
            evaluation_started = time.perf_counter()
            evaluation = runtime_env.evaluate(simulation=simulation, task=task, judge_model=judge_model)
            timing["evaluation_s"] = time.perf_counter() - evaluation_started
            break
        except Exception as exc:
            error_text = f"tau official runtime error: {type(exc).__name__}: {exc}"
            is_transient = _is_transient_tau_runtime_error(exc)
            if runtime_attempt < max_runtime_retries and is_transient:
                runtime_retry_count += 1
                time.sleep(_tau_runtime_retry_delay_s(runtime_attempt))
                continue
            agent.parse_errors.append(error_text)
            messages = list(getattr(orchestrator, "messages", []) or getattr(orchestrator, "_messages", []) or [])
            simulation = SimpleNamespace(
                task_id=record.task_id,
                messages=messages,
                agent_cost=float(getattr(orchestrator, "agent_cost", 0.0) or 0.0),
                user_cost=float(getattr(orchestrator, "user_cost", 0.0) or 0.0),
                termination_reason=error_text,
            )
            evaluation = SimpleNamespace(
                reward=0.0,
                is_passed=False,
                details={
                    "termination_reason": error_text,
                    "runtime_error": error_text,
                    "transient_runtime_error": bool(is_transient),
                    "runtime_retries_exhausted": bool(is_transient),
                },
            )
            break
    timing["runtime_retry_count"] = runtime_retry_count
    timing["total_attempt_s"] = time.perf_counter() - attempt_started
    return _tau_official_completion_payload(
        record=record,
        sample_index=sample_index,
        repeat_index=repeat_index,
        pass_index=pass_index,
        simulation=simulation,
        evaluation=evaluation,
        agent=agent,
        benchmark_name=run.benchmark_name,
        dataset_split=run.dataset_split,
        sampling_payload=sampling_payload,
        timing=timing,
    )


def _tau_runtime_retry_count() -> int:
    raw = os.environ.get("RWKV_TAU_RUNTIME_RETRIES", "4").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 4


def _tau_runtime_retry_delay_s(attempt_index: int) -> float:
    base_raw = os.environ.get("RWKV_TAU_RUNTIME_RETRY_BASE_S", "3").strip()
    cap_raw = os.environ.get("RWKV_TAU_RUNTIME_RETRY_CAP_S", "30").strip()
    try:
        base = max(0.1, float(base_raw))
    except ValueError:
        base = 3.0
    try:
        cap = max(base, float(cap_raw))
    except ValueError:
        cap = 30.0
    return min(cap, base * (2 ** max(0, int(attempt_index))))


def _is_transient_tau_runtime_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "当前系统繁忙",
        "rate limit",
        "ratelimit",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "apierror",
        "openaiexception",
        "server error",
        "http 429",
        "status 429",
        " 429",
        "http 500",
        "status 500",
        "http 502",
        "status 502",
        "http 503",
        "status 503",
        "http 504",
        "status 504",
        "connection reset",
        "connection aborted",
    )
    return any(marker in text for marker in markers)


def _tau_long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
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


def _is_lightweight_tau_record(record: TauManifestRecord) -> bool:
    version = str(record.benchmark_version).lower().strip()
    return version in {"tau_v3_light", "tau3_light", "tau_light"} or (
        record.domain == "mock" and version.startswith("tau_v3_light")
    )


def _requires_tau_v3_source(records: Sequence[TauManifestRecord]) -> bool:
    return any(str(record.benchmark_version).lower().strip() == "tau_v3" for record in records)


def _requires_tau_user_model(records: Sequence[TauManifestRecord]) -> bool:
    return any(not _is_lightweight_tau_record(record) for record in records)


def _apply_tau_model_overrides(args: argparse.Namespace) -> None:
    overrides = {
        "USER_MODEL_NAME": getattr(args, "user_model", None),
        "USER_API_KEY": getattr(args, "user_api_key", None),
        "USER_BASE_URL": getattr(args, "user_base_url", None),
        "JUDGE_MODEL": getattr(args, "judge_model", None),
        "JUDGE_API_KEY": getattr(args, "judge_api_key", None),
        "JUDGE_BASE_URL": getattr(args, "judge_base_url", None),
    }
    for env_name, raw_value in overrides.items():
        value = str(raw_value or "").strip()
        if value:
            os.environ[env_name] = value


def _tau_runtime_model_metadata(user_model: Any | None, judge_model: Any | None) -> dict[str, Any]:
    return {
        "user_model": getattr(user_model, "model_name", None),
        "user_base_url": getattr(user_model, "base_url", None),
        "judge_model": getattr(judge_model, "model_name", None),
        "judge_base_url": getattr(judge_model, "base_url", None),
        "static_user": user_model is None,
        "judge_configured": judge_model is not None,
    }


def _run_tau(
    args: argparse.Namespace,
    run: ResolvedFunctionCallingRun,
    *,
    run_context: "RunContext | None" = None,
) -> int:
    records = load_tau_manifest_records(run.dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("tau_bench/tau2_bench manifest is empty")
    if _requires_tau_v3_source(records):
        require_tau_v3_source(run.dataset_slug)

    plan = _resolve_function_calling_plan(run.dataset_slug, len(records), avg_ks=args.avg_k)
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    decision_sampling = resolve_sampling_config(
        run.dataset_slug,
        run.model_name,
        stage="final",
        fallback_templates="instruction_following_default",
    )
    if decision_sampling is None:
        raise ValueError(f"missing sampling config for dataset={run.dataset_slug}, model={run.model_name}")
    normalize_function_prompt_style(getattr(args, "prompt_style", None))
    decision_sampling = clamp_function_calling_sampling(
        decision_sampling,
        args.decision_max_tokens or DEFAULT_TAU_DECISION_MAX_TOKENS,
    )

    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]
    batch_size = max(1, int(args.batch_size or 16))
    max_steps = max(1, int(args.max_steps))
    max_tool_errors = max(1, int(args.max_tool_errors))
    prompt_max_chars = max(
        DEFAULT_TAU_PROMPT_MAX_CHARS,
        int(
            getattr(args, "prompt_max_chars", None)
            or os.environ.get("RWKV_TAU_PROMPT_MAX_CHARS", str(DEFAULT_TAU_PROMPT_MAX_CHARS))
        ),
    )
    long_doc_config = _tau_long_doc_config(args)
    tool_routing_config = tool_routing_config_from_args(args)
    retail_repeated_read_guard = bool(getattr(args, "tau_retail_repeated_read_guard", False))
    retail_tool_use_guard = bool(getattr(args, "tau_retail_tool_use_guard", False))
    retail_progressive_tool_disclosure = bool(getattr(args, "tau_retail_progressive_tool_disclosure", False))
    sampling_payload = attach_function_calling_context_metadata(
        normalize_sampling_config_by_stage([(1, decision_sampling)]),
        long_doc_config=long_doc_config,
        tool_routing_config=tool_routing_config,
        prompt_max_chars=prompt_max_chars,
    )
    sampling_payload["tau_adapter"] = {
        "semantic_fallbacks": True,
        "text_decision_recovery": True,
        "format_conversion": True,
        "retail_repeated_read_guard": retail_repeated_read_guard,
        "retail_tool_use_guard": retail_tool_use_guard,
        "retail_progressive_tool_disclosure": retail_progressive_tool_disclosure,
    }
    tau_history_cap = int(os.environ.get("RWKV_TAU_HISTORY_MAX_CHARS", str(DEFAULT_TAU_HISTORY_MAX_CHARS)))
    history_max_chars = max(0, min(int(args.history_max_chars), tau_history_cap))
    user_model = None
    judge_model = None
    if _requires_tau_user_model(records):
        _apply_tau_model_overrides(args)
        user_model = resolve_required_user_model_config()
        judge_model = resolve_judge_model_config(
            default_model=user_model.model_name,
            default_api_key=user_model.api_key,
            default_base_url=user_model.base_url,
        ) or user_model
        apply_openai_env(user_model)
    sampling_payload["tau_official_runtime"] = _tau_runtime_model_metadata(user_model, judge_model)

    runtime_cache: dict[str, TauOfficialRuntime] = {}

    def _runtime_for_domain(domain: str) -> TauOfficialRuntime:
        cached = runtime_cache.get(domain)
        if cached is None:
            cached = TauOfficialRuntime(domain=domain)
            runtime_cache[domain] = cached
        return cached

    if args.probe_only:
        repeated = repeat_probe_entries(selected_entries, batch_size=batch_size)
        decision_prompts: list[str] = []
        for _sample_index, record in repeated:
            runtime_env = _runtime_for_domain(record.domain)
            task = runtime_env.load_task(record.task)
            environment = runtime_env.create_environment(solo_mode=False)
            agent = RWKVTauOfficialAgent(
                engine=run.engine,
                sampling=decision_sampling,
                tools=environment.get_tools(),
                domain_policy=str(environment.get_policy()),
                domain=record.domain,
                history_max_chars=history_max_chars,
                prompt_max_chars=prompt_max_chars,
                long_doc_config=long_doc_config,
                tool_routing_config=tool_routing_config,
                retail_repeated_read_guard=retail_repeated_read_guard,
                retail_tool_use_guard=retail_tool_use_guard,
                retail_progressive_tool_disclosure=retail_progressive_tool_disclosure,
            )
            decision_prompts.append(
                agent._build_prompt(  # noqa: SLF001 - probe path intentionally inspects rendered first-turn prompt.
                    [{"role": "user", "content": str(getattr(task, "user_scenario", ""))}]
                )
            )
        run.engine.generate(
            decision_prompts,
            sampling=decision_sampling,
            batch_size=len(decision_prompts),
            progress_desc="TauOfficial-Probe",
            prompt_stop_suffixes=[list(JSON_CALL_STOP_SUFFIXES) for _ in decision_prompts],
            prompt_seeds=[
                sample_repeat_seed(sample_index, 0, stage=2)
                for sample_index, _record in repeated
            ],
        )
        print(f"probe-only run completed: {len(decision_prompts)} prompt(s)")
        return 0

    slug_lower = run.dataset_slug.lower()
    if slug_lower.startswith("tau3_bench"):
        default_job_name = "function_tau3_bench"
    elif slug_lower.startswith("tau2_bench"):
        default_job_name = "function_tau2_bench"
    else:
        default_job_name = "function_tau_bench"
    job_name = _resolve_job_name(default_job_name, run_context=run_context)
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
    _flush_partial_eval = build_partial_eval_flusher(
        ctx=ctx,
        completion_to_eval=_tau_completion_to_eval_payload,
        runner_name="tau_bench",
    )

    pending = list(build_pending_attempts(attempt_keys, records, skip_keys=ctx.skip_keys))
    max_attempt_workers = batch_size if run.engine.__class__.__name__ == "RemoteInferenceBackend" else 1

    try:
        with TaskRunSignalGuard(
            controller=runtime,
            writer=writer,
            close_timeout_s=float(args.db_close_timeout_s),
            on_interrupt=_flush_partial_eval,
        ):
            try:
                with ThreadPoolExecutor(max_workers=max(1, int(max_attempt_workers))) as executor:
                    futures = {
                        executor.submit(
                            _run_tau_official_attempt,
                            args=args,
                            run=run,
                            record=record,
                            sample_index=key.sample_index,
                            repeat_index=key.repeat_index,
                            pass_index=key.pass_index,
                            runtime_env=_runtime_for_domain(record.domain),
                            user_model=user_model,
                            judge_model=judge_model,
                            sampling=decision_sampling,
                            sampling_payload=sampling_payload,
                            history_max_chars=history_max_chars,
                            prompt_max_chars=prompt_max_chars,
                            long_doc_config=long_doc_config,
                            tool_routing_config=tool_routing_config,
                            retail_repeated_read_guard=retail_repeated_read_guard,
                            retail_tool_use_guard=retail_tool_use_guard,
                            retail_progressive_tool_disclosure=retail_progressive_tool_disclosure,
                            max_steps=max_steps,
                            max_tool_errors=max_tool_errors,
                        ): key
                        for key, record in pending
                    }
                    for future in as_completed(futures):
                        writer.enqueue(future.result())
            except Exception:
                runtime.handle_attempt_stage_failure(
                    writer,
                    timeout_s=float(args.db_close_timeout_s),
                    on_after_close=lambda: _flush_partial_eval("exception"),
                )
                raise

        completions_payloads, _eval_payloads, metrics = finalize_function_calling_run(
            ctx=ctx,
            completion_to_eval=_tau_completion_to_eval_payload,
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
                extra={"cot_mode": CoTMode.NO_COT.value},
            ),
        )
    except Exception as exc:
        if not ctx.runtime.state.is_terminal():
            ctx.runtime.fail_task(error=str(exc))
        raise
    print(f"tau function-calling done: {len(completions_payloads)} samples")
    return 0
