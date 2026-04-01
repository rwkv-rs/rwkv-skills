from __future__ import annotations

"""Run tau_bench / tau2_bench with a rwkv-rs-style function-calling loop."""

import argparse
import json
import os
import signal
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.db.async_writer import CompletionWriteWorker
from src.db.eval_db_service import EvalDbService
from src.db.orm import init_orm
from src.eval.agent_bench.envs.tau_v2 import TauV2Env
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import prepare_task_execution, run_checker_for_task
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import AttemptKey, avg_k_metric_key, build_attempt_keys, build_auto_avg_k_execution_plan, plan_attempt_count
from src.eval.field_common import build_plan_task_details, build_task_sampling_config, set_task_env
from src.eval.function_calling import (
    TauManifestRecord,
    TauToolCall,
    build_expected_context,
    build_tau_system_prompt,
    build_turn_completion_prompt,
    load_tau_manifest_records,
    parse_tool_call_or_final_answer,
    render_assistant_tool_message,
    render_tau_user_prompt,
    render_tool_result,
)
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split


DEFAULT_MAX_STEPS = 16
DEFAULT_MAX_TOOL_ERRORS = 4


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV tau_bench/tau2_bench function-calling evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="Prepared tau_bench/tau2_bench JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--max-samples", type=int, help="Limit source task count before avg@k planning")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum tool/answer turns per task")
    parser.add_argument(
        "--max-tool-errors",
        type=int,
        default=DEFAULT_MAX_TOOL_ERRORS,
        help="Abort one task after this many tool-call errors",
    )
    parser.add_argument("--cot-max-tokens", type=int, default=2048, help="Clamp CoT generation length")
    parser.add_argument("--decision-max-tokens", type=int, default=1024, help="Clamp tool/final-answer generation length")
    parser.add_argument("--db-write-queue", type=int, default=32, help="DB completion write queue max size")
    parser.add_argument("--db-close-timeout-s", type=float, default=30.0, help="DB close timeout")
    parser.add_argument("--probe-only", action="store_true", help="Run one CoT+decision probe batch and skip scoring")
    return parser.parse_args(argv)


def _default_job_name(dataset_slug: str) -> str:
    lower = dataset_slug.lower()
    if lower.startswith("tau2_bench"):
        return "function_tau2_bench"
    return "function_tau_bench"


def _assistant_tools_schema(runtime_env: TauV2Env, environment: Any) -> list[dict[str, Any]]:
    return runtime_env.tools_schema(environment)


def _user_tools_schema(runtime_env: TauV2Env, environment: Any) -> list[dict[str, Any]]:
    return runtime_env.user_tools_schema(environment)


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
                        "requestor": str(getattr(tool_call, "requestor", "assistant") or "assistant"),
                        "name": str(getattr(tool_call, "name", "") or ""),
                        "arguments": dict(getattr(tool_call, "arguments", {}) or {}),
                    }
                    blocks.append(f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```")
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
        assistant_tools=_assistant_tools_schema(runtime_env, environment),
        user_tools=_user_tools_schema(runtime_env, environment),
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sum_message_costs(trajectory: Sequence[Any]) -> float:
    total = 0.0
    for item in trajectory:
        total += _safe_float(getattr(item, "cost", None), 0.0)
    return total


def _ref_answer(state: _ActiveEpisode) -> str:
    task_id = str(getattr(state.task, "id", "") or state.record.task_id)
    return f"domain={state.record.domain}\ntask_id={task_id}\nbenchmark_version={state.record.benchmark_version}"


def _completion_payload(
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


def _completion_to_eval_payload(payload: dict[str, Any]) -> dict[str, Any]:
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


def _avg_metrics(eval_payloads: Sequence[dict[str, Any]], *, avg_k: float) -> dict[str, float]:
    from src.eval.metrics.at_k import compute_avg_at_k

    rows = [
        (
            int(item.get("sample_index", 0)),
            int(item.get("repeat_index", 0)),
            bool(item.get("is_passed", False)),
        )
        for item in eval_payloads
    ]
    metrics = compute_avg_at_k(rows, (avg_k,))
    total = len(rows)
    passed = sum(1 for _, _, ok in rows if ok)
    if total:
        metrics["success_rate"] = passed / total
    metrics.setdefault(avg_k_metric_key(avg_k), metrics.get("success_rate", 0.0))
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    from src.infer.engine import InferenceEngine
    from src.infer.model import ModelLoadConfig, load_rwkv_model

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    dataset_slug = infer_dataset_slug_from_path(str(dataset_path))
    benchmark_name, dataset_split = split_benchmark_and_split(dataset_slug)
    model_name = Path(args.model_path).stem

    records = load_tau_manifest_records(dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("tau_bench/tau2_bench manifest is empty")

    plan = build_auto_avg_k_execution_plan(dataset_slug, len(records))
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    cot_sampling = resolve_sampling_config(
        dataset_slug,
        model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    decision_sampling = resolve_sampling_config(
        dataset_slug,
        model_name,
        stage="final",
        fallback_templates="instruction_following_default",
    )
    if cot_sampling is None or decision_sampling is None:
        raise ValueError(f"missing sampling config for dataset={dataset_slug}, model={model_name}")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    decision_sampling = decision_sampling.clamp(args.decision_max_tokens)
    sampling_payload = normalize_sampling_config_by_stage([(1, cot_sampling), (2, decision_sampling)])

    model, tokenizer = load_rwkv_model(ModelLoadConfig(weights_path=args.model_path, device=args.device))
    engine = InferenceEngine(model, tokenizer)

    selected_entries = [(int(index), records[int(index)]) for index in plan.sample_indices]
    batch_size = max(1, int(args.batch_size))
    max_steps = max(1, int(args.max_steps))
    max_tool_errors = max(1, int(args.max_tool_errors))

    runtime_cache: dict[str, TauV2Env] = {}

    def _runtime_for_domain(domain: str) -> TauV2Env:
        cached = runtime_cache.get(domain)
        if cached is None:
            cached = TauV2Env(domain=domain, judge=None)
            runtime_cache[domain] = cached
        return cached

    if args.probe_only:
        probe_entries = selected_entries[:batch_size] or selected_entries
        repeated = probe_entries
        if repeated and len(repeated) < batch_size:
            repeat_factor = (batch_size + len(repeated) - 1) // len(repeated)
            repeated = (repeated * repeat_factor)[:batch_size]
        probe_states = [
            _start_episode(
                sample_index=sample_index,
                repeat_index=0,
                pass_index=0,
                record=record,
                runtime_env=_runtime_for_domain(record.domain),
            )
            for sample_index, record in repeated
        ]
        cot_prompts = [build_expected_context(state.system_prompt, state.prompt_messages) for state in probe_states]
        cot_outputs = engine.generate(
            cot_prompts,
            sampling=cot_sampling,
            batch_size=len(cot_prompts),
            progress_desc="TauBench-Probe-CoT",
            prompt_seeds=[
                sample_repeat_seed(state.sample_index, state.repeat_index, stage=1)
                for state in probe_states
            ],
        )
        by_index = {int(item.prompt_index): item for item in cot_outputs}
        decision_prompts = [
            build_turn_completion_prompt(cot_prompts[idx], by_index[idx].text if idx in by_index else "")
            for idx in range(len(cot_prompts))
        ]
        engine.generate(
            decision_prompts,
            sampling=decision_sampling,
            batch_size=len(decision_prompts),
            progress_desc="TauBench-Probe-Decision",
            prompt_seeds=[
                sample_repeat_seed(state.sample_index, state.repeat_index, stage=2)
                for state in probe_states
            ],
        )
        print(f"probe-only run completed: {len(probe_states)} prompt(s)")
        return 0

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    job_name = os.environ.get("RWKV_SKILLS_JOB_NAME", _default_job_name(dataset_slug))
    task_state = prepare_task_execution(
        service=service,
        dataset=str(dataset_slug),
        model=model_name,
        is_param_search=False,
        job_name=job_name,
        sampling_config=build_task_sampling_config(
            cot_mode=CoTMode.COT,
            avg_k=plan.avg_k,
            sampling_config=sampling_payload,
            effective_sample_count=plan.effective_sample_count,
        ),
    )
    task_id = task_state.task_id
    skip_keys = task_state.skip_keys
    set_task_env(task_id)

    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
    expected_count = plan_attempt_count(plan, max_pass_k=1)
    should_exit = {"active": False}
    original_signal_handlers: dict[signal.Signals, object] = {}

    def _restore_signal_handlers() -> None:
        for sig, handler in original_signal_handlers.items():
            signal.signal(sig, handler)
        original_signal_handlers.clear()

    def _flush_partial_eval(signame: str) -> None:
        try:
            completions_payloads = service.list_completion_payloads(task_id=task_id, status="Completed")
            eval_payloads = [_completion_to_eval_payload(item) for item in completions_payloads]
            service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
        except Exception as exc:
            print(f"failed to ingest partial tau_bench eval rows during {signame}: {exc}")

    def _handle_termination(signum: int, _frame: object) -> None:
        if should_exit["active"]:
            raise SystemExit(128 + signum)
        should_exit["active"] = True
        signame = signal.Signals(signum).name
        try:
            writer.close(timeout_s=float(args.db_close_timeout_s))
        finally:
            _flush_partial_eval(signame)
            try:
                service.update_task_status(task_id=task_id, status="failed")
            except Exception:
                pass
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        original_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_termination)

    pending: deque[tuple[AttemptKey, TauManifestRecord]] = deque(
        (key, records[int(key.sample_index)])
        for key in attempt_keys
        if key.as_tuple() not in skip_keys
    )

    try:
        active: list[_ActiveEpisode] = []
        while pending or active:
            while pending and len(active) < batch_size:
                key, record = pending.popleft()
                active.append(
                    _start_episode(
                        sample_index=key.sample_index,
                        repeat_index=key.repeat_index,
                        pass_index=key.pass_index,
                        record=record,
                        runtime_env=_runtime_for_domain(record.domain),
                    )
                )

            if not active:
                break

            cot_prompts = [build_expected_context(state.system_prompt, state.prompt_messages) for state in active]
            cot_outputs = engine.generate(
                cot_prompts,
                sampling=cot_sampling,
                batch_size=len(cot_prompts),
                progress_desc="TauBench-CoT",
                prompt_seeds=[
                    sample_repeat_seed(
                        state.sample_index,
                        state.repeat_index,
                        pass_index=state.pass_index,
                        stage=state.turn_count * 2 + 1,
                    )
                    for state in active
                ],
            )
            cot_by_index = {int(item.prompt_index): item for item in cot_outputs}

            decision_prompts = [
                build_turn_completion_prompt(cot_prompts[idx], cot_by_index.get(idx).text if cot_by_index.get(idx) else "")
                for idx in range(len(active))
            ]
            decision_outputs = engine.generate(
                decision_prompts,
                sampling=decision_sampling,
                batch_size=len(decision_prompts),
                progress_desc="TauBench-Decision",
                prompt_seeds=[
                    sample_repeat_seed(
                        state.sample_index,
                        state.repeat_index,
                        pass_index=state.pass_index,
                        stage=state.turn_count * 2 + 2,
                    )
                    for state in active
                ],
            )
            decision_by_index = {int(item.prompt_index): item for item in decision_outputs}

            finished_slots: list[int] = []
            for slot_index, state in enumerate(active):
                cot_output = cot_by_index.get(slot_index)
                decision_output = decision_by_index.get(slot_index)
                cot_text = cot_output.text if cot_output is not None else ""
                decision_text = decision_output.text if decision_output is not None else ""
                prior_context = cot_prompts[slot_index].replace("<|completions_of_cot|>", cot_text)
                state.stages.append(
                    StageRecord(
                        prompt=cot_prompts[slot_index],
                        completion=cot_text,
                        stop_reason=cot_output.finish_reason if cot_output is not None else "missing_output",
                    )
                )
                state.stages.append(
                    StageRecord(
                        prompt=prompt_delta(decision_prompts[slot_index], prior_context),
                        completion=decision_text,
                        stop_reason=decision_output.finish_reason if decision_output is not None else "missing_output",
                    )
                )
                state.turn_count += 1

                try:
                    decision = parse_tool_call_or_final_answer(decision_text)
                except Exception as exc:
                    state.termination_reason = "parse_error"
                    state.error = str(exc)
                    finished_slots.append(slot_index)
                    continue

                if decision.is_tool_call and decision.tool_call is not None:
                    tool_call = decision.tool_call
                    state.tool_calls.append(tool_call)
                    state.prompt_messages.append(
                        {"role": "assistant", "content": render_assistant_tool_message(cot_text, tool_call)}
                    )
                    try:
                        tool_call_model = state.runtime_env.build_tool_call(
                            tool_call_id=(
                                f"{state.sample_index}-{state.repeat_index}-{state.pass_index}-"
                                f"{uuid.uuid4().hex[:8]}"
                            ),
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            requestor=tool_call.requestor,
                        )
                        assistant_message = state.runtime_env.build_assistant_message(
                            content=None,
                            tool_calls=[tool_call_model],
                        )
                        state.trajectory.append(assistant_message)
                        tool_message = state.runtime_env.call_tool(
                            environment=state.environment,
                            tool_call=tool_call_model,
                        )
                        state.trajectory.append(tool_message)
                        ok, output, error_text = _tool_output_payload(tool_message)
                        state.prompt_messages.append(
                            {
                                "role": "user",
                                "content": render_tool_result(
                                    tool_call,
                                    ok=ok,
                                    output=output,
                                    error=error_text,
                                ),
                            }
                        )
                        if not ok:
                            state.tool_errors += 1
                    except Exception as exc:
                        state.tool_errors += 1
                        state.prompt_messages.append(
                            {
                                "role": "user",
                                "content": render_tool_result(tool_call, ok=False, error=str(exc)),
                            }
                        )

                    if state.tool_errors >= max_tool_errors:
                        state.termination_reason = "too_many_errors"
                        finished_slots.append(slot_index)
                        continue
                    if state.turn_count >= max_steps:
                        state.termination_reason = "max_steps"
                        finished_slots.append(slot_index)
                    continue

                state.final_answer = decision.final_answer.strip()
                state.prompt_messages.append({"role": "assistant", "content": state.final_answer})
                state.trajectory.append(
                    state.runtime_env.build_assistant_message(content=state.final_answer)
                )
                state.termination_reason = "agent_stop"
                finished_slots.append(slot_index)

            completed_payloads = [
                _completion_payload(
                    active[slot_index],
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sampling_payload=sampling_payload,
                )
                for slot_index in finished_slots
            ]
            for payload in completed_payloads:
                writer.enqueue(payload)

            for slot_index in reversed(finished_slots):
                active.pop(slot_index)

        writer.close(timeout_s=float(args.db_close_timeout_s))
    except BaseException:
        try:
            writer.close(timeout_s=float(args.db_close_timeout_s))
        finally:
            actual = service.count_completions(task_id=task_id, status="Completed")
            status = "failed" if should_exit["active"] else ("completed" if actual == expected_count else "failed")
            service.update_task_status(task_id=task_id, status=status)
        raise
    finally:
        _restore_signal_handlers()

    completions_payloads = service.list_completion_payloads(task_id=task_id, status="Completed")
    eval_payloads = [_completion_to_eval_payload(item) for item in completions_payloads]
    metrics = _avg_metrics(eval_payloads, avg_k=plan.avg_k)
    service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
    run_checker_for_task(service=service, task_id=task_id, model_name=model_name)
    score_payload = make_score_payload(
        dataset_slug,
        is_cot=True,
        model_name=model_name,
        metrics=metrics,
        samples=len(completions_payloads),
        problems=plan.sample_size,
        task=job_name,
        task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
        extra={"cot_mode": CoTMode.COT.value},
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    print(f"tau function-calling done: {len(completions_payloads)} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
