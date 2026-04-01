from __future__ import annotations

"""Run MCP-Bench with local RWKV planning and the official Python evaluator bridge."""

import argparse
import os
import signal
from pathlib import Path
from typing import Sequence

from src.db.async_writer import CompletionWriteWorker
from src.db.eval_db_service import EvalDbService
from src.db.orm import init_orm
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import prepare_task_execution, run_checker_for_task
from src.eval.env_config import load_env_file, resolve_judge_model_config
from src.eval.evaluators.common import SampleRecord, StageRecord, sample_repeat_seed
from src.eval.execution_plan import AttemptKey, avg_k_metric_key, build_attempt_keys, build_auto_avg_k_execution_plan, plan_attempt_count
from src.eval.field_common import build_plan_task_details, build_task_sampling_config, set_task_env
from src.eval.function_calling import (
    McpBenchExecutionResult,
    McpBenchWorkerClient,
    append_round_summary,
    build_context_summary,
    build_final_answer_prompt,
    build_mcp_bench_ref_answer,
    build_planning_context,
    build_planning_decision_prompt,
    collapse_mcp_bench_pass,
    load_mcp_bench_manifest_records,
    normalize_planned_tool_call,
    parse_planning_decision,
    render_trace,
    summarize_mcp_bench_evaluation,
)
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.config import DEFAULT_DB_CONFIG, REPO_ROOT
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV MCP-Bench evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="Prepared MCP-Bench JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--max-samples", type=int, help="Limit source task count before avg@k planning")
    parser.add_argument("--max-rounds", type=int, default=20, help="Maximum planning rounds per task")
    parser.add_argument("--planning-max-tokens", type=int, default=2048, help="Clamp planning CoT generation length")
    parser.add_argument("--decision-max-tokens", type=int, default=2048, help="Clamp planning JSON generation length")
    parser.add_argument("--final-max-tokens", type=int, default=3072, help="Clamp final synthesis generation length")
    parser.add_argument("--db-write-queue", type=int, default=8, help="DB completion write queue max size")
    parser.add_argument("--db-close-timeout-s", type=float, default=30.0, help="DB close timeout")
    parser.add_argument("--probe-only", action="store_true", help="Run one planning round probe and skip scoring")
    return parser.parse_args(argv)


def _completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
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


def _ingest_current_eval_payloads(service: EvalDbService, *, task_id: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completions_payloads = service.list_completion_payloads(task_id=task_id, status="Completed")
    eval_payloads = [_completion_to_eval_payload(item) for item in completions_payloads]
    service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
    return completions_payloads, eval_payloads


def _avg_metrics(eval_payloads: Sequence[dict[str, object]], *, avg_k: float) -> dict[str, float]:
    rows = [
        (
            int(payload.get("sample_index", 0)),
            int(payload.get("repeat_index", 0)),
            bool(payload.get("is_passed", False)),
        )
        for payload in eval_payloads
    ]
    metrics = compute_avg_at_k(rows, (avg_k,))
    total = len(rows)
    passed = sum(1 for _, _, ok in rows if ok)
    if total:
        metrics["success_rate"] = passed / total
    metrics.setdefault(avg_k_metric_key(avg_k), metrics.get("success_rate", 0.0))
    return metrics


def _presented_task(item) -> str:
    return item.task.fuzzy_description.strip() or item.task.task_description.strip()


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


def _render_attempt_context(
    item,
    *,
    available_tools: dict[str, dict[str, object]],
    steps: Sequence[dict[str, object]],
    accumulated_information: str,
    final_answer: str,
    evaluation_summary: str,
) -> str:
    tool_names = "\n".join(sorted(available_tools))
    parts = [
        build_context_summary(item),
        "",
        f"available_tools_count={len(available_tools)}",
        "available_tools:",
        tool_names,
    ]
    if steps:
        parts.extend(["", "trace:", render_trace(steps)])
    if accumulated_information.strip():
        parts.extend(["", "accumulated_information:", accumulated_information])
    if final_answer.strip():
        parts.extend(["", "final_answer:", final_answer])
    if evaluation_summary.strip():
        parts.extend(["", f"evaluation={evaluation_summary}"])
    return "\n".join(parts).strip()


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file(Path(".env"))
    args = parse_args(argv)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    dataset_slug = infer_dataset_slug_from_path(str(dataset_path))
    benchmark_name, dataset_split = split_benchmark_and_split(dataset_slug)
    model_name = Path(args.model_path).stem

    items = load_mcp_bench_manifest_records(dataset_path)
    if args.max_samples and args.max_samples > 0:
        items = items[: int(args.max_samples)]
    if not items:
        raise ValueError("MCP-Bench manifest is empty")

    plan = build_auto_avg_k_execution_plan(dataset_slug, len(items))
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    base_sampling = resolve_sampling_config(
        dataset_slug,
        model_name,
        stage="cot",
        fallback_templates="instruction_following_default",
    )
    if base_sampling is None:
        raise ValueError(f"missing sampling config for dataset={dataset_slug}, model={model_name}")
    planning_sampling = base_sampling.clamp(args.planning_max_tokens)
    decision_sampling = base_sampling.clamp(args.decision_max_tokens)
    final_sampling = base_sampling.clamp(args.final_max_tokens)

    model, tokenizer = load_rwkv_model(ModelLoadConfig(weights_path=args.model_path, device=args.device))
    engine = InferenceEngine(model, tokenizer)

    runtime_root = Path(items[0].runtime_root or "").expanduser().resolve()
    worker_script = REPO_ROOT / "src" / "eval" / "function_calling" / "mcp_bench_worker.py"
    if args.probe_only:
        worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
        try:
            available_tools = worker.open_task(items[0])
            prompt = build_planning_context(items[0], available_tools, "")
            engine.generate(
                [prompt],
                sampling=planning_sampling,
                batch_size=1,
                progress_desc="MCPBench-Probe",
            )
            worker.close_task()
        finally:
            worker.close()
        print("probe-only run completed: 1 task")
        return 0

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("MCP-Bench requires JUDGE_MODEL / judge_model_name and judge API key")

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    job_name = os.environ.get("RWKV_SKILLS_JOB_NAME", "function_mcp_bench")
    sampling_payload = normalize_sampling_config_by_stage(
        [(1, planning_sampling), (2, decision_sampling), (3, final_sampling)]
    )
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
            judger_model_name=judge_cfg.model_name,
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
    worker = McpBenchWorkerClient(runtime_root=runtime_root, worker_script=worker_script)
    expected_count = plan_attempt_count(plan, max_pass_k=1)
    should_exit = {"active": False}
    original_signal_handlers: dict[signal.Signals, object] = {}

    def _restore_signal_handlers() -> None:
        for sig, handler in original_signal_handlers.items():
            signal.signal(sig, handler)
        original_signal_handlers.clear()

    def _flush_partial_eval(signame: str) -> None:
        try:
            _ingest_current_eval_payloads(service, task_id=task_id)
        except Exception as exc:
            print(f"failed to ingest partial mcp_bench eval rows during {signame}: {exc}")

    def _handle_termination(signum: int, _frame: object) -> None:
        if should_exit["active"]:
            raise SystemExit(128 + signum)
        should_exit["active"] = True
        signame = signal.Signals(signum).name
        try:
            writer.close(timeout_s=float(args.db_close_timeout_s))
        finally:
            worker.close()
            _flush_partial_eval(signame)
            try:
                service.update_task_status(task_id=task_id, status="failed")
            except Exception:
                pass
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        original_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_termination)

    try:
        pending: list[tuple[AttemptKey, object]] = []
        for key in attempt_keys:
            if key.as_tuple() in skip_keys:
                continue
            pending.append((key, items[int(key.sample_index)]))

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
                total_planned_tools = 0
                valid_planned_tools = 0
                executed_rounds = 0
                for round_num in range(1, max_rounds + 1):
                    cot_context = build_planning_context(item, available_tools, accumulated_information)
                    cot_output = engine.generate(
                        [cot_context],
                        sampling=planning_sampling,
                        batch_size=1,
                        progress_desc="MCPBench-Plan",
                        prompt_seeds=[
                            sample_repeat_seed(
                                sample_index,
                                repeat_index,
                                pass_index=key.pass_index,
                                stage=(round_num - 1) * 3 + 1,
                            )
                        ],
                    )[0]
                    decision_prompt = build_planning_decision_prompt(cot_context, cot_output.text)
                    decision_output = engine.generate(
                        [decision_prompt],
                        sampling=decision_sampling,
                        batch_size=1,
                        progress_desc="MCPBench-Decision",
                        prompt_seeds=[
                            sample_repeat_seed(
                                sample_index,
                                repeat_index,
                                pass_index=key.pass_index,
                                stage=(round_num - 1) * 3 + 2,
                            )
                        ],
                    )[0]
                    stages.append(
                        StageRecord(
                            prompt=cot_context,
                            completion=cot_output.text,
                            stop_reason=cot_output.finish_reason,
                        )
                    )
                    stages.append(
                        StageRecord(
                            prompt=prompt_delta(decision_prompt, f"{cot_context}{cot_output.text}"),
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
                                "cot": cot_output.text,
                                "decision": {"raw": decision_output.text, "parse_error": fail_reason},
                                "executions": [],
                            }
                        )
                        break

                    round_executions: list[McpBenchExecutionResult] = []
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
                    steps.append(
                        {
                            "round_num": round_num,
                            "cot": cot_output.text,
                            "decision": {
                                "reasoning": decision.reasoning,
                                "should_continue": decision.should_continue,
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

                if not fail_reason:
                    final_prompt = build_final_answer_prompt(item, accumulated_information)
                    final_output = engine.generate(
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
                            "task": _presented_task(item),
                            "final_solution": final_answer,
                            "total_rounds": executed_rounds,
                            "available_tools": available_tools,
                            "planning_json_compliance": planning_json_compliance,
                            "accumulated_information": accumulated_information,
                            "concrete_task_description": item.task.task_description if item.task.fuzzy_description.strip() else "",
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
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
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
                payload["instruction"] = _presented_task(item)
                writer.enqueue(payload)
            except Exception as exc:
                ref_answer = build_mcp_bench_ref_answer(item)
                payload = SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
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
                payload["instruction"] = _presented_task(item)
                writer.enqueue(payload)
            finally:
                try:
                    worker.close_task()
                except Exception:
                    pass

        writer.close(timeout_s=float(args.db_close_timeout_s))
    except BaseException:
        try:
            writer.close(timeout_s=float(args.db_close_timeout_s))
        finally:
            worker.close()
            _flush_partial_eval("exception")
            actual = service.count_completions(task_id=task_id, status="Completed")
            status = "failed" if should_exit["active"] else ("completed" if actual == expected_count else "failed")
            service.update_task_status(task_id=task_id, status=status)
        raise
    finally:
        worker.close()
        _restore_signal_handlers()

    completions_payloads, eval_payloads = _ingest_current_eval_payloads(service, task_id=task_id)
    metrics = _avg_metrics(eval_payloads, avg_k=plan.avg_k)
    run_checker_for_task(service=service, task_id=task_id, model_name=model_name)
    score_payload = make_score_payload(
        dataset_slug,
        is_cot=True,
        model_name=model_name,
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
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    print(f"mcp_bench done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
