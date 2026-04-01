from __future__ import annotations

"""Run BrowseComp / BrowseComp-ZH evaluation with local RWKV inference."""

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
from src.eval.execution_plan import (
    AttemptKey,
    avg_k_metric_key,
    build_attempt_keys,
    build_auto_avg_k_execution_plan,
    plan_attempt_count,
)
from src.eval.field_common import build_plan_task_details, build_task_sampling_config, set_task_env
from src.eval.function_calling import (
    BrowseCompJudgeConfig,
    build_browsecomp_answer_prompt,
    build_browsecomp_expected_context,
    build_browsecomp_user_prompt,
    judge_browsecomp_answers,
    load_browsecomp_manifest_records,
)
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import make_eval_payload, normalize_sampling_config_by_stage, prompt_delta
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV BrowseComp evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="Prepared BrowseComp JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="Generation batch size")
    parser.add_argument("--max-samples", type=int, help="Limit source question count before avg@k planning")
    parser.add_argument("--cot-max-tokens", type=int, default=2048, help="Clamp CoT generation length")
    parser.add_argument("--answer-max-tokens", type=int, default=1024, help="Clamp final answer generation length")
    parser.add_argument("--db-write-queue", type=int, default=32, help="DB completion write queue max size")
    parser.add_argument("--db-close-timeout-s", type=float, default=30.0, help="DB close timeout")
    parser.add_argument("--probe-only", action="store_true", help="Run a minimal batch probe and skip scoring")
    return parser.parse_args(argv)


def _normalize_final_answer(text: str, *, locale: str) -> str:
    body = text.strip()
    if not body:
        return ""
    prefix = "解释:" if locale == "zh" else "Explanation:"
    return body if body.startswith(prefix) else f"{prefix} {body}"


def _completion_to_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    agent_result = payload.get("agent_result")
    if not isinstance(agent_result, dict):
        agent_result = {}
    agent_info = payload.get("agent_info")
    if not isinstance(agent_info, dict):
        agent_info = {}
    passed = bool(agent_result.get("is_passed", False))
    reason = str(agent_info.get("judge_reason") or "")
    return make_eval_payload(
        payload,
        is_passed=passed,
        fail_reason=reason if not passed else "",
        answer=str(agent_info.get("response") or ""),
        ref_answer=str(agent_info.get("reference_answer") or ""),
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


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file(Path(".env"))
    args = parse_args(argv)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    dataset_slug = infer_dataset_slug_from_path(str(dataset_path))
    benchmark_name, dataset_split = split_benchmark_and_split(dataset_slug)
    model_name = Path(args.model_path).stem

    records = load_browsecomp_manifest_records(dataset_path)
    if args.max_samples and args.max_samples > 0:
        records = records[: int(args.max_samples)]
    if not records:
        raise ValueError("BrowseComp manifest is empty")

    plan = build_auto_avg_k_execution_plan(dataset_slug, len(records))
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    cot_sampling = resolve_sampling_config(
        dataset_slug,
        model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    answer_sampling = resolve_sampling_config(
        dataset_slug,
        model_name,
        stage="final",
        fallback_templates="free_response_cot_default",
    )
    if cot_sampling is None or answer_sampling is None:
        raise ValueError(f"missing sampling config for dataset={dataset_slug}, model={model_name}")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    answer_sampling = answer_sampling.clamp(args.answer_max_tokens)

    model, tokenizer = load_rwkv_model(ModelLoadConfig(weights_path=args.model_path, device=args.device))
    engine = InferenceEngine(model, tokenizer)

    batch_size = max(1, int(args.batch_size))
    selected_entries = [
        (int(sample_index), records[int(sample_index)])
        for sample_index in plan.sample_indices
    ]

    if args.probe_only:
        probe_entries = selected_entries[:batch_size] or selected_entries
        repeated = probe_entries
        if repeated and len(repeated) < batch_size:
            repeat_factor = (batch_size + len(repeated) - 1) // len(repeated)
            repeated = (repeated * repeat_factor)[:batch_size]
        prompts = [
            build_browsecomp_expected_context(
                build_browsecomp_user_prompt(record.question, locale=record.locale)
            )
            for _, record in repeated
        ]
        engine.generate(
            prompts,
            sampling=cot_sampling,
            batch_size=len(prompts),
            progress_desc="BrowseComp-Probe",
        )
        print(f"probe-only run completed: {len(prompts)} prompt(s)")
        return 0

    judge_cfg = resolve_judge_model_config()
    if judge_cfg is None:
        raise ValueError("BrowseComp requires JUDGE_MODEL / judge_model_name and judge API key")
    judge = BrowseCompJudgeConfig(
        api_key=judge_cfg.api_key,
        model=judge_cfg.model_name,
        base_url=judge_cfg.base_url,
    )

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    job_name = os.environ.get("RWKV_SKILLS_JOB_NAME", "function_browsecomp")
    sampling_payload = normalize_sampling_config_by_stage([(1, cot_sampling), (2, answer_sampling)])
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
            judger_model_name=judge.model,
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
            _ingest_current_eval_payloads(service, task_id=task_id)
        except Exception as exc:
            print(f"failed to ingest partial browsecomp eval rows during {signame}: {exc}")

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

    try:
        pending: list[tuple[AttemptKey, object]] = []
        for key in attempt_keys:
            if key.as_tuple() in skip_keys:
                continue
            pending.append((key, records[int(key.sample_index)]))

        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            cot_prompts = [
                build_browsecomp_expected_context(
                    build_browsecomp_user_prompt(record.question, locale=record.locale)
                )
                for _key, record in chunk
            ]
            cot_outputs = engine.generate(
                cot_prompts,
                sampling=cot_sampling,
                batch_size=len(cot_prompts),
                progress_desc="BrowseComp-CoT",
                prompt_seeds=[
                    sample_repeat_seed(
                        key.sample_index,
                        key.repeat_index,
                        pass_index=key.pass_index,
                        stage=1,
                    )
                    for key, _record in chunk
                ],
            )
            answer_prompts: list[str] = []
            answer_stage_prompts: list[str] = []
            for (key, record), cot_output in zip(chunk, cot_outputs):
                answer_prompt = build_browsecomp_answer_prompt(
                    cot_prompts[len(answer_prompts)],
                    cot_output.text,
                    locale=record.locale,
                )
                answer_prompts.append(answer_prompt)
                answer_stage_prompts.append(prompt_delta(answer_prompt, f"{cot_output.prompt}{cot_output.text}"))
            answer_outputs = engine.generate(
                answer_prompts,
                sampling=answer_sampling,
                batch_size=len(answer_prompts),
                progress_desc="BrowseComp-Answer",
                prompt_seeds=[
                    sample_repeat_seed(
                        key.sample_index,
                        key.repeat_index,
                        pass_index=key.pass_index,
                        stage=2,
                    )
                    for key, _record in chunk
                ],
            )
            judged = judge_browsecomp_answers(
                [
                    (record, _normalize_final_answer(answer_output.text, locale=record.locale))
                    for (_key, record), answer_output in zip(chunk, answer_outputs)
                ],
                config=judge,
            )
            for index, ((key, record), cot_output, answer_output, outcome) in enumerate(
                zip(chunk, cot_outputs, answer_outputs, judged)
            ):
                final_answer = _normalize_final_answer(answer_output.text, locale=record.locale)
                stages = [
                    StageRecord(
                        prompt=cot_prompts[index],
                        completion=cot_output.text,
                        stop_reason=cot_output.finish_reason,
                    ),
                    StageRecord(
                        prompt=answer_stage_prompts[index],
                        completion=answer_output.text,
                        stop_reason=answer_output.finish_reason,
                    ),
                ]
                payload = SampleRecord(
                    benchmark_name=benchmark_name,
                    dataset_split=dataset_split,
                    sample_index=key.sample_index,
                    repeat_index=key.repeat_index,
                    pass_index=key.pass_index,
                    stages=stages,
                    sampling_config=sampling_payload,
                ).as_payload()
                payload["agent_result"] = {
                    "reward": 1.0 if outcome.is_passed else 0.0,
                    "num_turns": 2,
                    "cost": 0.0,
                    "is_passed": bool(outcome.is_passed),
                }
                payload["agent_info"] = {
                    "question": record.question,
                    "reference_answer": record.answer,
                    "response": final_answer,
                    "judge_reason": outcome.reason,
                    "locale": record.locale,
                    "cot_mode": CoTMode.COT.value,
                    "topic": record.topic or "",
                }
                payload["agent_trace"] = [
                    {"stage": "cot", "text": cot_output.text},
                    {"stage": "answer", "text": final_answer},
                ]
                payload["task_id"] = record.task_id
                payload["domain"] = "function_call"
                payload["instruction"] = record.question
                writer.enqueue(payload)
        writer.close(timeout_s=float(args.db_close_timeout_s))
    except BaseException:
        try:
            writer.close(timeout_s=float(args.db_close_timeout_s))
        finally:
            _flush_partial_eval("exception")
            actual = service.count_completions(task_id=task_id, status="Completed")
            status = "failed" if should_exit["active"] else ("completed" if actual == expected_count else "failed")
            service.update_task_status(task_id=task_id, status=status)
        raise
    finally:
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
        problems=len(records),
        task=job_name,
        task_details=build_plan_task_details(plan, cot_mode=CoTMode.COT.value),
        extra={
            "sampling_config": sampling_payload,
            "judger_model_name": judge.model,
            "cot_mode": CoTMode.COT.value,
        },
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    print(f"browsecomp done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
