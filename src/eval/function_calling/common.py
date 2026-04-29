"""Shared helpers for function-calling benchmark runners."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import TypeVar

from src.db.async_writer import CompletionWriteWorker
from src.db.database import init_db
from src.db.eval_db_service import EvalDbService
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunController, TaskRunState, attempt_tuple, prepare_task_execution
from src.eval.execution_plan import AttemptKey, avg_k_metric_key
from src.eval.field_common import build_task_sampling_config, set_task_env
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.scheduler.config import DEFAULT_DB_CONFIG

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

T = TypeVar("T")
AttemptTuple = tuple[int, int, int]


@dataclass(slots=True)
class FunctionCallingRunContext:
    service: EvalDbService
    runtime: TaskRunController
    writer: CompletionWriteWorker
    task_id: str
    skip_keys: frozenset[AttemptTuple]


CompletionToEval = Callable[[dict[str, object]], dict[str, object]]
ScorePayloadBuilder = Callable[
    [Sequence[dict[str, object]], Sequence[dict[str, object]], dict[str, float]],
    Mapping[str, object],
]


def compute_function_calling_metrics(
    eval_payloads: Sequence[Mapping[str, object]],
    *,
    avg_k: float,
) -> dict[str, float]:
    rows = [
        (
            attempt_tuple(payload)[0],
            attempt_tuple(payload)[1],
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


def build_pending_attempts(
    attempt_keys: Sequence[AttemptKey],
    records: Sequence[T],
    *,
    skip_keys: Collection[AttemptTuple],
) -> list[tuple[AttemptKey, T]]:
    pending: list[tuple[AttemptKey, T]] = []
    for key in attempt_keys:
        if key.as_tuple() in skip_keys:
            continue
        pending.append((key, records[int(key.sample_index)]))
    return pending


def repeat_probe_entries(entries: Sequence[T], *, batch_size: int) -> list[T]:
    repeated = list(entries[:batch_size] or entries)
    if repeated and len(repeated) < batch_size:
        repeat_factor = (batch_size + len(repeated) - 1) // len(repeated)
        repeated = (repeated * repeat_factor)[:batch_size]
    return repeated


def prepare_function_calling_run(
    *,
    dataset_slug: str,
    model_name: str,
    job_name: str,
    attempt_keys: Sequence[AttemptKey],
    expected_attempt_count: int,
    sampling_payload: Mapping[str, object],
    avg_k: float,
    effective_sample_count: int,
    db_write_queue: int,
    run_context: "RunContext | None" = None,
    judger_model_name: str | None = None,
) -> FunctionCallingRunContext:
    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    resolved_job_name = run_context.job_name if run_context is not None else job_name
    task_state = prepare_task_execution(
        service=service,
        dataset=str(dataset_slug),
        model=model_name,
        is_param_search=False,
        job_name=resolved_job_name,
        sampling_config=build_task_sampling_config(
            cot_mode=CoTMode.COT,
            avg_k=avg_k,
            sampling_config=sampling_payload,
            effective_sample_count=effective_sample_count,
            judger_model_name=judger_model_name,
        ),
        run_mode=(run_context.run_mode if run_context is not None else None),
    )
    task_run = TaskRunState.from_task_execution(
        execution_state=task_state,
        attempt_keys=attempt_keys,
        expected_attempt_count=expected_attempt_count,
    )
    runtime = TaskRunController(service=service, state=task_run)
    task_id = task_run.task_id
    set_task_env(task_id)
    writer = runtime.create_writer(max_queue=db_write_queue)
    return FunctionCallingRunContext(
        service=service,
        runtime=runtime,
        writer=writer,
        task_id=task_id,
        skip_keys=frozenset(task_state.skip_keys),
    )


def build_partial_eval_flusher(
    *,
    ctx: FunctionCallingRunContext,
    completion_to_eval: CompletionToEval,
    runner_name: str,
) -> Callable[[str], None]:
    def _flush_partial_eval(signame: str) -> None:
        try:
            completions_payloads = ctx.service.list_completion_payloads(task_id=ctx.task_id, status="Completed")
            eval_payloads = [completion_to_eval(item) for item in completions_payloads]
            ctx.runtime.ingest_eval_payloads(eval_payloads)
        except Exception as exc:
            print(f"failed to ingest partial {runner_name} eval rows during {signame}: {exc}")

    return _flush_partial_eval


def finalize_function_calling_run(
    *,
    ctx: FunctionCallingRunContext,
    completion_to_eval: CompletionToEval,
    model_name: str,
    avg_k: float,
    timeout_s: float | None,
    build_score_payload: ScorePayloadBuilder,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    completions_payloads = ctx.runtime.complete_attempt_stage(ctx.writer, timeout_s=timeout_s)
    eval_payloads = [completion_to_eval(item) for item in completions_payloads]
    ctx.runtime.ingest_eval_payloads(eval_payloads)
    metrics = compute_function_calling_metrics(eval_payloads, avg_k=avg_k)
    ctx.runtime.run_checker(model_name=model_name)
    ctx.runtime.record_score(build_score_payload(completions_payloads, eval_payloads, metrics))
    return completions_payloads, eval_payloads, metrics


__all__ = [
    "CompletionToEval",
    "FunctionCallingRunContext",
    "ScorePayloadBuilder",
    "build_partial_eval_flusher",
    "build_pending_attempts",
    "compute_function_calling_metrics",
    "finalize_function_calling_run",
    "prepare_function_calling_run",
    "repeat_probe_entries",
]
