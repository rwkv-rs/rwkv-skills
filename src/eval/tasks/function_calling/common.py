"""Shared helpers for function-calling benchmark runners."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import TypeVar

from src.db.async_writer import CompletionWriteWorker
from src.db.eval_service import create_eval_service, init_eval_store
from src.eval.benchmark_registry import CoTMode
from src.eval.evaluating import TaskRunController, TaskRunState, attempt_tuple, prepare_task_execution
from src.eval.execution_plan import AttemptKey, avg_k_metric_key
from src.eval.field_common import build_task_sampling_config, set_task_env
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.scheduler.config import DBConfig, DEFAULT_DB_CONFIG
from src.infer.sampling import SamplingConfig

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext
    from src.eval.long_doc_evidence import LongDocEvidenceConfig
    from src.eval.tasks.function_calling.tool_router import ToolRoutingConfig

T = TypeVar("T")
AttemptTuple = tuple[int, int, int]


@dataclass(slots=True)
class FunctionCallingRunContext:
    service: object
    runtime: TaskRunController
    writer: CompletionWriteWorker
    task_id: str
    skip_keys: frozenset[AttemptTuple]


CompletionToEval = Callable[[dict[str, object]], dict[str, object]]
ScorePayloadBuilder = Callable[
    [Sequence[dict[str, object]], Sequence[dict[str, object]], dict[str, float]],
    Mapping[str, object],
]
FUNCTION_CALLING_STOP_TOKENS = (0,)


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


def compute_function_calling_diagnostics(
    completions_payloads: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    prompt_lengths: list[int] = []
    sample_prompt_totals: list[int] = []
    completion_lengths: list[int] = []
    turn_counts: list[float] = []
    error_count = 0
    unknown_tool_errors = 0
    parse_errors = 0
    long_doc_prompt_hits = 0
    route_rows: list[Mapping[str, object]] = []

    for payload in completions_payloads:
        sample_prompt_total = 0
        for key, value in payload.items():
            key_text = str(key)
            if key_text.startswith("prompt") and isinstance(value, str):
                prompt_lengths.append(len(value))
                sample_prompt_total += len(value)
                if "Long document compacted" in value:
                    long_doc_prompt_hits += 1
            elif key_text.startswith("completion") and isinstance(value, str):
                completion_lengths.append(len(value))
        if sample_prompt_total:
            sample_prompt_totals.append(sample_prompt_total)

        result = payload.get("agent_result")
        if isinstance(result, Mapping):
            raw_error = result.get("error")
            error_text = str(raw_error or "")
            if error_text:
                error_count += 1
                lowered = error_text.lower()
                if "unknown" in lowered and "tool" in lowered:
                    unknown_tool_errors += 1
                if "parse" in lowered or "json" in lowered or "decision" in lowered:
                    parse_errors += 1
            try:
                turn_counts.append(float(result.get("num_turns", 0) or 0))
            except (TypeError, ValueError):
                pass

        route_rows.extend(_iter_tool_route_rows(payload))

    metrics: dict[str, float] = {}
    if prompt_lengths:
        avg_prompt_chars = sum(prompt_lengths) / len(prompt_lengths)
        metrics["avg_stage_prompt_chars"] = avg_prompt_chars
        metrics["avg_sample_prompt_chars"] = avg_prompt_chars
        metrics["max_stage_prompt_chars"] = float(max(prompt_lengths))
        metrics["long_doc_prompt_rate"] = long_doc_prompt_hits / len(prompt_lengths)
    if sample_prompt_totals:
        metrics["avg_sample_total_prompt_chars"] = sum(sample_prompt_totals) / len(sample_prompt_totals)
    if completion_lengths:
        metrics["avg_stage_completion_chars"] = sum(completion_lengths) / len(completion_lengths)
        metrics["max_stage_completion_chars"] = float(max(completion_lengths))
    if completions_payloads:
        total = len(completions_payloads)
        metrics["agent_error_rate"] = error_count / total
        metrics["unknown_tool_error_rate"] = unknown_tool_errors / total
        metrics["decision_parse_error_rate"] = parse_errors / total
    if turn_counts:
        metrics["avg_agent_turns"] = sum(turn_counts) / len(turn_counts)

    if route_rows:
        routed = sum(1 for row in route_rows if bool(row.get("routed", False)))
        selected_counts = [_selected_tool_count(row) for row in route_rows]
        total_tool_counts = [_float_value(row.get("total_tool_count")) for row in route_rows]
        catalog_chars = [_float_value(row.get("catalog_chars")) for row in route_rows]
        fallback_count = sum(1 for row in route_rows if "fallback" in str(row.get("reason") or ""))
        model_error_count = sum(1 for row in route_rows if "model_error" in str(row.get("reason") or ""))
        metrics["tool_route_count"] = float(len(route_rows))
        metrics["tool_route_routed_rate"] = routed / len(route_rows)
        metrics["tool_route_avg_selected_tools"] = sum(selected_counts) / len(selected_counts)
        if total_tool_counts:
            metrics["tool_route_avg_total_tools"] = sum(total_tool_counts) / len(total_tool_counts)
        if catalog_chars:
            metrics["tool_route_avg_catalog_chars"] = sum(catalog_chars) / len(catalog_chars)
        metrics["tool_route_fallback_rate"] = fallback_count / len(route_rows)
        metrics["tool_route_model_error_rate"] = model_error_count / len(route_rows)
    return metrics


def attach_function_calling_context_metadata(
    sampling_payload: Mapping[str, object],
    *,
    long_doc_config: "LongDocEvidenceConfig | None" = None,
    tool_routing_config: "ToolRoutingConfig | None" = None,
    candidate_router_config: object | None = None,
    prompt_max_chars: int | None = None,
) -> dict[str, object]:
    payload = dict(sampling_payload)
    context: dict[str, object] = {}
    if prompt_max_chars is not None:
        context["prompt_max_chars"] = int(prompt_max_chars)
    if long_doc_config is not None:
        context["long_doc"] = {
            "enabled": bool(getattr(long_doc_config, "enabled", True)),
            "mode": str(getattr(long_doc_config, "mode", "lexical")),
            "max_chunk_chars": int(getattr(long_doc_config, "max_chunk_chars", 0)),
            "overlap_lines": int(getattr(long_doc_config, "overlap_lines", 0)),
            "min_long_text_chars": int(getattr(long_doc_config, "min_long_text_chars", 0)),
            "max_evidence_chunks": int(getattr(long_doc_config, "max_evidence_chunks", 0)),
            "max_evidence_chars": int(getattr(long_doc_config, "max_evidence_chars", 0)),
        }
    if tool_routing_config is not None:
        context["tool_router"] = {
            "mode": str(getattr(tool_routing_config, "mode", "off")),
            "max_tools": int(getattr(tool_routing_config, "max_tools", 0)),
            "trigger_tool_count": int(getattr(tool_routing_config, "trigger_tool_count", 0)),
            "trigger_catalog_chars": int(getattr(tool_routing_config, "trigger_catalog_chars", 0)),
            "context_chars": int(getattr(tool_routing_config, "context_chars", 0)),
            "max_tokens": int(getattr(tool_routing_config, "max_tokens", 0)),
            "description_chars": int(getattr(tool_routing_config, "description_chars", 0)),
        }
    if candidate_router_config is not None:
        context["candidate_router"] = {
            "mode": "parallel",
            "chunk_tools": int(getattr(candidate_router_config, "chunk_tools", 0)),
            "batch_size": int(getattr(candidate_router_config, "batch_size", 0)),
            "context_chars": int(getattr(candidate_router_config, "context_chars", 0)),
            "prompt_max_chars": int(getattr(candidate_router_config, "prompt_max_chars", 0)),
            "candidate_max_tokens": int(getattr(candidate_router_config, "candidate_max_tokens", 0)),
            "aggregate_max_tokens": int(getattr(candidate_router_config, "aggregate_max_tokens", 0)),
            "max_candidates": int(getattr(candidate_router_config, "max_candidates", 0)),
            "tool_schema_mode": str(getattr(candidate_router_config, "tool_schema_mode", "")),
            "include_respond": bool(getattr(candidate_router_config, "include_respond", False)),
            "fallback_to_highest_confidence": bool(
                getattr(candidate_router_config, "fallback_to_highest_confidence", False)
            ),
            "evidence_chars": int(getattr(candidate_router_config, "evidence_chars", 0)),
            "policy_chars": int(getattr(candidate_router_config, "policy_chars", 0)),
            "ground_identifier_arguments": bool(getattr(candidate_router_config, "ground_identifier_arguments", False)),
        }
    if context:
        payload["long_context"] = context
    return payload


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


def clamp_function_calling_sampling(config: SamplingConfig, max_tokens: int | None) -> SamplingConfig:
    """Use role-boundary string suffixes instead of unsafe raw role tokens."""

    clamped = config.clamp(max_tokens)
    return replace(clamped, stop_tokens=FUNCTION_CALLING_STOP_TOKENS)


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
    db_config: DBConfig | None = None,
) -> FunctionCallingRunContext:
    init_eval_store(db_config or DEFAULT_DB_CONFIG)
    service = create_eval_service()
    resolved_job_name = run_context.job_name if run_context is not None else job_name
    task_state = prepare_task_execution(
        service=service,
        dataset=str(dataset_slug),
        model=model_name,
        is_param_search=False,
        job_name=resolved_job_name,
        sampling_config=build_task_sampling_config(
            cot_mode=CoTMode.NO_COT,
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
    metrics.update(compute_function_calling_diagnostics(completions_payloads))
    try:
        ctx.runtime.run_checker(model_name=model_name)
    except Exception as exc:
        metrics["checker_failed"] = 1.0
        print(f"warning: function-calling checker failed; score will still be recorded: {exc}")
    ctx.runtime.record_score(build_score_payload(completions_payloads, eval_payloads, metrics))
    return completions_payloads, eval_payloads, metrics


def _iter_tool_route_rows(value: object) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        if "selected_names" in value and "total_tool_count" in value and "routed" in value:
            rows.append(value)
        for key, child in value.items():
            if key == "tool_route" and isinstance(child, Mapping):
                rows.append(child)
                continue
            rows.extend(_iter_tool_route_rows(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            rows.extend(_iter_tool_route_rows(child))
    return rows


def _selected_tool_count(row: Mapping[str, object]) -> float:
    raw = row.get("selected_names")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return float(len(raw))
    return 0.0


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "CompletionToEval",
    "FunctionCallingRunContext",
    "ScorePayloadBuilder",
    "build_partial_eval_flusher",
    "build_pending_attempts",
    "clamp_function_calling_sampling",
    "attach_function_calling_context_metadata",
    "compute_function_calling_diagnostics",
    "compute_function_calling_metrics",
    "finalize_function_calling_run",
    "prepare_function_calling_run",
    "repeat_probe_entries",
]
