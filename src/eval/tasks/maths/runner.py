"""Field-oriented maths runner aligned with rwkv-rs maths datasets."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Mapping
from typing import Sequence

from src.eval.benchmark_registry import CoTMode
from src.eval.common.field_runner import (
    add_common_runner_args,
    resolve_prompt_profile,
    scoring_stage,
)
from src.eval.field_common import (
    build_plan_task_details,
    build_task_sampling_config,
    resolve_configured_k_plan,
    set_task_env,
)
from src.eval.tasks.maths.common import (
    JudgeMode,
    build_llm_judge,
    count_free_answer_records,
    default_db_drain_every,
    default_db_write_queue,
    default_job_name,
    resolve_generation_sampling,
    resolve_sampling_pair,
)

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext, TaskSpec
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    resolve_backend_model_name,
    validate_inference_backend_args,
)


@dataclass(frozen=True, slots=True)
class MathStageConfig:
    cot_prompt_template: str
    final_answer_template: str | None
    cot_sampling: Any
    final_sampling: Any | None
    prompt_max_chars: int | None = None


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _judge_error_summaries(judge_stats_by_group: Mapping[str, Mapping[str, object]]) -> list[str]:
    summaries: list[str] = []
    for group, stats in sorted(judge_stats_by_group.items()):
        total = _safe_int(stats.get("total"))
        invalid = _safe_int(stats.get("invalid_output_count"))
        request = _safe_int(stats.get("request_error_count"))
        errors = invalid + request
        if errors:
            summaries.append(
                f"{group} {errors}/{total} "
                f"(invalid_output={invalid}, request_error={request})"
            )
    return summaries


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV maths benchmark runner")
    add_common_runner_args(parser, batch_size_default=64, db_write_queue_default=None)
    add_inference_backend_arguments(parser)
    parser.add_argument("--max-tokens", type=int, help="Clamp full-response generation length")
    parser.add_argument("--cot-max-tokens", type=int, help="Compatibility alias for --max-tokens")
    parser.add_argument("--final-max-tokens", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--db-drain-every",
        type=int,
        help="Force DB writer to drain every N completion payloads (0 disables)",
    )
    parser.add_argument(
        "--db-close-timeout-s",
        type=float,
        default=30.0,
        help="Max seconds to wait for DB writer drain/close on shutdown",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run a single-sample probe and skip scoring",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to generate for and compute (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--judge-mode",
        choices=[mode.value for mode in JudgeMode],
        default=JudgeMode.EXACT.value,
        help="Maths verdict mode",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=("normal", "naive"),
        help="Prompt profile: normal uses benchmark configs; naive uses only the problem plus final-answer prefill.",
    )
    parser.add_argument(
        "--single-generation",
        action="store_true",
        help="Compatibility alias for --strategy-a-single-generation.",
    )
    parser.add_argument(
        "--strategy-a-single-generation",
        action="store_true",
        help="Generate strategy A as one full response while scoring B/C from the two-stage path.",
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        help="Override the main generation temperature.",
    )
    parser.add_argument(
        "--generation-repetition-penalty",
        type=float,
        help="Override the main generation repetition penalty/alpha_frequency.",
    )
    parser.add_argument(
        "--generation-penalty-decay",
        type=float,
        help="Override the main generation penalty decay/alpha_decay.",
    )
    parser.add_argument("--judge-model", help="LLM judge model name (env: JUDGE_MODEL)")
    parser.add_argument("--judge-api-key", help="API key for judge model (env: JUDGE_API_KEY)")
    parser.add_argument(
        "--judge-base-url",
        help="Optional base URL for judge model (env: JUDGE_BASE_URL)",
    )
    parser.add_argument("--judge-max-workers", type=int, help="Max concurrent workers for LLM judge")
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        help="Max judge completion tokens. Defaults to not passing max_tokens.",
    )
    parser.add_argument(
        "--run-checker",
        action="store_true",
        help="Run the optional wrong-answer checker before writing scores.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Only score the primary strategy group before writing scores.",
    )
    return parser.parse_args(argv)


def _env_enabled(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _should_run_checker(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "run_checker", False)):
        return True
    return _env_enabled("RWKV_MATH_RUN_CHECKER")


def _should_score_primary_only(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "primary_only", False)):
        return True
    return _env_enabled("RWKV_MATH_PRIMARY_ONLY")


def _apply_generation_sampling_overrides(
    stage_config: MathStageConfig,
    args: argparse.Namespace,
) -> MathStageConfig:
    from dataclasses import replace

    overrides: dict[str, float] = {}
    if args.generation_temperature is not None:
        overrides["temperature"] = float(args.generation_temperature)
    if args.generation_repetition_penalty is not None:
        overrides["alpha_frequency"] = float(args.generation_repetition_penalty)
    if args.generation_penalty_decay is not None:
        overrides["alpha_decay"] = float(args.generation_penalty_decay)
    if not overrides:
        return stage_config
    return replace(stage_config, cot_sampling=replace(stage_config.cot_sampling, **overrides))


def _math_sampling_payload(
    stage_config: MathStageConfig,
    *,
    strategy_a_config: MathStageConfig | None = None,
) -> dict[str, object]:
    from src.eval.results.schema import sampling_config_to_dict

    payload: dict[str, object] = {
        "stage1": sampling_config_to_dict(stage_config.cot_sampling),
    }
    if stage_config.final_sampling is None:
        raise ValueError("final_sampling is required for two-stage math generation")
    payload["stage2"] = sampling_config_to_dict(stage_config.final_sampling)
    if strategy_a_config is not None:
        payload["strategy_a"] = sampling_config_to_dict(strategy_a_config.cot_sampling)
    return payload


def _close_writer_and_mark_failed(
    service,
    writer,
    *,
    task_id: str,
    expected_count: int,
    close_timeout_s: float,
    interrupted: bool,
) -> None:
    try:
        writer.close(timeout_s=close_timeout_s)
    finally:
        actual = service.count_completions(task_id=task_id, status="Completed")
        status = "failed" if interrupted else ("completed" if actual == expected_count else "failed")
        service.update_task_status(task_id=task_id, status=status)


def main(
    argv: Sequence[str] | None = None,
    *,
    run_context: "RunContext | None" = None,
    task_spec: "TaskSpec | None" = None,
) -> int:
    del task_spec
    args = parse_args(argv)
    validate_inference_backend_args(args)

    from src.eval.benchmark_config import resolve_benchmark_model_config
    from src.eval.env_config import load_env_file
    from src.eval.evaluating import TaskRunController, TaskRunSignalGuard, TaskRunState, prepare_task_execution
    from src.eval.execution_plan import build_attempt_keys, plan_attempt_count
    from src.eval.tasks.maths.pipeline import FreeResponsePipeline
    from src.eval.metrics.free_response import (
        attach_strategy_task_ids,
        build_grouped_metrics_payload,
        evaluate_free_response,
    )
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService

    load_env_file(Path(".env"))
    judge_mode = JudgeMode(args.judge_mode)
    job_name = run_context.job_name if run_context is not None else os.environ.get("RWKV_SKILLS_JOB_NAME", default_job_name(judge_mode))
    prompt_profile = resolve_prompt_profile(args.prompt_profile, job_name)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    model_name = resolve_backend_model_name(args)
    total_records = count_free_answer_records(dataset_path, None)
    k_plan = resolve_configured_k_plan(
        slug=slug,
        model_name=model_name,
        dataset_len=total_records,
        args=args,
    )
    plan = k_plan.plan
    attempt_keys = build_attempt_keys(plan, max_pass_k=1)
    backend = build_inference_backend_from_args(args)
    pipeline = FreeResponsePipeline(backend)

    stage_config = _resolve_math_stage_config(
        slug,
        model_name,
        cot_max_tokens=args.max_tokens or args.cot_max_tokens,
        final_max_tokens=args.final_max_tokens,
        prompt_profile=prompt_profile,
    )
    strategy_a_single_generation = bool(
        args.strategy_a_single_generation or args.single_generation
    )
    strategy_a_config = None
    if strategy_a_single_generation:
        strategy_a_config = _resolve_single_generation_stage_config(
            slug,
            model_name,
            max_tokens=args.max_tokens or args.cot_max_tokens,
            prompt_profile=prompt_profile,
        )
        strategy_a_config = _apply_generation_sampling_overrides(strategy_a_config, args)
    else:
        stage_config = _apply_generation_sampling_overrides(stage_config, args)
    batch_size = max(1, args.batch_size)
    judge = None
    if judge_mode is JudgeMode.LLM:
        judge = build_llm_judge(
            judge_model=args.judge_model,
            judge_api_key=args.judge_api_key,
            judge_base_url=args.judge_base_url,
            judge_max_workers=args.judge_max_workers,
            judge_max_tokens=args.judge_max_tokens,
            required=True,
        )
    root_config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if judge is not None and root_config is not None and root_config.judge_prompt_template:
        judge.config.prompt_template = root_config.judge_prompt_template

    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    task_state = prepare_task_execution(
        service=service,
        dataset=str(slug),
        model=model_name,
        is_param_search=False,
        job_name=job_name,
        run_mode=(run_context.run_mode if run_context is not None else None),
        sampling_config=build_task_sampling_config(
            cot_mode=CoTMode.COT,
            avg_k=plan.avg_k,
            sampling_config=_math_sampling_payload(
                stage_config,
                strategy_a_config=strategy_a_config,
            ),
            effective_sample_count=plan.effective_sample_count,
            pass_ks=k_plan.pass_k,
            judger_model_name=(judge.config.model if judge is not None else None),
            prompt_profile=prompt_profile,
        ),
    )
    expected_count = plan_attempt_count(plan, max_pass_k=1)
    task_run = TaskRunState.from_task_execution(
        execution_state=task_state,
        attempt_keys=attempt_keys,
        expected_attempt_count=expected_count,
    )
    runtime = TaskRunController(service=service, state=task_run)
    task_id = task_run.task_id
    skip_keys = task_state.skip_keys
    set_task_env(task_id)

    if args.probe_only:
        pipeline.run(
            dataset_path=str(dataset_path),
            cot_prompt_template=stage_config.cot_prompt_template,
            cot_sampling=stage_config.cot_sampling,
            strategy_a_prompt_template=(
                strategy_a_config.cot_prompt_template if strategy_a_config is not None else None
            ),
            strategy_a_sampling=(
                strategy_a_config.cot_sampling if strategy_a_config is not None else None
            ),
            strategy_a_filter_correct=strategy_a_config is not None,
            final_answer_template=stage_config.final_answer_template,
            final_sampling=stage_config.final_sampling,
            batch_size=batch_size,
            sample_limit=batch_size,
            pad_to_batch=True,
            pass_k=(1,),
            samples_per_task=1,
            probe_only=True,
            prompt_max_chars=stage_config.prompt_max_chars,
        )
        print(f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}.")
        return 0

    db_write_queue = (
        args.db_write_queue
        if args.db_write_queue is not None
        else default_db_write_queue(judge_mode)
    )
    db_drain_every = (
        args.db_drain_every
        if args.db_drain_every is not None
        else default_db_drain_every(judge_mode)
    )
    close_timeout_s = max(0.0, float(args.db_close_timeout_s))
    writer = runtime.create_writer(max_queue=db_write_queue, drain_every=db_drain_every)
    with TaskRunSignalGuard(
        controller=runtime,
        writer=writer,
        close_timeout_s=close_timeout_s,
        on_interrupt=lambda signame: print(f"⚠️ Received {signame}; draining completion writer before exit..."),
    ):
        try:
            result = pipeline.run(
                dataset_path=str(dataset_path),
                cot_prompt_template=stage_config.cot_prompt_template,
                cot_sampling=stage_config.cot_sampling,
                strategy_a_prompt_template=(
                    strategy_a_config.cot_prompt_template if strategy_a_config is not None else None
                ),
                strategy_a_sampling=(
                    strategy_a_config.cot_sampling if strategy_a_config is not None else None
                ),
                strategy_a_filter_correct=strategy_a_config is not None,
                final_answer_template=stage_config.final_answer_template,
                final_sampling=stage_config.final_sampling,
                batch_size=batch_size,
                record_indices=plan.sample_indices,
                pad_to_batch=False,
                pass_k=k_plan.pass_k,
                samples_per_task=max(plan.repeat_count, 1),
                attempt_keys=attempt_keys,
                skip_keys=skip_keys,
                on_record=writer.enqueue,
                prompt_max_chars=stage_config.prompt_max_chars,
            )
        except Exception:
            runtime.handle_attempt_stage_failure(writer, timeout_s=close_timeout_s)
            raise

    completions_payloads = runtime.complete_attempt_stage(writer, timeout_s=close_timeout_s)
    with scoring_stage(runtime):
        evaluation = evaluate_free_response(
            completions_payloads,
            dataset_path=str(dataset_path),
            judge=judge,
            primary_only=_should_score_primary_only(args),
        )
        if judge_mode is JudgeMode.LLM and evaluation.judge_accuracy is None:
            raise RuntimeError("LLM judge 未返回有效 judge_accuracy，无法写入 judge-only 分数。")
        if judge is not None:
            judge_errors = _judge_error_summaries(evaluation.judge_stats_by_group)
            for summary in judge_errors:
                print(f"⚠️ LLM judge 存在异常样本：{summary}")
            if judge_mode is JudgeMode.LLM and judge_errors:
                print(
                    "⚠️ LLM judge 异常样本已按 judge_false 计入，并写入 judge_stats: "
                    + "; ".join(judge_errors)
                )

        strategy_task_ids = service.ingest_eval_payload_groups(
            task_id=task_id,
            completion_payloads=completions_payloads,
            payloads_by_group=evaluation.payloads_by_group,
            primary_group=evaluation.primary_group,
        )
        metrics_payload, metric_details = build_grouped_metrics_payload(
            evaluation,
            pass_k=k_plan.pass_k,
            avg_k=k_plan.avg_k,
            report_pass_k=k_plan.report_pass_k,
            report_avg_k=k_plan.report_avg_k,
        )
        attach_strategy_task_ids(metrics_payload, strategy_task_ids)

        task_details: dict[str, object] = build_plan_task_details(
            plan,
            cot_mode=CoTMode.COT.value,
            prompt_profile=prompt_profile,
        )
        task_details.update(metric_details)
        runtime.state.task_results = {
            (int(payload["sample_index"]), int(payload["repeat_index"]), int(payload.get("pass_index", 0))): bool(
                payload.get("is_passed", False)
            )
            for payload in evaluation.payloads
        }
        runtime.state.eval_count = len(runtime.state.task_results)
        if _should_run_checker(args):
            runtime.run_checker(model_name=model_name)
        score_payload = make_score_payload(
            slug,
            is_cot=True,
            model_name=model_name,
            metrics=metrics_payload,
            samples=evaluation.samples,
            problems=result.problem_count,
            task=job_name,
            task_details=task_details,
            extra={
                "cot_mode": CoTMode.COT.value,
                "prompt_profile": prompt_profile,
                "strategy_a_single_generation": strategy_a_single_generation,
            },
        )
        runtime.record_score(score_payload)
    if judge_mode is JudgeMode.LLM:
        print(f"✅ judge CoT done: {result.sample_count} samples")
    else:
        print(f"✅ CoT free-form done: {result.sample_count} samples")
    return 0


def _require_math_prompt_template(
    slug: str,
    model_name: str,
    *,
    stage: str,
    template: str | None,
) -> str:
    if template:
        return template
    expected_key = "cot_prompt_template" if stage == "cot" else "final_prompt_template"
    raise ValueError(
        f"math benchmark {slug!r} ({model_name}) requires configs/<benchmark>.toml "
        f"[{stage}] {expected_key}"
    )


def _resolve_single_generation_stage_config(
    slug: str,
    model_name: str,
    *,
    max_tokens: int | None = None,
    prompt_profile: str = "normal",
) -> MathStageConfig:
    from src.eval.benchmark_config import resolve_benchmark_model_config
    from src.eval.tasks.maths.pipeline import DEFAULT_COT_PROMPT

    cot_sampling = resolve_generation_sampling(
        slug,
        model_name,
        max_tokens=max_tokens,
    )
    root_config = resolve_benchmark_model_config(slug, model_name, stage=None)
    prompt_max_chars = getattr(root_config, "prompt_max_chars", None)
    if prompt_profile == "naive":
        return MathStageConfig(
            cot_prompt_template=DEFAULT_COT_PROMPT,
            final_answer_template=None,
            cot_sampling=cot_sampling,
            final_sampling=None,
            prompt_max_chars=prompt_max_chars,
        )
    cot_config = resolve_benchmark_model_config(slug, model_name, stage="cot")
    return MathStageConfig(
        cot_prompt_template=_require_math_prompt_template(
            slug,
            model_name,
            stage="cot",
            template=getattr(cot_config, "cot_prompt_template", None),
        ),
        final_answer_template=None,
        cot_sampling=cot_sampling,
        final_sampling=None,
        prompt_max_chars=prompt_max_chars,
    )


def _resolve_math_stage_config(
    slug: str,
    model_name: str,
    *,
    cot_max_tokens: int | None = None,
    final_max_tokens: int | None = None,
    prompt_profile: str = "normal",
) -> MathStageConfig:
    from src.eval.benchmark_config import resolve_benchmark_model_config
    from src.eval.tasks.maths.pipeline import DEFAULT_COT_PROMPT, DEFAULT_FINAL_PROMPT

    cot_sampling, final_sampling = resolve_sampling_pair(
        slug,
        model_name,
        cot_max_tokens=cot_max_tokens,
        final_max_tokens=final_max_tokens,
    )
    root_config = resolve_benchmark_model_config(slug, model_name, stage=None)
    prompt_max_chars = getattr(root_config, "prompt_max_chars", None)
    if prompt_profile == "naive":
        return MathStageConfig(
            cot_prompt_template=DEFAULT_COT_PROMPT,
            final_answer_template=DEFAULT_FINAL_PROMPT,
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            prompt_max_chars=prompt_max_chars,
        )
    cot_config = resolve_benchmark_model_config(slug, model_name, stage="cot")
    final_config = resolve_benchmark_model_config(slug, model_name, stage="final")
    return MathStageConfig(
        cot_prompt_template=_require_math_prompt_template(
            slug,
            model_name,
            stage="cot",
            template=getattr(cot_config, "cot_prompt_template", None),
        ),
        final_answer_template=_require_math_prompt_template(
            slug,
            model_name,
            stage="final",
            template=getattr(final_config, "final_prompt_template", None),
        ),
        cot_sampling=cot_sampling,
        final_sampling=final_sampling,
        prompt_max_chars=prompt_max_chars,
    )


__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
