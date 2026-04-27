from __future__ import annotations

"""Run CoT + answer generation for judge-style math datasets."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.eval.k_values import NumericK, filter_metrics_by_k, max_generation_k
from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.metrics.free_response import (
    DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE,
    LLMJudge,
    LLMJudgeConfig,
    compute_pass_at_k,
    compute_avg_at_k,
    evaluate_free_response,
)
from src.eval.benchmark_config import resolve_benchmark_model_config, resolve_sampling_config
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.job_env import ensure_job_id
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.evaluators.free_response import (
    DEFAULT_COT_PROMPT,
    DEFAULT_FINAL_PROMPT,
    FreeResponsePipeline,
)
from src.infer.model import ModelLoadConfig


DEFAULT_PASS_K: tuple[int, ...] = ()
DEFAULT_AVG_K: tuple[NumericK, ...] = ()


def _load_env_file(path: Path) -> None:
    """Lightweight .env loader (key=value, optional quotes, ignores comments)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _count_records(path: str | Path, limit: int | None) -> int:
    loader = JsonlFreeAnswerLoader(str(path))
    count = 0
    for _ in loader:
        count += 1
        if limit and count >= limit:
            break
    return count

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV judge CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--cot-max-tokens", type=int, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, help="Clamp final answer generation length")
    parser.add_argument("--db-write-queue", type=int, default=16, help="DB completion write queue max size")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to generate for and compute (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute from generated samples (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument("--judge-model", help="LLM judge model name (env: JUDGE_MODEL / LLM_JUDGE_MODEL)")
    parser.add_argument("--judge-api-key", help="API key for judge model (env: JUDGE_API_KEY / OPENAI_API_KEY / API_KEY)")
    parser.add_argument("--judge-base-url", help="Optional base URL for judge model (env: JUDGE_BASE_URL / LLM_JUDGE_BASE_URL / API_BASE)")
    parser.add_argument(
        "--judge-max-workers",
        type=int,
        help="Max concurrent workers for LLM judge (env: JUDGE_MAX_WORKERS / LLM_JUDGE_MAX_WORKERS)",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        help="Max judge completion tokens (env: JUDGE_MAX_TOKENS / LLM_JUDGE_MAX_TOKENS)",
    )
    return parser.parse_args(argv)


def _resolve_pass_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[int, ...]:
    if args.pass_k:
        return tuple(args.pass_k)
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.pass_k is not None:
        return config.pass_k
    return DEFAULT_PASS_K


def _resolve_avg_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[NumericK, ...]:
    if args.avg_k:
        return tuple(args.avg_k)
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.avg_k is not None:
        return config.avg_k
    return DEFAULT_AVG_K


def _report_pass_k(slug: str, model_name: str, pass_k: tuple[int, ...]) -> tuple[int, ...]:
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.report_pass_k is not None:
        return config.report_pass_k
    return pass_k


def _report_avg_k(
    slug: str,
    model_name: str,
    avg_k: tuple[NumericK, ...],
) -> tuple[NumericK, ...]:
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.report_avg_k is not None:
        return config.report_avg_k
    return avg_k


def _resolve_max_samples(slug: str, model_name: str, args: argparse.Namespace) -> int | None:
    if args.max_samples is not None:
        return args.max_samples
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    return config.max_samples if config is not None else None


def _resolve_prompt_templates(slug: str, model_name: str) -> tuple[str, str]:
    cot_config = resolve_benchmark_model_config(slug, model_name, stage="cot")
    final_config = resolve_benchmark_model_config(slug, model_name, stage="final")
    cot_prompt = (
        cot_config.cot_prompt_template
        if cot_config is not None and cot_config.cot_prompt_template
        else DEFAULT_COT_PROMPT
    )
    final_prompt = (
        final_config.final_prompt_template
        if final_config is not None and final_config.final_prompt_template
        else DEFAULT_FINAL_PROMPT
    )
    return cot_prompt, final_prompt


def _resolve_judge_prompt_template(slug: str, model_name: str) -> str | None:
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is None:
        return None
    return config.judge_prompt_template


def main(argv: Sequence[str] | None = None) -> int:
    _load_env_file(Path(".env"))
    args = parse_args(argv)
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    slug = infer_dataset_slug_from_path(str(dataset_path))
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = FreeResponsePipeline(config)
    model_name = Path(args.model_path).stem
    pass_k_final = _resolve_pass_k(slug, model_name, args)
    avg_k_final = _resolve_avg_k(slug, model_name, args)
    report_pass_k = _report_pass_k(slug, model_name, pass_k_final)
    report_avg_k = _report_avg_k(slug, model_name, avg_k_final)
    sample_limit = _resolve_max_samples(slug, model_name, args)
    cot_prompt_template, final_prompt_template = _resolve_prompt_templates(slug, model_name)
    judge_prompt_template = _resolve_judge_prompt_template(slug, model_name)

    cot_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="cot",
        fallback_templates="free_response_cot_default",
    )
    final_sampling = resolve_sampling_config(
        slug,
        model_name,
        stage="final",
        fallback_templates="free_response_final_default",
    )
    if cot_sampling is None or final_sampling is None:
        raise ValueError(f"缺少采样配置: {slug} ({model_name})")
    cot_sampling = cot_sampling.clamp(args.cot_max_tokens)
    final_sampling = final_sampling.clamp(args.final_max_tokens)
    generate_pass_k = (1,) if args.probe_only else pass_k_final
    samples_per_task = max(max_generation_k(pass_k_final), max_generation_k(avg_k_final), 1)
    init_orm(DEFAULT_DB_CONFIG)

    service = EvalDbService()
    expected_count = service.expected_completion_count(
        dataset=str(slug),
        sample_limit=sample_limit,
        repeats_per_problem=samples_per_task,
    )
    if expected_count is None:
        expected_count = _count_records(dataset_path, sample_limit) * samples_per_task
    force_new_task = os.environ.get("RWKV_SCHEDULER_OVERWRITE") == "1"

    # 三层级联检索：一次查询获取所有续跑信息
    ctx = service.get_resume_context(
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        force_new_task=force_new_task,
    )
    sampling_payload = {
        "cot": sampling_config_to_dict(cot_sampling),
        "final": sampling_config_to_dict(final_sampling),
    }
    task_id = service.create_task_from_context(
        ctx=ctx,
        job_name="eval_free_response_judge",
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config=sampling_payload,
    )
    skip_keys = ctx.completed_keys

    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id

    if args.probe_only:
        batch_size = max(1, args.batch_size)
        _ = pipeline.run(
            dataset_path=str(dataset_path),
            cot_prompt_template=cot_prompt_template,
            final_answer_template=final_prompt_template,
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=batch_size,
            pad_to_batch=True,
            pass_k=(1,),
            samples_per_task=1,
            probe_only=True,
        )
        print(f"🧪 probe-only run completed: {batch_size} sample(s) evaluated with batch {args.batch_size}.")
        return 0

    # Evaluate from canonical completions (no reference/metrics fields inside).
    judge_model = (
        args.judge_model
        or os.environ.get("JUDGE_MODEL")
        or os.environ.get("LLM_JUDGE_MODEL")
    )
    judge_api_key = (
        args.judge_api_key
        or os.environ.get("JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("API_KEY")
    )
    judge_base_url = (
        args.judge_base_url
        or os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("LLM_JUDGE_BASE_URL")
        or os.environ.get("API_BASE")
    )
    judge_max_workers = (
        args.judge_max_workers
        or int(
            os.environ.get("JUDGE_MAX_WORKERS")
            or os.environ.get("LLM_JUDGE_MAX_WORKERS")
            or "16"
        )
    )
    judge_max_tokens = (
        args.judge_max_tokens
        or int(
            os.environ.get("JUDGE_MAX_TOKENS")
            or os.environ.get("LLM_JUDGE_MAX_TOKENS")
            or "16"
        )
    )

    if not judge_model or not judge_api_key:
        raise ValueError(
            "free_response_judge 需要有效的 judge 配置："
            "请提供 --judge-model/--judge-api-key，或设置 JUDGE_MODEL + JUDGE_API_KEY。"
        )
    judge = LLMJudge(
        LLMJudgeConfig(
            api_key=judge_api_key,
            model=judge_model,
            base_url=judge_base_url,
            max_workers=judge_max_workers,
            prompt_template=judge_prompt_template or DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE,
            max_completion_tokens=judge_max_tokens,
        )
    )

    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
    try:
        result = pipeline.run(
            dataset_path=str(dataset_path),
            cot_prompt_template=cot_prompt_template,
            final_answer_template=final_prompt_template,
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=max(1, args.batch_size),
            sample_limit=sample_limit,
            pass_k=generate_pass_k,
            samples_per_task=samples_per_task,
            skip_keys=skip_keys,
            on_record=writer.enqueue,
        )
    except BaseException:
        try:
            writer.close()
        finally:
            actual = service.count_completions(task_id=task_id, status="answer")
            status = "completed" if actual == expected_count else "failed"
            service.update_task_status(task_id=task_id, status=status)
            session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
            if session_task_id:
                try:
                    service.update_task_session_status(task_id=session_task_id, session_status="failed")
                except Exception:
                    pass
        raise
    writer.close()
    completions_payloads = service.list_completion_payloads(
        task_id=task_id,
        status="answer",
    )
    evaluation = evaluate_free_response(
        completions_payloads,
        dataset_path=str(dataset_path),
        judge=judge,
    )
    judge_stats = judge.last_run_stats
    if judge_stats is not None and judge_stats.total > 0:
        if judge_stats.error_count:
            print(
                "⚠️ LLM judge 存在异常样本："
                f"{judge_stats.error_count}/{judge_stats.total} "
                f"(invalid_output={judge_stats.invalid_output_count}, "
                f"request_error={judge_stats.request_error_count})"
            )
            if judge_stats.request_error_examples:
                print("⚠️ request_error 示例：")
                for item in judge_stats.request_error_examples:
                    print(f"  - {item}")
            if judge_stats.invalid_output_examples:
                print("⚠️ invalid_output 示例：")
                for item in judge_stats.invalid_output_examples:
                    print(f"  - {item}")
        if judge_stats.parsed_count == 0:
            service.update_task_status(task_id=task_id, status="failed")
            session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
            if session_task_id:
                try:
                    service.update_task_session_status(task_id=session_task_id, session_status="failed")
                except Exception:
                    pass
            raise RuntimeError(
                "LLM judge 未成功解析任何样本，拒绝写入可能由 judge 故障导致的全 0 分数。"
            )
    pass_metrics_all = compute_pass_at_k(evaluation.rows, pass_k_final)
    avg_metrics_all = compute_avg_at_k(evaluation.rows, avg_k_final)
    if evaluation.judge_accuracy is None:
        raise RuntimeError("LLM judge 未返回有效 judge_accuracy，无法写入 judge-only 分数。")
    metrics_payload = {"judge_accuracy": evaluation.judge_accuracy}
    pass_payload = filter_metrics_by_k(pass_metrics_all, report_pass_k, "pass@")
    if report_pass_k and not pass_payload:
        pass_payload = pass_metrics_all or {}
    if pass_payload:
        metrics_payload.update(pass_payload)
    avg_payload = filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@")
    if report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    if avg_payload:
        metrics_payload.update(avg_payload)
    task_details: dict[str, object] = {}
    if judge_stats is not None:
        task_details["judge_stats"] = judge_stats.as_dict()
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all
    service.ingest_eval_payloads(
        payloads=evaluation.payloads,
        task_id=task_id,
    )
    score_payload = make_score_payload(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics=metrics_payload,
        samples=evaluation.samples,
        problems=result.problem_count,
        task="free_response_judge",
        task_details=task_details,
    )
    service.record_score_payload(
        payload=score_payload,
        task_id=task_id,
    )
    # Update session status on success
    session_task_id = os.environ.get("RWKV_SESSION_TASK_ID")
    if session_task_id:
        try:
            service.update_task_session_status(task_id=session_task_id, session_status="completed")
        except Exception:
            pass
    export_version_results(
        service,
        task_id=task_id,
    )
    print(f"✅ judge CoT done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
