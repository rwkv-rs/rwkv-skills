from __future__ import annotations

"""Run CoT + answer generation for judge-style math datasets."""

import argparse
import os
from pathlib import Path
from typing import Sequence

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.metrics.free_response import (
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
from src.db.database import DatabaseManager
from src.db.eval_db_service import EvalDbService
from src.db.async_writer import CompletionWriteWorker
from src.db.export_results import export_version_results
from src.eval.evaluators.free_response import FreeResponsePipeline
from src.infer.model import ModelLoadConfig


DEFAULT_PASS_K = (1,)
DEFAULT_AVG_K: tuple[int, ...] = ()


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
    parser.add_argument("--db-write-batch", type=int, default=1, help="DB completion write batch size")
    parser.add_argument("--db-write-queue", type=int, default=1, help="DB completion write queue max size")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to generate for and compute (default: 1; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument(
        "--avg-k",
        type=int,
        action="append",
        help="avg@k values to compute from generated samples (default: none; can be set in configs/<benchmark>.toml)",
    )
    parser.add_argument("--judge-model", help="LLM judge model name (env: JUDGE_MODEL / LLM_JUDGE_MODEL)")
    parser.add_argument("--judge-api-key", help="API key for judge model (env: JUDGE_API_KEY / OPENAI_API_KEY / API_KEY)")
    parser.add_argument("--judge-base-url", help="Optional base URL for judge model (env: JUDGE_BASE_URL / LLM_JUDGE_BASE_URL / API_BASE)")
    parser.add_argument("--judge-max-workers", type=int, default=32, help="Max concurrent workers for LLM judge")
    return parser.parse_args(argv)


def _max_k(values) -> int:
    return max(values) if values else 0


def _resolve_pass_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[int, ...]:
    if args.pass_k:
        return tuple(args.pass_k)
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.pass_k is not None:
        return config.pass_k
    return DEFAULT_PASS_K


def _resolve_avg_k(slug: str, model_name: str, args: argparse.Namespace) -> tuple[int, ...]:
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


def _report_avg_k(slug: str, model_name: str, avg_k: tuple[int, ...]) -> tuple[int, ...]:
    config = resolve_benchmark_model_config(slug, model_name, stage=None)
    if config is not None and config.report_avg_k is not None:
        return config.report_avg_k
    return avg_k


def _filter_metrics_by_k(metric_map, ks: tuple[int, ...], prefix: str) -> dict[str, float]:
    if not metric_map or not ks:
        return {}
    allowed = {int(k) for k in ks if int(k) > 0}
    filtered: dict[str, float] = {}
    for key, value in metric_map.items():
        if not key.startswith(prefix):
            continue
        suffix_text = key[len(prefix) :]
        if not suffix_text.isdigit():
            continue
        suffix = int(suffix_text)
        if suffix in allowed:
            filtered[key] = value
    return filtered


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
    sample_limit: int | None = args.max_samples
    generate_pass_k = (1,) if args.probe_only else pass_k_final
    samples_per_task = max(_max_k(pass_k_final), _max_k(avg_k_final), 1)
    expected_count = _count_records(dataset_path, args.max_samples) * samples_per_task
    if not DEFAULT_DB_CONFIG.enabled:
        raise RuntimeError("DB 未启用：当前仅支持数据库写入模式。")
    db = DatabaseManager.instance()
    db.initialize(DEFAULT_DB_CONFIG)
    service = EvalDbService(db)
    allow_resume = service.should_allow_resume(
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        is_cot=True,
    )
    sampling_payload = {
        "cot": sampling_config_to_dict(cot_sampling),
        "final": sampling_config_to_dict(final_sampling),
    }
    task_id = service.get_or_create_task(
        job_name="eval_free_response_judge",
        job_id=ensure_job_id("free_response_judge"),
        dataset=str(slug),
        model=Path(args.model_path).stem,
        is_param_search=False,
        sampling_config=sampling_payload,
        allow_resume=allow_resume,
    )
    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id
    skip_keys = service.list_completion_keys(
        task_id=task_id,
    )

    if args.probe_only:
        batch_size = max(1, args.batch_size)
        _ = pipeline.run(
            dataset_path=str(dataset_path),
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

    judge: LLMJudge | None = None
    if judge_model and judge_api_key:
        judge = LLMJudge(
            LLMJudgeConfig(
                api_key=judge_api_key,
                model=judge_model,
                base_url=judge_base_url,
                max_workers=args.judge_max_workers,
            )
        )

    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        batch_size=args.db_write_batch,
        max_queue=args.db_write_queue,
    )
    try:
        result = pipeline.run(
            dataset_path=str(dataset_path),
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
            actual = service.count_completions(task_id=task_id)
            status = "completed" if actual == expected_count else "failed"
            service.update_task_status(task_id=task_id, status=status)
        raise
    writer.close()
    completions_payloads = service.list_completion_payloads(
        task_id=task_id,
    )
    evaluation = evaluate_free_response(
        completions_payloads,
        dataset_path=str(dataset_path),
        judge=judge,
    )
    pass_metrics_all = compute_pass_at_k(evaluation.rows, pass_k_final)
    avg_metrics_all = compute_avg_at_k(evaluation.rows, avg_k_final)
    metrics_payload = {
        "exact_accuracy": evaluation.exact_accuracy,
        "judge_accuracy": evaluation.judge_accuracy,
    }
    pass_payload = _filter_metrics_by_k(pass_metrics_all, report_pass_k, "pass@")
    if report_pass_k and not pass_payload:
        pass_payload = pass_metrics_all or {}
    if pass_payload:
        metrics_payload.update(pass_payload)
    avg_payload = _filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@")
    if report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    if avg_payload:
        metrics_payload.update(avg_payload)
    task_details: dict[str, object] = {}
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
    export_version_results(
        service,
        task_id=task_id,
    )
    print(f"✅ judge CoT done: {result.sample_count} samples")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
