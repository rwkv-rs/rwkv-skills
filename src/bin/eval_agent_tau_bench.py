from __future__ import annotations

"""Run tau-bench v1 agent evaluation with RWKV local inference."""

import argparse
import os
import signal
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.db.async_writer import CompletionWriteWorker
from src.db.database import init_db
from src.db.eval_db_service import EvalDbService
from src.eval.agent_bench.chat_bridge import RWKVChatBridge
from src.eval.agent_bench.deps import ensure_tau_v1_runtime_dependencies
from src.eval.evaluating import prepare_task_execution, run_checker_for_task
from src.eval.env_config import (
    apply_openai_env,
    load_env_file,
    resolve_required_user_model_config,
)
from src.eval.agent_bench.envs import TauV1Env
from src.eval.agent_bench.metrics import compute_agent_metrics
from src.eval.agent_bench.payloads import completion_to_eval_payload, episode_to_completion_payload
from src.eval.agent_bench.runtime import run_tau_v1_episodes
from src.eval.agent_bench.tasks import infer_domain_from_slug, load_manifest
from src.eval.benchmark_config import resolve_sampling_config
from src.eval.results.payloads import make_score_payload
from src.eval.results.schema import sampling_config_to_dict
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, split_benchmark_and_split
from src.infer.backend import (
    add_inference_backend_arguments,
    build_inference_backend_from_args,
    resolve_backend_model_name,
    validate_inference_backend_args,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV tau-bench v1 evaluator")
    parser.epilog = "Requires .env with API_KEY (or OPENAI_API_KEY) and model_name (or MODEL_NAME)."
    parser.add_argument("--dataset", required=True, help="tau-bench manifest JSONL path")
    add_inference_backend_arguments(parser)
    parser.add_argument(
        "--user-strategy",
        default="llm",
        choices=("llm", "human"),
        help="User simulator strategy",
    )
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum agent turns per episode")
    parser.add_argument("--batch-size", type=int, default=1, help="Max concurrent agent episodes / RWKV batch size")
    parser.add_argument("--probe-only", action="store_true", help="Run a minimal batch-sizing probe")
    parser.add_argument("--max-samples", type=int, help="Limit number of tasks for quick runs")
    parser.add_argument("--num-trials", type=int, default=1, help="Number of repeats per task")
    parser.add_argument("--pass-k", type=int, action="append", help="pass@k values to report")
    parser.add_argument("--max-tokens", type=int, help="Clamp generation length")
    parser.add_argument("--temperature", type=float, help="Override temperature")
    parser.add_argument("--top-k", type=int, help="Override top-k")
    parser.add_argument("--top-p", type=float, help="Override top-p")
    parser.add_argument("--db-write-queue", type=int, default=1024, help="DB completion write queue max size")
    return parser.parse_args(argv)


def _resolve_sampling(args: argparse.Namespace, *, dataset_slug: str, model_name: str):
    sampling = resolve_sampling_config(
        dataset_slug,
        model_name,
        stage="final",
        fallback_templates="instruction_following_default",
    )
    if sampling is None:
        raise ValueError(f"missing sampling config for dataset={dataset_slug}, model={model_name}")
    if args.max_tokens:
        sampling = sampling.clamp(args.max_tokens)
    if args.temperature is not None:
        sampling = replace(sampling, temperature=float(args.temperature))
    if args.top_k is not None:
        sampling = replace(sampling, top_k=int(args.top_k))
    if args.top_p is not None:
        sampling = replace(sampling, top_p=float(args.top_p))
    return sampling


def _normalize_pass_k(raw: Sequence[int] | None, *, num_trials: int) -> tuple[int, ...]:
    if raw:
        values = sorted({int(item) for item in raw if int(item) > 0 and int(item) <= num_trials})
        if values:
            return tuple(values)
    return tuple(range(1, max(1, int(num_trials)) + 1))


def _ingest_current_eval_payloads(
    service: EvalDbService,
    *,
    task_id: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completions_payloads = service.list_completion_payloads(task_id=task_id, status="Completed")
    eval_payloads = [completion_to_eval_payload(item) for item in completions_payloads]
    service.ingest_eval_payloads(payloads=eval_payloads, task_id=task_id)
    return completions_payloads, eval_payloads


def main(argv: Sequence[str] | None = None) -> int:
    load_env_file(Path(".env"))
    args = parse_args(argv)
    validate_inference_backend_args(args)
    ensure_tau_v1_runtime_dependencies()
    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False)
    dataset_slug = infer_dataset_slug_from_path(str(dataset_path))
    domain = infer_domain_from_slug(dataset_slug)
    benchmark_name, dataset_split = split_benchmark_and_split(dataset_slug)
    num_trials = max(1, int(args.num_trials))
    batch_size = max(1, int(args.batch_size))
    pass_k = _normalize_pass_k(args.pass_k, num_trials=num_trials)
    if args.user_strategy == "human" and batch_size > 1:
        raise ValueError("human user_strategy 不支持 batch_size > 1")
    effective_max_samples = args.max_samples
    if args.probe_only and (effective_max_samples is None or effective_max_samples <= 0 or effective_max_samples > batch_size):
        effective_max_samples = batch_size

    manifest = load_manifest(dataset_path, max_samples=effective_max_samples)
    expected_count = len(manifest) * num_trials

    model_name = resolve_backend_model_name(args)
    sampling = _resolve_sampling(args, dataset_slug=dataset_slug, model_name=model_name)
    user_model = resolve_required_user_model_config()
    apply_openai_env(user_model)

    backend = build_inference_backend_from_args(args)
    bridge = RWKVChatBridge(engine=backend, default_sampling=sampling)

    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    sampling_payload = {
        "stage1": sampling_config_to_dict(sampling),
    }
    task_state = prepare_task_execution(
        service=service,
        dataset=str(dataset_slug),
        model=model_name,
        is_param_search=False,
        job_name="eval_agent_tau_bench",
        sampling_config=sampling_payload,
    )
    task_id = task_state.task_id
    skip_keys = task_state.skip_keys

    os.environ["RWKV_SKILLS_TASK_ID"] = task_id
    os.environ["RWKV_SKILLS_VERSION_ID"] = task_id

    writer = CompletionWriteWorker(
        service=service,
        task_id=task_id,
        max_queue=args.db_write_queue,
    )
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
            print(f"⚠️ Failed to ingest partial tau-bench eval rows during {signame}: {exc}")

    def _handle_termination(signum: int, _frame: object) -> None:
        if should_exit["active"]:
            raise SystemExit(128 + signum)
        should_exit["active"] = True
        signame = signal.Signals(signum).name
        print(f"⚠️ Received {signame}; draining tau-bench completions before exit...")
        try:
            writer.close(timeout_s=30.0)
        except Exception as exc:
            print(f"⚠️ Failed to close completion writer during {signame}: {exc}")
        _flush_partial_eval(signame)
        try:
            service.update_task_status(task_id=task_id, status="failed")
        except Exception as exc:
            print(f"⚠️ Failed to mark task {task_id} failed after {signame}: {exc}")
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        original_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _handle_termination)

    try:
        pending_requests: list[tuple[int, int, object]] = []
        task_indices: list[int] = []
        for sample_index, record in enumerate(manifest):
            for repeat_index in range(num_trials):
                key = (sample_index, repeat_index, 0)
                if key in skip_keys:
                    continue
                pending_requests.append((sample_index, repeat_index, record))
                task_indices.append(record.index)

        def _env_factory() -> TauV1Env:
            return TauV1Env(
                domain=domain,
                user_strategy=args.user_strategy,
                user_model=user_model.model_name,
                user_provider="openai",
                task_split=dataset_split or "test",
            )

        def _enqueue_episode(request_index: int, episode) -> None:
            sample_index, repeat_index, record = pending_requests[request_index]
            payload = episode_to_completion_payload(
                episode,
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                sample_index=sample_index,
                repeat_index=repeat_index,
                sampling_config=sampling_payload,
            )
            payload["task_id"] = record.task_id
            payload["domain"] = record.domain
            payload["instruction"] = record.instruction
            writer.enqueue(payload)

        run_tau_v1_episodes(
            bridge=bridge,
            env_factory=_env_factory,
            task_indices=task_indices,
            max_steps=args.max_turns,
            max_concurrency=batch_size,
            sampling=sampling,
            on_complete=_enqueue_episode,
        )
        writer.close(timeout_s=30.0)
    except BaseException:
        try:
            writer.close(timeout_s=30.0)
        finally:
            _flush_partial_eval("exception")
            actual = service.count_completions(task_id=task_id, status="Completed")
            status = "failed" if should_exit["active"] else ("completed" if actual == expected_count else "failed")
            service.update_task_status(task_id=task_id, status=status)
        raise
    finally:
        _restore_signal_handlers()

    completions_payloads, eval_payloads = _ingest_current_eval_payloads(service, task_id=task_id)
    run_checker_for_task(service=service, task_id=task_id, model_name=model_name)

    metrics = compute_agent_metrics(
        eval_payloads,
        pass_k=pass_k,
        expected_count=expected_count,
    )
    score_payload = make_score_payload(
        dataset_slug,
        is_cot=False,
        model_name=model_name,
        metrics=metrics,
        samples=len(completions_payloads),
        problems=len(manifest),
        task="agent_tau_bench",
    )
    service.record_score_payload(payload=score_payload, task_id=task_id)
    print(f"tau-bench done: samples={len(completions_payloads)}, metrics={metrics}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
