from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

from src.eval.agent_bench.tau_official import (
    DEFAULT_TAU_PROMPT_MAX_CHARS,
    RWKVTauOfficialAgent,
    TauOfficialRuntime,
    _append_tau_message,
    _build_tau_tool_facts_message,
    _tau_messages_to_prompt_messages,
)
from src.eval.agent_bench.tasks import require_tau_v3_source
from src.eval.env_config import (
    apply_openai_env,
    load_env_file,
    resolve_judge_model_config,
    resolve_required_user_model_config,
)
from src.eval.evaluators.common import StageRecord, sample_repeat_seed
from src.eval.experiments.parallel_candidate_router import (
    ParallelCandidateRouterConfig,
    route_parallel_candidate_tool_call,
)
from src.eval.function_calling.tau_bench import TauManifestRecord, load_tau_manifest_records
from src.eval.function_calling.tau_runner import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOOL_ERRORS,
    DEFAULT_TAU_HISTORY_MAX_CHARS,
    _requires_tau_user_model,
    _requires_tau_v3_source,
    _sum_message_costs,
    _tau_completion_to_eval_payload,
    _tau_official_completion_payload,
    _tau_runtime_model_metadata,
)
from src.eval.long_doc_evidence import (
    LongDocEvidenceConfig,
    compact_long_text,
    compact_messages_for_long_context,
    infer_query_from_messages,
)
from src.eval.results.payloads import make_score_payload
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, safe_slug, split_benchmark_and_split
from src.infer.backend import RemoteInferenceBackend, RemoteInferenceConfig, normalize_api_base
from src.infer.sampling import SamplingConfig

DEFAULT_MODELS = (
    "rwkv7-g1f-2.9b-20260420-ctx8192",
    "rwkv7-g1g-13.3b-20260523-ctx8192",
    "rwkv7-g1g-2.9b-20260526-ctx8192",
)


class ParallelCandidateTauOfficialAgent(RWKVTauOfficialAgent):
    def __init__(
        self,
        *,
        candidate_config: ParallelCandidateRouterConfig,
        include_router_prompts: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._candidate_config = candidate_config
        self._include_router_prompts = bool(include_router_prompts)

    def generate_next_message(self, message: Any, state: list[Any] | None) -> tuple[Any, list[Any]]:
        step_started = time.perf_counter()
        history = list(state or [])
        if message is not None:
            _append_tau_message(history, message, MultiToolMessage=self._MultiToolMessage)

        prompt_messages = _tau_messages_to_prompt_messages(
            history,
            ToolMessage=self._ToolMessage,
            UserMessage=self._UserMessage,
        )
        prompt_seed = None if self._seed is None else int(self._seed) + self._turn_index
        route_started = time.perf_counter()
        route, route_prompt_chars = self._route_candidate_decision(prompt_messages, prompt_seed=prompt_seed)
        route_s = time.perf_counter() - route_started
        parse_started = time.perf_counter()
        parse_error: str | None = None
        recovered = False
        selected = route.selected
        if selected is None:
            parse_error = route.aggregate_error or "parallel candidate router did not select a tool call"
            self.parse_errors.append(parse_error)
            assistant_message = self._AssistantMessage(
                role="assistant",
                content="I am unable to continue safely. ###STOP###",
            )
        else:
            try:
                self._current_tool_names = set(self._tool_names)
                assistant_message = self._decision_to_assistant_message(
                    selected.name,
                    selected.arguments,
                    prompt_messages=prompt_messages,
                )
            except Exception as exc:  # noqa: BLE001 - trace and stop this sample cleanly.
                parse_error = str(exc)
                self.parse_errors.append(parse_error)
                assistant_message = self._AssistantMessage(
                    role="assistant",
                    content="I am unable to continue safely. ###STOP###",
                )
        parse_s = time.perf_counter() - parse_started
        self.stages.append(
            StageRecord(
                prompt=route.aggregate_prompt,
                completion=route.aggregate_completion,
                stop_reason=route.aggregate_finish_reason,
            )
        )
        self.step_timings.append(
            {
                "turn_index": int(self._turn_index),
                "prompt_chars": int(route_prompt_chars),
                "completion_chars": len(route.aggregate_completion),
                "prompt_build_s": 0.0,
                "generation_s": route_s,
                "candidate_route_s": route_s,
                "parse_s": parse_s,
                "total_s": time.perf_counter() - step_started,
                "finish_reason": route.aggregate_finish_reason,
                "format_prefill": "json_object_open",
                "parse_input_prefill_applied": False,
                "parse_input_chars": len(route.aggregate_completion),
                "parse_recovered": recovered,
                "parse_error": parse_error,
            }
        )
        self._turn_index += 1
        history.append(assistant_message)
        return assistant_message, history

    def _route_candidate_decision(
        self,
        prompt_messages: Sequence[Mapping[str, object]],
        *,
        prompt_seed: int | None,
    ) -> tuple[Any, int]:
        long_doc_query = infer_query_from_messages(
            prompt_messages,
            skip_longer_than=max(1, int(self._long_doc_config.min_long_text_chars)),
        )
        long_doc_seed = None if self._seed is None else int(self._seed) + 20_000 + self._turn_index
        compacted_messages = compact_messages_for_long_context(
            prompt_messages,
            query=long_doc_query,
            config=self._long_doc_config,
            engine=self._engine,
            sampling=self._sampling,
            progress_desc="TauCandidate-LongDoc",
            prompt_seed=long_doc_seed,
        ).messages
        facts_message = _build_tau_tool_facts_message(prompt_messages, max_chars=1100)
        facts_text = facts_message["content"] if facts_message is not None else None
        if facts_message is not None:
            compacted_messages = [*compacted_messages, facts_message]
        policy_result = compact_long_text(
            self._domain_policy,
            query=long_doc_query,
            config=self._long_doc_config,
            label="domain_policy",
            engine=self._engine,
            sampling=self._sampling,
            progress_desc="TauCandidate-Policy",
            prompt_seed=None if long_doc_seed is None else long_doc_seed + 5_000,
        )
        domain_policy = _prepend_candidate_policy_focus(
            domain=self._domain,
            full_policy=self._domain_policy,
            compacted_policy=policy_result.text,
            messages=prompt_messages,
        )
        route = route_parallel_candidate_tool_call(
            tools=self._tools,
            messages=compacted_messages,
            domain_policy=domain_policy,
            domain=self._domain,
            facts_text=facts_text,
            engine=self._engine,
            sampling=self._sampling,
            config=self._candidate_config,
            progress_desc="TauCandidate-Router",
            prompt_seed=None if prompt_seed is None else int(prompt_seed) + 10_000,
        )
        route_trace = route.trace_payload(include_prompts=self._include_router_prompts)
        route_trace["turn_index"] = self._turn_index
        route_trace["total_tool_count"] = len(self._tools)
        route_trace["catalog_chars"] = sum(len(str(getattr(tool, "openai_schema", tool))) for tool in self._tools)
        self.tool_routes.append(route_trace)
        prompt_chars = len(route.aggregate_prompt) + sum(len(chunk.prompt) for chunk in route.chunks)
        return route, prompt_chars


class ScriptedTauUser:
    """No-LLM smoke user for router/tool execution diagnostics, not formal scoring."""

    def __init__(self, *, task: Any) -> None:
        message_module = __import__("tau2.data_model.message", fromlist=["UserMessage"])
        base_module = __import__("tau2.user.user_simulator_base", fromlist=["UserState"])
        self._UserMessage = getattr(message_module, "UserMessage")
        self._UserState = getattr(base_module, "UserState")
        self._initial_content = _scripted_initial_user_content(task)
        self._followup_content = _scripted_followup_user_content(task)
        self._turn_index = 0

    def get_init_state(self, message_history: list[Any] | None = None) -> Any:
        return self._UserState(system_messages=[], messages=list(message_history or []))

    @classmethod
    def is_stop(cls, message: Any) -> bool:
        content = getattr(message, "content", "")
        return isinstance(content, str) and "###STOP###" in content

    def generate_next_message(self, message: Any, state: Any) -> tuple[Any, Any]:
        if self._turn_index <= 0:
            content = self._initial_content
        elif self._followup_content and _assistant_message_looks_like_refusal(message):
            content = self._followup_content
        else:
            content = "###STOP###"
        self._turn_index += 1
        user_message = self._UserMessage(role="user", content=content, cost=0.0)
        state.messages.append(user_message)
        return user_message, state

    def set_seed(self, seed: int) -> None:
        del seed

    def stop(self, message: Any | None = None, state: Any | None = None) -> None:
        del message, state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run JSON-only TAU parallel-candidate-router experiments")
    parser.add_argument("--dataset", default="tau2_bench_airline/base", help="TAU manifest path or dataset slug")
    parser.add_argument("--output-dir", help="Output directory; defaults to out/parallel_candidate_router/<timestamp>")
    parser.add_argument("--sample-offset", type=int, default=0, help="Start from this zero-based sample index")
    parser.add_argument("--max-samples", type=int, default=2, help="Samples per model")
    parser.add_argument("--max-steps", type=int, default=6, help="Official TAU max turns per sample")
    parser.add_argument("--max-tool-errors", type=int, default=DEFAULT_MAX_TOOL_ERRORS)
    parser.add_argument("--infer-base-url", default="http://127.0.0.1:19083", help="Forwarded router base URL")
    parser.add_argument("--infer-model", action="append", dest="infer_models", help="Model name; repeatable")
    parser.add_argument("--infer-api-key", default="")
    parser.add_argument("--infer-timeout-s", type=float, default=900.0)
    parser.add_argument("--infer-max-workers", type=int, default=64)
    parser.add_argument("--infer-max-retries", type=int, default=3)
    parser.add_argument(
        "--infer-protocol",
        "--infer-mode",
        choices=("openai", "vllm", "completions"),
        dest="infer_protocol",
        default="completions",
    )
    parser.add_argument("--infer-seed-policy", choices=("preserve", "omit-for-contents"), default="preserve")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--candidate-chunk-tools", type=int, default=2)
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--candidate-max-tokens", type=int, default=192)
    parser.add_argument("--aggregate-max-tokens", type=int, default=192)
    parser.add_argument("--candidate-context-chars", type=int, default=6000)
    parser.add_argument("--candidate-policy-chars", type=int, default=1200)
    parser.add_argument("--candidate-tool-schema-mode", choices=("minimal", "compact", "full"), default="compact")
    parser.add_argument("--prompt-max-chars", type=int, default=12288)
    parser.add_argument("--history-max-chars", type=int, default=8000)
    parser.add_argument("--long-doc-mode", choices=("off", "lexical"), default="lexical")
    parser.add_argument("--long-doc-max-chars", type=int, default=900)
    parser.add_argument("--long-doc-min-chars", type=int, default=1200)
    parser.add_argument("--long-doc-max-evidence-chunks", type=int, default=3)
    parser.add_argument("--long-doc-max-evidence-chars", type=int, default=3000)
    parser.add_argument("--long-doc-model-max-tokens", type=int, default=96)
    parser.add_argument("--long-doc-model-parallel-batch-size", type=int, default=8)
    parser.add_argument("--user-model")
    parser.add_argument("--user-api-key")
    parser.add_argument("--user-base-url")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--judge-base-url")
    parser.add_argument(
        "--tau-runtime-retries",
        type=int,
        default=3,
        help="Retry official TAU runtime/user/judge transport errors per sample before recording failure",
    )
    parser.add_argument("--static-user", action="store_true", help="Use StaticStopTauUser; useful only for smoke runs")
    parser.add_argument(
        "--scripted-user",
        action="store_true",
        help="Use a deterministic no-LLM user from task instructions; smoke only, not formal scoring",
    )
    parser.add_argument("--include-router-prompts", action="store_true", help="Store full candidate prompts in traces")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file()
    _apply_model_overrides(args)
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_or_prepare_dataset(args.dataset, verbose=False, record_stats=False)
    records = load_tau_manifest_records(dataset_path)
    sample_offset = max(0, int(args.sample_offset or 0))
    if sample_offset:
        records = records[sample_offset:]
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise ValueError(f"empty TAU manifest: {dataset_path}")
    if _requires_tau_v3_source(records):
        require_tau_v3_source(infer_dataset_slug_from_path(str(dataset_path)))
    user_model = None
    judge_model = None
    if not args.static_user and not args.scripted_user and _requires_tau_user_model(records):
        user_model = resolve_required_user_model_config()
        judge_model = resolve_judge_model_config(
            default_model=user_model.model_name,
            default_api_key=user_model.api_key,
            default_base_url=user_model.base_url,
        ) or user_model
        apply_openai_env(user_model)

    slug = infer_dataset_slug_from_path(str(dataset_path))
    benchmark_name, dataset_split = split_benchmark_and_split(slug)
    health = _health_snapshot(args.infer_base_url)
    (output_dir / "health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
    experiment_summary: dict[str, Any] = {
        "created_at": _now_stamp(),
        "dataset": str(dataset_path),
        "dataset_slug": slug,
        "sample_offset": sample_offset,
        "samples_per_model": len(records),
        "infer_base_url": args.infer_base_url,
        "health": health,
        "models": {},
    }

    models, model_source = _resolve_models(args, health)
    experiment_summary["model_source"] = model_source
    experiment_summary["model_order"] = list(models)
    for model_name in models:
        model_dir = output_dir / safe_slug(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"running model={model_name} samples={len(records)} output={model_dir}")
        engine = RemoteInferenceBackend(
            RemoteInferenceConfig(
                base_url=args.infer_base_url,
                model=model_name,
                api_key=args.infer_api_key,
                timeout_s=float(args.infer_timeout_s),
                max_workers=int(args.infer_max_workers),
                max_retries=max(0, int(args.infer_max_retries)),
                protocol=args.infer_protocol,
                seed_policy=args.infer_seed_policy,
            )
        )
        sampling = SamplingConfig(
            max_generate_tokens=max(int(args.candidate_max_tokens), int(args.aggregate_max_tokens)),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
        )
        completions = _run_model_records(
            args=args,
            records=records,
            engine=engine,
            sampling=sampling,
            user_model=user_model,
            judge_model=judge_model,
            benchmark_name=benchmark_name,
            dataset_split=dataset_split,
            model_dir=model_dir,
        )
        eval_payloads = [_tau_completion_to_eval_payload(item) for item in completions]
        _write_jsonl(model_dir / "completions.jsonl", completions)
        _write_jsonl(model_dir / "eval.jsonl", eval_payloads)
        metrics = _compute_metrics(completions, eval_payloads)
        score = make_score_payload(
            slug,
            is_cot=True,
            model_name=model_name,
            metrics=metrics,
            samples=len(completions),
            problems=len(records),
            task="parallel_candidate_router_experiment",
            extra={"cot_mode": "cot", "router_mode": "parallel_candidate"},
        )
        (model_dir / "score.json").write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
        model_summary = {
            "model": model_name,
            "samples": len(completions),
            "metrics": metrics,
            "score_path": str(model_dir / "score.json"),
            "completions_path": str(model_dir / "completions.jsonl"),
            "eval_path": str(model_dir / "eval.jsonl"),
        }
        (model_dir / "summary.json").write_text(json.dumps(model_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        experiment_summary["models"][model_name] = model_summary
        (output_dir / "summary.json").write_text(
            json.dumps(experiment_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"experiment done: {output_dir}")
    return 0


def _run_model_records(
    *,
    args: argparse.Namespace,
    records: Sequence[TauManifestRecord],
    engine: RemoteInferenceBackend,
    sampling: SamplingConfig,
    user_model: Any | None,
    judge_model: Any | None,
    benchmark_name: str,
    dataset_split: str,
    model_dir: Path,
) -> list[dict[str, Any]]:
    runtime_cache: dict[str, TauOfficialRuntime] = {}
    completions: list[dict[str, Any]] = []
    sampling_payload = {
        "router_mode": "parallel_candidate",
        "candidate_chunk_tools": int(args.candidate_chunk_tools),
        "candidate_batch_size": int(args.candidate_batch_size),
        "candidate_max_tokens": int(args.candidate_max_tokens),
        "aggregate_max_tokens": int(args.aggregate_max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "tau_official_runtime": _tau_runtime_model_metadata(user_model, judge_model),
        "scripted_user": bool(args.scripted_user),
    }
    sample_offset = max(0, int(args.sample_offset or 0))
    for sample_index, record in enumerate(records, start=sample_offset):
        print(f"  sample={sample_index} task_id={record.task_id} domain={record.domain}")
        runtime_env = runtime_cache.get(record.domain)
        if runtime_env is None:
            runtime_env = TauOfficialRuntime(domain=record.domain)
            runtime_cache[record.domain] = runtime_env
        max_runtime_attempts = 1 if args.scripted_user else max(1, int(args.tau_runtime_retries or 1))
        payload: dict[str, Any] | None = None
        for runtime_attempt in range(1, max_runtime_attempts + 1):
            payload = _run_one_attempt(
                args=args,
                record=record,
                sample_index=sample_index,
                runtime_env=runtime_env,
                engine=engine,
                sampling=sampling,
                sampling_payload=sampling_payload,
                user_model=user_model,
                judge_model=judge_model,
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
            )
            error_text = str((payload.get("agent_result") or {}).get("error") or "")
            if not error_text.startswith("tau official runtime error:"):
                break
            if runtime_attempt >= max_runtime_attempts:
                break
            delay_s = min(30.0, 5.0 * runtime_attempt)
            print(
                f"    official runtime error; retrying sample={sample_index} "
                f"attempt={runtime_attempt + 1}/{max_runtime_attempts} after {delay_s:.0f}s"
            )
            time.sleep(delay_s)
        assert payload is not None
        completions.append(payload)
        _write_jsonl(model_dir / "completions.partial.jsonl", completions)
    return completions


def _run_one_attempt(
    *,
    args: argparse.Namespace,
    record: TauManifestRecord,
    sample_index: int,
    runtime_env: TauOfficialRuntime,
    engine: RemoteInferenceBackend,
    sampling: SamplingConfig,
    sampling_payload: Mapping[str, Any],
    user_model: Any | None,
    judge_model: Any | None,
    benchmark_name: str,
    dataset_split: str,
) -> dict[str, Any]:
    attempt_started = time.perf_counter()
    task = runtime_env.load_task(record.task)
    environment = runtime_env.create_environment(solo_mode=False)
    agent = ParallelCandidateTauOfficialAgent(
        engine=engine,
        sampling=sampling,
        tools=environment.get_tools(),
        domain_policy=str(environment.get_policy()),
        domain=record.domain,
        history_max_chars=max(0, min(int(args.history_max_chars), DEFAULT_TAU_HISTORY_MAX_CHARS)),
        prompt_max_chars=int(args.prompt_max_chars or DEFAULT_TAU_PROMPT_MAX_CHARS),
        long_doc_config=_long_doc_config(args),
        tool_routing_config=None,
        candidate_config=ParallelCandidateRouterConfig(
            chunk_tools=max(1, int(args.candidate_chunk_tools)),
            batch_size=max(1, int(args.candidate_batch_size)),
            context_chars=max(1, int(args.candidate_context_chars)),
            prompt_max_chars=max(1024, int(args.prompt_max_chars)),
            candidate_max_tokens=max(1, int(args.candidate_max_tokens)),
            aggregate_max_tokens=max(1, int(args.aggregate_max_tokens)),
            policy_chars=max(200, int(args.candidate_policy_chars)),
            tool_schema_mode=str(args.candidate_tool_schema_mode),
        ),
        include_router_prompts=bool(args.include_router_prompts),
    )
    if args.scripted_user:
        user = ScriptedTauUser(task=task)
    else:
        user = runtime_env.build_user(task=task, environment=environment, user_model=user_model)
    seed = sample_repeat_seed(sample_index, 0, pass_index=0, stage=1)
    orchestrator = runtime_env.build_orchestrator(
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max(1, int(args.max_steps or DEFAULT_MAX_STEPS)),
        max_errors=max(1, int(args.max_tool_errors or DEFAULT_MAX_TOOL_ERRORS)),
        seed=seed,
        validate_communication=True,
    )
    timing: dict[str, Any] = {}
    try:
        run_started = time.perf_counter()
        simulation = orchestrator.run()
        timing["orchestrator_run_s"] = time.perf_counter() - run_started
        evaluation_started = time.perf_counter()
        if args.scripted_user:
            evaluation = SimpleNamespace(
                reward=0.0,
                is_passed=False,
                details={
                    "termination_reason": str(getattr(simulation, "termination_reason", "")),
                    "scripted_user": True,
                    "evaluation_skipped": "scripted_user_smoke",
                },
            )
        else:
            evaluation = runtime_env.evaluate(simulation=simulation, task=task, judge_model=judge_model)
        timing["evaluation_s"] = time.perf_counter() - evaluation_started
    except Exception as exc:  # noqa: BLE001 - keep JSON artifact for failed samples.
        error_text = f"tau official runtime error: {type(exc).__name__}: {exc}"
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
            details={"termination_reason": error_text, "runtime_error": error_text},
        )
    timing["total_attempt_s"] = time.perf_counter() - attempt_started
    payload = _tau_official_completion_payload(
        record=record,
        sample_index=sample_index,
        repeat_index=0,
        pass_index=0,
        simulation=simulation,
        evaluation=evaluation,
        agent=agent,
        benchmark_name=benchmark_name,
        dataset_split=dataset_split,
        sampling_payload=dict(sampling_payload),
        timing=timing,
    )
    payload["agent_result"]["cost"] = _sum_message_costs(list(getattr(simulation, "messages", []) or []))
    return payload


def _long_doc_config(args: argparse.Namespace) -> LongDocEvidenceConfig:
    mode = str(args.long_doc_mode or "lexical").strip().lower()
    enabled = mode != "off"
    if mode == "off":
        mode = "lexical"
    return LongDocEvidenceConfig(
        enabled=enabled,
        mode=mode,  # type: ignore[arg-type]
        max_chunk_chars=max(1, int(args.long_doc_max_chars)),
        overlap_lines=2,
        min_long_text_chars=max(1, int(args.long_doc_min_chars)),
        max_evidence_chunks=max(1, int(args.long_doc_max_evidence_chunks)),
        max_evidence_chars=max(1, int(args.long_doc_max_evidence_chars)),
        model_max_tokens=max(1, int(args.long_doc_model_max_tokens)),
        model_parallel_batch_size=max(1, int(args.long_doc_model_parallel_batch_size)),
    )


def _compute_metrics(
    completions: Sequence[Mapping[str, Any]],
    eval_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    total = len(eval_payloads)
    passed = sum(1 for item in eval_payloads if bool(item.get("is_passed", False)))
    errors = 0
    parse_errors = 0
    fallback_routes = 0
    route_count = 0
    candidate_counts: list[float] = []
    turns: list[float] = []
    for payload in completions:
        result = payload.get("agent_result")
        if isinstance(result, Mapping):
            if result.get("error"):
                errors += 1
            try:
                turns.append(float(result.get("num_turns") or 0.0))
            except (TypeError, ValueError):
                pass
        info = payload.get("agent_info")
        if isinstance(info, Mapping):
            parse_errors += len(info.get("parse_errors") or [])
            for row in info.get("tool_routes") or []:
                if not isinstance(row, Mapping):
                    continue
                route_count += 1
                if row.get("fallback_used"):
                    fallback_routes += 1
                try:
                    candidate_counts.append(float(row.get("candidate_count") or 0.0))
                except (TypeError, ValueError):
                    pass
    metrics: dict[str, float] = {
        "avg@1": (passed / total) if total else 0.0,
        "success_rate": (passed / total) if total else 0.0,
        "agent_error_rate": (errors / len(completions)) if completions else 0.0,
        "decision_parse_error_rate": (parse_errors / len(completions)) if completions else 0.0,
    }
    if route_count:
        metrics["tool_route_count"] = float(route_count)
        metrics["tool_route_fallback_rate"] = fallback_routes / route_count
    if candidate_counts:
        metrics["candidate_avg_count"] = sum(candidate_counts) / len(candidate_counts)
    if turns:
        metrics["avg_agent_turns"] = sum(turns) / len(turns)
    return metrics


def _scripted_initial_user_content(task: Any) -> str:
    instructions = _task_user_instruction_mapping(task)
    reason = _clean_scripted_user_text(instructions.get("reason_for_call"))
    known = _clean_scripted_user_text(instructions.get("known_info"))
    task_notes = _clean_scripted_user_text(instructions.get("task_instructions"))
    parts = ["Hello, I need help."]
    if reason:
        parts.append(reason)
    if known:
        parts.append(known)
    if task_notes and "refund" in task_notes.lower():
        parts.append("I do not want to continue unless the refund condition is clear.")
    return _clean_scripted_user_text(" ".join(parts)) or "Hello, I need help with my request."


def _scripted_followup_user_content(task: Any) -> str:
    task_notes = _clean_scripted_user_text(_task_user_instruction_mapping(task).get("task_instructions"))
    if not task_notes:
        return ""
    return task_notes


def _task_user_instruction_mapping(task: Any) -> dict[str, Any]:
    scenario = getattr(task, "user_scenario", None)
    if isinstance(scenario, Mapping):
        instructions = scenario.get("instructions")
    else:
        instructions = getattr(scenario, "instructions", None)
    if isinstance(instructions, Mapping):
        return dict(instructions)
    model_dump = getattr(instructions, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if instructions is None:
        return {}
    return {
        "reason_for_call": getattr(instructions, "reason_for_call", ""),
        "known_info": getattr(instructions, "known_info", ""),
        "task_instructions": getattr(instructions, "task_instructions", ""),
    }


def _clean_scripted_user_text(value: Any) -> str:
    text = " ".join(str(value or "").replace("\t", " ").split())
    return text.strip()


def _assistant_message_looks_like_refusal(message: Any) -> bool:
    text = str(getattr(message, "content", "") or "").lower()
    if not text:
        return False
    markers = ("cannot", "can't", "unable", "not able", "not possible", "sorry", "refuse", "no refund")
    return any(marker in text for marker in markers)


def _prepend_candidate_policy_focus(
    *,
    domain: str,
    full_policy: str,
    compacted_policy: str,
    messages: Sequence[Mapping[str, object]],
) -> str:
    if str(domain or "").strip().lower() != "airline":
        return compacted_policy
    if not _messages_mention_airline_cancel(messages):
        return compacted_policy
    section = _markdown_section(full_policy, "## Cancel flight")
    if not section:
        return compacted_policy
    if "The API does not check that cancellation rules are met" in compacted_policy:
        return compacted_policy
    return (
        "Critical airline cancellation policy:\n"
        f"{section}\n\n"
        "Other selected policy evidence:\n"
        f"{compacted_policy}"
    )


def _messages_mention_airline_cancel(messages: Sequence[Mapping[str, object]]) -> bool:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return bool(re.search(r"\b(cancel|cancelled|canceled|cancellation|refund)\b", text, re.IGNORECASE))


def _markdown_section(text: str, heading: str) -> str:
    normalized = str(text or "").replace("\r\n", "\n")
    start = normalized.find(heading)
    if start < 0:
        return ""
    next_heading = normalized.find("\n## ", start + len(heading))
    if next_heading < 0:
        next_heading = len(normalized)
    return normalized[start:next_heading].strip()


def _apply_model_overrides(args: argparse.Namespace) -> None:
    overrides = {
        "USER_MODEL_NAME": args.user_model,
        "USER_API_KEY": args.user_api_key,
        "USER_BASE_URL": args.user_base_url,
        "JUDGE_MODEL": args.judge_model,
        "JUDGE_API_KEY": args.judge_api_key,
        "JUDGE_BASE_URL": args.judge_base_url,
    }
    for name, value in overrides.items():
        text = str(value or "").strip()
        if text:
            os.environ[name] = text


def _resolve_models(args: argparse.Namespace, health: Mapping[str, Any]) -> tuple[tuple[str, ...], str]:
    if args.infer_models:
        return tuple(str(model).strip() for model in args.infer_models if str(model).strip()), "cli"
    health_models = _models_from_health(health)
    if health_models:
        return health_models, "health"
    return DEFAULT_MODELS, "defaults"


def _models_from_health(health: Mapping[str, Any]) -> tuple[str, ...]:
    candidates: list[str] = []
    healthz = health.get("healthz")
    if isinstance(healthz, Mapping):
        payload = healthz.get("payload")
        if isinstance(payload, Mapping):
            models = payload.get("models")
            if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
                candidates.extend(str(model).strip() for model in models if str(model).strip())
    models_response = health.get("models")
    if isinstance(models_response, Mapping):
        payload = models_response.get("payload")
        if isinstance(payload, Mapping):
            data = payload.get("data")
            if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
                for row in data:
                    if isinstance(row, Mapping):
                        model_id = str(row.get("id") or "").strip()
                        if model_id:
                            candidates.append(model_id)
    return tuple(dict.fromkeys(candidates))


def _resolve_output_dir(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser()
    return Path("out") / "parallel_candidate_router" / _now_stamp()


def _now_stamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%z")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _health_snapshot(base_url: str) -> dict[str, Any]:
    base = str(base_url or "").rstrip("/")
    api_base = normalize_api_base(base)
    return {
        "healthz": _fetch_json(f"{base}/healthz"),
        "models": _fetch_json(f"{api_base}/models"),
    }


def _fetch_json(url: str) -> dict[str, Any]:
    try:
        with urllib_request.urlopen(url, timeout=10.0) as response:
            body = response.read().decode("utf-8", errors="replace")
        try:
            payload: Any = json.loads(body)
        except json.JSONDecodeError:
            payload = body[:1000]
        return {"ok": True, "url": url, "payload": payload}
    except (OSError, urllib_error.URLError) as exc:
        return {"ok": False, "url": url, "error": str(exc)}


if __name__ == "__main__":
    raise SystemExit(main())
