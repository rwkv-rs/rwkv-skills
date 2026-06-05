from __future__ import annotations

"""Dedicated official TAU2/TAU3 execution path.

This runner keeps the existing eval_function_call_agent DB/write/scoring
contract, but avoids the generic local agent pipeline that repeatedly rebuilds
unbounded TAU prompts.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from src.eval.agent_bench.tau_official import (
    DEFAULT_TAU_PROMPT_MAX_CHARS,
    RWKVTauOfficialAgent,
    TauOfficialRuntime,
    tau_trajectory_dump,
)
from src.eval.datasets.data_struct.function_call import FunctionCallTaskRecord
from src.eval.evaluators.common import sample_repeat_seed
from src.eval.function_calling.agent.pipeline import load_agent_records
from src.eval.function_calling.common.payload import FunctionCallRunStats, build_agent_completion_payload
from src.eval.function_calling.long_context_router import (
    LongContextRoutingConfig,
    long_context_routing_config_to_payload,
)
from src.eval.function_calling.tool_router import (
    ToolRoutingConfig,
    tool_routing_config_to_payload,
)
from src.eval.results.schema import dataset_slug_parts, normalize_sampling_config_by_stage
from src.infer.engine import InferenceEngine
from src.infer.sampling import SamplingConfig

if TYPE_CHECKING:
    from src.infer.model import ModelLoadConfig

DEFAULT_TAU_MAX_STEPS = 200
DEFAULT_TAU_MAX_TOOL_ERRORS = 10
DEFAULT_TAU_HISTORY_MAX_CHARS = 16000
DEFAULT_TAU_DECISION_MAX_TOKENS = 1024
DEFAULT_TAU_MAX_REPEATED_TOOL_CALLS = 2


@dataclass(slots=True)
class TauLLMConfig:
    model_name: str
    api_key: str
    base_url: str | None = None


@dataclass(slots=True)
class TauOfficialRunnerOptions:
    max_steps: int = DEFAULT_TAU_MAX_STEPS
    max_tool_errors: int = DEFAULT_TAU_MAX_TOOL_ERRORS
    history_max_chars: int = DEFAULT_TAU_HISTORY_MAX_CHARS
    prompt_max_chars: int = DEFAULT_TAU_PROMPT_MAX_CHARS
    decision_max_tokens: int = DEFAULT_TAU_DECISION_MAX_TOKENS
    max_repeated_tool_calls: int = DEFAULT_TAU_MAX_REPEATED_TOOL_CALLS
    user_model: str | None = None
    user_api_key: str | None = None
    user_base_url: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str | None = None

    @classmethod
    def from_sources(cls, args: Any, config: Any | None) -> TauOfficialRunnerOptions:
        def int_value(name: str, env_name: str, default: int) -> int:
            raw = getattr(args, name, None)
            if raw is None and config is not None:
                raw = getattr(config, name, None)
            if raw is None:
                raw = os.environ.get(env_name)
            return _positive_int(raw, default)

        def str_value(name: str, *env_names: str) -> str | None:
            raw = getattr(args, name, None)
            if raw is None and config is not None:
                raw = getattr(config, name, None)
            if raw is None:
                for env_name in env_names:
                    value = os.environ.get(env_name)
                    if value:
                        raw = value
                        break
            text = str(raw or "").strip()
            return text or None

        return cls(
            max_steps=int_value("max_steps", "RWKV_TAU_MAX_STEPS", DEFAULT_TAU_MAX_STEPS),
            max_tool_errors=int_value(
                "max_tool_errors",
                "RWKV_TAU_MAX_TOOL_ERRORS",
                DEFAULT_TAU_MAX_TOOL_ERRORS,
            ),
            history_max_chars=int_value(
                "history_max_chars",
                "RWKV_TAU_HISTORY_MAX_CHARS",
                DEFAULT_TAU_HISTORY_MAX_CHARS,
            ),
            prompt_max_chars=int_value(
                "prompt_max_chars",
                "RWKV_TAU_PROMPT_MAX_CHARS",
                DEFAULT_TAU_PROMPT_MAX_CHARS,
            ),
            decision_max_tokens=int_value(
                "decision_max_tokens",
                "RWKV_TAU_DECISION_MAX_TOKENS",
                DEFAULT_TAU_DECISION_MAX_TOKENS,
            ),
            max_repeated_tool_calls=int_value(
                "max_repeated_tool_calls",
                "RWKV_TAU_MAX_REPEATED_TOOL_CALLS",
                DEFAULT_TAU_MAX_REPEATED_TOOL_CALLS,
            ),
            user_model=str_value("user_model", "USER_MODEL_NAME", "model_name", "MODEL_NAME"),
            user_api_key=str_value("user_api_key", "USER_API_KEY", "API_KEY", "OPENAI_API_KEY"),
            user_base_url=_normalize_openai_base_url(
                str_value("user_base_url", "USER_BASE_URL", "OPENAI_BASE_URL", "API_BASE", "BASE_URL")
            ),
            judge_model=str_value("judge_model", "judge_model_name", "JUDGE_MODEL", "LLM_JUDGE_MODEL"),
            judge_api_key=str_value("judge_api_key", "JUDGE_API_KEY", "API_KEY", "OPENAI_API_KEY"),
            judge_base_url=_normalize_openai_base_url(
                str_value("judge_base_url", "JUDGE_BASE_URL", "OPENAI_BASE_URL", "API_BASE", "BASE_URL")
            ),
        )


@dataclass(slots=True)
class TauOfficialPipelineResult:
    dataset: str
    sample_count: int
    payloads: list[dict[str, Any]]


class TauOfficialAgentPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        from src.infer.model import load_rwkv_model

        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)
        self.model_path = model_config.weights_path

    def run(
        self,
        dataset_path: str,
        *,
        sampling: SamplingConfig,
        options: TauOfficialRunnerOptions,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        samples_per_task: int | None = None,
        skip_keys: set[tuple[int, int]] | None = None,
        tool_routing_config: ToolRoutingConfig | None = None,
        long_context_routing_config: LongContextRoutingConfig | None = None,
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> TauOfficialPipelineResult:
        records, resolved_name = load_agent_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        benchmark_name, dataset_split = dataset_slug_parts(dataset_name)
        repeats = max(1, int(samples_per_task or 1))
        skip_keys = skip_keys or set()
        routing = tool_routing_config or ToolRoutingConfig()
        long_context_routing = long_context_routing_config or LongContextRoutingConfig(
            mode="lexical" if routing.enabled else "off"
        )
        sampling = sampling.clamp(options.decision_max_tokens)

        entries: list[tuple[int, FunctionCallTaskRecord, int]] = []
        for idx, record in enumerate(records):
            for sample_id in range(repeats):
                if (idx, sample_id) not in skip_keys:
                    entries.append((idx, record, sample_id))
        if not entries:
            return TauOfficialPipelineResult(dataset_name, 0, [])

        user_model = _resolve_user_model(records, options)
        judge_model = _resolve_judge_model(options, default_model=getattr(user_model, "model_name", None))
        if user_model is not None:
            _apply_openai_env(user_model)
        sampling_config = normalize_sampling_config_by_stage([(1, sampling)])
        sampling_config["tau_official_runtime"] = _tau_runtime_model_metadata(user_model, judge_model)
        sampling_config["tau_limits"] = {
            "history_max_chars": int(options.history_max_chars),
            "prompt_max_chars": int(options.prompt_max_chars),
            "max_steps": int(options.max_steps),
            "max_tool_errors": int(options.max_tool_errors),
            "decision_max_tokens": int(options.decision_max_tokens),
            "max_repeated_tool_calls": int(options.max_repeated_tool_calls),
        }
        sampling_config["tool_routing"] = tool_routing_config_to_payload(routing)
        sampling_config["long_context_routing"] = long_context_routing_config_to_payload(long_context_routing)

        runtime_cache: dict[str, TauOfficialRuntime] = {}

        def runtime_for_domain(domain: str) -> TauOfficialRuntime:
            cached = runtime_cache.get(domain)
            if cached is None:
                cached = TauOfficialRuntime(domain=domain)
                runtime_cache[domain] = cached
            return cached

        payloads: list[dict[str, Any]] = []
        for record_idx, record, sample_id in entries:
            payload = self._run_one(
                record=record,
                sample_index=record_idx,
                repeat_index=sample_id,
                sampling=sampling,
                sampling_config=sampling_config,
                benchmark_name=benchmark_name,
                dataset_split=dataset_split,
                options=options,
                tool_routing_config=routing,
                long_context_routing_config=long_context_routing,
                runtime=runtime_for_domain(_record_domain(record)),
                user_model=user_model,
                judge_model=judge_model,
            )
            if on_record is not None:
                on_record(payload)
            payloads.append(payload)
        return TauOfficialPipelineResult(dataset_name, len(entries), payloads)

    def _run_one(
        self,
        *,
        record: FunctionCallTaskRecord,
        sample_index: int,
        repeat_index: int,
        sampling: SamplingConfig,
        sampling_config: dict[str, Any],
        benchmark_name: str,
        dataset_split: str,
        options: TauOfficialRunnerOptions,
        tool_routing_config: ToolRoutingConfig,
        long_context_routing_config: LongContextRoutingConfig,
        runtime: TauOfficialRuntime,
        user_model: TauLLMConfig | None,
        judge_model: TauLLMConfig | None,
    ) -> dict[str, Any]:
        task_payload = _record_task_payload(record)
        env_kwargs = _tau_env_kwargs(record)
        task = runtime.load_task(task_payload)
        environment = runtime.create_environment(solo_mode=False, env_kwargs=env_kwargs)
        agent = RWKVTauOfficialAgent(
            engine=self.engine,
            sampling=sampling,
            tools=environment.get_tools(),
            domain_policy=str(environment.get_policy()),
            history_max_chars=options.history_max_chars,
            prompt_max_chars=options.prompt_max_chars,
            tool_routing_config=tool_routing_config,
            long_context_routing_config=long_context_routing_config,
            max_repeated_tool_calls=options.max_repeated_tool_calls,
        )
        agent.set_seed(sample_repeat_seed(sample_index, repeat_index, stage=1))
        user = runtime.build_user(task=task, environment=environment, user_model=user_model)
        orchestrator = runtime.build_orchestrator(
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=min(options.max_steps, record.max_steps or options.max_steps),
            max_errors=options.max_tool_errors,
            seed=sample_repeat_seed(sample_index, repeat_index, stage=2),
        )
        try:
            simulation = orchestrator.run()
            evaluation = runtime.evaluate(
                simulation=simulation,
                task=task,
                judge_model=judge_model,
                solo_mode=False,
                env_kwargs=env_kwargs,
            )
        except Exception as exc:
            error_text = f"tau official runtime error: {type(exc).__name__}: {exc}"
            agent.parse_errors.append(error_text)
            simulation = _fallback_simulation(record, task, orchestrator, error_text)
            evaluation = _failed_evaluation(error_text)

        messages = list(getattr(simulation, "messages", []) or [])
        details = dict(getattr(evaluation, "details", {}) or {})
        details.update(
            {
                "benchmark": "tau_official",
                "domain": _record_domain(record),
                "task_id": str(getattr(task, "id", "") or record.task_id),
                "benchmark_version": _record_benchmark_version(record),
                "score": float(evaluation.reward),
                "steps": len(agent.stages),
                "parse_error_count": len(agent.parse_errors),
                "invalid_action_count": len(agent.parse_errors),
                "tool_routes": list(agent.tool_routes),
                "finish_reason": str(details.get("termination_reason") or getattr(simulation, "termination_reason", "")),
                "max_steps": int(options.max_steps),
                "max_tool_errors": int(options.max_tool_errors),
                "prompt_max_chars": int(options.prompt_max_chars),
                "history_max_chars": int(options.history_max_chars),
                "max_repeated_tool_calls": int(options.max_repeated_tool_calls),
                "runtime": "official_tau_orchestrator",
            }
        )
        if agent.parse_errors:
            details["parse_errors"] = list(agent.parse_errors)
        final_answer = _final_answer_from_messages(messages)
        events = tau_trajectory_dump(messages)
        prompts = [stage.prompt for stage in agent.stages]
        completions = [stage.completion for stage in agent.stages]
        stats = FunctionCallRunStats(
            steps=len(agent.stages),
            tool_calls=_count_tau_tool_calls(messages),
            prompt_chars=sum(len(prompt) for prompt in prompts),
            completion_chars=sum(len(completion) for completion in completions),
        )
        payload = build_agent_completion_payload(
            benchmark_name=benchmark_name,
            dataset_split=dataset_split,
            sample_index=sample_index,
            repeat_index=repeat_index,
            sampling_config=sampling_config,  # type: ignore[arg-type]
            prompts=prompts,
            completions=completions,
            events=events,
            stats=stats,
            env_type="tau_official",
            scorer_type="tau_official",
            success=bool(evaluation.is_passed),
            official_score=float(evaluation.reward),
            details=details,
        )
        payload["final_answer"] = final_answer
        payload["agent_trace"] = events
        payload["tau_official_result"] = details
        return payload



def _record_domain(record: FunctionCallTaskRecord) -> str:
    domain = str(record.env.get("domain") or record.metadata.get("domain") or "").strip()
    if not domain:
        raise ValueError("TAU official record is missing domain")
    return domain


def _record_task_payload(record: FunctionCallTaskRecord) -> dict[str, Any]:
    task_payload = record.metadata.get("task") or record.env.get("task")
    if not isinstance(task_payload, Mapping):
        raise ValueError("TAU official record is missing task payload")
    return dict(task_payload)


def _record_benchmark_version(record: FunctionCallTaskRecord) -> str:
    return str(record.metadata.get("benchmark_version") or record.env.get("benchmark_version") or "")


def _tau_env_kwargs(record: FunctionCallTaskRecord) -> dict[str, Any]:
    raw = record.env.get("env_kwargs") or record.metadata.get("env_kwargs")
    kwargs = dict(raw) if isinstance(raw, Mapping) else {}
    retrieval_config = record.env.get("retrieval_config") or record.metadata.get("retrieval_config")
    if retrieval_config and "retrieval_variant" not in kwargs:
        kwargs["retrieval_variant"] = str(retrieval_config)
    return kwargs


def _is_lightweight_tau_record(record: FunctionCallTaskRecord) -> bool:
    version = _record_benchmark_version(record).lower().strip()
    return version in {"tau_v3_light", "tau3_light", "tau_light"} or (
        _record_domain(record) == "mock" and version.startswith("tau_v3_light")
    )


def _resolve_user_model(records: Sequence[FunctionCallTaskRecord], options: TauOfficialRunnerOptions) -> TauLLMConfig | None:
    if not any(not _is_lightweight_tau_record(record) for record in records):
        return None
    model = (options.user_model or "").strip()
    api_key = (options.user_api_key or "").strip()
    if not model or not api_key:
        raise ValueError(
            "TAU official runtime requires user simulator config: set user_model/user_api_key "
            "in the function-calling TOML or USER_MODEL_NAME/USER_API_KEY in the environment."
        )
    return TauLLMConfig(model_name=model, api_key=api_key, base_url=options.user_base_url)


def _resolve_judge_model(options: TauOfficialRunnerOptions, *, default_model: str | None) -> TauLLMConfig | None:
    model = (options.judge_model or default_model or "").strip()
    api_key = (options.judge_api_key or options.user_api_key or "").strip()
    if not model:
        return None
    if not api_key:
        raise ValueError("TAU judge model was configured but no judge_api_key/user_api_key is available.")
    return TauLLMConfig(
        model_name=model,
        api_key=api_key,
        base_url=options.judge_base_url or options.user_base_url,
    )


def _apply_openai_env(config: TauLLMConfig) -> None:
    os.environ["OPENAI_API_KEY"] = config.api_key
    os.environ["API_KEY"] = config.api_key
    base_url = _normalize_openai_base_url(config.base_url)
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["API_BASE"] = base_url


def _tau_runtime_model_metadata(user_model: TauLLMConfig | None, judge_model: TauLLMConfig | None) -> dict[str, Any]:
    return {
        "user_model": getattr(user_model, "model_name", None),
        "user_base_url": getattr(user_model, "base_url", None),
        "judge_model": getattr(judge_model, "model_name", None),
        "judge_base_url": getattr(judge_model, "base_url", None),
        "static_user": user_model is None,
        "judge_configured": judge_model is not None,
    }


def _tau_litellm_model_name(model_config: TauLLMConfig) -> str:
    model_name = str(model_config.model_name or "").strip()
    if not model_name or "/" in model_name:
        return model_name
    base_url = _normalize_openai_base_url(model_config.base_url) or ""
    if "api.deepseek.com" in base_url and model_name.startswith("deepseek-"):
        return f"deepseek/{model_name}"
    return model_name


def _tau_llm_timeout_args() -> dict[str, float]:
    timeout_s = _first_positive_float_env("RWKV_TAU_LLM_TIMEOUT_S", "RWKV_TAU_USER_TIMEOUT_S", "RWKV_LLM_TIMEOUT_S")
    if timeout_s is None:
        return {}
    return {"timeout": timeout_s}


def _first_positive_float_env(*names: str) -> float | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = float(value.strip())
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


def _normalize_openai_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().rstrip("/")
    if not text:
        return None
    suffix = "/chat/completions"
    if text.endswith(suffix):
        text = text[: -len(suffix)].rstrip("/")
    marker = "://"
    if marker in text:
        after_scheme = text.split(marker, 1)[1]
        if "/" not in after_scheme:
            text = f"{text}/v1"
    return text or None


def _count_tau_tool_calls(messages: Sequence[Any]) -> int:
    total = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            total += len(list(tool_calls))
    return total


def _final_answer_from_messages(messages: Sequence[Any]) -> str:
    for message in reversed(messages):
        if str(getattr(message, "role", "") or "").strip().lower() != "assistant":
            continue
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.replace("###STOP###", "").strip()
    return ""


def _fallback_simulation(record: FunctionCallTaskRecord, task: Any, orchestrator: Any, error_text: str) -> Any:
    messages = list(getattr(orchestrator, "messages", []) or getattr(orchestrator, "_messages", []) or [])
    return type(
        "TauFailedSimulation",
        (),
        {
            "task_id": str(getattr(task, "id", "") or record.task_id),
            "messages": messages,
            "agent_cost": float(getattr(orchestrator, "agent_cost", 0.0) or 0.0),
            "user_cost": float(getattr(orchestrator, "user_cost", 0.0) or 0.0),
            "termination_reason": error_text,
        },
    )()


def _failed_evaluation(error_text: str) -> Any:
    return type(
        "TauFailedEvaluation",
        (),
        {
            "reward": 0.0,
            "is_passed": False,
            "details": {"termination_reason": error_text, "runtime_error": error_text},
        },
    )()


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(1, parsed)


__all__ = [
    "DEFAULT_TAU_DECISION_MAX_TOKENS",
    "DEFAULT_TAU_HISTORY_MAX_CHARS",
    "DEFAULT_TAU_MAX_REPEATED_TOOL_CALLS",
    "DEFAULT_TAU_MAX_STEPS",
    "DEFAULT_TAU_MAX_TOOL_ERRORS",
    "DEFAULT_TAU_PROMPT_MAX_CHARS",
    "TauOfficialAgentPipeline",
    "TauOfficialPipelineResult",
    "TauOfficialRunnerOptions",
]
