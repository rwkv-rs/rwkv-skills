from __future__ import annotations

"""Shared scheduler launch profiles.

Profiles are the stable interface for routine benchmark starts.  They collapse
the long dispatch CLI surface into a small TOML file plus a few explicit
overrides, while still producing the same DispatchOptions used by the legacy
queue/dispatch commands.
"""

import json
import re
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.benchmark_registry import BenchmarkField
from src.eval.evaluating import RunMode, collect_benchmark_dataset_slugs
from src.infer.backend import REMOTE_INFERENCE_PROTOCOL_CHOICES

from .actions_base import CodingConfig, DispatchOptions, FunctionCallingConfig, InferenceConfig, KnowledgeConfig, MathConfig
from .config import (
    DEFAULT_ADMIN_STATE_DIR,
    DEFAULT_DISPATCH_POLL_SECONDS,
    DEFAULT_GPU_IDLE_MAX_MEM,
    DEFAULT_INFER_MAX_WORKERS,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL_GLOBS,
    DEFAULT_PID_DIR,
    DEFAULT_RUN_LOG_DIR,
    REPO_ROOT,
)
from .dataset_utils import canonical_slug, canonicalize_benchmark_list
from .jobs import JOB_CATALOGUE, JOB_ORDER
from .models import MODEL_SELECT_CHOICES
from .remote_slots import INFER_WORKER_PROFILE_CHOICES


PROFILE_DIR = REPO_ROOT / "configs" / "scheduler"
DEFAULT_PROFILE_NAME = "local-6gpu-full"

_KNOWN_DATASET_SLUGS: tuple[str, ...] = tuple(
    sorted({canonical_slug(slug) for spec in JOB_CATALOGUE.values() for slug in spec.dataset_slugs})
)


@dataclass(slots=True)
class SchedulerLaunchRequest:
    """Serializable scheduler start request used by CLI, profiles, and admin."""

    profile: str = ""
    run_tag: str = ""
    log_dir: str = str(DEFAULT_LOG_DIR)
    pid_dir: str = str(DEFAULT_PID_DIR)
    run_log_dir: str = str(DEFAULT_RUN_LOG_DIR)
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODEL_GLOBS))
    infer_base_url: str = ""
    infer_models: list[str] = field(default_factory=list)
    infer_model_groups: list[dict[str, Any]] = field(default_factory=list)
    infer_slots_per_model: int = 1
    infer_api_key: str = ""
    infer_timeout_s: float = 600.0
    infer_max_workers: int = DEFAULT_INFER_MAX_WORKERS
    infer_worker_profile: str = "fixed"
    infer_protocol: str = "openai"
    infer_seed_policy: str = "preserve"
    remote_batch_size: int | None = None
    plain_choice_batch_size: int | None = None
    plain_choice_timeout_s: float | None = None
    sample_workers: int | None = None
    infer_backpressure: bool = True
    infer_backpressure_timeout_s: float = 2.0
    infer_backpressure_pending_high_watermark: int = 0
    infer_budget_min_workers: int = 1
    distributed_claims: bool = False
    scheduler_node_id: str = ""
    lease_duration_s: int = 900
    model_regex: list[str] = field(default_factory=list)
    model_select: str = "latest-data"
    min_param_b: float | None = None
    max_param_b: float | None = None
    only_jobs: list[str] = field(default_factory=list)
    skip_jobs: list[str] = field(default_factory=list)
    job_order: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    benchmark_fields: list[str] = field(default_factory=list)
    extra_benchmarks: list[str] = field(default_factory=list)
    only_datasets: list[str] = field(default_factory=list)
    skip_datasets: list[str] = field(default_factory=list)
    enable_param_search: bool = False
    run_mode: str = RunMode.AUTO.value
    dispatch_poll_seconds: int = DEFAULT_DISPATCH_POLL_SECONDS
    gpu_idle_max_mem: int = DEFAULT_GPU_IDLE_MAX_MEM
    skip_missing_dataset: bool = False
    clean_param_swap: bool = False
    batch_cache: str | None = None
    benchmark_config_root: str = ""
    overwrite: bool = False
    disable_checker: bool = False
    coding_eval_workers: int | None = None
    max_active_coding_runners: int | None = None
    math_judge_max_workers: int | None = None
    math_prompt_max_chars: int | None = None
    math_long_doc_mode: str | None = None
    knowledge_prompt_max_chars: int | None = None
    knowledge_long_doc_mode: str | None = None
    function_prompt_style: str | None = None
    function_tool_catalog_format: str | None = None
    function_cot_max_tokens: int | None = None
    function_decision_max_tokens: int | None = None
    function_planning_max_tokens: int | None = None
    function_final_max_tokens: int | None = None
    function_answer_max_tokens: int | None = None
    function_judge_max_workers: int | None = None
    function_history_max_chars: int | None = None
    function_prompt_max_chars: int | None = None
    function_long_doc_mode: str | None = None
    function_tool_router_mode: str | None = None
    function_tool_router_max_tools: int | None = None
    function_tool_router_trigger_tool_count: int | None = None
    function_tool_router_trigger_catalog_chars: int | None = None
    function_candidate_router_mode: str | None = None
    function_candidate_router_chunk_tools: int | None = None
    function_candidate_router_batch_size: int | None = None
    function_candidate_router_prompt_max_chars: int | None = None
    function_candidate_router_context_chars: int | None = None
    function_candidate_router_candidate_max_tokens: int | None = None
    function_candidate_router_aggregate_max_tokens: int | None = None
    function_candidate_router_max_candidates: int | None = None
    function_candidate_router_tool_schema_mode: str | None = None
    function_candidate_router_evidence_chars: int | None = None
    function_candidate_router_policy_chars: int | None = None
    function_max_rounds: int | None = None
    function_max_steps: int | None = None
    function_max_tool_errors: int | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SchedulerLaunchRequest":
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown fields: {', '.join(unknown)}")
        return cls(**dict(payload))

    def copy(self) -> "SchedulerLaunchRequest":
        return replace(
            self,
            models=list(self.models),
            infer_models=list(self.infer_models),
            infer_model_groups=[dict(item) for item in self.infer_model_groups],
            model_regex=list(self.model_regex),
            only_jobs=list(self.only_jobs),
            skip_jobs=list(self.skip_jobs),
            job_order=list(self.job_order),
            domains=list(self.domains),
            benchmark_fields=list(self.benchmark_fields),
            extra_benchmarks=list(self.extra_benchmarks),
            only_datasets=list(self.only_datasets),
            skip_datasets=list(self.skip_datasets),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_dispatch_options(self) -> DispatchOptions:
        if self.model_select not in MODEL_SELECT_CHOICES:
            raise ValueError(f"unknown model_select={self.model_select!r}")
        infer_protocol = str(self.infer_protocol or "openai")
        if infer_protocol not in REMOTE_INFERENCE_PROTOCOL_CHOICES:
            raise ValueError(f"unknown infer_protocol={infer_protocol!r}")
        infer_seed_policy = str(self.infer_seed_policy or "preserve")
        if infer_seed_policy == "omit-for-contents":
            infer_seed_policy = "omit"
        if infer_seed_policy not in {"preserve", "omit"}:
            raise ValueError(f"unknown infer_seed_policy={infer_seed_policy!r}")
        infer_worker_profile = str(self.infer_worker_profile or "fixed")
        if infer_worker_profile not in INFER_WORKER_PROFILE_CHOICES:
            raise ValueError(f"unknown infer_worker_profile={infer_worker_profile!r}")

        job_list = resolve_job_list(self.only_jobs, self.skip_jobs, self.domains)
        if not job_list:
            raise ValueError("no schedulable jobs remain after filtering")

        skip_dataset_slugs = canonicalize_slugs(self.skip_datasets)
        benchmark_fields = tuple(BenchmarkField(value) for value in self.benchmark_fields)
        selected: set[str] = set()
        if benchmark_fields or self.extra_benchmarks:
            selected.update(
                collect_benchmark_dataset_slugs(
                    fields=benchmark_fields,
                    extra_benchmark_names=tuple(self.extra_benchmarks),
                )
            )
        selected.update(canonicalize_slugs(self.only_datasets))
        only_dataset_slugs = tuple(sorted(selected))
        model_patterns = compile_model_patterns(self.model_regex)
        job_priority = resolve_job_priority(self.job_order, job_list)

        explicit = self.run_mode
        if self.overwrite and explicit not in (RunMode.AUTO.value, RunMode.RERUN.value):
            raise ValueError("--overwrite only supports run_mode auto/rerun")
        run_mode = RunMode.RERUN if self.overwrite else RunMode.parse(explicit)

        infer_base_url = str(self.infer_base_url or "").strip() or None
        infer_models = expand_infer_model_groups(
            self.infer_models,
            self.infer_model_groups,
            slots_per_model=int(self.infer_slots_per_model or 1),
        )
        if infer_base_url or infer_models:
            if not infer_base_url:
                raise ValueError("remote inference mode requires infer_base_url")
            if not infer_models:
                raise ValueError("remote inference mode requires infer_models")
            model_globs: tuple[str, ...] = ()
        else:
            model_globs = tuple(self.models)

        batch_cache_path = Path(self.batch_cache) if self.batch_cache else None
        return DispatchOptions(
            log_dir=Path(self.log_dir),
            pid_dir=Path(self.pid_dir),
            run_log_dir=Path(self.run_log_dir),
            job_order=job_list,
            job_priority=job_priority,
            model_select=self.model_select,
            min_param_b=self.min_param_b,
            max_param_b=self.max_param_b,
            skip_dataset_slugs=skip_dataset_slugs,
            model_globs=model_globs,
            only_dataset_slugs=only_dataset_slugs,
            model_name_patterns=model_patterns,
            enable_param_search=self.enable_param_search,
            run_mode=run_mode,
            inference=InferenceConfig(
                base_url=infer_base_url,
                models=infer_models,
                api_key=str(self.infer_api_key or ""),
                timeout_s=float(self.infer_timeout_s),
                max_workers=int(self.infer_max_workers),
                worker_profile=infer_worker_profile,
                protocol=infer_protocol,
                seed_policy=infer_seed_policy,
                remote_batch_size=(
                    int(self.remote_batch_size)
                    if self.remote_batch_size is not None
                    else None
                ),
                plain_choice_batch_size=(
                    int(self.plain_choice_batch_size)
                    if self.plain_choice_batch_size is not None
                    else None
                ),
                plain_choice_timeout_s=(
                    float(self.plain_choice_timeout_s)
                    if self.plain_choice_timeout_s is not None
                    else None
                ),
                sample_workers=(
                    int(self.sample_workers)
                    if self.sample_workers is not None
                    else None
                ),
                backpressure=bool(self.infer_backpressure),
                backpressure_timeout_s=float(self.infer_backpressure_timeout_s),
                backpressure_pending_high_watermark=int(self.infer_backpressure_pending_high_watermark),
                budget_min_workers=int(self.infer_budget_min_workers),
            ),
            functions=FunctionCallingConfig(
                prompt_style=self.function_prompt_style,
                tool_catalog_format=self.function_tool_catalog_format,
                cot_max_tokens=self.function_cot_max_tokens,
                decision_max_tokens=self.function_decision_max_tokens,
                planning_max_tokens=self.function_planning_max_tokens,
                final_max_tokens=self.function_final_max_tokens,
                answer_max_tokens=self.function_answer_max_tokens,
                judge_max_workers=self.function_judge_max_workers,
                history_max_chars=self.function_history_max_chars,
                prompt_max_chars=self.function_prompt_max_chars,
                long_doc_mode=self.function_long_doc_mode,
                tool_router_mode=self.function_tool_router_mode,
                tool_router_max_tools=self.function_tool_router_max_tools,
                tool_router_trigger_tool_count=self.function_tool_router_trigger_tool_count,
                tool_router_trigger_catalog_chars=self.function_tool_router_trigger_catalog_chars,
                candidate_router_mode=self.function_candidate_router_mode,
                candidate_router_chunk_tools=self.function_candidate_router_chunk_tools,
                candidate_router_batch_size=self.function_candidate_router_batch_size,
                candidate_router_prompt_max_chars=self.function_candidate_router_prompt_max_chars,
                candidate_router_context_chars=self.function_candidate_router_context_chars,
                candidate_router_candidate_max_tokens=self.function_candidate_router_candidate_max_tokens,
                candidate_router_aggregate_max_tokens=self.function_candidate_router_aggregate_max_tokens,
                candidate_router_max_candidates=self.function_candidate_router_max_candidates,
                candidate_router_tool_schema_mode=self.function_candidate_router_tool_schema_mode,
                candidate_router_evidence_chars=self.function_candidate_router_evidence_chars,
                candidate_router_policy_chars=self.function_candidate_router_policy_chars,
                max_rounds=self.function_max_rounds,
                max_steps=self.function_max_steps,
                max_tool_errors=self.function_max_tool_errors,
            ),
            coding=CodingConfig(
                eval_workers=(
                    int(self.coding_eval_workers)
                    if self.coding_eval_workers is not None
                    else None
                ),
                max_active_runners=(
                    int(self.max_active_coding_runners)
                    if self.max_active_coding_runners is not None
                    else None
                ),
            ),
            math=MathConfig(
                judge_max_workers=(
                    int(self.math_judge_max_workers)
                    if self.math_judge_max_workers is not None
                    else None
                ),
                prompt_max_chars=(
                    int(self.math_prompt_max_chars)
                    if self.math_prompt_max_chars is not None
                    else None
                ),
                long_doc_mode=self.math_long_doc_mode,
            ),
            knowledge=KnowledgeConfig(
                prompt_max_chars=(
                    int(self.knowledge_prompt_max_chars)
                    if self.knowledge_prompt_max_chars is not None
                    else None
                ),
                long_doc_mode=self.knowledge_long_doc_mode,
            ),
            distributed_claims=bool(self.distributed_claims),
            scheduler_node_id=(str(self.scheduler_node_id or "").strip() or None),
            lease_duration_s=int(self.lease_duration_s),
            dispatch_poll_seconds=int(self.dispatch_poll_seconds),
            gpu_idle_max_mem=int(self.gpu_idle_max_mem),
            skip_missing_dataset=self.skip_missing_dataset,
            clean_param_swap=self.clean_param_swap,
            batch_cache_path=batch_cache_path,
            disable_checker=self.disable_checker,
            benchmark_config_root=(
                Path(self.benchmark_config_root).expanduser()
                if str(self.benchmark_config_root or "").strip()
                else None
            ),
        )


def resolve_job_list(
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    domains: Sequence[str] | None,
) -> tuple[str, ...]:
    order = list(JOB_ORDER)
    if domains:
        allowed_domains = set(domains)
        order = [job for job in order if JOB_CATALOGUE[job].domain in allowed_domains]
    if include:
        allowed_jobs = set(include)
        order = [job for job in order if job in allowed_jobs]
    if exclude:
        blocked_jobs = set(exclude)
        order = [job for job in order if job not in blocked_jobs]
    return tuple(order)


def canonicalize_slugs(slugs: Sequence[str] | None) -> tuple[str, ...]:
    if not slugs:
        return tuple()
    return canonicalize_benchmark_list(slugs, known_slugs=_KNOWN_DATASET_SLUGS)


def compile_model_patterns(patterns: Sequence[str] | None) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for raw in patterns or ():
        compiled.append(re.compile(raw))
    return tuple(compiled)


def resolve_job_priority(priority: Sequence[str] | None, available: Sequence[str]) -> tuple[str, ...] | None:
    if not priority:
        return None
    allowed = set(available)
    ordered: list[str] = []
    for job in priority:
        if job in allowed and job not in ordered:
            ordered.append(job)
    return tuple(ordered) if ordered else None


def expand_infer_model_groups(
    infer_models: Sequence[str],
    model_groups: Sequence[Mapping[str, Any]],
    *,
    slots_per_model: int,
) -> tuple[str, ...]:
    expanded: list[str] = []
    seen: set[str] = set()

    def _add(spec: str) -> None:
        text = str(spec or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        expanded.append(text)

    for group in model_groups:
        model = str(group.get("model") or "").strip()
        if not model:
            raise ValueError("inference.model_groups entry requires model")
        slot_prefix = str(group.get("slot_prefix") or group.get("slot") or model).strip()
        slots = max(1, int(group.get("slots", 1)))
        start = int(group.get("start", 1))
        width = max(1, int(group.get("width", 2)))
        raw_base_urls = group.get("base_urls")
        if raw_base_urls is None:
            base_urls: tuple[str, ...] = ()
        elif isinstance(raw_base_urls, Sequence) and not isinstance(raw_base_urls, (str, bytes)):
            base_urls = tuple(str(url).strip() for url in raw_base_urls)
        else:
            raise ValueError("inference.model_groups base_urls must be an array")
        if base_urls and (len(base_urls) != slots or any(not url for url in base_urls)):
            raise ValueError("inference.model_groups base_urls must contain one URL per slot")
        shared_base_url = str(group.get("base_url") or "").strip()
        for offset, index in enumerate(range(start, start + slots)):
            spec = f"{slot_prefix}_s{index:0{width}d}={model}"
            base_url = base_urls[offset] if base_urls else shared_base_url
            _add(f"{spec}|{base_url}" if base_url else spec)

    explicit_slots = max(1, int(slots_per_model))
    for raw in infer_models:
        text = str(raw).strip()
        if not text:
            continue
        if "=" in text or explicit_slots == 1:
            _add(text)
            continue
        for index in range(explicit_slots):
            _add(f"{text}-s{index}={text}")
    return tuple(expanded)


def load_launch_profile(profile: str | Path = DEFAULT_PROFILE_NAME) -> SchedulerLaunchRequest:
    path = resolve_profile_path(profile)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    payload = _flatten_profile_payload(raw)
    payload.setdefault("profile", path.stem)
    request = SchedulerLaunchRequest.from_payload(payload)
    return _expand_path_templates(request)


def resolve_profile_path(profile: str | Path) -> Path:
    raw = Path(profile).expanduser()
    candidates: list[Path] = []
    if raw.suffix:
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append((PROFILE_DIR / raw).resolve())
    else:
        candidates.append(raw)
        candidates.append(PROFILE_DIR / f"{raw}.toml")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"scheduler profile not found: {profile!r}; searched {searched}")


def _flatten_profile_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    direct_keys = set(SchedulerLaunchRequest.__dataclass_fields__)
    section_maps: dict[str, dict[str, str]] = {
        "paths": {
            "log_dir": "log_dir",
            "pid_dir": "pid_dir",
            "run_log_dir": "run_log_dir",
        },
        "inference": {
            "base_url": "infer_base_url",
            "models": "infer_models",
            "model_groups": "infer_model_groups",
            "slots_per_model": "infer_slots_per_model",
            "api_key": "infer_api_key",
            "timeout_s": "infer_timeout_s",
            "max_workers": "infer_max_workers",
            "worker_profile": "infer_worker_profile",
            "protocol": "infer_protocol",
            "seed_policy": "infer_seed_policy",
            "remote_batch_size": "remote_batch_size",
            "plain_choice_batch_size": "plain_choice_batch_size",
            "plain_choice_timeout_s": "plain_choice_timeout_s",
            "sample_workers": "sample_workers",
            "backpressure": "infer_backpressure",
            "backpressure_timeout_s": "infer_backpressure_timeout_s",
            "backpressure_pending_high_watermark": "infer_backpressure_pending_high_watermark",
            "budget_min_workers": "infer_budget_min_workers",
        },
        "selection": {
            "models": "models",
            "model_regex": "model_regex",
            "model_select": "model_select",
            "min_param_b": "min_param_b",
            "max_param_b": "max_param_b",
            "only_jobs": "only_jobs",
            "skip_jobs": "skip_jobs",
            "job_order": "job_order",
            "domains": "domains",
            "benchmark_fields": "benchmark_fields",
            "extra_benchmarks": "extra_benchmarks",
            "only_datasets": "only_datasets",
            "skip_datasets": "skip_datasets",
            "enable_param_search": "enable_param_search",
        },
        "runtime": {
            "run_mode": "run_mode",
            "dispatch_poll_seconds": "dispatch_poll_seconds",
            "gpu_idle_max_mem": "gpu_idle_max_mem",
            "skip_missing_dataset": "skip_missing_dataset",
            "clean_param_swap": "clean_param_swap",
            "batch_cache": "batch_cache",
            "benchmark_config_root": "benchmark_config_root",
            "overwrite": "overwrite",
            "disable_checker": "disable_checker",
            "distributed_claims": "distributed_claims",
            "scheduler_node_id": "scheduler_node_id",
            "lease_duration_s": "lease_duration_s",
        },
        "coding": {
            "eval_workers": "coding_eval_workers",
            "max_active_runners": "max_active_coding_runners",
        },
        "math": {
            "judge_max_workers": "math_judge_max_workers",
            "prompt_max_chars": "math_prompt_max_chars",
            "long_doc_mode": "math_long_doc_mode",
        },
        "knowledge": {
            "prompt_max_chars": "knowledge_prompt_max_chars",
            "long_doc_mode": "knowledge_long_doc_mode",
        },
        "function_calling": {
            "prompt_style": "function_prompt_style",
            "tool_catalog_format": "function_tool_catalog_format",
            "cot_max_tokens": "function_cot_max_tokens",
            "decision_max_tokens": "function_decision_max_tokens",
            "planning_max_tokens": "function_planning_max_tokens",
            "final_max_tokens": "function_final_max_tokens",
            "answer_max_tokens": "function_answer_max_tokens",
            "judge_max_workers": "function_judge_max_workers",
            "history_max_chars": "function_history_max_chars",
            "prompt_max_chars": "function_prompt_max_chars",
            "long_doc_mode": "function_long_doc_mode",
            "tool_router_mode": "function_tool_router_mode",
            "tool_router_max_tools": "function_tool_router_max_tools",
            "tool_router_trigger_tool_count": "function_tool_router_trigger_tool_count",
            "tool_router_trigger_catalog_chars": "function_tool_router_trigger_catalog_chars",
            "candidate_router_mode": "function_candidate_router_mode",
            "candidate_router_chunk_tools": "function_candidate_router_chunk_tools",
            "candidate_router_batch_size": "function_candidate_router_batch_size",
            "candidate_router_prompt_max_chars": "function_candidate_router_prompt_max_chars",
            "candidate_router_context_chars": "function_candidate_router_context_chars",
            "candidate_router_candidate_max_tokens": "function_candidate_router_candidate_max_tokens",
            "candidate_router_aggregate_max_tokens": "function_candidate_router_aggregate_max_tokens",
            "candidate_router_max_candidates": "function_candidate_router_max_candidates",
            "candidate_router_tool_schema_mode": "function_candidate_router_tool_schema_mode",
            "candidate_router_evidence_chars": "function_candidate_router_evidence_chars",
            "candidate_router_policy_chars": "function_candidate_router_policy_chars",
            "max_rounds": "function_max_rounds",
            "max_steps": "function_max_steps",
            "max_tool_errors": "function_max_tool_errors",
        },
    }
    for key, value in raw.items():
        if key in section_maps:
            if not isinstance(value, Mapping):
                raise ValueError(f"profile section {key!r} must be a table")
            for section_key, section_value in value.items():
                if section_key not in section_maps[key]:
                    raise ValueError(f"unknown scheduler profile key: {key}.{section_key}")
                payload[section_maps[key][section_key]] = section_value
            continue
        if key not in direct_keys:
            raise ValueError(f"unknown scheduler profile key: {key}")
        payload[key] = value
    return payload


def _expand_path_templates(request: SchedulerLaunchRequest) -> SchedulerLaunchRequest:
    if not request.run_tag:
        return request
    values = {
        "run_tag": request.run_tag,
        "profile": request.profile,
        "admin_state_dir": str(DEFAULT_ADMIN_STATE_DIR),
    }

    def _format(text: str) -> str:
        return str(text).format_map(_SafeFormatDict(values))

    return replace(
        request,
        log_dir=_format(request.log_dir),
        pid_dir=_format(request.pid_dir),
        run_log_dir=_format(request.run_log_dir),
        batch_cache=(_format(request.batch_cache) if request.batch_cache else None),
    )


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def launch_request_to_json(request: SchedulerLaunchRequest) -> str:
    return json.dumps(request.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "PROFILE_DIR",
    "SchedulerLaunchRequest",
    "canonicalize_slugs",
    "compile_model_patterns",
    "expand_infer_model_groups",
    "launch_request_to_json",
    "load_launch_profile",
    "resolve_job_list",
    "resolve_job_priority",
    "resolve_profile_path",
]
