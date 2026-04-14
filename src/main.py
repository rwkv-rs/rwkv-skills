from __future__ import annotations

"""Unified config-driven entrypoint for single-run evaluation flows."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import inspect
import json
import os
from pathlib import Path
import tomllib
from typing import Any, Iterator, Mapping, Sequence

from src.eval.benchmark_registry import BenchmarkField, BenchmarkMetadata, CoTMode, resolve_benchmark_metadata
from src.eval.evaluating import RunContext, RunMode, TaskSpec, resolve_registered_benchmark_name
from src.eval.runner_registry import RunnerGroup, RunnerSpec, resolve_runner
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import canonical_slug, infer_dataset_slug_from_path, make_dataset_slug, split_benchmark_and_split
from src.eval.scheduler.datasets import DATASET_ROOTS, find_dataset_file

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_CONFIG_ROOT = REPO_ROOT / "configs" / "run"


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a table/object")


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _maybe_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a boolean")


def _maybe_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    raise TypeError(f"{field_name} must be an integer")


def _maybe_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{field_name} must be a number")


def _tuple_str(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return tuple(str(item) for item in value)


def _tuple_int(value: object, *, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    values: list[int] = []
    for item in value:
        values.append(_maybe_int(item, field_name=field_name) or 0)
    return tuple(values)


def _tuple_float(value: object, *, field_name: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    values: list[float] = []
    for item in value:
        parsed = _maybe_float(item, field_name=field_name)
        if parsed is None:
            continue
        values.append(parsed)
    return tuple(values)


@dataclass(frozen=True, slots=True)
class RunSection:
    mode: str = "eval"
    id: str | None = None
    job: str | None = None
    run_mode: RunMode = RunMode.AUTO
    batch_size: int | None = None
    max_samples: int | None = None
    probe_only: bool = False
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RunSection":
        raw_run_mode = _maybe_str(payload.get("run_mode")) or RunMode.AUTO.value
        return cls(
            mode=_maybe_str(payload.get("mode")) or "eval",
            id=_maybe_str(payload.get("id")),
            job=_maybe_str(payload.get("job")),
            run_mode=RunMode(raw_run_mode),
            batch_size=_maybe_int(payload.get("batch_size"), field_name="run.batch_size"),
            max_samples=_maybe_int(payload.get("max_samples"), field_name="run.max_samples"),
            probe_only=bool(_maybe_bool(payload.get("probe_only"), field_name="run.probe_only") or False),
            extra_args=_tuple_str(payload.get("extra_args"), field_name="run.extra_args"),
        )


@dataclass(frozen=True, slots=True)
class DatasetSection:
    name: str | None = None
    split: str | None = None
    path: str | None = None
    prepare: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetSection":
        return cls(
            name=_maybe_str(payload.get("name")),
            split=_maybe_str(payload.get("split")),
            path=_maybe_str(payload.get("path")),
            prepare=bool(_maybe_bool(payload.get("prepare"), field_name="dataset.prepare") is not False),
        )


@dataclass(frozen=True, slots=True)
class ModelSection:
    path: str | None = None
    device: str = "cuda"
    infer_base_url: str | None = None
    infer_model: str | None = None
    infer_api_key: str = ""
    infer_timeout_s: float | None = None
    infer_max_workers: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelSection":
        return cls(
            path=_maybe_str(payload.get("path")),
            device=_maybe_str(payload.get("device")) or "cuda",
            infer_base_url=_maybe_str(payload.get("infer_base_url")),
            infer_model=_maybe_str(payload.get("infer_model")),
            infer_api_key=_maybe_str(payload.get("infer_api_key")) or "",
            infer_timeout_s=_maybe_float(payload.get("infer_timeout_s"), field_name="model.infer_timeout_s"),
            infer_max_workers=_maybe_int(payload.get("infer_max_workers"), field_name="model.infer_max_workers"),
        )


@dataclass(frozen=True, slots=True)
class RunnerSection:
    cot_mode: str | None = None
    judge_mode: str | None = None
    benchmark_kind: str | None = None
    target_token_format: str | None = None
    db_write_queue: int | None = None
    db_drain_every: int | None = None
    db_close_timeout_s: float | None = None
    cot_max_tokens: int | None = None
    final_max_tokens: int | None = None
    answer_max_tokens: int | None = None
    planning_max_tokens: int | None = None
    decision_max_tokens: int | None = None
    max_rounds: int | None = None
    max_steps: int | None = None
    max_tool_errors: int | None = None
    history_max_chars: int | None = None
    enable_think: bool = False
    stop_tokens: tuple[int, ...] = ()
    ban_tokens: tuple[int, ...] = ()
    avg_ks: tuple[float, ...] = ()
    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    eval_timeout: float | None = None
    eval_workers: int | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str | None = None
    judge_max_workers: int | None = None
    extra_args: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RunnerSection":
        return cls(
            cot_mode=_maybe_str(payload.get("cot_mode")),
            judge_mode=_maybe_str(payload.get("judge_mode")),
            benchmark_kind=_maybe_str(payload.get("benchmark_kind")),
            target_token_format=_maybe_str(payload.get("target_token_format")),
            db_write_queue=_maybe_int(payload.get("db_write_queue"), field_name="runner.db_write_queue"),
            db_drain_every=_maybe_int(payload.get("db_drain_every"), field_name="runner.db_drain_every"),
            db_close_timeout_s=_maybe_float(payload.get("db_close_timeout_s"), field_name="runner.db_close_timeout_s"),
            cot_max_tokens=_maybe_int(payload.get("cot_max_tokens"), field_name="runner.cot_max_tokens"),
            final_max_tokens=_maybe_int(payload.get("final_max_tokens"), field_name="runner.final_max_tokens"),
            answer_max_tokens=_maybe_int(payload.get("answer_max_tokens"), field_name="runner.answer_max_tokens"),
            planning_max_tokens=_maybe_int(payload.get("planning_max_tokens"), field_name="runner.planning_max_tokens"),
            decision_max_tokens=_maybe_int(payload.get("decision_max_tokens"), field_name="runner.decision_max_tokens"),
            max_rounds=_maybe_int(payload.get("max_rounds"), field_name="runner.max_rounds"),
            max_steps=_maybe_int(payload.get("max_steps"), field_name="runner.max_steps"),
            max_tool_errors=_maybe_int(payload.get("max_tool_errors"), field_name="runner.max_tool_errors"),
            history_max_chars=_maybe_int(payload.get("history_max_chars"), field_name="runner.history_max_chars"),
            enable_think=bool(_maybe_bool(payload.get("enable_think"), field_name="runner.enable_think") or False),
            stop_tokens=_tuple_int(payload.get("stop_tokens"), field_name="runner.stop_tokens"),
            ban_tokens=_tuple_int(payload.get("ban_tokens"), field_name="runner.ban_tokens"),
            avg_ks=_tuple_float(payload.get("avg_ks"), field_name="runner.avg_ks"),
            max_tokens=_maybe_int(payload.get("max_tokens"), field_name="runner.max_tokens"),
            temperature=_maybe_float(payload.get("temperature"), field_name="runner.temperature"),
            top_k=_maybe_int(payload.get("top_k"), field_name="runner.top_k"),
            top_p=_maybe_float(payload.get("top_p"), field_name="runner.top_p"),
            eval_timeout=_maybe_float(payload.get("eval_timeout"), field_name="runner.eval_timeout"),
            eval_workers=_maybe_int(payload.get("eval_workers"), field_name="runner.eval_workers"),
            judge_model=_maybe_str(payload.get("judge_model")),
            judge_api_key=_maybe_str(payload.get("judge_api_key")),
            judge_base_url=_maybe_str(payload.get("judge_base_url")),
            judge_max_workers=_maybe_int(payload.get("judge_max_workers"), field_name="runner.judge_max_workers"),
            extra_args=_tuple_str(payload.get("extra_args"), field_name="runner.extra_args"),
        )


@dataclass(frozen=True, slots=True)
class RunConfig:
    run: RunSection = field(default_factory=RunSection)
    dataset: DatasetSection = field(default_factory=DatasetSection)
    model: ModelSection = field(default_factory=ModelSection)
    runner: RunnerSection = field(default_factory=RunnerSection)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, source_path: Path | None = None) -> "RunConfig":
        run = RunSection.from_mapping(_as_mapping(payload.get("run"), field_name="run"))
        dataset = DatasetSection.from_mapping(_as_mapping(payload.get("dataset"), field_name="dataset"))
        model = ModelSection.from_mapping(_as_mapping(payload.get("model"), field_name="model"))
        runner = RunnerSection.from_mapping(_as_mapping(payload.get("runner"), field_name="runner"))
        return cls(run=run, dataset=dataset, model=model, runner=runner, source_path=source_path)


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    config: RunConfig
    benchmark: BenchmarkMetadata
    dataset_slug: str
    dataset_path: Path
    runner: RunnerSpec
    task_spec: TaskSpec
    run_context: RunContext
    argv: tuple[str, ...]
    env: Mapping[str, str]

    @property
    def module(self) -> str:
        return self.runner.module


def load_run_config(config_path: str | Path) -> RunConfig:
    path = resolve_run_config_path(config_path)
    payload = _load_config_mapping(path)
    return RunConfig.from_mapping(payload, source_path=path)


def resolve_run_config_path(config_path: str | Path | None = None, *, benchmark: str | None = None) -> Path:
    if benchmark:
        benchmark_slug = canonical_slug(benchmark)
        candidate = RUN_CONFIG_ROOT / f"{benchmark_slug}.toml"
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(
            f"run config for benchmark {benchmark_slug!r} not found: {candidate}"
        )

    if config_path is None:
        raise ValueError("config path or benchmark name is required")

    raw = str(config_path).strip()
    if not raw:
        raise ValueError("config path must not be empty")

    direct = Path(raw).expanduser()
    if direct.is_file():
        return direct.resolve()

    candidate = RUN_CONFIG_ROOT / raw
    if candidate.is_file():
        return candidate.resolve()

    if direct.suffix:
        raise FileNotFoundError(f"config file not found: {direct.resolve()}")

    named = RUN_CONFIG_ROOT / f"{canonical_slug(raw)}.toml"
    if named.is_file():
        return named.resolve()

    raise FileNotFoundError(
        f"config file not found: {raw!r}; searched {direct.resolve()} and {named}"
    )


def _load_config_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        return data
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, Mapping):
            return data
        raise TypeError("JSON config must be an object")
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML config requires PyYAML; install it or use TOML") from exc
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, Mapping):
            return data
        raise TypeError("YAML config must be an object")
    raise ValueError(f"unsupported config format: {path.suffix}")


def resolve_run_config(config: RunConfig) -> ResolvedRun:
    if config.run.mode not in {"eval", "param_search"}:
        raise ValueError(f"unsupported run.mode: {config.run.mode!r}; expected 'eval' or 'param_search'")
    benchmark = _resolve_benchmark(config.dataset)
    dataset_slug = canonical_slug(make_dataset_slug(benchmark.dataset, config.dataset.split or benchmark.default_split))
    dataset_path = _resolve_dataset_path(config.dataset, dataset_slug=dataset_slug)
    runner = _resolve_runner_for_config(config, benchmark)
    argv = _build_runner_argv(config, runner=runner, benchmark=benchmark, dataset_path=dataset_path)
    benchmark_name, benchmark_split = split_benchmark_and_split(dataset_slug)
    task_spec = TaskSpec(
        run_kind=config.run.mode,
        runner_name=runner.name,
        dataset_slug=dataset_slug,
        dataset_path=dataset_path,
        benchmark_name=benchmark_name,
        benchmark_split=benchmark_split,
        model_name=_resolve_model_name(config.model),
        model_path=config.model.path,
        config_path=config.source_path,
    )
    run_context = RunContext(
        job_name=runner.name,
        run_mode=config.run.run_mode,
        run_id=config.run.id,
    )
    env = run_context.env_overrides(dataset_slug=dataset_slug)
    return ResolvedRun(
        config=config,
        benchmark=benchmark,
        dataset_slug=dataset_slug,
        dataset_path=dataset_path,
        runner=runner,
        task_spec=task_spec,
        run_context=run_context,
        argv=tuple(argv),
        env=env,
    )


def _resolve_benchmark(dataset: DatasetSection) -> BenchmarkMetadata:
    if dataset.name:
        if dataset.split:
            return resolve_benchmark_metadata(make_dataset_slug(dataset.name, dataset.split))
        return resolve_benchmark_metadata(resolve_registered_benchmark_name(dataset.name))
    if dataset.path:
        return resolve_benchmark_metadata(infer_dataset_slug_from_path(dataset.path))
    raise ValueError("dataset.name or dataset.path is required")


def _resolve_dataset_path(dataset: DatasetSection, *, dataset_slug: str) -> Path:
    if dataset.path:
        path = Path(dataset.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dataset path does not exist: {path}")
        inferred_slug = canonical_slug(infer_dataset_slug_from_path(str(path)))
        if inferred_slug != dataset_slug:
            raise ValueError(
                f"dataset.path resolves to slug {inferred_slug!r}, expected {dataset_slug!r}; "
                "use a canonical file name or omit dataset.path"
            )
        return path
    if dataset.prepare:
        return resolve_or_prepare_dataset(dataset_slug, verbose=False)
    path = find_dataset_file(dataset_slug, DATASET_ROOTS)
    if path is None:
        search = ", ".join(str(root) for root in DATASET_ROOTS)
        raise FileNotFoundError(
            f"dataset {dataset_slug!r} not found in configured roots: {search}; "
            "set dataset.prepare=true or provide dataset.path"
        )
    return path


def _resolve_runner_for_config(config: RunConfig, benchmark: BenchmarkMetadata) -> RunnerSpec:
    job_name = canonical_slug(config.run.job) if config.run.job else _default_job_name(config, benchmark)
    if config.run.mode == "param_search":
        runner = resolve_runner(job_name)
        if runner.group is not RunnerGroup.PARAM_SEARCH:
            raise ValueError(f"job {job_name!r} is not a param_search runner")
        if config.model.path is None:
            raise ValueError("run.mode='param_search' currently requires model.path")
        if runner.name != "param_search_select":
            target_job = runner.name.removeprefix("param_search_")
            if target_job not in benchmark.scheduler_jobs:
                allowed = ", ".join(benchmark.scheduler_jobs)
                raise ValueError(
                    f"param_search job {job_name!r} does not match benchmark {benchmark.name!r}; "
                    f"expected param_search variant of: {allowed}"
                )
        return runner
    if job_name not in benchmark.scheduler_jobs:
        allowed = ", ".join(benchmark.scheduler_jobs)
        raise ValueError(f"job {job_name!r} is not valid for benchmark {benchmark.name!r}; allowed: {allowed}")
    return resolve_runner(job_name)


def _default_job_name(config: RunConfig, benchmark: BenchmarkMetadata) -> str:
    if config.run.mode == "param_search":
        if benchmark.field is not BenchmarkField.MATHS:
            raise ValueError("run.mode='param_search' only supports maths benchmarks right now")
        if "free_response_judge" in benchmark.scheduler_jobs:
            return "param_search_free_response_judge"
        if "free_response" in benchmark.scheduler_jobs:
            return "param_search_free_response"
        raise ValueError(
            f"benchmark {benchmark.name!r} does not have a compatible free-response runner for param_search"
        )
    jobs = benchmark.scheduler_jobs
    if len(jobs) == 1:
        return jobs[0]
    if benchmark.field is BenchmarkField.KNOWLEDGE:
        cot_mode = _resolve_cot_mode(config.runner.cot_mode, benchmark=benchmark, default=CoTMode.NO_COT)
        return {
            CoTMode.NO_COT: "multi_choice_plain",
            CoTMode.FAKE_COT: "multi_choice_fake_cot",
            CoTMode.COT: "multi_choice_cot",
        }[cot_mode]
    if benchmark.field is BenchmarkField.MATHS:
        judge_mode = config.runner.judge_mode or "exact"
        return "free_response_judge" if judge_mode == "llm" else "free_response"
    if benchmark.field is BenchmarkField.CODING:
        cot_mode = _resolve_cot_mode(config.runner.cot_mode, benchmark=benchmark, default=CoTMode.NO_COT)
        return {
            CoTMode.NO_COT: "code_mbpp",
            CoTMode.FAKE_COT: "code_mbpp_fake_cot",
            CoTMode.COT: "code_mbpp_cot",
        }[cot_mode]
    raise ValueError(f"benchmark {benchmark.name!r} requires an explicit run.job")


def _resolve_cot_mode(raw_mode: str | None, *, benchmark: BenchmarkMetadata, default: CoTMode) -> CoTMode:
    resolved = CoTMode(raw_mode) if raw_mode is not None else default
    if resolved not in benchmark.cot_modes:
        supported = ", ".join(mode.value for mode in benchmark.cot_modes)
        raise ValueError(f"benchmark {benchmark.name!r} does not support cot_mode={resolved.value!r}; supported: {supported}")
    return resolved


def _append_flag(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _append_repeatable(argv: list[str], flag: str, values: Sequence[object]) -> None:
    for value in values:
        argv.extend([flag, str(value)])


def _append_backend_args(argv: list[str], model: ModelSection) -> None:
    if model.path:
        argv.extend(["--model-path", model.path, "--device", model.device])
        return
    if model.infer_base_url:
        argv.extend(["--infer-base-url", model.infer_base_url])
    if model.infer_model:
        argv.extend(["--infer-model", model.infer_model])
    if model.infer_api_key:
        argv.extend(["--infer-api-key", model.infer_api_key])
    if model.infer_timeout_s is not None:
        argv.extend(["--infer-timeout-s", str(model.infer_timeout_s)])
    if model.infer_max_workers is not None:
        argv.extend(["--infer-max-workers", str(model.infer_max_workers)])


def _resolve_model_name(model: ModelSection) -> str:
    if model.path:
        return Path(model.path).stem
    if model.infer_model:
        return model.infer_model
    if model.infer_base_url:
        return model.infer_base_url.rstrip("/").rsplit("/", 1)[-1] or "remote-model"
    return "unknown-model"


def _function_calling_benchmark_kind(job_name: str) -> str:
    if job_name == "function_browsecomp":
        return "browsecomp"
    if job_name == "function_mcp_bench":
        return "mcp_bench"
    if job_name == "function_bfcl_v3":
        return "bfcl_v3"
    if job_name == "function_tau2_bench":
        return "tau2_bench"
    return "tau_bench"


def _coding_benchmark_kind(job_name: str) -> str:
    if job_name == "code_human_eval":
        return "human_eval"
    if job_name == "code_livecodebench":
        return "livecodebench"
    return "mbpp"


def _build_runner_argv(
    config: RunConfig,
    *,
    runner: RunnerSpec,
    benchmark: BenchmarkMetadata,
    dataset_path: Path,
) -> list[str]:
    argv = ["--dataset", str(dataset_path)]
    _append_backend_args(argv, config.model)
    if config.run.max_samples is not None:
        _append_flag(argv, "--max-samples", config.run.max_samples)
    if config.run.batch_size is not None:
        if runner.batch_flag is None:
            raise ValueError(f"runner {runner.name!r} does not accept batch_size")
        _append_flag(argv, runner.batch_flag, config.run.batch_size)
    if config.run.probe_only:
        if runner.probe_flag is None:
            raise ValueError(f"runner {runner.name!r} does not support probe_only")
        argv.append(runner.probe_flag)
    argv.extend(runner.extra_args)

    group = runner.group
    runner_cfg = config.runner
    if group is RunnerGroup.KNOWLEDGE:
        _append_flag(argv, "--target-token-format", runner_cfg.target_token_format)
        _append_flag(argv, "--db-write-queue", runner_cfg.db_write_queue)
    elif group is RunnerGroup.MATHS:
        _append_flag(argv, "--cot-max-tokens", runner_cfg.cot_max_tokens)
        _append_flag(argv, "--final-max-tokens", runner_cfg.final_max_tokens)
        _append_flag(argv, "--db-write-queue", runner_cfg.db_write_queue)
        _append_flag(argv, "--db-drain-every", runner_cfg.db_drain_every)
        _append_flag(argv, "--db-close-timeout-s", runner_cfg.db_close_timeout_s)
        _append_flag(argv, "--judge-model", runner_cfg.judge_model)
        _append_flag(argv, "--judge-api-key", runner_cfg.judge_api_key)
        _append_flag(argv, "--judge-base-url", runner_cfg.judge_base_url)
        _append_flag(argv, "--judge-max-workers", runner_cfg.judge_max_workers)
    elif group is RunnerGroup.CODING:
        _append_flag(argv, "--benchmark-kind", runner_cfg.benchmark_kind or _coding_benchmark_kind(runner.name))
        _append_flag(argv, "--max-tokens", runner_cfg.max_tokens)
        _append_flag(argv, "--temperature", runner_cfg.temperature)
        _append_flag(argv, "--top-k", runner_cfg.top_k)
        _append_flag(argv, "--top-p", runner_cfg.top_p)
        _append_flag(argv, "--eval-timeout", runner_cfg.eval_timeout)
        _append_flag(argv, "--eval-workers", runner_cfg.eval_workers)
        _append_flag(argv, "--db-write-queue", runner_cfg.db_write_queue)
    elif group is RunnerGroup.INSTRUCTION_FOLLOWING:
        if runner_cfg.enable_think:
            argv.append("--enable-think")
        _append_repeatable(argv, "--stop-token", runner_cfg.stop_tokens)
        _append_repeatable(argv, "--ban-token", runner_cfg.ban_tokens)
        _append_repeatable(argv, "--avg-k", runner_cfg.avg_ks)
        _append_flag(argv, "--db-write-queue", runner_cfg.db_write_queue)
    elif group is RunnerGroup.FUNCTION_CALLING:
        _append_flag(argv, "--benchmark-kind", runner_cfg.benchmark_kind or _function_calling_benchmark_kind(runner.name))
        _append_flag(argv, "--db-write-queue", runner_cfg.db_write_queue)
        _append_flag(argv, "--db-close-timeout-s", runner_cfg.db_close_timeout_s)
        _append_flag(argv, "--history-max-chars", runner_cfg.history_max_chars)
        _append_repeatable(argv, "--avg-k", runner_cfg.avg_ks)
        if runner.name == "function_browsecomp":
            _append_flag(argv, "--cot-max-tokens", runner_cfg.cot_max_tokens)
            _append_flag(argv, "--answer-max-tokens", runner_cfg.answer_max_tokens)
        elif runner.name == "function_mcp_bench":
            _append_flag(argv, "--planning-max-tokens", runner_cfg.planning_max_tokens)
            _append_flag(argv, "--decision-max-tokens", runner_cfg.decision_max_tokens)
            _append_flag(argv, "--final-max-tokens", runner_cfg.final_max_tokens)
            _append_flag(argv, "--max-rounds", runner_cfg.max_rounds)
        else:
            _append_flag(argv, "--max-steps", runner_cfg.max_steps)
            _append_flag(argv, "--max-tool-errors", runner_cfg.max_tool_errors)
            _append_flag(argv, "--cot-max-tokens", runner_cfg.cot_max_tokens)
            _append_flag(argv, "--decision-max-tokens", runner_cfg.decision_max_tokens)
    elif group is RunnerGroup.PARAM_SEARCH:
        _append_flag(argv, "--db-write-queue", runner_cfg.db_write_queue)
        _append_flag(argv, "--cot-max-tokens", runner_cfg.cot_max_tokens)
        _append_flag(argv, "--final-max-tokens", runner_cfg.final_max_tokens)
        if runner.name == "param_search_free_response_judge":
            _append_flag(argv, "--judge-model", runner_cfg.judge_model)
            _append_flag(argv, "--judge-api-key", runner_cfg.judge_api_key)
            _append_flag(argv, "--judge-base-url", runner_cfg.judge_base_url)
    else:  # pragma: no cover - unsupported future runner groups
        raise ValueError(f"unsupported runner group for unified main: {group.value}")

    argv.extend(config.run.extra_args)
    argv.extend(runner_cfg.extra_args)
    return argv


def run_from_config(config: RunConfig) -> int:
    resolved = resolve_run_config(config)
    module = importlib.import_module(resolved.module)
    target_main = getattr(module, "main", None)
    if not callable(target_main):
        raise TypeError(f"runner module {resolved.module!r} does not expose callable main()")
    explicit_kwargs = _explicit_contract_kwargs(target_main=target_main, resolved=resolved)
    if explicit_kwargs:
        result = target_main(
            list(resolved.argv),
            **explicit_kwargs,
        )
    else:
        with _patched_environ(resolved.env):
            result = target_main(list(resolved.argv))
    return int(result)


def _explicit_contract_kwargs(*, target_main: Any, resolved: ResolvedRun) -> dict[str, object]:
    try:
        params = inspect.signature(target_main).parameters
    except (TypeError, ValueError):
        return {}
    kwargs: dict[str, object] = {}
    if "run_context" in params:
        kwargs["run_context"] = resolved.run_context
    if "task_spec" in params:
        kwargs["task_spec"] = resolved.task_spec
    return kwargs


@contextmanager
def _patched_environ(overrides: Mapping[str, str]) -> Iterator[None]:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    os.environ.update({key: str(value) for key, value in overrides.items()})
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RWKV unified config-driven entry")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--config",
        help="Run config path (.toml/.json/.yaml) or config name under configs/run/",
    )
    target.add_argument(
        "--benchmark",
        help="Benchmark name resolved to configs/run/<benchmark>.toml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and print runner invocation without executing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = resolve_run_config_path(args.config, benchmark=args.benchmark)
    config = load_run_config(config_path)
    resolved = resolve_run_config(config)
    if args.dry_run:
        payload = {
            "config_path": str(config_path),
            "module": resolved.module,
            "job": resolved.runner.name,
            "dataset_slug": resolved.dataset_slug,
            "dataset_path": str(resolved.dataset_path),
            "argv": list(resolved.argv),
            "env": dict(resolved.env),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return run_from_config(config)


__all__ = [
    "DatasetSection",
    "ModelSection",
    "ResolvedRun",
    "RunConfig",
    "RunSection",
    "RunnerSection",
    "build_parser",
    "load_run_config",
    "main",
    "resolve_run_config_path",
    "resolve_run_config",
    "run_from_config",
]


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
