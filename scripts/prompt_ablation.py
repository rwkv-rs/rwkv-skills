from __future__ import annotations

"""Build isolated prompt-ablation scheduler runs.

Prompt-set TOML example:

    [trial.asdiv_short_v1]
    jobs = ["free_response"]
    benchmarks = ["asdiv_cot", "svamp_cot"]
    max_samples = 80
    cot_prompt_template = "User: Solve the problem. Preserve the requested answer form.\\n<Q>\\n\\nAssistant: <think"
    final_prompt_template = "<Q><COT>\\nAnswer only with the final answer:"

    [trial.gpqa_short_v1]
    jobs = ["multi_choice_cot"]
    benchmarks = ["gpqa_main_cot"]
    cot_prompt_template = "User: Choose the single best option.\\n<Q>\\n<CHOICES>\\n\\nAssistant: <think"
    final_prompt_template = "<Q><COT>\\nTherefore, the answer is"

The script writes one config overlay per trial and a commands.sh file. The
overlay only replaces fields present in the trial; normal configs/ values still
provide sampling parameters and k-metric settings.
"""

import argparse
import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for existing server envs.
    import tomli as tomllib

from src.eval.scheduler.config import REPO_ROOT
from src.eval.scheduler.dataset_utils import (
    canonical_slug,
    safe_slug,
    split_benchmark_and_split,
)


TRIAL_ROOT_KEY = "trial"
PROMPT_FIELDS_BY_SECTION: dict[str, tuple[str, ...]] = {
    "direct": ("direct_prompt_template",),
    "cot": ("cot_prompt_template",),
    "final": ("final_prompt_template",),
}
DEFAULT_FIELDS = {
    "avg_k",
    "agent_system_template",
    "agent_user_template",
    "function_call_system_template",
    "function_call_user_template",
    "judge_prompt_template",
    "max_samples",
    "pass_k",
    "report_avg_k",
    "report_pass_k",
}
IGNORED_TRIAL_FIELDS = {"benchmarks", "description", "jobs", "notes"}


@dataclass(frozen=True)
class PromptTrial:
    name: str
    jobs: tuple[str, ...]
    benchmarks: tuple[str, ...]
    config_benchmarks: tuple[str, ...]
    config: dict[str, dict[str, Any]]
    description: str | None = None


@dataclass(frozen=True)
class ExperimentCommand:
    trial: PromptTrial
    trial_dir: Path
    config_root: Path
    command: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prompt ablation scheduler runs")
    parser.add_argument("--prompt-set", required=True, help="TOML file containing [trial.<name>] tables")
    parser.add_argument("--output-root", default="results/prompt_ablation", help="Experiment output directory")
    parser.add_argument("--run-id", help="Optional run id; defaults to current timestamp")
    parser.add_argument("--only-trials", nargs="+", help="Only build named trials from the prompt set")
    parser.add_argument("--jobs", nargs="+", help="Default jobs for trials without jobs=...")
    parser.add_argument("--benchmarks", nargs="+", help="Default/restricting benchmark list")
    parser.add_argument("--models", nargs="*", default=(), help="Model paths/globs passed to scheduler")
    parser.add_argument("--model-select", default="all", help="Scheduler --model-select value")
    parser.add_argument("--model-regex", nargs="+", help="Scheduler --model-regex values")
    parser.add_argument("--min-param-b", type=float, help="Scheduler --min-param-b value")
    parser.add_argument("--max-param-b", type=float, help="Scheduler --max-param-b value")
    parser.add_argument("--overwrite", action="store_true", help="Add scheduler --overwrite")
    parser.add_argument("--disable-checker", action="store_true", help="Add scheduler --disable-checker")
    parser.add_argument("--skip-missing-dataset", action="store_true", help="Add scheduler --skip-missing-dataset")
    parser.add_argument("--pg-host", help="PG_HOST exported into generated commands.sh")
    parser.add_argument("--pg-port", help="PG_PORT exported into generated commands.sh")
    parser.add_argument("--pg-user", help="PG_USER exported into generated commands.sh")
    parser.add_argument("--pg-password", help="PG_PASSWORD exported into generated commands.sh")
    parser.add_argument("--pg-dbname", help="PG_DBNAME exported into generated commands.sh")
    parser.add_argument("--execute", action="store_true", help="Run generated commands sequentially")
    return parser.parse_args(argv)


def load_trials(
    prompt_set: Path,
    *,
    default_jobs: Sequence[str] = (),
    benchmark_filter: Sequence[str] = (),
    only_trials: Sequence[str] = (),
) -> list[PromptTrial]:
    payload = tomllib.loads(prompt_set.read_text(encoding="utf-8"))
    raw_trials = payload.get(TRIAL_ROOT_KEY)
    if not isinstance(raw_trials, Mapping):
        raise ValueError(f"{prompt_set} must contain [trial.<name>] tables")

    allowed_trials = {name for name in only_trials}
    trials: list[PromptTrial] = []
    for name, raw_table in raw_trials.items():
        trial_name = safe_slug(str(name))
        if allowed_trials and trial_name not in allowed_trials and str(name) not in allowed_trials:
            continue
        if not isinstance(raw_table, Mapping):
            continue
        jobs = _resolve_list(raw_table.get("jobs"), default_jobs)
        if not jobs:
            raise ValueError(f"trial {trial_name!r} needs jobs=... or CLI --jobs")
        benchmarks = _resolve_benchmarks(raw_table.get("benchmarks"), benchmark_filter)
        if not benchmarks:
            if raw_table.get("benchmarks") is not None and benchmark_filter:
                continue
            raise ValueError(f"trial {trial_name!r} needs benchmarks=... or CLI --benchmarks")
        config = _trial_config(raw_table)
        if not any(config.values()):
            raise ValueError(f"trial {trial_name!r} does not set any supported prompt/config field")
        trials.append(
            PromptTrial(
                name=trial_name,
                jobs=jobs,
                benchmarks=benchmarks,
                config_benchmarks=_unique(_config_benchmark_name(item) for item in benchmarks),
                config=config,
                description=raw_table.get("description") if isinstance(raw_table.get("description"), str) else None,
            )
        )
    if not trials:
        raise ValueError("no prompt trials selected")
    return trials


def build_experiment(
    trials: Sequence[PromptTrial],
    *,
    output_root: Path,
    run_id: str | None = None,
    models: Sequence[str] = (),
    model_select: str = "all",
    model_regex: Sequence[str] = (),
    min_param_b: float | None = None,
    max_param_b: float | None = None,
    overwrite: bool = False,
    disable_checker: bool = False,
    skip_missing_dataset: bool = False,
    db_env: Mapping[str, str] | None = None,
) -> tuple[Path, list[ExperimentCommand]]:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = output_root / safe_slug(run_id)
    commands: list[ExperimentCommand] = []
    for trial in trials:
        trial_dir = experiment_root / trial.name
        config_root = trial_dir / "configs"
        _write_trial_configs(trial, config_root)
        command = _scheduler_command(
            trial,
            config_root=config_root,
            trial_dir=trial_dir,
            models=models,
            model_select=model_select,
            model_regex=model_regex,
            min_param_b=min_param_b,
            max_param_b=max_param_b,
            overwrite=overwrite,
            disable_checker=disable_checker,
            skip_missing_dataset=skip_missing_dataset,
        )
        commands.append(
            ExperimentCommand(
                trial=trial,
                trial_dir=trial_dir,
                config_root=config_root,
                command=tuple(command),
            )
        )
    _write_manifest(experiment_root, commands, db_env=db_env or {})
    _write_command_script(experiment_root, commands, db_env=db_env or {})
    return experiment_root, commands


def _resolve_list(raw: object, fallback: Sequence[str]) -> tuple[str, ...]:
    if raw is None:
        return tuple(str(item) for item in fallback)
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        return tuple(str(item) for item in raw)
    raise ValueError(f"expected string or list, got {type(raw).__name__}")


def _resolve_benchmarks(raw: object, cli_filter: Sequence[str]) -> tuple[str, ...]:
    trial_values = tuple(_normalize_benchmark_for_scheduler(item) for item in _resolve_list(raw, cli_filter))
    filter_values = tuple(_normalize_benchmark_for_scheduler(item) for item in cli_filter)
    if not raw or not filter_values:
        return _unique(trial_values)
    allowed = set(filter_values)
    return _unique(item for item in trial_values if item in allowed)


def _normalize_benchmark_for_scheduler(value: object) -> str:
    slug = canonical_slug(str(value))
    for suffix in ("__cot", "_cot", "_nocot"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    return slug


def _config_benchmark_name(value: str) -> str:
    slug = safe_slug(_normalize_benchmark_for_scheduler(value)).lower()
    if (REPO_ROOT / "configs" / f"{slug}.toml").exists():
        return slug
    benchmark, _ = split_benchmark_and_split(slug)
    base = safe_slug(benchmark).lower()
    if (REPO_ROOT / "configs" / f"{base}.toml").exists():
        return base
    return slug


def _unique(values: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _trial_config(raw_table: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {"default": {}, "direct": {}, "cot": {}, "final": {}}
    for key, value in raw_table.items():
        if key in IGNORED_TRIAL_FIELDS:
            continue
        if key in DEFAULT_FIELDS:
            config["default"][key] = value
            continue
        section = _section_for_field(key)
        if section:
            config[section][key] = value
            continue
        raise ValueError(f"unsupported trial field {key!r}")
    return config


def _section_for_field(field: str) -> str | None:
    for section, fields in PROMPT_FIELDS_BY_SECTION.items():
        if field in fields:
            return section
    return None


def _write_trial_configs(trial: PromptTrial, config_root: Path) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    content = _render_config(trial.config)
    for benchmark in trial.config_benchmarks:
        path = config_root / f"{benchmark}.toml"
        path.write_text(content, encoding="utf-8")


def _render_config(config: Mapping[str, Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for section in ("default", "direct", "cot", "final"):
        values = config.get(section) or {}
        if not values:
            continue
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ValueError(f"unsupported TOML value type: {type(value).__name__}")


def _scheduler_command(
    trial: PromptTrial,
    *,
    config_root: Path,
    trial_dir: Path,
    models: Sequence[str],
    model_select: str,
    model_regex: Sequence[str],
    min_param_b: float | None,
    max_param_b: float | None,
    overwrite: bool,
    disable_checker: bool,
    skip_missing_dataset: bool,
) -> list[str]:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "src.eval.scheduler.cli",
        "dispatch",
        "--benchmark-config-root",
        str(config_root.resolve()),
        "--log-dir",
        str((trial_dir / "scheduler_logs").resolve()),
        "--run-log-dir",
        str((trial_dir / "run_logs").resolve()),
        "--pid-dir",
        str((trial_dir / "pids").resolve()),
        "--model-select",
        model_select,
        "--only-jobs",
        *trial.jobs,
        "--only-datasets",
        *trial.benchmarks,
    ]
    if models:
        command.extend(["--models", *models])
    if model_regex:
        command.extend(["--model-regex", *model_regex])
    if min_param_b is not None:
        command.extend(["--min-param-b", str(min_param_b)])
    if max_param_b is not None:
        command.extend(["--max-param-b", str(max_param_b)])
    if overwrite:
        command.append("--overwrite")
    if disable_checker:
        command.append("--disable-checker")
    if skip_missing_dataset:
        command.append("--skip-missing-dataset")
    return command


def _write_manifest(
    experiment_root: Path,
    commands: Sequence[ExperimentCommand],
    *,
    db_env: Mapping[str, str],
) -> None:
    experiment_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_root": str(experiment_root.resolve()),
        "commands_sh": str((experiment_root / "commands.sh").resolve()),
        "db_env": dict(db_env),
        "trials": [
            {
                "name": item.trial.name,
                "description": item.trial.description,
                "jobs": list(item.trial.jobs),
                "benchmarks": list(item.trial.benchmarks),
                "config_benchmarks": list(item.trial.config_benchmarks),
                "config_root": str(item.config_root.resolve()),
                "command": list(item.command),
            }
            for item in commands
        ],
    }
    (experiment_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_command_script(
    experiment_root: Path,
    commands: Sequence[ExperimentCommand],
    *,
    db_env: Mapping[str, str],
) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if db_env:
        lines.append("# Isolate prompt-ablation writes from the main score database.")
        for key, value in db_env.items():
            lines.append(f"export {key}={shlex.quote(value)}")
        lines.append("")
    for item in commands:
        lines.append(f"# trial: {item.trial.name}")
        lines.append(" ".join(shlex.quote(part) for part in item.command))
        lines.append("")
    script_path = experiment_root / "commands.sh"
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)


def _db_env_from_args(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "PG_HOST": args.pg_host,
        "PG_PORT": args.pg_port,
        "PG_USER": args.pg_user,
        "PG_PASSWORD": args.pg_password,
        "PG_DBNAME": args.pg_dbname,
    }
    return {key: str(value) for key, value in values.items() if value is not None}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    db_env = _db_env_from_args(args)
    trials = load_trials(
        Path(args.prompt_set),
        default_jobs=args.jobs or (),
        benchmark_filter=args.benchmarks or (),
        only_trials=args.only_trials or (),
    )
    experiment_root, commands = build_experiment(
        trials,
        output_root=Path(args.output_root),
        run_id=args.run_id,
        models=args.models,
        model_select=args.model_select,
        model_regex=args.model_regex or (),
        min_param_b=args.min_param_b,
        max_param_b=args.max_param_b,
        overwrite=args.overwrite,
        disable_checker=args.disable_checker,
        skip_missing_dataset=args.skip_missing_dataset,
        db_env=db_env,
    )
    print(f"Wrote {len(commands)} prompt trial(s) to {experiment_root}")
    print(f"Commands: {experiment_root / 'commands.sh'}")
    if db_env:
        print("DB env: " + ", ".join(f"{key}={value}" for key, value in db_env.items()))
    if args.execute:
        env = {**os.environ, **db_env}
        for item in commands:
            print(f"Running trial {item.trial.name}")
            subprocess.run(item.command, cwd=REPO_ROOT, check=True, env=env)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
