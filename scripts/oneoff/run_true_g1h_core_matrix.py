#!/usr/bin/env python3
from __future__ import annotations

"""Run true-g1h non-function-calling scores.

The selector is registry-driven and mode-aware.  It skips pairs that already
have a non-naive score for the same model/benchmark/split/cot_mode, and skips
currently running pairs.  SWE-bench variants are skipped by default because
they require the official Docker harness.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import psycopg
from psycopg.rows import dict_row

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark_registry import BenchmarkField, CoTMode, get_benchmarks_with_field
from src.eval.env_config import load_env_file
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_utils import make_dataset_slug, split_benchmark_and_split


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    base_url: str
    batch_size: int
    infer_workers: int


@dataclass(frozen=True, slots=True)
class RunSpec:
    field: BenchmarkField
    benchmark: str
    dataset: str
    split: str
    job: str
    cot_mode: CoTMode
    data_path: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="name=...,base_url=...,batch=...,workers=...")
    parser.add_argument("--field", action="append", choices=("maths", "knowledge", "coding", "instruction_following"))
    parser.add_argument("--benchmark", action="append", help="Restrict benchmark name. Repeatable.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--stamp", default=os.getenv("STAMP") or time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--log-dir", help="Default: results/logs/true_g1h_core_<stamp>")
    parser.add_argument("--api-key", default=os.getenv("INFER_API_KEY", "rwkv-skills"))
    parser.add_argument("--infer-timeout-s", type=float, default=float(os.getenv("INFER_TIMEOUT_S", "900")))
    parser.add_argument("--judge-max-workers", type=int, default=int(os.getenv("JUDGE_MAX_WORKERS_OVERRIDE", "12")))
    parser.add_argument("--db-write-queue", type=int, default=int(os.getenv("DB_WRITE_QUEUE", "256")))
    parser.add_argument("--include-swebench", action="store_true")
    parser.add_argument("--run-mode", default=os.getenv("RUN_MODE", "fresh"), choices=("auto", "fresh", "rerun", "new", "resume"))
    parser.add_argument("--rerun-scored", action="store_true", help="Do not skip already-scored pairs; create rerun tasks.")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else repo_root / "results" / "logs" / f"true_g1h_core_{args.stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    models = tuple(parse_model_spec(raw) for raw in args.model)
    requested_fields = {BenchmarkField(item) for item in (args.field or ("maths", "knowledge", "coding", "instruction_following"))}
    requested_benchmarks = {item.strip() for item in (args.benchmark or ()) if item.strip()}

    load_env_file()
    with psycopg.connect(
        host=DEFAULT_DB_CONFIG.host,
        port=DEFAULT_DB_CONFIG.port,
        user=DEFAULT_DB_CONFIG.user,
        password=DEFAULT_DB_CONFIG.password,
        dbname=DEFAULT_DB_CONFIG.dbname,
        row_factory=dict_row,
    ) as conn:
        runs = build_run_specs(
            repo_root=repo_root,
            fields=requested_fields,
            requested_benchmarks=requested_benchmarks,
            include_swebench=bool(args.include_swebench),
        )
        print(f"core_matrix_start models={len(models)} runs={len(runs)} log_dir={log_dir}", flush=True)
        started = 0
        for model in models:
            for spec in runs:
                if args.max_runs and started >= args.max_runs:
                    print(f"max_runs_reached limit={args.max_runs}", flush=True)
                    return 0
                state = db_state(conn, model.name, spec)
                if state == "running" or (state == "scored" and not args.rerun_scored):
                    print(
                        f"skip_{state} model={model.name} field={spec.field.value} benchmark={spec.benchmark} "
                        f"split={spec.split} cot_mode={spec.cot_mode.value}",
                        flush=True,
                    )
                    continue
                rc = run_one(args, repo_root, log_dir, model, spec)
                started += 1
                print(
                    f"run_result rc={rc} model={model.name} field={spec.field.value} benchmark={spec.benchmark} "
                    f"split={spec.split} cot_mode={spec.cot_mode.value}",
                    flush=True,
                )
    print(f"core_matrix_done started={started}", flush=True)
    return 0


def parse_model_spec(raw: str) -> ModelSpec:
    parts: dict[str, str] = {}
    for item in str(raw).split(","):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"invalid model spec item {item!r}; expected key=value")
        parts[key.strip()] = value.strip()
    return ModelSpec(
        name=parts["name"],
        base_url=parts["base_url"].rstrip("/"),
        batch_size=positive_int(parts.get("batch"), 64),
        infer_workers=positive_int(parts.get("workers"), 64),
    )


def build_run_specs(
    *,
    repo_root: Path,
    fields: set[BenchmarkField],
    requested_benchmarks: set[str],
    include_swebench: bool,
) -> tuple[RunSpec, ...]:
    specs: list[RunSpec] = []
    for field in (
        BenchmarkField.MATHS,
        BenchmarkField.KNOWLEDGE,
        BenchmarkField.CODING,
        BenchmarkField.INSTRUCTION_FOLLOWING,
    ):
        if field not in fields:
            continue
        for meta in get_benchmarks_with_field(field):
            if requested_benchmarks and meta.name not in requested_benchmarks:
                continue
            job = primary_job(meta.scheduler_jobs)
            if not job:
                continue
            if job == "code_swe_bench" and not include_swebench:
                print(f"skip_swebench_harness_required benchmark={meta.name}", flush=True)
                continue
            data_path = resolve_data_path(meta.dataset, meta.default_split, repo_root=repo_root)
            if data_path is None:
                print(f"skip_missing_data benchmark={meta.name} dataset={meta.dataset} split={meta.default_split}", flush=True)
                continue
            for cot_mode in meta.cot_modes:
                specs.append(
                    RunSpec(
                        field=field,
                        benchmark=meta.name,
                        dataset=meta.dataset,
                        split=meta.default_split,
                        job=job_for_mode(job, cot_mode),
                        cot_mode=cot_mode,
                        data_path=data_path,
                    )
                )
    return tuple(specs)


def primary_job(jobs: tuple[str, ...]) -> str:
    for job in jobs:
        if not job.endswith("_naive"):
            return job
    return jobs[0] if jobs else ""


def job_for_mode(job: str, cot_mode: CoTMode) -> str:
    if job.startswith("multi_choice"):
        return "multi_choice_cot" if cot_mode is CoTMode.COT else "multi_choice_plain"
    return job


def resolve_data_path(dataset: str, split: str, *, repo_root: Path) -> Path | None:
    candidates = local_dataset_candidates(dataset, split, repo_root=repo_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def local_dataset_candidates(dataset: str, split: str, *, repo_root: Path) -> tuple[Path, ...]:
    slug = make_dataset_slug(dataset, split)
    names = (dataset, dataset.replace("_", "-"), slug, slug.replace("_", "-"))
    splits = (split, "test") if split != "test" else ("test",)
    paths: list[Path] = []
    for name in dict.fromkeys(names):
        for item_split in dict.fromkeys(splits):
            paths.append(repo_root / "data" / name / f"{item_split}.jsonl")
    return tuple(dict.fromkeys(paths))


def db_state(conn, model_name: str, spec: RunSpec) -> str:
    cot_db_value = "CoT" if spec.cot_mode is CoTMode.COT else "NoCoT"
    benchmark_name = benchmark_db_name(spec)
    score = conn.execute(
        """
        SELECT 1
        FROM scores s
        JOIN task t ON t.task_id = s.task_id
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        JOIN model m ON m.model_id = t.model_id
        WHERE m.model_name = %s
          AND b.benchmark_name = %s
          AND b.benchmark_split = %s
          AND coalesce(t.is_tmp, false) = false
          AND coalesce(t.sampling_config->>'cot_mode', '') = %s
          AND coalesce(t.sampling_config->>'prompt_profile', '') NOT IN ('naive')
        LIMIT 1
        """,
        (model_name, benchmark_name, spec.split, cot_db_value),
    ).fetchone()
    if score:
        return "scored"
    running = conn.execute(
        """
        SELECT 1
        FROM task t
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        JOIN model m ON m.model_id = t.model_id
        WHERE m.model_name = %s
          AND b.benchmark_name = %s
          AND b.benchmark_split = %s
          AND coalesce(t.is_tmp, false) = false
          AND t.status = 'Running'
          AND coalesce(t.sampling_config->>'cot_mode', '') = %s
        LIMIT 1
        """,
        (model_name, benchmark_name, spec.split, cot_db_value),
    ).fetchone()
    return "running" if running else "missing"


def benchmark_db_name(spec: RunSpec) -> str:
    """Return the exact DB benchmark identity implied by the dataset path.

    A benchmark display name and its materialized dataset name can differ
    (for example, ``gpqa_main`` uses ``gpqa`` with split ``main``).  Expanding
    spelling and dataset aliases here is unsafe: one historical score can then
    suppress a different registered pair.  Resolve only the canonical
    dataset-base/split pair that the runner itself will persist.
    """
    benchmark_name, dataset_split = split_benchmark_and_split(
        make_dataset_slug(spec.dataset, spec.split)
    )
    if dataset_split != spec.split:
        raise ValueError(
            f"dataset identity split mismatch for {spec.benchmark}: "
            f"expected {spec.split!r}, got {dataset_split!r}"
        )
    return benchmark_name


def run_one(args: argparse.Namespace, repo_root: Path, log_dir: Path, model: ModelSpec, spec: RunSpec) -> int:
    safe = safe_name(f"{model.name}__{spec.field.value}__{spec.benchmark}__{spec.split}__{spec.cot_mode.value}")
    log_path = log_dir / f"{safe}.log"
    command = base_command(args, model, spec)
    print(
        f"run_start model={model.name} field={spec.field.value} benchmark={spec.benchmark} "
        f"split={spec.split} cot_mode={spec.cot_mode.value} log={log_path}",
        flush=True,
    )
    print("$ " + " ".join(command), flush=True)
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env["RWKV_SKILLS_JOB_NAME"] = spec.job
    env["RWKV_TASK_DESC"] = f"true_g1h_core field={spec.field.value} benchmark={spec.benchmark} model={model.name} stamp={args.stamp}"
    env["RWKV_SKILLS_LOG_PATH"] = str(log_path)
    env["RWKV_OMIT_PENALTY_DECAY"] = "1"
    env["RWKV_FUNCTION_CALLING_ALPHA_DECAY"] = "0"
    effective_run_mode = "rerun" if args.rerun_scored else str(args.run_mode)
    env["RWKV_EVAL_RUN_MODE"] = effective_run_mode
    env["RWKV_SCHEDULER_OVERWRITE"] = "1" if effective_run_mode == "rerun" else "0"
    with log_path.open("w", encoding="utf-8") as fh:
        result = subprocess.run(command, cwd=repo_root, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return int(result.returncode)


def base_command(args: argparse.Namespace, model: ModelSpec, spec: RunSpec) -> list[str]:
    module = {
        BenchmarkField.MATHS: "src.eval.tasks.maths.runner",
        BenchmarkField.KNOWLEDGE: "src.eval.tasks.knowledge.runner",
        BenchmarkField.CODING: "src.eval.tasks.coding.runner",
        BenchmarkField.INSTRUCTION_FOLLOWING: "src.eval.tasks.instruction_following.runner",
    }[spec.field]
    command = [
        str(args.python),
        "-m",
        module,
        "--dataset",
        str(spec.data_path),
        "--batch-size",
        str(model.batch_size),
        "--infer-base-url",
        model.base_url,
        "--infer-model",
        model.name,
        "--infer-api-key",
        str(args.api_key),
        "--infer-protocol",
        "completions",
        "--infer-seed-policy",
        "omit",
        "--infer-timeout-s",
        str(float(args.infer_timeout_s)),
        "--infer-max-workers",
        str(model.infer_workers),
        "--db-write-queue",
        str(int(args.db_write_queue)),
    ]
    if spec.field is BenchmarkField.MATHS:
        command.extend(["--strategy-a-single-generation", "--db-close-timeout-s", "120"])
        if spec.job == "free_response_judge":
            command.extend(["--judge-mode", "llm", "--judge-max-workers", str(int(args.judge_max_workers))])
        else:
            command.extend(["--judge-mode", "exact"])
    elif spec.field is BenchmarkField.KNOWLEDGE:
        command.extend(["--cot-mode", spec.cot_mode.value, "--target-token-format", " <LETTER>"])
    elif spec.field is BenchmarkField.CODING:
        command.extend(["--eval-timeout", "30", "--eval-workers", "8"])
        if spec.job == "code_human_eval":
            command.extend(["--benchmark-kind", "human_eval"])
        elif spec.job == "code_mbpp":
            command.extend(["--benchmark-kind", "mbpp", "--cot-mode", spec.cot_mode.value])
        elif spec.job == "code_livecodebench":
            command.extend(["--benchmark-kind", "livecodebench"])
    elif spec.field is BenchmarkField.INSTRUCTION_FOLLOWING:
        command.extend(["--prompt-profile", "normal"])
    return command


def positive_int(raw: str | None, default: int) -> int:
    try:
        return max(1, int(raw)) if raw is not None else int(default)
    except (TypeError, ValueError):
        return int(default)


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
