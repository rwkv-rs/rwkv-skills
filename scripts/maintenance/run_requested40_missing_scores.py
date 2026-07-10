#!/usr/bin/env python3
from __future__ import annotations

"""Run missing requested-40 benchmark/model pairs sequentially.

This is a conservative queue helper: it uses the DB audit matrix as the source
of truth, skips existing scores, skips missing datasets, skips running tasks by
default, writes a temporary src.main TOML config per pair, and logs each run.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.maintenance.audit_agent_score_matrix import DEFAULT_MODELS, classify_pairs, default_dataset_specs
from src.eval.benchmark_sources import REQUESTED_BENCHMARKS_BY_NAME

DEFAULT_BASE_URL = "http://127.0.0.1:19183/v1"
DEFAULT_API_KEY = "rwkv-skills"
RUNNABLE_STATES = {"no_task", "failed_zero_completion", "failed_with_completions"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--stamp", default=os.getenv("STAMP") or time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--log-dir", help="Default: <repo>/results/logs/requested40_missing_<stamp>")
    parser.add_argument("--wait-screen", default=os.getenv("WAIT_SCREEN", ""))
    parser.add_argument("--wait-seconds", type=int, default=int(os.getenv("WAIT_SECONDS", "300")))
    parser.add_argument("--domain", action="append", help="Limit to domain. Repeatable.")
    parser.add_argument("--dataset", action="append", help="Limit to requested benchmark name. Repeatable.")
    parser.add_argument("--model", action="append", help="Limit to model. Repeatable.")
    parser.add_argument("--state", action="append", choices=sorted(RUNNABLE_STATES), help="Runnable state to include.")
    parser.add_argument("--include-running", action="store_true", help="Also rerun running_no_score pairs.")
    parser.add_argument("--run-mode", default=os.getenv("RUN_MODE", "fresh"), choices=("auto", "fresh", "rerun", "new"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "1")))
    parser.add_argument("--infer-max-workers", type=int, default=int(os.getenv("INFER_MAX_WORKERS", "1")))
    parser.add_argument("--infer-timeout-s", type=float, default=float(os.getenv("INFER_TIMEOUT_S", "900")))
    parser.add_argument("--judge-max-workers", type=int, default=int(os.getenv("JUDGE_MAX_WORKERS", "2")))
    parser.add_argument("--max-runs", type=int, default=int(os.getenv("MAX_RUNS", "0")))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    opts = parse_args()
    repo_root = Path(opts.repo_root).resolve()
    log_dir = Path(opts.log_dir) if opts.log_dir else repo_root / "results" / "logs" / f"requested40_missing_{opts.stamp}"
    config_dir = log_dir / "configs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.log"

    def log(message: str) -> None:
        line = f"{time.strftime('%F %T %Z')} {message}"
        print(line, flush=True)
        with summary_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"requested40_missing_start repo={repo_root} base_url={opts.base_url} dry_run={opts.dry_run}")
    wait_for_screen(opts.wait_screen, opts.wait_seconds, log=log)

    models = tuple(opts.model or DEFAULT_MODELS)
    states = classify_pairs(repo_root, default_dataset_specs("requested40"), models)
    runnable_states = set(opts.state or RUNNABLE_STATES)
    if opts.include_running:
        runnable_states.add("running_no_score")
    selected = [
        state
        for state in states
        if state.state in runnable_states
        and state.data_rows is not None
        and (not opts.domain or state.domain in set(opts.domain))
        and (not opts.dataset or state.dataset in set(opts.dataset))
    ]

    log(f"selected_pairs count={len(selected)} states={','.join(sorted(runnable_states))}")
    runs_started = 0
    for state in selected:
        if opts.max_runs and runs_started >= opts.max_runs:
            log(f"max_runs_reached limit={opts.max_runs}")
            break
        source = REQUESTED_BENCHMARKS_BY_NAME.get(state.dataset)
        if source is None:
            log(f"skip_unknown_requested_source dataset={state.dataset} model={state.model}")
            continue
        config_path = config_dir / f"{safe_name(state.dataset)}__{safe_name(state.model)}.toml"
        model_log = log_dir / f"{safe_name(state.dataset)}__{safe_name(state.model)}.log"
        config_path.write_text(build_config_text(state, source.scheduler_job, opts), encoding="utf-8")
        command = [".venv/bin/python", "-m", "src.main", "--config", str(config_path)]
        log(
            "run_start "
            f"dataset={state.dataset} split={state.split} domain={state.domain} "
            f"model={state.model} state={state.state} job={source.scheduler_job} log={model_log}"
        )
        if opts.dry_run:
            dry_command = command + ["--dry-run"]
            with model_log.open("w", encoding="utf-8") as fh:
                result = subprocess.run(dry_command, cwd=repo_root, stdout=fh, stderr=subprocess.STDOUT, check=False)
            log(f"dry_run_done rc={result.returncode} dataset={state.dataset} model={state.model}")
            runs_started += 1
            continue
        env = os.environ.copy()
        env["RWKV_TASK_DESC"] = f"requested40_missing_scores dataset={state.dataset} model={state.model} stamp={opts.stamp}"
        env["RWKV_SKILLS_LOG_PATH"] = str(model_log)
        with model_log.open("w", encoding="utf-8") as fh:
            result = subprocess.run(command, cwd=repo_root, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
        runs_started += 1
        if result.returncode == 0:
            log(f"run_done dataset={state.dataset} model={state.model}")
        else:
            log(f"run_failed rc={result.returncode} dataset={state.dataset} model={state.model}")
    log(f"requested40_missing_done runs_started={runs_started}")
    return 0


def wait_for_screen(screen_name: str, wait_seconds: int, *, log) -> None:
    if not screen_name:
        return
    while True:
        result = subprocess.run(["screen", "-ls"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if screen_name not in result.stdout:
            log(f"wait_screen_done screen={screen_name}")
            return
        log(f"waiting_for_screen screen={screen_name} sleep_s={wait_seconds}")
        time.sleep(max(1, wait_seconds))


def build_config_text(state: Any, scheduler_job: str, opts: argparse.Namespace) -> str:
    dataset_name = run_dataset_name(state)
    runner_fields = runner_config_fields(state, scheduler_job, opts)
    lines = [
        "[run]",
        f"run_mode = {toml_str(opts.run_mode)}",
        f"job = {toml_str(scheduler_job)}",
        f"batch_size = {int(opts.batch_size)}",
        "",
        "[dataset]",
        f"name = {toml_str(dataset_name)}",
        f"split = {toml_str(state.split)}",
        "",
        "[model]",
        f"infer_base_url = {toml_str(opts.base_url)}",
        f"infer_model = {toml_str(state.model)}",
        f"infer_api_key = {toml_str(opts.api_key)}",
        f"infer_timeout_s = {float(opts.infer_timeout_s)}",
        f"infer_max_workers = {int(opts.infer_max_workers)}",
        'infer_protocol = "completions"',
        'infer_seed_policy = "omit"',
        "",
        "[runner]",
    ]
    for key, value in runner_fields.items():
        lines.append(f"{key} = {toml_value(value)}")
    lines.append("")
    return "\n".join(lines)


def run_dataset_name(state: Any) -> str:
    if state.score_benchmark:
        return str(state.score_benchmark)
    if state.latest_benchmark:
        return str(state.latest_benchmark)
    if state.dataset == "gpqa_diamond":
        return "gpqa"
    return str(state.benchmark)


def runner_config_fields(state: Any, scheduler_job: str, opts: argparse.Namespace) -> dict[str, Any]:
    fields: dict[str, Any] = {"db_write_queue": 8, "db_close_timeout_s": 120.0}
    if scheduler_job == "free_response_judge":
        fields.update(
            {
                "judge_mode": "llm",
                "judge_max_workers": int(opts.judge_max_workers),
                "max_tokens": 4096,
                "final_max_tokens": 2048,
            }
        )
    elif scheduler_job == "free_response":
        fields.update({"judge_mode": "exact", "max_tokens": 4096, "final_max_tokens": 2048})
    elif scheduler_job == "code_swe_bench":
        fields.update(
            {
                "benchmark_kind": "swe_bench",
                "max_tokens": 4096,
                "temperature": 0.0,
                "top_p": 1.0,
                "eval_timeout": 900.0,
                "eval_workers": 1,
                "long_doc_mode": "lexical",
                "swebench_max_prompt_chars": 24000,
            }
        )
    elif scheduler_job == "function_agent_loop":
        fields.update(
            {
                "benchmark_kind": "agent_loop",
                "history_max_chars": 24000,
                "long_doc_mode": "lexical",
                "candidate_router_mode": "auto",
                "max_steps": 16,
                "max_tool_errors": 5,
                "decision_max_tokens": 1024,
            }
        )
    elif scheduler_job == "function_browsecomp":
        fields.update(
            {
                "benchmark_kind": "browsecomp",
                "cot_max_tokens": 2048,
                "answer_max_tokens": 1024,
                "judge_max_workers": int(opts.judge_max_workers),
            }
        )
    elif scheduler_job.startswith("multi_choice"):
        fields.update({"cot_mode": "cot" if scheduler_job.endswith("_cot") else "no_cot"})
    return {key: value for key, value in fields.items() if value not in (None, "")}


def toml_str(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return toml_str(str(value))


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
