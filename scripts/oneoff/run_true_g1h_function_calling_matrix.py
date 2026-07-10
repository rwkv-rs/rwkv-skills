#!/usr/bin/env python3
from __future__ import annotations

"""Run the true-g1h function-calling matrix with per-model concurrency.

This helper is intentionally separate from the older g1f/g1g matrix runner so
the true g1h endpoints can use different base URLs and worker budgets.
"""

import argparse
import concurrent.futures
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark_registry import resolve_benchmark_metadata
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import make_dataset_slug


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    base_url: str
    batch_size: int
    infer_workers: int
    jobs: int
    sample_workers: int


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    split: str
    job: str
    kind: str
    batch: bool = True
    needs_checker: bool = False
    extra_args: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        return make_dataset_slug(self.name, self.split)


DEFAULT_MODELS: tuple[str, ...] = (
    "name=rwkv7-g1h-1.5b-20260710-ctx10240,base_url=http://127.0.0.1:29610,batch=256,workers=256,jobs=4,sample_workers=24",
    "name=rwkv7-g1h-2.9b-20260710-ctx10240,base_url=http://127.0.0.1:29611,batch=224,workers=224,jobs=4,sample_workers=24",
    "name=rwkv7-g1h-7.2b-20260710-ctx10240,base_url=http://127.0.0.1:29272,batch=160,workers=160,jobs=3,sample_workers=16",
    "name=rwkv7-g1h-13.3b-20260710-ctx10240,base_url=http://127.0.0.1:29313,batch=128,workers=128,jobs=2,sample_workers=12",
)


DEFAULT_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("bfcl_v3", "test", "function_bfcl_v3", "bfcl_v3"),
    BenchmarkSpec("bfcl_simple_python", "test", "function_bfcl_ast", "bfcl_ast"),
    BenchmarkSpec("bfcl_multiple", "test", "function_bfcl_ast", "bfcl_ast"),
    BenchmarkSpec("bfcl_exec_simple_ast", "test", "function_bfcl_ast", "bfcl_ast"),
    BenchmarkSpec("bfcl_exec_multiple_ast", "test", "function_bfcl_ast", "bfcl_ast"),
    BenchmarkSpec("bfcl_exec_simple", "test", "function_bfcl_exec", "bfcl_exec"),
    BenchmarkSpec("bfcl_exec_multiple", "test", "function_bfcl_exec", "bfcl_exec"),
    BenchmarkSpec("bfcl_exec_parallel", "test", "function_bfcl_exec", "bfcl_exec"),
    BenchmarkSpec("bfcl_exec_parallel_multiple", "test", "function_bfcl_exec", "bfcl_exec"),
    BenchmarkSpec(
        "complexfuncbench_official",
        "test",
        "function_complexfuncbench",
        "complexfuncbench",
        extra_args=("--complexfuncbench-offline-compare",),
    ),
    BenchmarkSpec(
        "complexfuncbench_subset",
        "test",
        "function_complexfuncbench",
        "complexfuncbench",
        extra_args=("--complexfuncbench-offline-compare",),
    ),
    BenchmarkSpec("toolalpaca_eval_simulated", "test", "function_toolalpaca", "toolalpaca"),
    BenchmarkSpec("toolalpaca_eval_real", "test", "function_toolalpaca", "toolalpaca"),
    BenchmarkSpec("tau_bench_airline", "test", "function_tau_bench", "tau_bench"),
    BenchmarkSpec("tau_bench_retail", "test", "function_tau_bench", "tau_bench"),
    BenchmarkSpec("tau_bench_telecom", "test", "function_tau_bench", "tau_bench"),
    BenchmarkSpec("tau2_bench_airline", "base", "function_tau2_bench", "tau2_bench"),
    BenchmarkSpec("tau2_bench_retail", "base", "function_tau2_bench", "tau2_bench"),
    BenchmarkSpec("tau2_bench_telecom", "base", "function_tau2_bench", "tau2_bench"),
    BenchmarkSpec("tau3_bench_mock", "base", "function_tau3_bench", "tau3_bench"),
    BenchmarkSpec("tau3_bench_mock_long_context", "base", "function_tau3_bench", "tau3_bench"),
    BenchmarkSpec("tau3_bench_airline", "base", "function_tau3_bench", "tau3_bench"),
    BenchmarkSpec("tau3_bench_retail", "base", "function_tau3_bench", "tau3_bench"),
    BenchmarkSpec("tau3_bench_telecom", "base", "function_tau3_bench", "tau3_bench"),
    BenchmarkSpec("tau3_bench_banking_knowledge", "base", "function_tau3_bench", "tau3_bench"),
    BenchmarkSpec("mcp_bench", "test", "function_mcp_bench", "mcp_bench", batch=False, needs_checker=True),
    BenchmarkSpec("mcp_bench_single", "test", "function_mcp_bench", "mcp_bench", batch=False, needs_checker=True),
    BenchmarkSpec("mcp_bench_multi_2server", "test", "function_mcp_bench", "mcp_bench", batch=False, needs_checker=True),
    BenchmarkSpec("mcp_bench_multi_3server", "test", "function_mcp_bench", "mcp_bench", batch=False, needs_checker=True),
    BenchmarkSpec("apibank_l1", "test", "function_api_bank", "api_bank"),
    BenchmarkSpec("apibank_l2", "test", "function_api_bank", "api_bank"),
    BenchmarkSpec("apibank_level1", "test", "function_api_bank", "api_bank"),
    BenchmarkSpec("apibank_level2", "test", "function_api_bank", "api_bank"),
    BenchmarkSpec("longbench", "test", "function_longbench", "longbench"),
    BenchmarkSpec("longbench_qa", "test", "function_longbench", "longbench"),
    BenchmarkSpec("longbench_qa_balanced", "test", "function_longbench", "longbench"),
    BenchmarkSpec("longcodeqa", "test", "function_longcodebench", "longcodebench"),
)


AGENTBENCH_BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("agentbench_db", "test", "function_agentbench", "agentbench", batch=False),
    BenchmarkSpec("agentbench_kg", "test", "function_agentbench", "agentbench", batch=False),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", dest="models", help="Model spec: name=...,base_url=...,batch=...,workers=...,jobs=...")
    parser.add_argument("--benchmark", action="append", dest="benchmarks", help="Benchmark name subset. Repeatable.")
    parser.add_argument("--include-agentbench", action="store_true", help="Also run AgentBench controller-backed tasks.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--log-dir", help="Default: results/logs/true_g1h_function_calling_<stamp>")
    parser.add_argument("--stamp", default=os.getenv("STAMP") or time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--run-mode", default=os.getenv("RUN_MODE", "fresh"), choices=("auto", "fresh", "rerun", "new", "resume"))
    parser.add_argument("--api-key", default=os.getenv("INFER_API_KEY", "rwkv-skills"))
    parser.add_argument("--infer-timeout-s", type=float, default=float(os.getenv("INFER_TIMEOUT_S", "900")))
    parser.add_argument("--sample-cap", type=int, default=int(os.getenv("SAMPLE_CAP", "500")), help="Randomly sample datasets above this many rows. 0 disables.")
    parser.add_argument("--sample-seed", type=int, default=int(os.getenv("SAMPLE_SEED", "20260710")))
    parser.add_argument("--db-write-queue", type=int, default=int(os.getenv("DB_WRITE_QUEUE", "128")))
    parser.add_argument("--judge-max-workers", type=int, default=int(os.getenv("JUDGE_MAX_WORKERS_OVERRIDE", "16")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else repo_root / "results" / "logs" / f"true_g1h_function_calling_{args.stamp}"
    config_dir = log_dir / "configs"
    data_dir = log_dir / "sampled_data"
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    models = tuple(parse_model_spec(raw) for raw in (args.models or DEFAULT_MODELS))
    benchmarks = select_benchmarks(args)
    write_manifest(log_dir / "matrix.json", models=models, benchmarks=benchmarks, args=args)

    print(f"matrix_start models={len(models)} benchmarks={len(benchmarks)} log_dir={log_dir}", flush=True)
    results: list[tuple[str, str, int]] = []
    pending_by_model = {model.name: list(benchmarks) for model in models}
    with concurrent.futures.ThreadPoolExecutor(max_workers=sum(max(1, model.jobs) for model in models)) as pool:
        future_to_pair: dict[concurrent.futures.Future[int], tuple[ModelSpec, BenchmarkSpec]] = {}
        models_by_name = {model.name: model for model in models}

        def submit_next(model: ModelSpec) -> None:
            pending = pending_by_model[model.name]
            if not pending:
                return
            spec = pending.pop(0)
            future = pool.submit(
                run_one,
                args,
                repo_root,
                log_dir,
                config_dir,
                data_dir,
                model,
                spec,
            )
            future_to_pair[future] = (model, spec)

        for model in models:
            for _ in range(max(1, model.jobs)):
                submit_next(model)

        while future_to_pair:
            done, _ = concurrent.futures.wait(
                future_to_pair,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                model, spec = future_to_pair.pop(future)
                try:
                    rc = int(future.result())
                except BaseException as exc:  # noqa: BLE001
                    rc = 99
                    print(f"run_exception model={model.name} benchmark={spec.name} error={exc}", flush=True)
                results.append((model.name, spec.name, rc))
                status = "ok" if rc == 0 else f"failed:{rc}"
                print(f"run_result model={model.name} benchmark={spec.name} status={status}", flush=True)
                submit_next(models_by_name[model.name])
    failed = [(model, bench, rc) for model, bench, rc in results if rc != 0]
    write_jsonl(log_dir / "results.jsonl", ({"model": m, "benchmark": b, "returncode": rc} for m, b, rc in results))
    print(f"matrix_done total={len(results)} failed={len(failed)}", flush=True)
    return 1 if failed and not args.keep_going else 0


def run_one_guarded(
    semaphore: threading.Semaphore,
    args: argparse.Namespace,
    repo_root: Path,
    log_dir: Path,
    config_dir: Path,
    data_dir: Path,
    model: ModelSpec,
    spec: BenchmarkSpec,
) -> int:
    with semaphore:
        return run_one(args, repo_root, log_dir, config_dir, data_dir, model, spec)


def parse_model_spec(raw: str) -> ModelSpec:
    parts: dict[str, str] = {}
    for item in str(raw).split(","):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"invalid model spec item {item!r}; expected key=value")
        parts[key.strip()] = value.strip()
    name = parts.get("name", "")
    base_url = parts.get("base_url", "")
    if not name or not base_url:
        raise ValueError(f"model spec requires name and base_url: {raw!r}")
    return ModelSpec(
        name=name,
        base_url=base_url.rstrip("/"),
        batch_size=positive_int(parts.get("batch"), 64),
        infer_workers=positive_int(parts.get("workers"), 64),
        jobs=positive_int(parts.get("jobs"), 1),
        sample_workers=positive_int(parts.get("sample_workers"), 8),
    )


def select_benchmarks(args: argparse.Namespace) -> tuple[BenchmarkSpec, ...]:
    specs = list(DEFAULT_BENCHMARKS)
    if args.include_agentbench:
        specs.extend(AGENTBENCH_BENCHMARKS)
    requested = {item.strip() for item in (args.benchmarks or ()) if item.strip()}
    if requested:
        specs = [spec for spec in specs if spec.name in requested]
        missing = requested.difference(spec.name for spec in specs)
        if missing:
            raise ValueError(f"unknown benchmark(s): {', '.join(sorted(missing))}")
    return tuple(specs)


def run_one(
    args: argparse.Namespace,
    repo_root: Path,
    log_dir: Path,
    config_dir: Path,
    data_dir: Path,
    model: ModelSpec,
    spec: BenchmarkSpec,
) -> int:
    run_name = f"{safe_name(model.name)}__{safe_name(spec.name)}"
    model_log = log_dir / f"{run_name}.log"
    dataset_path = prepare_dataset_path(spec, data_dir=data_dir, sample_cap=int(args.sample_cap), seed=int(args.sample_seed))
    config_path = config_dir / f"{run_name}.toml"
    config_path.write_text(build_config_text(args, model, spec, dataset_path), encoding="utf-8")
    command = [str(args.python), "-m", "src.main", "--config", str(config_path)]
    env = os.environ.copy()
    env["RWKV_TASK_DESC"] = f"true_g1h_function_calling model={model.name} benchmark={spec.name} stamp={args.stamp}"
    env["RWKV_OMIT_PENALTY_DECAY"] = "1"
    env["RWKV_FUNCTION_CALLING_ALPHA_DECAY"] = "0"
    env["JUDGE_MAX_WORKERS"] = str(int(args.judge_max_workers))
    env.setdefault("JUDGE_TIMEOUT_S", "180")
    if not spec.needs_checker:
        env["RWKV_SKILLS_DISABLE_CHECKER"] = "1"
    print(f"run_start model={model.name} benchmark={spec.name} config={config_path} log={model_log}", flush=True)
    if args.dry_run:
        print("$ " + " ".join(command), flush=True)
        return 0
    with model_log.open("w", encoding="utf-8") as fh:
        result = subprocess.run(command, cwd=repo_root, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    return int(result.returncode)


def prepare_dataset_path(spec: BenchmarkSpec, *, data_dir: Path, sample_cap: int, seed: int) -> Path:
    source = resolve_or_prepare_dataset(spec.slug, verbose=False, record_stats=False)
    if sample_cap <= 0:
        return source
    lines = source.read_text(encoding="utf-8").splitlines()
    non_empty = [line for line in lines if line.strip()]
    if len(non_empty) <= sample_cap:
        return source
    target = data_dir / spec.name / f"{spec.split}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and count_lines(target) == sample_cap:
        return target
    rng = random.Random(seed + stable_int(spec.slug))
    selected = sorted(rng.sample(range(len(non_empty)), sample_cap))
    target.write_text("\n".join(non_empty[index] for index in selected) + "\n", encoding="utf-8")
    manifest = {
        "source": str(source),
        "target": str(target),
        "source_rows": len(non_empty),
        "sample_rows": sample_cap,
        "seed": seed,
        "slug": spec.slug,
        "indices_sha256": hashlib.sha256(",".join(map(str, selected)).encode("utf-8")).hexdigest(),
    }
    target.with_suffix(target.suffix + ".sample.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def build_config_text(args: argparse.Namespace, model: ModelSpec, spec: BenchmarkSpec, dataset_path: Path) -> str:
    runner = runner_config(args, model, spec)
    payload: dict[str, Mapping[str, Any]] = {
        "run": {
            "run_mode": args.run_mode,
            **({"batch_size": model.batch_size} if spec.batch else {}),
        },
        "dataset": {
            "name": spec.name,
            "split": spec.split,
            "path": str(dataset_path),
        },
        "model": {
            "infer_base_url": model.base_url,
            "infer_model": model.name,
            "infer_api_key": args.api_key,
            "infer_timeout_s": float(args.infer_timeout_s),
            "infer_max_workers": model.infer_workers,
            "infer_protocol": "completions",
            "infer_seed_policy": "omit",
        },
        "runner": runner,
    }
    return render_toml(payload)


def runner_config(args: argparse.Namespace, model: ModelSpec, spec: BenchmarkSpec) -> dict[str, Any]:
    extra_args = ["--sample-workers", str(model.sample_workers), *spec.extra_args]
    config: dict[str, Any] = {
        "benchmark_kind": spec.kind,
        "prompt_style": "rwkv_official_json",
        "tool_catalog_format": "json",
        "tool_call_io": "rwkv-json",
        "avg_ks": [1],
        "db_write_queue": int(args.db_write_queue),
        "db_close_timeout_s": 60.0,
        "history_max_chars": 24000,
        "prompt_max_chars": 28000,
        "tool_router_mode": "lexical",
        "tool_router_max_tools": 32,
        "tool_router_trigger_tool_count": 16,
        "tool_router_trigger_catalog_chars": 6000,
        "tool_router_context_chars": 6000,
        "tool_router_max_tokens": 192,
        "tool_router_description_chars": 512,
        "max_steps": 200,
        "max_tool_errors": 10,
        "cot_max_tokens": 2048,
        "decision_max_tokens": 1024,
        "final_max_tokens": 3072,
        "planning_max_tokens": 2048,
        "max_rounds": 20,
        "judge_max_workers": int(args.judge_max_workers),
        "extra_args": extra_args,
    }
    if spec.kind == "bfcl_v3":
        config.update(candidate_router_defaults(mode="parallel"))
        config["history_max_chars"] = 24000
        config["prompt_max_chars"] = 28000
    elif spec.job == "function_bfcl_ast":
        config.update(candidate_router_defaults(mode="auto"))
    elif spec.kind in {"longbench", "longcodebench"}:
        config["answer_max_tokens"] = 512
        config["history_max_chars"] = 24000
        config["prompt_max_chars"] = 28000
    elif spec.kind.startswith("tau"):
        config["history_max_chars"] = 16000 if spec.kind != "tau3_bench" else 12000
        config["prompt_max_chars"] = 28000
    return config


def candidate_router_defaults(*, mode: str) -> dict[str, Any]:
    batch_size = positive_int(os.getenv("RWKV_CANDIDATE_ROUTER_BATCH_SIZE"), 16)
    max_candidates = positive_int(os.getenv("RWKV_CANDIDATE_ROUTER_MAX_CANDIDATES"), 8)
    return {
        "candidate_router_mode": mode,
        "candidate_router_chunk_tools": 2,
        "candidate_router_batch_size": batch_size,
        "candidate_router_context_chars": 6000,
        "candidate_router_prompt_max_chars": 8192,
        "candidate_router_candidate_max_tokens": 192,
        "candidate_router_aggregate_max_tokens": 192,
        "candidate_router_max_candidates": max_candidates,
        "candidate_router_tool_schema_mode": "compact",
        "candidate_router_evidence_chars": 1200,
        "candidate_router_policy_chars": 2000,
    }


def render_toml(payload: Mapping[str, Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for section, values in payload.items():
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {toml_value(value)}")
    return "\n".join(lines).rstrip() + "\n"


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def positive_int(raw: str | None, default: int) -> int:
    if raw is None or raw == "":
        return default
    return max(1, int(raw))


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in text)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def write_manifest(path: Path, *, models: Sequence[ModelSpec], benchmarks: Sequence[BenchmarkSpec], args: argparse.Namespace) -> None:
    payload = {
        "models": [asdict(model) for model in models],
        "benchmarks": [asdict(spec) for spec in benchmarks],
        "sample_cap": int(args.sample_cap),
        "sample_seed": int(args.sample_seed),
        "run_mode": str(args.run_mode),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
