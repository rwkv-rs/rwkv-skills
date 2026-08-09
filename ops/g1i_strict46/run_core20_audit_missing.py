#!/usr/bin/env python3
"""Run only missing or invalid cells from the G1g/G1i Core20 audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from urllib.request import Request, urlopen

from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import JOB_CATALOGUE


REPO = Path(__file__).resolve().parents[2]
AUDITOR = REPO / "ops/g1i_strict46/audit_core20_dual.py"
SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RECOVERABLE_STATES = {"missing", "invalid"}


def audit_cell_to_dataset(cell: str) -> str:
    benchmark, separator, split = str(cell).partition("__")
    if not separator or not benchmark or not split:
        raise ValueError(f"invalid Core20 benchmark key: {cell!r}")
    if benchmark == "gpqa":
        if split not in {"diamond", "main", "extended"}:
            raise ValueError(f"unsupported GPQA split: {cell!r}")
        return f"gpqa_{split}"
    return benchmark


def _catalogue_slug(cell: str) -> str:
    benchmark, separator, split = str(cell).partition("__")
    if not separator or not benchmark or not split:
        raise ValueError(f"invalid Core20 benchmark key: {cell!r}")
    return canonical_slug(f"{benchmark}_{split}")


def job_for_cell(cell: dict[str, Any]) -> str:
    benchmark_key = str(cell.get("benchmark") or "")
    benchmark = benchmark_key.partition("__")[0]
    domain = str(cell.get("domain") or "")
    mode = str(cell.get("mode") or "")
    if mode not in {"CoT", "NoCoT"}:
        raise ValueError(f"invalid Core20 mode: {mode!r}")
    if domain == "knowledge":
        return "multi_choice_cot_naive" if mode == "CoT" else "multi_choice_plain_naive"
    if domain == "math":
        suffix = "_naive" if mode == "CoT" else "_plain_naive"
        candidates = (f"free_response{suffix}", f"free_response_judge{suffix}")
        slug = _catalogue_slug(benchmark_key)
        matching = [job for job in candidates if slug in JOB_CATALOGUE[job].dataset_slugs]
        if len(matching) != 1:
            raise ValueError(
                f"Core20 math cell must map to exactly one evaluator: "
                f"{benchmark_key} {mode} -> {matching}"
            )
        return matching[0]
    if domain == "coding":
        if mode != "NoCoT":
            raise ValueError(f"coding cell cannot use CoT: {benchmark_key}")
        if benchmark in {"human_eval", "human_eval_plus"}:
            return "code_human_eval_naive"
        if benchmark == "mbpp_plus":
            return "code_mbpp_naive"
        if benchmark == "livecodebench":
            return "code_livecodebench_plain_naive"
        raise ValueError(f"unsupported Core20 coding benchmark: {benchmark_key}")
    if domain == "instruction_following":
        if mode != "NoCoT":
            raise ValueError(f"instruction cell cannot use CoT: {benchmark_key}")
        return "instruction_following_naive"
    raise ValueError(f"unsupported Core20 domain: {domain!r}")


def recovery_plan(audit: dict[str, Any], model: str) -> dict[str, list[str]]:
    cells = audit.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Core20 audit has no cells list")
    model_cells = [cell for cell in cells if isinstance(cell, dict) and cell.get("model") == model]
    if not model_cells:
        raise ValueError(f"model is absent from Core20 audit: {model}")
    grouped: defaultdict[str, set[str]] = defaultdict(set)
    for cell in model_cells:
        if cell.get("state") not in RECOVERABLE_STATES:
            continue
        grouped[job_for_cell(cell)].add(audit_cell_to_dataset(str(cell["benchmark"])))
    return {job: sorted(datasets) for job, datasets in sorted(grouped.items())}


def verify_endpoint_model(base_url: str, expected_model: str) -> None:
    url = base_url.rstrip("/") + "/models"
    request = Request(url, headers={"Authorization": "Bearer rwkv-skills"})
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)
    identities = {
        str(row.get("id"))
        for row in payload.get("data", [])
        if isinstance(row, dict) and row.get("id")
    }
    if expected_model not in identities:
        raise RuntimeError(
            f"endpoint model mismatch: expected {expected_model!r}, got {sorted(identities)!r}"
        )


def _dispatch_command(
    *,
    model: str,
    base_url: str,
    tag: str,
    job: str,
    datasets: list[str],
) -> list[str]:
    lane = f"{tag}__{job}"
    return [
        sys.executable,
        "-m",
        "src.eval.scheduler.cli",
        "dispatch",
        "--log-dir",
        str(REPO / "logs/scheduler" / lane),
        "--pid-dir",
        str(REPO / "logs/pids" / lane),
        "--run-log-dir",
        str(REPO / "logs/runs" / lane),
        "--only-jobs",
        job,
        "--only-datasets",
        *datasets,
        "--infer-base-url",
        base_url,
        "--infer-models",
        model,
        "--infer-api-key",
        "rwkv-skills",
        "--infer-timeout-s",
        "1800",
        "--infer-max-workers",
        "64",
        "--infer-slots-per-model",
        "8",
        "--infer-protocol",
        "vllm",
        "--infer-seed-policy",
        "omit",
        "--remote-batch-size",
        "64",
        "--plain-choice-batch-size",
        "128",
        "--coding-eval-workers",
        "32",
        "--max-active-coding-runners",
        "2",
        "--math-judge-max-workers",
        "32",
        "--run-mode",
        "fresh",
        "--disable-checker",
        "--disable-infer-backpressure",
        "--dispatch-poll-seconds",
        "3",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("infer_base_url")
    parser.add_argument("tag")
    parser.add_argument("--dbname", default="chase_rwkv_skills_frontend46_20260804")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("logs/audits/g1g_g1i_core20_recovery_current.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SAFE_TAG.fullmatch(args.tag):
        parser.error("tag contains unsafe characters")
    output = args.audit_output if args.audit_output.is_absolute() else REPO / args.audit_output
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PG_DBNAME=args.dbname, PYTHONPATH=str(REPO))
    subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            "--family",
            "all",
            "--dbname",
            args.dbname,
            "--output",
            str(output),
        ],
        cwd=REPO,
        env=env,
        check=True,
    )
    audit = json.loads(output.read_text(encoding="utf-8"))
    plan = recovery_plan(audit, args.model)
    print(json.dumps({"model": args.model, "plan": plan}, ensure_ascii=False, indent=2))
    if not plan or args.dry_run:
        return 0
    verify_endpoint_model(args.infer_base_url, args.model)
    for job, datasets in plan.items():
        subprocess.run(
            _dispatch_command(
                model=args.model,
                base_url=args.infer_base_url,
                tag=args.tag,
                job=job,
                datasets=datasets,
            ),
            cwd=REPO,
            env=env,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
