#!/usr/bin/env python3
"""Run only strict-46 cells rejected or missing in the current audit.

This is the recovery lane used after a model's main fresh scheduler exits.
The strict audit remains the source of truth: valid cells are never queued,
while missing, failed, unscored, or protocol-invalid cells are rerun as fresh
tasks so their completions cannot be mixed with an older partial task.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ops.g1i_strict46.runtime_state import RUN_ID_RE, prepare_run_state


RUNTIME_REPO = Path(__file__).resolve().parents[2]
GATE = RUNTIME_REPO / "ops" / "g1i_strict46" / "require_global_protocol_gate.py"
JOBS = (
    "multi_choice_plain_naive",
    "free_response_naive",
    "free_response_judge_naive",
    "code_human_eval_naive",
    "code_mbpp_naive",
    "code_livecodebench_plain_naive",
    "instruction_following_naive",
)


def require_protocol_gate(
    *,
    env: dict[str, str],
    model: str,
    infer_base_url: str,
    phase: str,
    frozen_runtime: Path,
) -> None:
    """Revalidate the immutable protocol lock at each state boundary."""

    subprocess.run(
        [
            sys.executable,
            "-I",
            str(GATE),
            "--phase",
            phase,
            "--model",
            model,
            "--infer-base-url",
            infer_base_url,
            "--infer-api-key",
            "rwkv-skills",
            "--frozen-runtime",
            str(frozen_runtime),
            "--require-current-python",
        ],
        cwd=RUNTIME_REPO,
        env=env,
        check=True,
    )


def audit_cell_to_dataset(cell: str) -> str:
    """Map the audit's logical benchmark/split key to scheduler dataset slug."""

    benchmark, separator, split = str(cell).partition("__")
    if not separator or not benchmark or not split:
        raise ValueError(f"invalid strict-46 audit benchmark key: {cell!r}")
    if benchmark == "gpqa":
        if split not in {"diamond", "main", "extended"}:
            raise ValueError(f"unsupported GPQA split in audit: {cell!r}")
        return f"gpqa_{split}"
    # Scheduler aliases for the strict-46 catalogue identify these datasets
    # by benchmark name; their physical/logical split is resolved internally.
    return benchmark


def missing_datasets(audit: dict[str, object], model: str) -> list[str]:
    models = audit.get("models")
    if not isinstance(models, dict) or model not in models:
        raise ValueError(f"model is absent from strict-46 audit: {model}")
    model_row = models[model]
    if not isinstance(model_row, dict):
        raise ValueError(f"invalid strict-46 model audit row: {model}")
    missing = model_row.get("missing_cells")
    if not isinstance(missing, list):
        raise ValueError(f"strict-46 audit has no missing_cells list: {model}")
    datasets = {
        audit_cell_to_dataset(str(row.get("benchmark")))
        for row in missing
        if isinstance(row, dict) and row.get("benchmark")
    }
    return sorted(datasets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("infer_base_url")
    parser.add_argument("tag")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("logs/audits/g1i_strict46_recovery_current.json"),
    )
    args = parser.parse_args()

    env = dict(os.environ)
    run_id = env.get("RWKV_STRICT_RUN_ID", "").strip()
    if not RUN_ID_RE.fullmatch(run_id):
        raise SystemExit("RWKV_STRICT_RUN_ID is missing or unsafe")
    state_root = prepare_run_state(run_id, create=False)
    configured_state = env.get("RWKV_STRICT_STATE_ROOT", "").strip()
    if not configured_state or Path(configured_state).resolve(strict=True) != state_root:
        raise SystemExit("RWKV_STRICT_STATE_ROOT does not match the verified run state")
    if not RUN_ID_RE.fullmatch(args.tag):
        raise SystemExit("recovery tag is unsafe")
    frozen_raw = env.get("RWKV_STRICT_FROZEN_RUNTIME", "").strip()
    if not frozen_raw:
        raise SystemExit(
            "RWKV_STRICT_FROZEN_RUNTIME is mandatory; mutable-repository recovery is forbidden"
        )
    frozen_runtime = Path(frozen_raw).expanduser().resolve(strict=True)
    env.update(
        PG_DBNAME="chase_rwkv_skills_frontend46_20260804",
        RWKV_BENCHMARK_CONFIG_ROOT=str(frozen_runtime / "configs" / "g1h"),
        PYTHONPATH=str(frozen_runtime),
    )
    if not args.audit_output.is_absolute():
        args.audit_output = state_root / args.audit_output
    args.audit_output = args.audit_output.resolve()
    try:
        args.audit_output.relative_to(state_root)
    except ValueError as exc:
        raise SystemExit("audit output must remain inside the verified run state") from exc
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    require_protocol_gate(
        env=env,
        model=args.model,
        infer_base_url=args.infer_base_url,
        phase="audit",
        frozen_runtime=frozen_runtime,
    )
    subprocess.run(
        [
            sys.executable,
            str(frozen_runtime / "ops/g1i_strict46/audit_current.py"),
            "--output",
            str(args.audit_output),
        ],
        cwd=RUNTIME_REPO,
        env=env,
        check=True,
    )
    audit = json.loads(args.audit_output.read_text(encoding="utf-8"))
    datasets = missing_datasets(audit, args.model)
    if not datasets:
        print(f"{args.model}: strict-46 audit has no missing cells")
        return 0

    print(
        f"{args.model}: recovering {len(datasets)} strict-46 datasets: "
        + ", ".join(datasets),
        flush=True,
    )
    # The audit may take hours.  Revalidate immediately before dispatch so a
    # source/config change during the audit fails closed (TOCTOU protection).
    require_protocol_gate(
        env=env,
        model=args.model,
        infer_base_url=args.infer_base_url,
        phase="recovery",
        frozen_runtime=frozen_runtime,
    )
    command = [
        sys.executable,
        "-m",
        "src.eval.scheduler.cli",
        "dispatch",
        "--log-dir",
        str(state_root / "logs/scheduler" / args.tag),
        "--pid-dir",
        str(state_root / "logs/pids" / args.tag),
        "--run-log-dir",
        str(state_root / "logs/runs" / args.tag),
        "--only-jobs",
        *JOBS,
        "--only-datasets",
        *datasets,
        "--infer-base-url",
        args.infer_base_url,
        "--infer-models",
        args.model,
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
    return subprocess.run(command, cwd=RUNTIME_REPO, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
