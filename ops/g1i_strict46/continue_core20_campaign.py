#!/usr/bin/env python3
"""Finish the server-side G1g/G1i Core20 campaign without duplicate lanes.

The process is intended to run on 157 as an unprivileged user.  It owns only
GPU3 on 157 and GPU2 on 8222.  Every model handoff is gated by a fresh matrix
audit and the inference endpoint identity; the reserved 8222 GPU3/18073
service is never addressed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


REPO = Path(__file__).resolve().parents[2]
DBNAME = "chase_rwkv_skills_frontend46_20260804"
AUDIT = REPO / "ops/g1i_strict46/audit_core20_dual.py"
RECOVERY = REPO / "ops/g1i_strict46/run_core20_audit_missing.py"
INSPECT = REPO / "ops/g1i_strict46/inspect_task_completions.py"
API_KEY = "rwkv-skills"


@dataclass(frozen=True)
class ModelSpec:
    model: str
    weight: str
    context: int
    service: str


@dataclass(frozen=True)
class LaneSpec:
    name: str
    remote: bool
    repo: str
    user: str
    gpu: int
    port: int
    endpoint_for_eval: str
    initial_scheduler_units: tuple[str, ...]
    current_service: str
    models: tuple[ModelSpec, ...]


LANES = {
    "157": LaneSpec(
        name="157",
        remote=False,
        repo="/home/rwkv/chase/rwkv-skills",
        user="rwkv",
        gpu=3,
        port=19439,
        endpoint_for_eval="http://127.0.0.1:19439/v1",
        initial_scheduler_units=(
            "rwkv-g1i-core20-dual-15-frontend46-20260808.service",
            "rwkv-g1i-core20-gap-15-frontend46-20260808.service",
        ),
        current_service="rwkv-g1i-1p5b-gpu3-16k-c640.service",
        models=(
            ModelSpec(
                "rwkv7-g1i-1.5b-20260805-ctx16384",
                "/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1i-1.5b-20260805-ctx16384.pth",
                16384,
                "rwkv-core20-g1i-15-gpu3.service",
            ),
            ModelSpec(
                "rwkv7-g1i-2.9b-20260805-ctx16384",
                "/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1i-2.9b-20260805-ctx16384.pth",
                16384,
                "rwkv-core20-g1i-29-gpu3.service",
            ),
            ModelSpec(
                "rwkv7-g1g-1.5b-20260526-ctx8192",
                "/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-1.5b-20260526-ctx8192.pth",
                8192,
                "rwkv-core20-g1g-15-gpu3.service",
            ),
            ModelSpec(
                "rwkv7-g1g-2.9b-20260526-ctx8192",
                "/home/rwkv/chase/rwkv-skills/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-2.9b-20260526-ctx8192.pth",
                8192,
                "rwkv-core20-g1g-29-gpu3.service",
            ),
        ),
    ),
    "8222": LaneSpec(
        name="8222",
        remote=True,
        repo="/home/chase/rwkv-skills",
        user="chase",
        gpu=2,
        port=18074,
        endpoint_for_eval="http://127.0.0.1:29574/v1",
        initial_scheduler_units=(
            "rwkv-g1i-core20-dual-133-frontend46-20260808.service",
            "rwkv-g1i-core20-gap-133-frontend46-20260808.service",
        ),
        current_service="rwkv-g1i-13p3b-gpu2-16k-c640.service",
        models=(
            ModelSpec(
                "rwkv7-g1i-13.3b-20260805-ctx16384",
                "/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1i-13.3b-20260805-ctx16384.pth",
                16384,
                "rwkv-core20-g1i-133-gpu2.service",
            ),
            ModelSpec(
                "rwkv7-g1i-7.2b-20260805-ctx16384",
                "/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1i-7.2b-20260805-ctx16384.pth",
                16384,
                "rwkv-core20-g1i-72-gpu2.service",
            ),
            ModelSpec(
                "rwkv7-g1g-7.2b-20260523-ctx8192",
                "/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-7.2b-20260523-ctx8192.pth",
                8192,
                "rwkv-core20-g1g-72-gpu2.service",
            ),
            ModelSpec(
                "rwkv7-g1g-13.3b-20260523-ctx8192",
                "/home/chase/weights/BlinkDL__rwkv7-g1/rwkv7-g1g-13.3b-20260523-ctx8192.pth",
                8192,
                "rwkv-core20-g1g-133-gpu2.service",
            ),
        ),
    ),
}


def _ssh_prefix() -> list[str]:
    return [
        "ssh",
        "-p",
        "8222",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "chase@47.115.88.183",
    ]


def _command(spec: LaneSpec, args: Sequence[str]) -> list[str]:
    if spec.remote:
        return _ssh_prefix() + list(args)
    return list(args)


def _run(
    spec: LaneSpec,
    args: Sequence[str],
    *,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    command = _command(spec, args)
    print("$ " + shlex.join(command), flush=True)
    return subprocess.run(
        command,
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def _user_systemctl(spec: LaneSpec, *args: str, check: bool = True) -> str:
    result = _run(spec, ["systemctl", "--user", *args], check=check, timeout=45)
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output[-4000:], flush=True)
    return result.stdout.strip()


def _local_user_systemctl(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["systemctl", "--user", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=45,
        check=check,
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output[-4000:], flush=True)
    return result.stdout.strip()


def _is_active(spec: LaneSpec, unit: str) -> bool:
    result = _run(spec, ["systemctl", "--user", "is-active", unit], check=False, timeout=30)
    return result.returncode == 0 and result.stdout.strip() == "active"


def _local_is_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "active"


def _wait_units_inactive(spec: LaneSpec, units: Sequence[str]) -> None:
    while True:
        # All evaluation schedulers run on 157.  The lane host only applies to
        # the inference service; 8222 does not own these units.
        active = [unit for unit in units if _local_is_active(unit)]
        if not active:
            print(f"{spec.name}: initial schedulers are inactive", flush=True)
            return
        print(f"{spec.name}: waiting for schedulers: {', '.join(active)}", flush=True)
        time.sleep(60)


def _audit_path(lane: str, label: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return REPO / "logs" / "audits" / f"core20_{lane}_{label}_{timestamp}.json"


def _audit() -> dict[str, Any]:
    output = _audit_path("campaign", "orchestrator")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--family",
            "all",
            "--dbname",
            DBNAME,
            "--output",
            str(output),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    print(result.stdout.strip(), flush=True)
    return json.loads(output.read_text(encoding="utf-8"))


def _model_cells(audit: dict[str, Any], model: str) -> list[dict[str, Any]]:
    cells = [cell for cell in audit.get("cells", []) if cell.get("model") == model]
    if len(cells) != 34:
        raise RuntimeError(f"{model}: expected 34 cells, got {len(cells)}")
    return cells


def _model_state(audit: dict[str, Any], model: str) -> dict[str, int]:
    cells = _model_cells(audit, model)
    return {
        state: sum(cell.get("state") == state for cell in cells)
        for state in ("valid", "running", "missing", "invalid")
    }


def _verify_endpoint(spec: LaneSpec, model: str) -> None:
    result = _run(
        spec,
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "20",
            "-H",
            f"Authorization: Bearer {API_KEY}",
            f"http://127.0.0.1:{spec.port}/v1/models",
        ],
        timeout=30,
    )
    payload = json.loads(result.stdout)
    identities = {
        str(row.get("id")) for row in payload.get("data", []) if isinstance(row, dict)
    }
    if model not in identities:
        raise RuntimeError(f"{spec.name}: endpoint model mismatch, got {sorted(identities)}")
    if spec.name == "8222":
        forwarded = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                "20",
                "-H",
                f"Authorization: Bearer {API_KEY}",
                "http://127.0.0.1:29574/v1/models",
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        forwarded_models = {
            str(row.get("id"))
            for row in json.loads(forwarded.stdout).get("data", [])
            if isinstance(row, dict)
        }
        if model not in forwarded_models:
            raise RuntimeError(
                f"8222 forwarded endpoint mismatch, got {sorted(forwarded_models)}"
            )


def _recover_model(spec: LaneSpec, model: str, tag: str) -> None:
    for attempt in range(1, 4):
        audit = _audit()
        state = _model_state(audit, model)
        print(f"{spec.name}: {model} state {state}", flush=True)
        if state["valid"] == 34:
            _semantic_check(model)
            return
        if state["running"]:
            print(f"{spec.name}: waiting for {state['running']} running cells", flush=True)
            time.sleep(60)
            continue
        if not state["missing"] and not state["invalid"]:
            raise RuntimeError(f"{spec.name}: unresolved state for {model}: {state}")
        _verify_endpoint(spec, model)
        recovery_tag = f"{tag}_{attempt}"
        result = subprocess.run(
            [
                sys.executable,
                str(RECOVERY),
                model,
                spec.endpoint_for_eval,
                recovery_tag,
                "--dbname",
                DBNAME,
                "--audit-output",
                str(_audit_path(spec.name, f"recovery_{attempt}")),
            ],
            cwd=REPO,
            text=True,
            check=True,
        )
        print(result.stdout[-4000:] if result.stdout else "", flush=True)
    raise RuntimeError(f"{spec.name}: {model} did not reach 34 valid cells")


def _semantic_check(model: str) -> None:
    audit = _audit()
    task_ids = [
        str(int(cell["task_id"]))
        for cell in _model_cells(audit, model)
        if cell.get("state") == "valid"
    ]
    result = subprocess.run(
        [sys.executable, str(INSPECT), *task_ids, "--summary", "--limit", "0"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    summaries = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.startswith('{"type": "summary"')
    ]
    if len(summaries) != 34:
        raise RuntimeError(f"{model}: semantic inspection returned {len(summaries)}/34 summaries")
    failures = []
    for summary in summaries:
        if any(
            int(summary.get(field) or 0)
            for field in ("blank_completion_count", "leading_orphan_close_count")
        ):
            failures.append(summary)
        if int(summary.get("completion_count") or 0) != int(summary.get("evaluated_count") or 0):
            failures.append(summary)
    if failures:
        raise RuntimeError(f"{model}: semantic completion inspection failed: {failures[:3]}")
    truncation = sum(int(summary.get("final_stage_truncated_count") or 0) for summary in summaries)
    print(f"{model}: semantic inspection passed; final-stage truncations={truncation}", flush=True)


def _stop_service(spec: LaneSpec, unit: str) -> None:
    if not _is_active(spec, unit):
        return
    _user_systemctl(spec, "stop", unit)
    for _ in range(30):
        if not _is_active(spec, unit):
            return
        time.sleep(5)
    raise RuntimeError(f"{spec.name}: inference service did not stop: {unit}")


def _stop_initial_schedulers(spec: LaneSpec) -> None:
    # All evaluation schedulers run on 157; the 8222 lane only controls its
    # inference service.  Stopping known names is idempotent.
    _local_user_systemctl("stop", *spec.initial_scheduler_units, check=False)


def _busy_eval_processes(spec: LaneSpec) -> list[str]:
    """Find evaluation runners still attached to this lane's endpoint."""

    targets = (f":{spec.port}/v1", f"--infer-base-url http://127.0.0.1:{spec.port}/v1")
    commands: list[str] = []
    result = _run(
        spec,
        ["ps", "-eww", "-eo", "pid=,args="],
        check=False,
        timeout=30,
    )
    commands.extend(result.stdout.splitlines())
    # The 8222 evaluation clients run on 157 through the reverse forward.
    # They are invisible to 8222's process table, so inspect the local table
    # too before stopping GPU2.
    if spec.remote:
        local = subprocess.run(
            ["ps", "-eww", "-eo", "pid=,args="],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        commands.extend(local.stdout.splitlines())
        targets += (":29574/v1", "--infer-base-url http://127.0.0.1:29574/v1")
    return [
        line.strip()
        for line in commands
        if "src.eval." in line and any(target in line for target in targets)
    ]


def _wait_endpoint_idle(spec: LaneSpec) -> None:
    for _ in range(60):
        busy = _busy_eval_processes(spec)
        if not busy:
            print(f"{spec.name}: evaluation endpoint is idle", flush=True)
            return
        print(f"{spec.name}: waiting for {len(busy)} evaluation runners to exit", flush=True)
        time.sleep(10)
    raise RuntimeError(f"{spec.name}: evaluation endpoint stayed busy before handoff")


def _start_service(spec: LaneSpec, model: ModelSpec) -> None:
    if not Path(model.weight).name:
        raise RuntimeError("empty model weight")
    args = [
        "systemd-run",
        "--user",
        "--unit",
        model.service,
        "--property=Restart=on-failure",
        "--property=RestartSec=5s",
        "--property=KillMode=control-group",
        "--setenv",
        f"CUDA_VISIBLE_DEVICES={spec.gpu}",
        "--setenv",
        f"PYTHONPATH={('/home/chase/vllm-rwkv' if spec.remote else '/home/rwkv/chase/vllm-rwkv')}",
        "--setenv",
        "VLLM_USE_V2_MODEL_RUNNER=1",
        "--setenv",
        "VLLM_RWKV7_WKV_MODE=fp32io16",
        "--setenv",
        "VLLM_USE_RAPID_SAMPLER=1",
        "--setenv",
        "VLLM_RWKV_RAPID_SAMPLER=1",
        "--setenv",
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        "--working-directory",
        spec.repo,
        "/home/chase/.venv-vllm-56b463bf6/bin/vllm" if spec.remote else "/home/rwkv/chase/.venv-vllm-fcb31d859/bin/vllm",
        "serve",
        model.weight,
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--api-key",
        API_KEY,
        "--tokenizer-mode",
        "rwkv",
        "--trust-request-chat-template",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "rwkv",
        "--max-model-len",
        str(model.context),
        "--served-model-name",
        model.model,
        "--gpu-memory-utilization",
        "0.98",
        "--max-num-batched-tokens",
        "98304",
        "--max-num-seqs",
        "640",
        "--override-generation-config",
        '{"temperature":1e-5}',
    ]
    _user_systemctl(spec, "reset-failed", model.service, check=False)
    _run(spec, args, timeout=45)
    for _ in range(60):
        try:
            _verify_endpoint(spec, model.model)
            print(f"{spec.name}: endpoint ready for {model.model}", flush=True)
            return
        except (subprocess.CalledProcessError, json.JSONDecodeError, RuntimeError, URLError) as exc:
            print(f"{spec.name}: waiting for endpoint: {exc}", flush=True)
            time.sleep(10)
    raise RuntimeError(f"{spec.name}: endpoint did not become ready for {model.model}")


def run_lane(spec: LaneSpec) -> None:
    print(f"starting Core20 lane {spec.name}", flush=True)
    _wait_units_inactive(spec, spec.initial_scheduler_units)
    _recover_model(spec, spec.models[0].model, f"core20_{spec.name}_initial_recovery")
    _stop_initial_schedulers(spec)
    _wait_endpoint_idle(spec)

    for previous, next_model in zip(spec.models, spec.models[1:]):
        _stop_service(spec, previous.service if previous is not spec.models[0] else spec.current_service)
        _start_service(spec, next_model)
        _recover_model(spec, next_model.model, f"core20_{spec.name}_{next_model.model.replace('-', '_')}")

    final = _audit()
    summary = _model_state(final, spec.models[-1].model)
    print(f"lane {spec.name} completed final model {spec.models[-1].model}: {summary}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANES), required=True)
    args = parser.parse_args()
    run_lane(LANES[args.lane])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
