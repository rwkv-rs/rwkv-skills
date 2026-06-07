from __future__ import annotations

"""Verified launch helper for the nano-vLLM inference-swap formal eval path."""

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


DEFAULT_INFER_BASE_URL = "http://127.0.0.1:29082"
DEFAULT_INFER_MODEL = "rwkv7-g1g-2.9b-20260526-ctx8192"
DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_PORT = 5432
DEFAULT_DB_USER = "postgres"
DEFAULT_DB_NAME = "rwkv-eval"
DEFAULT_OUTPUT_JSON = "/tmp/rwkv-skills-infer-swap-preflight-29082-localdb.json"
DEFAULT_PHASE_GATE_JSON = "/tmp/rwkv-skills-infer-swap-phase-gate.json"
DEFAULT_LAUNCH_BUNDLE_JSON = "/tmp/rwkv-skills-infer-swap-launch-bundle.json"
MIN_PHASE_GATE_SCHEMA_VERSION = 2
DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE = 2
DEFAULT_JOBS = (
    "function_tau2_bench",
    "function_tau3_bench",
    "function_mcp_bench",
    "function_bfcl_v3",
)
DEFAULT_DATASETS = (
    "tau2_bench_airline_base",
    "tau2_bench_retail_base",
    "tau2_bench_telecom_base",
    "tau3_bench_airline_base",
    "tau3_bench_banking_knowledge_base",
    "tau3_bench_retail_base",
    "tau3_bench_telecom_base",
    "mcp_bench_test",
    "bfcl_v3_test",
)
PROFILE_CONCURRENCY = {
    "full-load": (2048, 2048),
    "throughput-peak": (1536, 1536),
    "low-risk": (1024, 1024),
}
REQUIRED_PHASE_GATE_PHASES_FOR_DISPATCH = (
    "backend_protocol",
    "router_fleet",
    "remote_comparison_and_probe",
    "formal_eval_guard",
    "compile_targets",
    "git_diff_check",
    "readiness_json",
    "probe_json",
    "summary_json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the verified inference-swap eval gate or dispatch")
    parser.add_argument(
        "--action",
        choices=("queue", "dispatch"),
        default="queue",
        help="queue is a safe preview; dispatch launches the formal benchmark",
    )
    parser.add_argument(
        "--confirm-dispatch",
        action="store_true",
        help="Required with --action dispatch to avoid accidental benchmark launches",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_CONCURRENCY),
        default="full-load",
        help="Concurrency profile from the 8222 GPU-full probe",
    )
    parser.add_argument("--infer-base-url", default=DEFAULT_INFER_BASE_URL)
    parser.add_argument("--infer-model", default=DEFAULT_INFER_MODEL)
    parser.add_argument("--infer-timeout-s", type=float, default=600.0)
    parser.add_argument("--infer-max-workers", type=int, help="Override profile infer workers")
    parser.add_argument("--remote-batch-size", type=int, help="Override profile remote batch size")
    parser.add_argument("--max-concurrent-jobs", type=int, default=1)
    parser.add_argument("--run-mode", default="new", choices=("auto", "new", "resume", "rerun"))
    parser.add_argument("--only-jobs", nargs="+", default=list(DEFAULT_JOBS))
    parser.add_argument("--only-datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--db-host", default=DEFAULT_DB_HOST)
    parser.add_argument("--db-port", type=int, default=DEFAULT_DB_PORT)
    parser.add_argument("--db-user", default=DEFAULT_DB_USER)
    parser.add_argument("--db-name", default=DEFAULT_DB_NAME)
    parser.add_argument("--db-sslmode", default="prefer")
    parser.add_argument("--db-timeout-s", type=float, default=5.0)
    parser.add_argument("--preflight-output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--phase-gate-json",
        default=DEFAULT_PHASE_GATE_JSON,
        help="Required successful phase-gate report before --action dispatch",
    )
    parser.add_argument(
        "--skip-phase-gate",
        action="store_true",
        help="Skip phase-gate evidence validation before dispatch",
    )
    parser.add_argument(
        "--launch-bundle-json",
        default=DEFAULT_LAUNCH_BUNDLE_JSON,
        help="Required launch bundle JSON before --action dispatch",
    )
    parser.add_argument(
        "--skip-launch-bundle",
        action="store_true",
        help="Skip launch-bundle identity validation before dispatch",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--print-scheduler-args",
        action="store_true",
        help="Print the scheduler argv before running queue/dispatch",
    )
    return parser.parse_args(argv)


def resolve_profile_concurrency(args: argparse.Namespace) -> tuple[int, int]:
    default_workers, default_batch = PROFILE_CONCURRENCY[str(args.profile)]
    workers = int(args.infer_max_workers) if args.infer_max_workers is not None else default_workers
    batch_size = int(args.remote_batch_size) if args.remote_batch_size is not None else default_batch
    if workers <= 0 or batch_size <= 0:
        raise ValueError("infer workers and remote batch size must be positive")
    return workers, batch_size


def db_config_from_args(args: argparse.Namespace):
    from src.eval.scheduler.config import DBConfig

    return DBConfig(
        host=str(args.db_host),
        port=int(args.db_port),
        user=str(args.db_user),
        password=str(os.environ.get("PG_PASSWORD", "") or ""),
        dbname=str(args.db_name),
        sslmode=str(args.db_sslmode),
    )


def apply_db_env(args: argparse.Namespace) -> None:
    from src.eval.env_config import load_env_file

    load_env_file(Path(".env"))
    os.environ["PG_HOST"] = str(args.db_host)
    os.environ["PG_PORT"] = str(int(args.db_port))
    os.environ["PG_USER"] = str(args.db_user)
    os.environ["PG_DBNAME"] = str(args.db_name)
    os.environ["RWKV_EVAL_SPACE_DB_HOST"] = str(args.db_host)
    os.environ["RWKV_EVAL_SPACE_DB_PORT"] = str(int(args.db_port))
    os.environ["RWKV_EVAL_SPACE_DB_USERNAME"] = str(args.db_user)
    os.environ["RWKV_EVAL_SPACE_DB_DATABASE_NAME"] = str(args.db_name)
    if args.db_sslmode:
        os.environ["PG_SSLMODE"] = str(args.db_sslmode)
        os.environ["RWKV_EVAL_SPACE_DB_SSLMODE"] = str(args.db_sslmode)
    if os.environ.get("PG_PASSWORD"):
        os.environ["RWKV_EVAL_SPACE_DB_PASSWORD"] = str(os.environ["PG_PASSWORD"])


def build_scheduler_args(args: argparse.Namespace) -> list[str]:
    workers, batch_size = resolve_profile_concurrency(args)
    scheduler_args = [
        str(args.action),
        "--infer-base-url",
        str(args.infer_base_url),
        "--infer-models",
        str(args.infer_model),
        "--infer-protocol",
        "nano-vllm-contents",
        "--infer-seed-policy",
        "omit-for-contents",
        "--infer-timeout-s",
        str(float(args.infer_timeout_s)),
        "--infer-max-workers",
        str(workers),
        "--remote-batch-size",
        str(batch_size),
        "--max-concurrent-jobs",
        str(int(args.max_concurrent_jobs)),
        "--only-jobs",
        *[str(item) for item in args.only_jobs],
        "--only-datasets",
        *[str(item) for item in args.only_datasets],
        "--run-mode",
        str(args.run_mode),
    ]
    return scheduler_args


def run_scheduler_cli(argv: Sequence[str]) -> int:
    from src.eval.scheduler import cli as scheduler_cli

    return int(scheduler_cli.main(list(argv)))


def load_preflight_module():
    from src.bin import preflight_remote_eval

    return preflight_remote_eval


def validate_phase_gate_for_dispatch(
    path: str | Path,
    dispatch_args: argparse.Namespace | None = None,
) -> tuple[str, ...]:
    phase_gate_path = Path(path).expanduser()
    if not phase_gate_path.exists():
        return (f"phase gate json missing: {phase_gate_path}",)
    try:
        payload = json.loads(phase_gate_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - dispatch guard should report malformed evidence.
        return (f"phase gate json load failed: {exc}",)
    if not isinstance(payload, dict):
        return (f"phase gate json root is not an object: {phase_gate_path}",)
    return validate_phase_gate_payload(payload, dispatch_args=dispatch_args)


def validate_launch_bundle_for_dispatch(
    path: str | Path,
    dispatch_args: argparse.Namespace,
) -> tuple[str, ...]:
    bundle_path = Path(path).expanduser()
    if not bundle_path.exists():
        return (f"launch bundle json missing: {bundle_path}",)
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - dispatch guard should report malformed evidence.
        return (f"launch bundle json load failed: {exc}",)
    if not isinstance(payload, dict):
        return (f"launch bundle json root is not an object: {bundle_path}",)
    return validate_launch_bundle_payload(payload, dispatch_args=dispatch_args)


def validate_launch_bundle_payload(
    payload: Mapping[str, Any],
    dispatch_args: argparse.Namespace,
) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("ok") is not True:
        errors.append("launch bundle ok is not true")
    if payload.get("readiness_ok") is not True:
        errors.append("launch bundle readiness_ok is not true")
    if payload.get("phase_gate_ok") is not True:
        errors.append("launch bundle phase_gate_ok is not true")
    if payload.get("readiness_errors"):
        errors.append(f"launch bundle readiness errors present: {payload.get('readiness_errors')}")
    if payload.get("phase_gate_errors"):
        errors.append(f"launch bundle phase gate errors present: {payload.get('phase_gate_errors')}")
    if not payload.get("generated_at_utc"):
        errors.append("launch bundle generated_at_utc missing")
    _validate_launch_bundle_phase_gate_path(payload, dispatch_args, errors)

    params = payload.get("launch_parameters")
    if not isinstance(params, Mapping):
        return tuple([*errors, "launch bundle launch_parameters missing"])

    errors.extend(_validate_launch_parameters_against_args(params, dispatch_args, label="launch bundle"))
    _validate_launch_bundle_tunnel_metadata(payload, dispatch_args, errors)
    _validate_launch_bundle_queue_argv(payload, params, errors)
    _validate_launch_bundle_dispatch_argv(payload, params, errors)
    _validate_launch_bundle_summary_argvs(payload, params, errors)
    _validate_launch_bundle_speedup_doc_argv(payload, errors)
    return tuple(errors)


def _validate_launch_parameters_against_args(
    params: Mapping[str, Any],
    dispatch_args: argparse.Namespace,
    *,
    label: str,
) -> list[str]:
    errors: list[str] = []
    expected_workers, expected_batch = resolve_profile_concurrency(dispatch_args)
    expected_jobs = tuple(str(item) for item in dispatch_args.only_jobs)
    expected_datasets = tuple(str(item) for item in dispatch_args.only_datasets)
    expected_db_target = _db_target_from_args(dispatch_args)
    checks: tuple[tuple[str, Any, Any], ...] = (
        ("profile", params.get("profile"), str(dispatch_args.profile)),
        ("base url", params.get("infer_base_url"), str(dispatch_args.infer_base_url)),
        ("model", params.get("infer_model"), str(dispatch_args.infer_model)),
        ("timeout", _optional_float(params.get("infer_timeout_s")), float(dispatch_args.infer_timeout_s)),
        ("workers", _optional_int(params.get("infer_max_workers")), expected_workers),
        ("remote batch", _optional_int(params.get("remote_batch_size")), expected_batch),
        ("max concurrent jobs", _optional_int(params.get("max_concurrent_jobs")), int(dispatch_args.max_concurrent_jobs)),
        ("run mode", params.get("run_mode"), str(dispatch_args.run_mode)),
        ("job count", _optional_int(params.get("job_count")), len(expected_jobs)),
        ("dataset count", _optional_int(params.get("dataset_count")), len(expected_datasets)),
        ("expected queue count", _optional_int(params.get("expected_queue_count")), len(expected_datasets)),
        ("db target", params.get("db_target"), expected_db_target),
        ("db sslmode", params.get("db_sslmode"), str(dispatch_args.db_sslmode)),
    )
    for name, actual, expected in checks:
        if actual != expected:
            errors.append(f"{label} {name} mismatch: {actual} != {expected}")
    phase_timeout_s = _optional_float(params.get("phase_timeout_s"))
    if phase_timeout_s is None or phase_timeout_s <= 0:
        errors.append(f"{label} phase timeout missing or invalid: {params.get('phase_timeout_s')}")
    summary_watch_interval_s = _optional_float(params.get("summary_watch_interval_s"))
    if summary_watch_interval_s is None or summary_watch_interval_s <= 0:
        errors.append(
            f"{label} summary watch interval missing or invalid: {params.get('summary_watch_interval_s')}"
        )

    jobs = params.get("only_jobs")
    if not isinstance(jobs, list):
        errors.append(f"{label} job list missing")
    elif tuple(str(item) for item in jobs) != expected_jobs:
        errors.append(f"{label} job list mismatch")
    datasets = params.get("only_datasets")
    if not isinstance(datasets, list):
        errors.append(f"{label} dataset list missing")
    elif tuple(str(item) for item in datasets) != expected_datasets:
        errors.append(f"{label} dataset list mismatch")
    return errors


def validate_phase_gate_payload(
    payload: Mapping[str, Any],
    dispatch_args: argparse.Namespace | None = None,
) -> tuple[str, ...]:
    errors: list[str] = []
    schema_version = _optional_int(payload.get("schema_version"))
    if schema_version is None or schema_version < MIN_PHASE_GATE_SCHEMA_VERSION:
        errors.append(f"phase gate schema_version too old: {payload.get('schema_version')}")
    if not payload.get("generated_at_utc"):
        errors.append("phase gate generated_at_utc missing")
    errors.extend(_validate_phase_gate_source_freshness(payload))
    required_names_payload = payload.get("required_phase_names")
    required_names = set(str(name) for name in required_names_payload) if isinstance(required_names_payload, list) else set()
    for name in REQUIRED_PHASE_GATE_PHASES_FOR_DISPATCH:
        if name not in required_names:
            errors.append(f"phase gate required_phase_names missing: {name}")
    if payload.get("ok") is not True:
        errors.append("phase gate ok is not true")
    phases = payload.get("phases")
    if not isinstance(phases, list):
        return tuple([*errors, "phase gate phases is not a list"])
    by_name = {
        str(item.get("name")): item
        for item in phases
        if isinstance(item, dict) and item.get("name") is not None
    }
    for name in REQUIRED_PHASE_GATE_PHASES_FOR_DISPATCH:
        phase = by_name.get(name)
        if phase is None:
            errors.append(f"phase gate missing phase: {name}")
            continue
        if phase.get("ok") is not True:
            errors.append(f"phase gate phase not ok: {name}")
        phase_errors = phase.get("errors")
        if phase_errors:
            errors.append(f"phase gate phase has errors: {name}: {phase_errors}")
    readiness = _phase_details(by_name.get("readiness_json"))
    if readiness:
        if readiness.get("ready_to_dispatch") is not True:
            errors.append("phase gate readiness ready_to_dispatch is not true")
        if readiness.get("queue_pending_count") != readiness.get("expected_queue_count"):
            errors.append("phase gate readiness queue count mismatch")
        if dispatch_args is not None:
            dispatch_workers, dispatch_batch = resolve_profile_concurrency(dispatch_args)
            if _optional_int(readiness.get("expected_infer_max_workers")) != dispatch_workers:
                errors.append(
                    "phase gate readiness worker mismatch: "
                    f"{readiness.get('expected_infer_max_workers')} != {dispatch_workers}"
                )
            if _optional_int(readiness.get("expected_remote_batch_size")) != dispatch_batch:
                errors.append(
                    "phase gate readiness remote batch mismatch: "
                    f"{readiness.get('expected_remote_batch_size')} != {dispatch_batch}"
                )
            required_concurrency = max(dispatch_workers, dispatch_batch)
            readiness_gpu_full = _optional_int(readiness.get("probe_gpu_full_concurrency"))
            if readiness_gpu_full is None or readiness_gpu_full < required_concurrency:
                errors.append(
                    "phase gate readiness gpu_full_concurrency insufficient: "
                    f"{readiness.get('probe_gpu_full_concurrency')} < {required_concurrency}"
                )
            readiness_largest = _optional_int(readiness.get("probe_largest_successful_concurrency"))
            if readiness_largest is None or readiness_largest < required_concurrency:
                errors.append(
                    "phase gate readiness largest_successful_concurrency insufficient: "
                    f"{readiness.get('probe_largest_successful_concurrency')} < {required_concurrency}"
                )
            if readiness.get("probe_model") != str(dispatch_args.infer_model):
                errors.append(f"phase gate readiness model mismatch: {readiness.get('probe_model')}")
            if readiness.get("protocol_smoke_ok") is not True:
                errors.append(f"phase gate readiness protocol_smoke_ok is not true: {readiness.get('protocol_smoke_ok')}")
            protocol_smoke_protocols = readiness.get("protocol_smoke_protocols")
            if not isinstance(protocol_smoke_protocols, list):
                errors.append("phase gate readiness protocol_smoke_protocols missing")
            else:
                errors.extend(_validate_phase_gate_protocol_smoke(protocol_smoke_protocols))
            expected_count = len(tuple(dispatch_args.only_datasets))
            if _optional_int(readiness.get("expected_queue_count")) != expected_count:
                errors.append(
                    "phase gate readiness expected queue count mismatch: "
                    f"{readiness.get('expected_queue_count')} != {expected_count}"
                )
    elif dispatch_args is not None:
        errors.append("phase gate readiness details missing")
    probe = _phase_details(by_name.get("probe_json"))
    if probe:
        if dispatch_args is not None:
            dispatch_workers, dispatch_batch = resolve_profile_concurrency(dispatch_args)
            required_concurrency = max(dispatch_workers, dispatch_batch)
            if probe.get("model") != str(dispatch_args.infer_model):
                errors.append(f"phase gate probe model mismatch: {probe.get('model')}")
            if _optional_int(probe.get("required_concurrency")) != required_concurrency:
                errors.append(
                    "phase gate probe required_concurrency mismatch: "
                    f"{probe.get('required_concurrency')} != {required_concurrency}"
                )
            probe_gpu_full = _optional_int(probe.get("gpu_full_concurrency"))
            if probe_gpu_full is None or probe_gpu_full < required_concurrency:
                errors.append(
                    "phase gate probe gpu_full_concurrency insufficient: "
                    f"{probe.get('gpu_full_concurrency')} < {required_concurrency}"
                )
            probe_largest = _optional_int(probe.get("largest_successful_concurrency"))
            if probe_largest is None or probe_largest < required_concurrency:
                errors.append(
                    "phase gate probe largest_successful_concurrency insufficient: "
                    f"{probe.get('largest_successful_concurrency')} < {required_concurrency}"
                )
        if probe.get("protocol") != "nano-vllm-contents":
            errors.append(f"phase gate probe protocol mismatch: {probe.get('protocol')}")
    elif dispatch_args is not None:
        errors.append("phase gate probe details missing")
    summary = _phase_details(by_name.get("summary_json"))
    if summary:
        if dispatch_args is None and _optional_int(summary.get("total_count")) != len(DEFAULT_DATASETS):
            errors.append(f"phase gate summary total_count mismatch: {summary.get('total_count')}")
        if dispatch_args is not None:
            dispatch_datasets = tuple(str(item) for item in dispatch_args.only_datasets)
            if _optional_int(summary.get("total_count")) != len(dispatch_datasets):
                errors.append(
                    "phase gate summary dispatch dataset count mismatch: "
                    f"{summary.get('total_count')} != {len(dispatch_datasets)}"
                )
            summary_datasets = summary.get("datasets")
            if not isinstance(summary_datasets, list):
                errors.append("phase gate summary dataset list missing")
            elif tuple(str(item) for item in summary_datasets) != dispatch_datasets:
                errors.append("phase gate summary dataset list mismatch")
    elif dispatch_args is not None:
        errors.append("phase gate summary details missing")
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    apply_db_env(args)
    db_config = db_config_from_args(args)
    scheduler_args = build_scheduler_args(args)
    if str(args.action) == "dispatch" and not bool(args.confirm_dispatch):
        print("refusing to dispatch without --confirm-dispatch", file=sys.stderr, flush=True)
        return 2
    if str(args.action) == "dispatch" and not bool(args.skip_phase_gate):
        phase_gate_errors = validate_phase_gate_for_dispatch(args.phase_gate_json, dispatch_args=args)
        if phase_gate_errors:
            for error in phase_gate_errors:
                print(f"phase gate failed: {error}", file=sys.stderr, flush=True)
            return 3
    if str(args.action) == "dispatch" and not bool(args.skip_launch_bundle):
        launch_bundle_errors = validate_launch_bundle_for_dispatch(args.launch_bundle_json, dispatch_args=args)
        if launch_bundle_errors:
            for error in launch_bundle_errors:
                print(f"launch bundle failed: {error}", file=sys.stderr, flush=True)
            return 4

    if not bool(args.skip_preflight):
        preflight_module = load_preflight_module()
        result = preflight_module.run_preflight(
            infer_base_url=str(args.infer_base_url),
            infer_model=str(args.infer_model),
            infer_timeout_s=float(args.infer_timeout_s),
            protocols=("openai", "nano-vllm-contents"),
            batch_size=DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
            max_tokens=16,
            check_db=True,
            db_timeout_s=float(args.db_timeout_s),
            db_config=db_config,
        )
        print(preflight_module.format_preflight_summary(result), flush=True)
        if args.preflight_output_json:
            preflight_module.write_preflight_result(Path(args.preflight_output_json).expanduser(), result)
        if not result.ok:
            return 1

    if bool(args.print_scheduler_args):
        print("scheduler_args=" + " ".join(scheduler_args), flush=True)
    return run_scheduler_cli(scheduler_args)


def _phase_details(phase: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if phase is None:
        return {}
    details = phase.get("details")
    return details if isinstance(details, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_phase_gate_protocol_smoke(protocols: Sequence[Mapping[str, Any]]) -> list[str]:
    by_protocol = {str(item.get("protocol")): item for item in protocols if item.get("protocol") is not None}
    errors: list[str] = []
    for protocol in ("openai", "nano-vllm-contents"):
        item = by_protocol.get(protocol)
        if item is None:
            errors.append(f"phase gate readiness protocol smoke missing: {protocol}")
            continue
        if item.get("ok") is not True:
            errors.append(f"phase gate readiness protocol smoke failed: {protocol}")
        request_count = _optional_int(item.get("request_count"))
        nonempty_count = _optional_int(item.get("nonempty_output_count"))
        if request_count is None or request_count < DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE:
            errors.append(
                "phase gate readiness protocol smoke request_count below batched smoke requirement: "
                f"{protocol}={request_count}/{DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE}"
            )
        elif nonempty_count != request_count:
            errors.append(
                "phase gate readiness protocol smoke nonempty output mismatch: "
                f"{protocol}={nonempty_count}/{request_count}"
            )
    return errors


def _validate_phase_gate_source_freshness(payload: Mapping[str, Any]) -> list[str]:
    digest = payload.get("source_digest")
    manifest = payload.get("source_manifest")
    errors: list[str] = []
    if not isinstance(digest, str) or not digest:
        errors.append("phase gate source_digest missing")
    if not isinstance(manifest, Mapping) or not manifest:
        errors.append("phase gate source_manifest missing")
        return errors
    try:
        from src.bin import validate_infer_swap_phases

        current_manifest = validate_infer_swap_phases.build_source_manifest()
        current_digest = validate_infer_swap_phases.source_digest(current_manifest)
    except Exception as exc:  # noqa: BLE001 - dispatch guard should report freshness failures.
        errors.append(f"phase gate source freshness check failed: {exc}")
        return errors
    recorded_manifest = {str(key): str(value) for key, value in manifest.items()}
    expected_paths = set(current_manifest)
    recorded_paths = set(recorded_manifest)
    missing_paths = sorted(expected_paths - recorded_paths)
    extra_paths = sorted(recorded_paths - expected_paths)
    if missing_paths:
        errors.append("phase gate source_manifest missing paths: " + ",".join(missing_paths))
    if extra_paths:
        errors.append("phase gate source_manifest extra paths: " + ",".join(extra_paths))
    changed = sorted(path for path in expected_paths & recorded_paths if recorded_manifest[path] != current_manifest[path])
    if changed:
        errors.append("phase gate source_manifest changed paths: " + ",".join(changed))
    if isinstance(digest, str) and digest and digest != current_digest:
        errors.append(f"phase gate source_digest mismatch: {digest} != {current_digest}")
    return errors


def _db_target_from_args(args: argparse.Namespace) -> str:
    return f"{args.db_host}:{int(args.db_port)}/{args.db_name} user={args.db_user}"


def _validate_launch_bundle_phase_gate_path(
    payload: Mapping[str, Any],
    dispatch_args: argparse.Namespace,
    errors: list[str],
) -> None:
    bundle_phase_gate = payload.get("phase_gate_json")
    if not bundle_phase_gate:
        errors.append("launch bundle phase_gate_json missing")
        return
    expected = _normalized_path(str(dispatch_args.phase_gate_json))
    actual = _normalized_path(str(bundle_phase_gate))
    if actual != expected:
        errors.append(f"launch bundle phase gate path mismatch: {actual} != {expected}")


def _validate_launch_bundle_dispatch_argv(
    payload: Mapping[str, Any],
    params: Mapping[str, Any],
    errors: list[str],
) -> None:
    argv_payload = payload.get("dispatch_argv")
    if not isinstance(argv_payload, list) or not argv_payload:
        errors.append("launch bundle dispatch_argv missing")
        return
    argv = tuple(str(item) for item in argv_payload)
    module_argv = _strip_run_infer_swap_eval_prefix(argv)
    if module_argv is None:
        errors.append("launch bundle dispatch_argv does not call src.bin.run_infer_swap_eval")
        return
    for required_flag in ("--confirm-dispatch", "--phase-gate-json", "--launch-bundle-json"):
        if required_flag not in module_argv:
            errors.append(f"launch bundle dispatch_argv missing {required_flag}")
    try:
        bundle_args = parse_args(module_argv)
    except SystemExit as exc:
        errors.append(f"launch bundle dispatch_argv parse failed: {exc}")
        return
    if str(bundle_args.action) != "dispatch":
        errors.append(f"launch bundle dispatch_argv action mismatch: {bundle_args.action}")
    if not bool(bundle_args.confirm_dispatch):
        errors.append("launch bundle dispatch_argv confirm missing")
    if bool(bundle_args.skip_phase_gate):
        errors.append("launch bundle dispatch_argv skips phase gate")
    if bool(bundle_args.skip_launch_bundle):
        errors.append("launch bundle dispatch_argv skips launch bundle")
    errors.extend(
        _validate_launch_parameters_against_args(
            params,
            bundle_args,
            label="launch bundle dispatch_argv",
        )
    )
    bundle_phase_gate = payload.get("phase_gate_json")
    if bundle_phase_gate and _normalized_path(str(bundle_args.phase_gate_json)) != _normalized_path(str(bundle_phase_gate)):
        errors.append("launch bundle dispatch_argv phase gate path mismatch")
    bundle_json = payload.get("bundle_json")
    if bundle_json:
        actual_bundle_path = _normalized_path(str(bundle_args.launch_bundle_json))
        expected_bundle_path = _normalized_path(str(bundle_json))
        if actual_bundle_path != expected_bundle_path:
            errors.append("launch bundle dispatch_argv bundle path mismatch")
    command = payload.get("dispatch_command")
    if not isinstance(command, str) or not command:
        errors.append("launch bundle dispatch_command missing")
    else:
        expected_command = "rtk " + shlex.join(argv)
        if command != expected_command:
            errors.append("launch bundle dispatch_command mismatch")


def _validate_launch_bundle_queue_argv(
    payload: Mapping[str, Any],
    params: Mapping[str, Any],
    errors: list[str],
) -> None:
    argv_payload = payload.get("queue_argv")
    if not isinstance(argv_payload, list) or not argv_payload:
        errors.append("launch bundle queue_argv missing")
        return
    argv = tuple(str(item) for item in argv_payload)
    module_argv = _strip_run_infer_swap_eval_prefix(argv)
    if module_argv is None:
        errors.append("launch bundle queue_argv does not call src.bin.run_infer_swap_eval")
        return
    try:
        queue_args = parse_args(module_argv)
    except SystemExit as exc:
        errors.append(f"launch bundle queue_argv parse failed: {exc}")
        return
    if str(queue_args.action) != "queue":
        errors.append(f"launch bundle queue_argv action mismatch: {queue_args.action}")
    if bool(queue_args.skip_preflight):
        errors.append("launch bundle queue_argv skips preflight")
    if not bool(queue_args.print_scheduler_args):
        errors.append("launch bundle queue_argv missing --print-scheduler-args")
    errors.extend(
        _validate_launch_parameters_against_args(
            params,
            queue_args,
            label="launch bundle queue_argv",
        )
    )
    command = payload.get("queue_command")
    if not isinstance(command, str) or not command:
        errors.append("launch bundle queue_command missing")
    else:
        expected_command = "rtk " + shlex.join(argv)
        if command != expected_command:
            errors.append("launch bundle queue_command mismatch")


def _validate_launch_bundle_tunnel_metadata(
    payload: Mapping[str, Any],
    dispatch_args: argparse.Namespace,
    errors: list[str],
) -> None:
    params = payload.get("tunnel_parameters")
    if not isinstance(params, Mapping):
        errors.append("launch bundle tunnel_parameters missing")
        return
    parsed = urlparse(str(dispatch_args.infer_base_url))
    if not parsed.hostname or parsed.port is None:
        errors.append("launch bundle dispatch infer_base_url missing host or port")
        return
    checks: tuple[tuple[str, Any, Any], ...] = (
        ("local host", params.get("local_host"), str(parsed.hostname)),
        ("local port", _optional_int(params.get("local_port")), int(parsed.port)),
    )
    for name, actual, expected in checks:
        if actual != expected:
            errors.append(f"launch bundle tunnel {name} mismatch: {actual} != {expected}")
    required_keys = ("ssh_user", "ssh_host", "ssh_port", "remote_host", "remote_port")
    for key in required_keys:
        if params.get(key) in (None, ""):
            errors.append(f"launch bundle tunnel {key} missing")
    tunnel_argv_payload = payload.get("tunnel_argv")
    if not isinstance(tunnel_argv_payload, list) or not tunnel_argv_payload:
        errors.append("launch bundle tunnel_argv missing")
        return
    tunnel_argv = tuple(str(item) for item in tunnel_argv_payload)
    expected_argv = _expected_tunnel_argv(params)
    if expected_argv is not None and tunnel_argv != expected_argv:
        errors.append("launch bundle tunnel_argv mismatch")
    tunnel_command = payload.get("tunnel_command")
    if not isinstance(tunnel_command, str) or not tunnel_command:
        errors.append("launch bundle tunnel_command missing")
    else:
        expected_command = "rtk " + shlex.join(tunnel_argv)
        if tunnel_command != expected_command:
            errors.append("launch bundle tunnel_command mismatch")


def _validate_launch_bundle_summary_argvs(
    payload: Mapping[str, Any],
    params: Mapping[str, Any],
    errors: list[str],
) -> None:
    _validate_launch_bundle_summary_argv(
        payload,
        params,
        errors,
        argv_key="summary_argv",
        command_key="summary_command",
        require_watch=False,
        require_markdown=False,
    )
    _validate_launch_bundle_summary_argv(
        payload,
        params,
        errors,
        argv_key="summary_watch_argv",
        command_key="summary_watch_command",
        require_watch=True,
        require_markdown=False,
    )
    _validate_launch_bundle_summary_argv(
        payload,
        params,
        errors,
        argv_key="evidence_argv",
        command_key="evidence_command",
        require_watch=False,
        require_markdown=True,
    )


def _validate_launch_bundle_summary_argv(
    payload: Mapping[str, Any],
    params: Mapping[str, Any],
    errors: list[str],
    *,
    argv_key: str,
    command_key: str,
    require_watch: bool,
    require_markdown: bool,
) -> None:
    argv_payload = payload.get(argv_key)
    if not isinstance(argv_payload, list) or not argv_payload:
        errors.append(f"launch bundle {argv_key} missing")
        return
    argv = tuple(str(item) for item in argv_payload)
    module_argv = _strip_module_prefix(argv, "src.bin.summarize_infer_swap_eval")
    if module_argv is None:
        errors.append(f"launch bundle {argv_key} does not call src.bin.summarize_infer_swap_eval")
        return
    try:
        from src.bin import summarize_infer_swap_eval

        summary_args = summarize_infer_swap_eval.parse_args(module_argv)
    except SystemExit as exc:
        errors.append(f"launch bundle {argv_key} parse failed: {exc}")
        return
    expected_datasets = tuple(str(item) for item in params.get("only_datasets", ()))
    if str(summary_args.model) != str(params.get("infer_model")):
        errors.append(f"launch bundle {argv_key} model mismatch")
    if tuple(str(item) for item in summary_args.datasets) != expected_datasets:
        errors.append(f"launch bundle {argv_key} dataset list mismatch")
    if _db_target_from_args(summary_args) != str(params.get("db_target")):
        errors.append(f"launch bundle {argv_key} db target mismatch")
    if str(summary_args.db_sslmode) != str(params.get("db_sslmode")):
        errors.append(f"launch bundle {argv_key} db sslmode mismatch")

    summary_json = payload.get("summary_json")
    if not summary_json:
        errors.append("launch bundle summary_json missing")
    elif _normalized_path(str(summary_args.output_json or "")) != _normalized_path(str(summary_json)):
        errors.append(f"launch bundle {argv_key} summary output mismatch")

    if bool(summary_args.watch) != require_watch:
        errors.append(f"launch bundle {argv_key} watch flag mismatch")
    if require_watch and float(summary_args.watch_interval_s) <= 0:
        errors.append(f"launch bundle {argv_key} watch interval must be positive")
    expected_watch_interval = _optional_float(params.get("summary_watch_interval_s"))
    if (
        require_watch
        and expected_watch_interval is not None
        and float(summary_args.watch_interval_s) != expected_watch_interval
    ):
        errors.append(f"launch bundle {argv_key} watch interval mismatch")

    if require_markdown:
        probe_json = payload.get("probe_json")
        evidence_md = payload.get("evidence_md")
        if not probe_json:
            errors.append("launch bundle probe_json missing")
        elif _normalized_path(str(summary_args.probe_json or "")) != _normalized_path(str(probe_json)):
            errors.append(f"launch bundle {argv_key} probe path mismatch")
        if not evidence_md:
            errors.append("launch bundle evidence_md missing")
        elif _normalized_path(str(summary_args.output_md or "")) != _normalized_path(str(evidence_md)):
            errors.append(f"launch bundle {argv_key} markdown output mismatch")
    else:
        if summary_args.probe_json:
            errors.append(f"launch bundle {argv_key} unexpected probe path")
        if summary_args.output_md:
            errors.append(f"launch bundle {argv_key} unexpected markdown output")

    command = payload.get(command_key)
    if not isinstance(command, str) or not command:
        errors.append(f"launch bundle {command_key} missing")
    else:
        expected_command = "rtk " + shlex.join(argv)
        if command != expected_command:
            errors.append(f"launch bundle {command_key} mismatch")


def _validate_launch_bundle_speedup_doc_argv(
    payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    speedup_md = payload.get("speedup_md")
    if not speedup_md:
        errors.append("launch bundle speedup_md missing")
    argv_payload = payload.get("speedup_doc_argv")
    if not isinstance(argv_payload, list) or not argv_payload:
        errors.append("launch bundle speedup_doc_argv missing")
        return
    argv = tuple(str(item) for item in argv_payload)
    module_argv = _strip_module_prefix(argv, "src.bin.draft_infer_swap_speedup_doc")
    if module_argv is None:
        errors.append("launch bundle speedup_doc_argv does not call src.bin.draft_infer_swap_speedup_doc")
        return
    try:
        from src.bin import draft_infer_swap_speedup_doc

        speedup_args = draft_infer_swap_speedup_doc.parse_args(module_argv)
    except SystemExit as exc:
        errors.append(f"launch bundle speedup_doc_argv parse failed: {exc}")
        return

    expected_paths = (
        ("summary_json", "summary_json", speedup_args.summary_json),
        ("probe_json", "probe_json", speedup_args.probe_json),
        ("readiness_json", "readiness_json", speedup_args.readiness_json),
        ("bundle_json", "launch_bundle_json", speedup_args.launch_bundle_json),
    )
    for payload_key, label, actual in expected_paths:
        expected = payload.get(payload_key)
        if not expected:
            errors.append(f"launch bundle {payload_key} missing")
        elif _normalized_path(str(actual or "")) != _normalized_path(str(expected)):
            errors.append(f"launch bundle speedup_doc_argv {label} mismatch")
    if not str(speedup_args.output_md or ""):
        errors.append("launch bundle speedup_doc_argv output_md missing")
    if speedup_md and _normalized_path(str(speedup_args.output_md or "")) != _normalized_path(str(speedup_md)):
        errors.append("launch bundle speedup_doc_argv output_md mismatch")
    if bool(speedup_args.allow_incomplete):
        errors.append("launch bundle speedup_doc_argv allows incomplete summary")

    command = payload.get("speedup_doc_command")
    if not isinstance(command, str) or not command:
        errors.append("launch bundle speedup_doc_command missing")
    else:
        expected_command = "rtk " + shlex.join(argv)
        if command != expected_command:
            errors.append("launch bundle speedup_doc_command mismatch")


def _expected_tunnel_argv(params: Mapping[str, Any]) -> tuple[str, ...] | None:
    ssh_port = _optional_int(params.get("ssh_port"))
    local_port = _optional_int(params.get("local_port"))
    remote_port = _optional_int(params.get("remote_port"))
    ssh_user = params.get("ssh_user")
    ssh_host = params.get("ssh_host")
    local_host = params.get("local_host")
    remote_host = params.get("remote_host")
    if None in (ssh_port, local_port, remote_port) or not all(
        isinstance(item, str) and item
        for item in (ssh_user, ssh_host, local_host, remote_host)
    ):
        return None
    return (
        "ssh",
        "-p",
        str(ssh_port),
        "-N",
        "-L",
        f"{local_host}:{local_port}:{remote_host}:{remote_port}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        f"{ssh_user}@{ssh_host}",
    )


def _strip_run_infer_swap_eval_prefix(argv: Sequence[str]) -> tuple[str, ...] | None:
    return _strip_module_prefix(argv, "src.bin.run_infer_swap_eval")


def _strip_module_prefix(argv: Sequence[str], module_name: str) -> tuple[str, ...] | None:
    prefix = ("uv", "run", "python", "-m", "src.bin.run_infer_swap_eval")
    if module_name != "src.bin.run_infer_swap_eval":
        prefix = ("uv", "run", "python", "-m", module_name)
    if tuple(argv[: len(prefix)]) == prefix:
        return tuple(argv[len(prefix) :])
    module_flag = "-m"
    try:
        module_index = tuple(argv).index(module_flag)
    except ValueError:
        return None
    module_name_index = module_index + 1
    if module_name_index >= len(argv) or argv[module_name_index] != module_name:
        return None
    return tuple(argv[module_name_index + 1 :])


def _normalized_path(path: str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


__all__ = [
    "DEFAULT_DATASETS",
    "DEFAULT_INFER_BASE_URL",
    "DEFAULT_LAUNCH_BUNDLE_JSON",
    "DEFAULT_INFER_MODEL",
    "DEFAULT_JOBS",
    "DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE",
    "MIN_PHASE_GATE_SCHEMA_VERSION",
    "PROFILE_CONCURRENCY",
    "REQUIRED_PHASE_GATE_PHASES_FOR_DISPATCH",
    "apply_db_env",
    "build_scheduler_args",
    "db_config_from_args",
    "load_preflight_module",
    "main",
    "parse_args",
    "resolve_profile_concurrency",
    "run_scheduler_cli",
    "validate_launch_bundle_for_dispatch",
    "validate_launch_bundle_payload",
    "validate_phase_gate_for_dispatch",
    "validate_phase_gate_payload",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
