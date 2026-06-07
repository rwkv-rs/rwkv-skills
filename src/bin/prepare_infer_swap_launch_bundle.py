from __future__ import annotations

"""Prepare the non-dispatch launch evidence bundle for the inference swap."""

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shlex
from typing import Any, Sequence
from urllib.parse import urlparse

from src.bin import audit_infer_swap_readiness, draft_infer_swap_speedup_doc, run_infer_swap_eval, validate_infer_swap_phases


DEFAULT_BUNDLE_JSON = "/tmp/rwkv-skills-infer-swap-launch-bundle.json"
DEFAULT_TUNNEL_SSH_HOST = "47.115.88.183"
DEFAULT_TUNNEL_SSH_PORT = 8222
DEFAULT_TUNNEL_SSH_USER = "chase"
DEFAULT_REMOTE_INFER_HOST = "127.0.0.1"
DEFAULT_REMOTE_INFER_PORT = 19082


@dataclass(slots=True, frozen=True)
class InferSwapLaunchBundle:
    ok: bool
    readiness_ok: bool
    phase_gate_ok: bool
    launch_parameters: dict[str, Any]
    readiness_json: str
    summary_json: str
    evidence_md: str
    speedup_md: str
    probe_json: str
    phase_gate_json: str
    bundle_json: str | None
    readiness_errors: tuple[str, ...]
    phase_gate_errors: tuple[str, ...]
    generated_at_utc: str = ""
    queue_argv: tuple[str, ...] = ()
    dispatch_argv: tuple[str, ...] = ()
    queue_command: str = ""
    dispatch_command: str = ""
    tunnel_parameters: dict[str, Any] | None = None
    tunnel_argv: tuple[str, ...] = ()
    tunnel_command: str = ""
    summary_argv: tuple[str, ...] = ()
    summary_watch_argv: tuple[str, ...] = ()
    evidence_argv: tuple[str, ...] = ()
    speedup_doc_argv: tuple[str, ...] = ()
    summary_command: str = ""
    summary_watch_command: str = ""
    evidence_command: str = ""
    speedup_doc_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare inference-swap launch evidence without dispatching")
    parser.add_argument("--profile", choices=tuple(run_infer_swap_eval.PROFILE_CONCURRENCY), default="full-load")
    parser.add_argument("--infer-base-url", default=run_infer_swap_eval.DEFAULT_INFER_BASE_URL)
    parser.add_argument("--infer-model", default=run_infer_swap_eval.DEFAULT_INFER_MODEL)
    parser.add_argument("--infer-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--infer-protocol",
        choices=run_infer_swap_eval.REMOTE_INFERENCE_PROTOCOL_CHOICES,
        default=run_infer_swap_eval.DEFAULT_INFER_PROTOCOL,
    )
    parser.add_argument(
        "--infer-seed-policy",
        choices=run_infer_swap_eval.REMOTE_INFERENCE_SEED_POLICY_CHOICES,
        default=run_infer_swap_eval.DEFAULT_INFER_SEED_POLICY,
    )
    parser.add_argument("--infer-max-workers", type=int)
    parser.add_argument("--remote-batch-size", type=int)
    parser.add_argument("--max-concurrent-jobs", type=int, default=1)
    parser.add_argument("--run-mode", default="new", choices=("auto", "new", "resume", "rerun"))
    parser.add_argument("--only-jobs", nargs="+", default=list(run_infer_swap_eval.DEFAULT_JOBS))
    parser.add_argument("--only-datasets", nargs="+", default=list(run_infer_swap_eval.DEFAULT_DATASETS))
    parser.add_argument("--expected-queue-count", type=int, default=len(run_infer_swap_eval.DEFAULT_DATASETS))
    parser.add_argument("--db-host", default=run_infer_swap_eval.DEFAULT_DB_HOST)
    parser.add_argument("--db-port", type=int, default=run_infer_swap_eval.DEFAULT_DB_PORT)
    parser.add_argument("--db-user", default=run_infer_swap_eval.DEFAULT_DB_USER)
    parser.add_argument("--db-name", default=run_infer_swap_eval.DEFAULT_DB_NAME)
    parser.add_argument("--db-sslmode", default="prefer")
    parser.add_argument("--db-timeout-s", type=float, default=5.0)
    parser.add_argument("--probe-json", default=audit_infer_swap_readiness.DEFAULT_PROBE_JSON)
    parser.add_argument("--readiness-output-json", default=audit_infer_swap_readiness.DEFAULT_AUDIT_JSON)
    parser.add_argument("--preflight-output-json", default=run_infer_swap_eval.DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-output-json", default=audit_infer_swap_readiness.DEFAULT_SUMMARY_JSON)
    parser.add_argument("--evidence-output-md", default=audit_infer_swap_readiness.DEFAULT_EVIDENCE_MD)
    parser.add_argument("--speedup-output-md", default=draft_infer_swap_speedup_doc.DEFAULT_OUTPUT_MD)
    parser.add_argument("--summary-watch-interval-s", type=float, default=60.0)
    parser.add_argument("--phase-gate-output-json", default=run_infer_swap_eval.DEFAULT_PHASE_GATE_JSON)
    parser.add_argument("--bundle-output-json", default=DEFAULT_BUNDLE_JSON)
    parser.add_argument("--tunnel-ssh-host", default=DEFAULT_TUNNEL_SSH_HOST)
    parser.add_argument("--tunnel-ssh-port", type=int, default=DEFAULT_TUNNEL_SSH_PORT)
    parser.add_argument("--tunnel-ssh-user", default=DEFAULT_TUNNEL_SSH_USER)
    parser.add_argument("--tunnel-remote-host", default=DEFAULT_REMOTE_INFER_HOST)
    parser.add_argument("--tunnel-remote-port", type=int, default=DEFAULT_REMOTE_INFER_PORT)
    parser.add_argument("--tunnel-local-host", help="Local tunnel bind host; defaults to infer-base-url host")
    parser.add_argument("--tunnel-local-port", type=int, help="Local tunnel bind port; defaults to infer-base-url port")
    parser.add_argument("--pytest-bin", default=validate_infer_swap_phases.sys.executable)
    parser.add_argument(
        "--phase-timeout-s",
        type=float,
        default=validate_infer_swap_phases.DEFAULT_PHASE_TIMEOUT_S,
        help="Maximum seconds allowed for each phase-gate command",
    )
    parser.add_argument("--skip-phase-tests", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-diff-check", action="store_true")
    parser.add_argument(
        "--stdout",
        choices=("summary", "json", "none"),
        default="summary",
        help="Stdout format",
    )
    return parser.parse_args(argv)


def run_launch_bundle(args: argparse.Namespace) -> InferSwapLaunchBundle:
    phase_args = validate_infer_swap_phases.parse_args(
        [
            "--output-json",
            str(args.phase_gate_output_json),
            "--pytest-bin",
            str(args.pytest_bin),
            "--phase-timeout-s",
            str(float(args.phase_timeout_s)),
            "--stdout",
            "none",
        ]
        + (["--skip-tests"] if bool(args.skip_phase_tests) else [])
        + (["--skip-compile"] if bool(args.skip_compile) else [])
        + (["--skip-diff-check"] if bool(args.skip_diff_check) else [])
    )
    base_phase_gate = validate_infer_swap_phases.run_phase_gates(phase_args)

    audit_args = audit_infer_swap_readiness.parse_args(
        [
            "--profile",
            str(args.profile),
            "--infer-base-url",
            str(args.infer_base_url),
            "--infer-model",
            str(args.infer_model),
            "--infer-timeout-s",
            str(float(args.infer_timeout_s)),
            "--infer-protocol",
            str(args.infer_protocol),
            "--infer-seed-policy",
            str(args.infer_seed_policy),
            "--max-concurrent-jobs",
            str(int(args.max_concurrent_jobs)),
            "--run-mode",
            str(args.run_mode),
            "--only-jobs",
            *[str(item) for item in args.only_jobs],
            "--only-datasets",
            *[str(item) for item in args.only_datasets],
            "--expected-queue-count",
            str(int(args.expected_queue_count)),
            "--db-host",
            str(args.db_host),
            "--db-port",
            str(int(args.db_port)),
            "--db-user",
            str(args.db_user),
            "--db-name",
            str(args.db_name),
            "--db-sslmode",
            str(args.db_sslmode),
            "--db-timeout-s",
            str(float(args.db_timeout_s)),
            "--probe-json",
            str(args.probe_json),
            "--output-json",
            str(args.readiness_output_json),
            "--preflight-output-json",
            str(args.preflight_output_json),
            "--summary-output-json",
            str(args.summary_output_json),
            "--evidence-output-md",
            str(args.evidence_output_md),
            "--stdout",
            "none",
        ]
        + _optional_pair("--infer-max-workers", args.infer_max_workers)
        + _optional_pair("--remote-batch-size", args.remote_batch_size)
    )
    readiness = audit_infer_swap_readiness.run_audit(audit_args)
    audit_infer_swap_readiness.write_audit(Path(str(args.readiness_output_json)).expanduser(), readiness)

    readiness_phase = validate_infer_swap_phases.validate_readiness_json(
        Path(str(args.readiness_output_json)).expanduser()
    )
    probe_phase = validate_infer_swap_phases.validate_probe_json(
        Path(str(args.probe_json)).expanduser(),
        readiness_phase=readiness_phase,
    )
    summary_phase = validate_infer_swap_phases.validate_summary_json(
        Path(str(args.summary_output_json)).expanduser(),
        require_all_scored=False,
    )
    generated_at_utc = datetime.now(UTC).isoformat(timespec="seconds")
    source_manifest = validate_infer_swap_phases.build_source_manifest()
    phase_gate = validate_infer_swap_phases.InferSwapPhaseGateReport(
        ok=all(phase.ok for phase in (*base_phase_gate.phases, readiness_phase, probe_phase, summary_phase)),
        phases=(*base_phase_gate.phases, readiness_phase, probe_phase, summary_phase),
        generated_at_utc=generated_at_utc,
        required_phase_names=(
            *base_phase_gate.required_phase_names,
            "readiness_json",
            "probe_json",
            "summary_json",
        ),
        source_digest=validate_infer_swap_phases.source_digest(source_manifest),
        source_manifest=source_manifest,
    )
    validate_infer_swap_phases.write_report(Path(str(args.phase_gate_output_json)).expanduser(), phase_gate)
    phase_gate_errors = tuple(
        error for phase in phase_gate.phases for error in phase.errors
    )
    queue_argv = build_run_infer_swap_eval_argv(args, action="queue")
    dispatch_argv = build_run_infer_swap_eval_argv(args, action="dispatch")
    tunnel_argv = build_tunnel_argv(args)
    summary_argv = build_summary_argv(args, watch=False, include_markdown=False)
    summary_watch_argv = build_summary_argv(args, watch=True, include_markdown=False)
    evidence_argv = build_summary_argv(args, watch=False, include_markdown=True)
    speedup_doc_argv = build_speedup_doc_argv(args)
    bundle = InferSwapLaunchBundle(
        ok=readiness.ready_to_dispatch and phase_gate.ok,
        readiness_ok=readiness.ready_to_dispatch,
        phase_gate_ok=phase_gate.ok,
        launch_parameters=build_launch_parameters(args),
        readiness_json=str(args.readiness_output_json),
        summary_json=str(args.summary_output_json),
        evidence_md=str(args.evidence_output_md),
        speedup_md=str(args.speedup_output_md),
        probe_json=str(args.probe_json),
        phase_gate_json=str(args.phase_gate_output_json),
        bundle_json=str(args.bundle_output_json) if args.bundle_output_json else None,
        readiness_errors=readiness.errors,
        phase_gate_errors=phase_gate_errors,
        generated_at_utc=generated_at_utc,
        queue_argv=queue_argv,
        dispatch_argv=dispatch_argv,
        queue_command=format_shell_command(queue_argv),
        dispatch_command=format_shell_command(dispatch_argv),
        tunnel_parameters=build_tunnel_parameters(args),
        tunnel_argv=tunnel_argv,
        tunnel_command=format_shell_command(tunnel_argv),
        summary_argv=summary_argv,
        summary_watch_argv=summary_watch_argv,
        evidence_argv=evidence_argv,
        speedup_doc_argv=speedup_doc_argv,
        summary_command=format_shell_command(summary_argv),
        summary_watch_command=format_shell_command(summary_watch_argv),
        evidence_command=format_shell_command(evidence_argv),
        speedup_doc_command=format_shell_command(speedup_doc_argv),
    )
    if args.bundle_output_json:
        write_bundle(Path(str(args.bundle_output_json)).expanduser(), bundle)
    return bundle


def write_bundle(path: Path, bundle: InferSwapLaunchBundle) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def format_bundle_summary(bundle: InferSwapLaunchBundle) -> str:
    lines = [
        f"ok={str(bundle.ok).lower()} readiness_ok={str(bundle.readiness_ok).lower()} phase_gate_ok={str(bundle.phase_gate_ok).lower()}",
        "launch="
        f"profile={bundle.launch_parameters.get('profile')} "
        f"model={bundle.launch_parameters.get('infer_model')} "
        f"workers={bundle.launch_parameters.get('infer_max_workers')} "
        f"remote_batch={bundle.launch_parameters.get('remote_batch_size')} "
        f"datasets={bundle.launch_parameters.get('dataset_count')}",
        f"readiness_json={bundle.readiness_json}",
        f"summary_json={bundle.summary_json}",
        f"phase_gate_json={bundle.phase_gate_json}",
    ]
    for error in bundle.readiness_errors:
        lines.append(f"readiness_error={error}")
    for error in bundle.phase_gate_errors:
        lines.append(f"phase_gate_error={error}")
    return "\n".join(lines)


def build_launch_parameters(args: argparse.Namespace) -> dict[str, Any]:
    workers, batch_size = run_infer_swap_eval.resolve_profile_concurrency(args)
    only_jobs = tuple(str(item) for item in args.only_jobs)
    only_datasets = tuple(str(item) for item in args.only_datasets)
    return {
        "profile": str(args.profile),
        "infer_base_url": str(args.infer_base_url),
        "infer_model": str(args.infer_model),
        "infer_timeout_s": float(args.infer_timeout_s),
        "infer_protocol": str(args.infer_protocol),
        "infer_seed_policy": str(args.infer_seed_policy),
        "infer_max_workers": workers,
        "remote_batch_size": batch_size,
        "max_concurrent_jobs": int(args.max_concurrent_jobs),
        "run_mode": str(args.run_mode),
        "only_jobs": list(only_jobs),
        "job_count": len(only_jobs),
        "only_datasets": list(only_datasets),
        "dataset_count": len(only_datasets),
        "expected_queue_count": int(args.expected_queue_count),
        "phase_timeout_s": float(args.phase_timeout_s),
        "summary_watch_interval_s": float(args.summary_watch_interval_s),
        "db_target": f"{args.db_host}:{int(args.db_port)}/{args.db_name} user={args.db_user}",
        "db_sslmode": str(args.db_sslmode),
    }


def build_tunnel_parameters(args: argparse.Namespace) -> dict[str, Any]:
    local_host, local_port = _resolve_tunnel_local_endpoint(args)
    return {
        "ssh_user": str(args.tunnel_ssh_user),
        "ssh_host": str(args.tunnel_ssh_host),
        "ssh_port": int(args.tunnel_ssh_port),
        "local_host": local_host,
        "local_port": local_port,
        "remote_host": str(args.tunnel_remote_host),
        "remote_port": int(args.tunnel_remote_port),
    }


def build_tunnel_argv(args: argparse.Namespace) -> tuple[str, ...]:
    params = build_tunnel_parameters(args)
    return (
        "ssh",
        "-p",
        str(params["ssh_port"]),
        "-N",
        "-L",
        f"{params['local_host']}:{params['local_port']}:{params['remote_host']}:{params['remote_port']}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        f"{params['ssh_user']}@{params['ssh_host']}",
    )


def build_run_infer_swap_eval_argv(args: argparse.Namespace, *, action: str) -> tuple[str, ...]:
    if action not in {"queue", "dispatch"}:
        raise ValueError(f"unsupported action: {action}")
    argv: list[str] = [
        "uv",
        "run",
        "python",
        "-m",
        "src.bin.run_infer_swap_eval",
        "--action",
        action,
        "--profile",
        str(args.profile),
        "--infer-base-url",
        str(args.infer_base_url),
        "--infer-model",
        str(args.infer_model),
        "--infer-timeout-s",
        str(float(args.infer_timeout_s)),
        "--infer-protocol",
        str(args.infer_protocol),
        "--infer-seed-policy",
        str(args.infer_seed_policy),
        "--max-concurrent-jobs",
        str(int(args.max_concurrent_jobs)),
        "--run-mode",
        str(args.run_mode),
        "--only-jobs",
        *[str(item) for item in args.only_jobs],
        "--only-datasets",
        *[str(item) for item in args.only_datasets],
        "--db-host",
        str(args.db_host),
        "--db-port",
        str(int(args.db_port)),
        "--db-user",
        str(args.db_user),
        "--db-name",
        str(args.db_name),
        "--db-sslmode",
        str(args.db_sslmode),
        "--db-timeout-s",
        str(float(args.db_timeout_s)),
        "--preflight-output-json",
        str(args.preflight_output_json),
    ]
    argv.extend(_optional_pair("--infer-max-workers", args.infer_max_workers))
    argv.extend(_optional_pair("--remote-batch-size", args.remote_batch_size))
    if action == "queue":
        argv.append("--print-scheduler-args")
    else:
        argv.extend(
            [
                "--confirm-dispatch",
                "--phase-gate-json",
                str(args.phase_gate_output_json),
                "--launch-bundle-json",
                str(args.bundle_output_json or run_infer_swap_eval.DEFAULT_LAUNCH_BUNDLE_JSON),
            ]
        )
    return tuple(argv)


def build_summary_argv(
    args: argparse.Namespace,
    *,
    watch: bool,
    include_markdown: bool,
) -> tuple[str, ...]:
    argv: list[str] = [
        "uv",
        "run",
        "python",
        "-m",
        "src.bin.summarize_infer_swap_eval",
        "--model",
        str(args.infer_model),
        "--datasets",
        *[str(item) for item in args.only_datasets],
        "--db-host",
        str(args.db_host),
        "--db-port",
        str(int(args.db_port)),
        "--db-user",
        str(args.db_user),
        "--db-name",
        str(args.db_name),
        "--db-sslmode",
        str(args.db_sslmode),
        "--db-timeout-s",
        str(float(args.db_timeout_s)),
        "--output-json",
        str(args.summary_output_json),
    ]
    if watch:
        argv.extend(["--watch", "--watch-interval-s", str(float(args.summary_watch_interval_s))])
    if include_markdown:
        argv.extend(
            [
                "--probe-json",
                str(args.probe_json),
                "--output-md",
                str(args.evidence_output_md),
            ]
        )
    return tuple(argv)


def build_speedup_doc_argv(args: argparse.Namespace) -> tuple[str, ...]:
    return (
        "uv",
        "run",
        "python",
        "-m",
        "src.bin.draft_infer_swap_speedup_doc",
        "--summary-json",
        str(args.summary_output_json),
        "--probe-json",
        str(args.probe_json),
        "--readiness-json",
        str(args.readiness_output_json),
        "--launch-bundle-json",
        str(args.bundle_output_json or DEFAULT_BUNDLE_JSON),
        "--output-md",
        str(args.speedup_output_md),
    )


def format_shell_command(argv: Sequence[str]) -> str:
    return "rtk " + shlex.join(str(item) for item in argv)


def _resolve_tunnel_local_endpoint(args: argparse.Namespace) -> tuple[str, int]:
    parsed = urlparse(str(args.infer_base_url))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise ValueError(f"infer-base-url must include scheme, host, and port: {args.infer_base_url}")
    local_host = str(args.tunnel_local_host) if args.tunnel_local_host else str(parsed.hostname)
    local_port = int(args.tunnel_local_port) if args.tunnel_local_port is not None else int(parsed.port)
    if local_port <= 0:
        raise ValueError("tunnel local port must be positive")
    if int(args.tunnel_remote_port) <= 0:
        raise ValueError("tunnel remote port must be positive")
    return local_host, local_port


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = run_launch_bundle(args)
    if args.stdout == "json":
        print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), flush=True)
    elif args.stdout == "summary":
        print(format_bundle_summary(bundle), flush=True)
    return 0 if bundle.ok else 1


def _optional_pair(name: str, value: object | None) -> list[str]:
    return [name, str(value)] if value is not None else []


__all__ = [
    "DEFAULT_BUNDLE_JSON",
    "DEFAULT_REMOTE_INFER_HOST",
    "DEFAULT_REMOTE_INFER_PORT",
    "DEFAULT_TUNNEL_SSH_HOST",
    "DEFAULT_TUNNEL_SSH_PORT",
    "DEFAULT_TUNNEL_SSH_USER",
    "InferSwapLaunchBundle",
    "build_run_infer_swap_eval_argv",
    "build_launch_parameters",
    "build_speedup_doc_argv",
    "build_summary_argv",
    "build_tunnel_argv",
    "build_tunnel_parameters",
    "format_shell_command",
    "format_bundle_summary",
    "main",
    "parse_args",
    "run_launch_bundle",
    "write_bundle",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
