from __future__ import annotations

"""Run the non-dispatch phase gates for the inference engine swap."""

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from src.bin import audit_infer_swap_readiness, run_infer_swap_eval


DEFAULT_OUTPUT_JSON = "/tmp/rwkv-skills-infer-swap-phase-gate.json"
PHASE_GATE_SCHEMA_VERSION = 2
DEFAULT_PHASE_TIMEOUT_S = 600.0
DEFAULT_COMPILE_TARGETS = (
    "src/infer/backend.py",
    "src/infer/api.py",
    "src/infer/server.py",
    "src/bin/run_infer_router.py",
    "src/bin/run_infer_fleet.py",
    "src/eval/scheduler/remote_profiler.py",
    "src/bin/verify_remote_infer_swap.py",
    "src/bin/probe_remote_infer.py",
    "src/bin/preflight_remote_eval.py",
    "src/bin/run_infer_swap_eval.py",
    "src/bin/summarize_infer_swap_eval.py",
    "src/bin/audit_infer_swap_readiness.py",
    "src/bin/draft_infer_swap_speedup_doc.py",
    "src/bin/prepare_infer_swap_launch_bundle.py",
    "src/bin/validate_infer_swap_phases.py",
)
DEFAULT_DIFF_CHECK_ARGV = ("git", "diff", "--check")

PHASE_TESTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "backend_protocol",
        (
            "tests/test_infer_split.py",
            "tests/test_scheduler_remote_inference.py",
            "tests/test_main_config.py",
        ),
    ),
    (
        "router_fleet",
        (
            "tests/test_infer_router.py",
            "tests/test_infer_fleet.py",
        ),
    ),
    (
        "remote_comparison_and_probe",
        (
            "tests/test_verify_remote_infer_swap.py",
            "tests/test_preflight_remote_eval.py",
            "tests/test_probe_remote_infer.py",
        ),
    ),
    (
        "formal_eval_guard",
        (
            "tests/test_run_infer_swap_eval.py",
            "tests/test_summarize_infer_swap_eval.py",
            "tests/test_audit_infer_swap_readiness.py",
            "tests/test_draft_infer_swap_speedup_doc.py",
            "tests/test_prepare_infer_swap_launch_bundle.py",
            "tests/test_validate_infer_swap_phases.py",
        ),
    ),
)
REQUIRED_PHASE_NAMES = tuple(name for name, _tests in PHASE_TESTS) + (
    "compile_targets",
    "git_diff_check",
)


@dataclass(slots=True, frozen=True)
class PhaseGateResult:
    name: str
    kind: str
    ok: bool
    rc: int | None = None
    argv: tuple[str, ...] = ()
    elapsed_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    details: Mapping[str, Any] | None = None
    errors: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class InferSwapPhaseGateReport:
    ok: bool
    phases: tuple[PhaseGateResult, ...]
    schema_version: int = PHASE_GATE_SCHEMA_VERSION
    generated_at_utc: str = ""
    required_phase_names: tuple[str, ...] = REQUIRED_PHASE_NAMES
    source_digest: str = ""
    source_manifest: Mapping[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "ok": self.ok,
            "required_phase_names": list(self.required_phase_names),
            "source_digest": self.source_digest,
            "source_manifest": dict(self.source_manifest or {}),
            "phases": [asdict(phase) for phase in self.phases],
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference-swap non-dispatch phase gates")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--pytest-bin", default=sys.executable, help="Python executable used with -m pytest")
    parser.add_argument(
        "--phase-timeout-s",
        type=float,
        default=DEFAULT_PHASE_TIMEOUT_S,
        help="Maximum seconds allowed for each command phase",
    )
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-diff-check", action="store_true")
    parser.add_argument("--readiness-json", help="Optional readiness audit JSON to validate")
    parser.add_argument("--probe-json", help="Optional remote probe JSON to validate")
    parser.add_argument("--summary-json", help="Optional formal eval summary JSON to include")
    parser.add_argument("--require-summary-all-scored", action="store_true")
    parser.add_argument(
        "--stdout",
        choices=("summary", "json", "none"),
        default="summary",
        help="Stdout format",
    )
    return parser.parse_args(argv)


def run_phase_gates(args: argparse.Namespace) -> InferSwapPhaseGateReport:
    phases: list[PhaseGateResult] = []
    required_phase_names = list(REQUIRED_PHASE_NAMES)
    phase_timeout_s = max(1.0, float(args.phase_timeout_s))
    if not bool(args.skip_tests):
        for name, tests in PHASE_TESTS:
            phases.append(
                run_command_phase(
                    name,
                    [str(args.pytest_bin), "-m", "pytest", "-q", *tests],
                    timeout_s=phase_timeout_s,
                )
            )
    else:
        for name, _tests in PHASE_TESTS:
            _remove_required_phase(required_phase_names, name)
    if not bool(args.skip_compile):
        phases.append(
            run_command_phase(
                "compile_targets",
                [sys.executable, "-m", "compileall", "-q", *DEFAULT_COMPILE_TARGETS],
                timeout_s=phase_timeout_s,
            )
        )
    else:
        _remove_required_phase(required_phase_names, "compile_targets")
    if not bool(args.skip_diff_check):
        phases.append(run_command_phase("git_diff_check", DEFAULT_DIFF_CHECK_ARGV, timeout_s=phase_timeout_s))
    else:
        _remove_required_phase(required_phase_names, "git_diff_check")
    if args.readiness_json:
        phases.append(validate_readiness_json(Path(str(args.readiness_json)).expanduser()))
        required_phase_names.append("readiness_json")
    if args.probe_json:
        phases.append(validate_probe_json(Path(str(args.probe_json)).expanduser(), readiness_phase=_find_phase(phases, "readiness_json")))
        required_phase_names.append("probe_json")
    if args.summary_json:
        phases.append(
            validate_summary_json(
                Path(str(args.summary_json)).expanduser(),
                require_all_scored=bool(args.require_summary_all_scored),
            )
        )
        required_phase_names.append("summary_json")
    source_manifest = build_source_manifest()
    return InferSwapPhaseGateReport(
        ok=all(phase.ok for phase in phases),
        phases=tuple(phases),
        generated_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        required_phase_names=tuple(dict.fromkeys(required_phase_names)),
        source_digest=source_digest(source_manifest),
        source_manifest=source_manifest,
    )


def run_command_phase(name: str, argv: Sequence[str], *, timeout_s: float | None = None) -> PhaseGateResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603 - argv is built without shell=True.
            list(argv),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return PhaseGateResult(
            name=name,
            kind="command",
            ok=False,
            rc=None,
            argv=tuple(str(item) for item in argv),
            elapsed_s=max(0.0, time.perf_counter() - started),
            stdout_tail=_tail(stdout),
            stderr_tail=_tail(stderr),
            errors=(f"command timed out after {float(timeout_s or 0.0):.1f}s",),
        )
    return PhaseGateResult(
        name=name,
        kind="command",
        ok=completed.returncode == 0,
        rc=int(completed.returncode),
        argv=tuple(str(item) for item in argv),
        elapsed_s=max(0.0, time.perf_counter() - started),
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def validate_readiness_json(path: Path) -> PhaseGateResult:
    payload, load_errors = _load_json_object(path)
    if payload is None:
        return PhaseGateResult(
            name="readiness_json",
            kind="evidence",
            ok=False,
            details={"path": str(path)},
            errors=tuple(load_errors),
        )
    errors = validate_readiness_payload(payload)
    return PhaseGateResult(
        name="readiness_json",
        kind="evidence",
        ok=not errors,
        details={
            "path": str(path),
            "ready_to_dispatch": payload.get("ready_to_dispatch"),
            "queue_pending_count": payload.get("queue_pending_count"),
            "expected_queue_count": payload.get("expected_queue_count"),
            "probe_model": payload.get("probe_model"),
            "probe_protocol": payload.get("probe_protocol"),
            "probe_gpu_full_concurrency": payload.get("probe_gpu_full_concurrency"),
            "probe_largest_successful_concurrency": payload.get("probe_largest_successful_concurrency"),
            "expected_infer_max_workers": payload.get("expected_infer_max_workers"),
            "expected_remote_batch_size": payload.get("expected_remote_batch_size"),
            "protocol_smoke_ok": payload.get("protocol_smoke_ok"),
            "protocol_smoke_protocols": payload.get("protocol_smoke_protocols"),
        },
        errors=tuple(errors),
    )


def validate_readiness_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("ready_to_dispatch") is not True:
        errors.append("readiness ready_to_dispatch is not true")
    payload_errors = payload.get("errors")
    if payload_errors:
        errors.append(f"readiness errors not empty: {payload_errors}")
    if payload.get("queue_pending_count") != payload.get("expected_queue_count"):
        errors.append(
            "readiness queue count mismatch: "
            f"{payload.get('queue_pending_count')} != {payload.get('expected_queue_count')}"
        )
    if payload.get("probe_model") != run_infer_swap_eval.DEFAULT_INFER_MODEL:
        errors.append(f"readiness probe model mismatch: {payload.get('probe_model')}")
    if payload.get("probe_protocol") != run_infer_swap_eval.DEFAULT_INFER_PROTOCOL:
        errors.append(f"readiness probe protocol mismatch: {payload.get('probe_protocol')}")
    if payload.get("protocol_smoke_ok") is not True:
        errors.append(f"readiness protocol_smoke_ok is not true: {payload.get('protocol_smoke_ok')}")
    protocol_smoke_protocols = payload.get("protocol_smoke_protocols")
    if not isinstance(protocol_smoke_protocols, list):
        errors.append("readiness protocol_smoke_protocols missing")
    else:
        errors.extend(audit_infer_swap_readiness.validate_protocol_smoke(protocol_smoke_protocols))
    expected_workers = _as_int(payload.get("expected_infer_max_workers"))
    expected_batch = _as_int(payload.get("expected_remote_batch_size"))
    required = max(expected_workers or 0, expected_batch or 0)
    gpu_full = _as_int(payload.get("probe_gpu_full_concurrency"))
    largest_successful = _as_int(payload.get("probe_largest_successful_concurrency"))
    if required <= 0:
        errors.append("readiness expected workers/batch missing")
    if gpu_full is None or gpu_full < required:
        errors.append(f"readiness gpu_full_concurrency insufficient: {gpu_full} < {required}")
    if largest_successful is None or largest_successful < required:
        errors.append(f"readiness largest_successful_concurrency insufficient: {largest_successful} < {required}")
    return tuple(errors)


def validate_probe_json(path: Path, *, readiness_phase: PhaseGateResult | None = None) -> PhaseGateResult:
    payload, load_errors = _load_json_object(path)
    if payload is None:
        return PhaseGateResult(
            name="probe_json",
            kind="evidence",
            ok=False,
            details={"path": str(path)},
            errors=tuple(load_errors),
        )
    expected_workers = _detail_int(readiness_phase, "expected_infer_max_workers")
    expected_batch = _detail_int(readiness_phase, "expected_remote_batch_size")
    if expected_workers is None:
        expected_workers = int(payload.get("suggested_infer_max_workers") or 0)
    if expected_batch is None:
        expected_batch = int(payload.get("suggested_remote_batch_size") or 0)
    probe_args = audit_infer_swap_readiness.parse_args(
        [
            "--infer-model",
            run_infer_swap_eval.DEFAULT_INFER_MODEL,
            "--infer-max-workers",
            str(max(1, expected_workers)),
            "--remote-batch-size",
            str(max(1, expected_batch)),
        ]
    )
    errors = list(
        audit_infer_swap_readiness.validate_probe_payload(
            payload,
            probe_args,
            expected_workers=max(1, expected_workers),
            expected_batch_size=max(1, expected_batch),
        )
    )
    return PhaseGateResult(
        name="probe_json",
        kind="evidence",
        ok=not errors,
        details={
            "path": str(path),
            "model": payload.get("model"),
            "protocol": payload.get("protocol"),
            "gpu_full_concurrency": payload.get("gpu_full_concurrency"),
            "throughput_best_concurrency": payload.get("throughput_best_concurrency"),
            "largest_successful_concurrency": payload.get("largest_successful_concurrency"),
            "required_concurrency": max(1, expected_workers, expected_batch),
        },
        errors=tuple(errors),
    )


def validate_summary_json(path: Path, *, require_all_scored: bool) -> PhaseGateResult:
    payload, load_errors = _load_json_object(path)
    if payload is None:
        return PhaseGateResult(
            name="summary_json",
            kind="evidence",
            ok=False,
            details={"path": str(path)},
            errors=tuple(load_errors),
        )
    errors: list[str] = []
    total_count = _as_int(payload.get("total_count"))
    if total_count != len(run_infer_swap_eval.DEFAULT_DATASETS):
        errors.append(f"summary total_count mismatch: expected 9, got {total_count}")
    if require_all_scored and payload.get("all_scored") is not True:
        errors.append("summary all_scored is not true")
    return PhaseGateResult(
        name="summary_json",
        kind="evidence",
        ok=not errors,
        details={
            "path": str(path),
            "all_scored": payload.get("all_scored"),
            "task_count": payload.get("task_count"),
            "scored_count": payload.get("scored_count"),
            "total_count": payload.get("total_count"),
            "datasets": [
                item.get("dataset")
                for item in payload.get("datasets", [])
                if isinstance(item, dict) and item.get("dataset") is not None
            ],
        },
        errors=tuple(errors),
    )


def write_report(path: Path, report: InferSwapPhaseGateReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def phase_gate_source_paths() -> tuple[str, ...]:
    paths: list[str] = list(DEFAULT_COMPILE_TARGETS)
    for _name, tests in PHASE_TESTS:
        paths.extend(tests)
    return tuple(dict.fromkeys(paths))


def build_source_manifest(paths: Sequence[str] | None = None) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for raw_path in paths or phase_gate_source_paths():
        path = Path(str(raw_path))
        if not path.exists():
            manifest[str(raw_path)] = "missing"
            continue
        manifest[str(raw_path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def source_digest(manifest: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(manifest.items())), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def format_report_summary(report: InferSwapPhaseGateReport) -> str:
    lines = [f"ok={str(report.ok).lower()} phases={len(report.phases)}"]
    for phase in report.phases:
        prefix = f"{phase.name}: ok={str(phase.ok).lower()} kind={phase.kind}"
        if phase.rc is not None:
            prefix += f" rc={phase.rc}"
        if phase.errors:
            prefix += " errors=" + "; ".join(phase.errors)
        lines.append(prefix)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_phase_gates(args)
    if args.output_json:
        write_report(Path(str(args.output_json)).expanduser(), report)
    if args.stdout == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), flush=True)
    elif args.stdout == "summary":
        print(format_report_summary(report), flush=True)
    return 0 if report.ok else 1


def _load_json_object(path: Path) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    if not path.exists():
        return None, (f"json missing: {path}",)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report load failures as phase errors.
        return None, (f"json load failed: {exc}",)
    if not isinstance(payload, dict):
        return None, (f"json root is not an object: {path}",)
    return payload, ()


def _find_phase(phases: Sequence[PhaseGateResult], name: str) -> PhaseGateResult | None:
    for phase in phases:
        if phase.name == name:
            return phase
    return None


def _remove_required_phase(names: list[str], name: str) -> None:
    try:
        names.remove(name)
    except ValueError:
        pass


def _detail_int(phase: PhaseGateResult | None, key: str) -> int | None:
    if phase is None or phase.details is None:
        return None
    return _as_int(phase.details.get(key))


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _tail(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


__all__ = [
    "DEFAULT_COMPILE_TARGETS",
    "DEFAULT_DIFF_CHECK_ARGV",
    "DEFAULT_OUTPUT_JSON",
    "InferSwapPhaseGateReport",
    "PHASE_TESTS",
    "PHASE_GATE_SCHEMA_VERSION",
    "PhaseGateResult",
    "REQUIRED_PHASE_NAMES",
    "build_source_manifest",
    "format_report_summary",
    "main",
    "parse_args",
    "phase_gate_source_paths",
    "run_command_phase",
    "run_phase_gates",
    "source_digest",
    "validate_probe_json",
    "validate_readiness_json",
    "validate_readiness_payload",
    "validate_summary_json",
    "write_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
