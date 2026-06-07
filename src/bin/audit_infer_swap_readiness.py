from __future__ import annotations

"""One-command readiness audit for the inference-swap formal eval path."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.bin import run_infer_swap_eval

if TYPE_CHECKING:
    from src.bin import summarize_infer_swap_eval


DEFAULT_PROBE_JSON = "/tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json"
DEFAULT_AUDIT_JSON = "/tmp/rwkv-skills-infer-swap-readiness-audit.json"
DEFAULT_SUMMARY_JSON = "/tmp/rwkv-skills-infer-swap-eval-summary.json"
DEFAULT_EVIDENCE_MD = "/tmp/rwkv-skills-infer-swap-eval-evidence.md"


@dataclass(slots=True, frozen=True)
class QueuePreviewResult:
    rc: int
    argv: tuple[str, ...]
    stdout: str


@dataclass(slots=True, frozen=True)
class InferenceSwapReadinessAudit:
    preflight_ok: bool
    queue_rc: int | None
    queue_argv: tuple[str, ...]
    queue_stdout: str
    queue_pending_count: int | None
    expected_queue_count: int
    summary_all_scored: bool | None
    summary_task_count: int | None
    summary_scored_count: int | None
    summary_total_count: int | None
    protocol_smoke_ok: bool | None
    protocol_smoke_protocols: tuple[dict[str, Any], ...]
    probe_path: str | None
    probe_loaded: bool
    probe_model: str | None
    probe_protocol: str | None
    probe_gpu_full_concurrency: int | None
    probe_throughput_best_concurrency: int | None
    probe_largest_successful_concurrency: int | None
    expected_infer_max_workers: int
    expected_remote_batch_size: int
    errors: tuple[str, ...] = ()

    @property
    def ready_to_dispatch(self) -> bool:
        return self.preflight_ok and self.queue_rc == 0 and self.probe_loaded and not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ready_to_dispatch"] = self.ready_to_dispatch
        return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit inference-swap formal eval readiness without dispatching")
    parser.add_argument("--profile", choices=tuple(run_infer_swap_eval.PROFILE_CONCURRENCY), default="full-load")
    parser.add_argument("--infer-base-url", default=run_infer_swap_eval.DEFAULT_INFER_BASE_URL)
    parser.add_argument("--infer-model", default=run_infer_swap_eval.DEFAULT_INFER_MODEL)
    parser.add_argument("--infer-timeout-s", type=float, default=600.0)
    parser.add_argument("--infer-max-workers", type=int)
    parser.add_argument("--remote-batch-size", type=int)
    parser.add_argument("--max-concurrent-jobs", type=int, default=1)
    parser.add_argument("--run-mode", default="new", choices=("auto", "new", "resume", "rerun"))
    parser.add_argument("--only-jobs", nargs="+", default=list(run_infer_swap_eval.DEFAULT_JOBS))
    parser.add_argument("--only-datasets", nargs="+", default=list(run_infer_swap_eval.DEFAULT_DATASETS))
    parser.add_argument(
        "--expected-queue-count",
        type=int,
        default=len(run_infer_swap_eval.DEFAULT_DATASETS),
        help="Require the scheduler queue preview to report this many pending tasks",
    )
    parser.add_argument("--db-host", default=run_infer_swap_eval.DEFAULT_DB_HOST)
    parser.add_argument("--db-port", type=int, default=run_infer_swap_eval.DEFAULT_DB_PORT)
    parser.add_argument("--db-user", default=run_infer_swap_eval.DEFAULT_DB_USER)
    parser.add_argument("--db-name", default=run_infer_swap_eval.DEFAULT_DB_NAME)
    parser.add_argument("--db-sslmode", default="prefer")
    parser.add_argument("--db-timeout-s", type=float, default=5.0)
    parser.add_argument("--probe-json", default=DEFAULT_PROBE_JSON)
    parser.add_argument("--allow-missing-probe", action="store_true")
    parser.add_argument("--output-json", default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--preflight-output-json", default=run_infer_swap_eval.DEFAULT_OUTPUT_JSON)
    parser.add_argument("--summary-output-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--evidence-output-md", default=DEFAULT_EVIDENCE_MD)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-queue", action="store_true")
    parser.add_argument(
        "--stdout",
        choices=("summary", "json", "none"),
        default="summary",
        help="Audit stdout format",
    )
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> InferenceSwapReadinessAudit:
    errors: list[str] = []
    run_infer_swap_eval.apply_db_env(args)
    expected_workers, expected_batch_size = run_infer_swap_eval.resolve_profile_concurrency(args)
    preflight_ok = True if bool(args.skip_preflight) else False
    protocol_smoke_ok: bool | None = None
    protocol_smoke_protocols: tuple[dict[str, Any], ...] = ()
    preflight_module = run_infer_swap_eval.load_preflight_module()
    if not bool(args.skip_preflight):
        preflight_result = preflight_module.run_preflight(
            infer_base_url=str(args.infer_base_url),
            infer_model=str(args.infer_model),
            infer_timeout_s=float(args.infer_timeout_s),
            protocols=("openai", "nano-vllm-contents"),
            batch_size=run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE,
            max_tokens=16,
            check_db=True,
            db_timeout_s=float(args.db_timeout_s),
            db_config=run_infer_swap_eval.db_config_from_args(args),
        )
        preflight_ok = bool(preflight_result.ok)
        protocol_smoke_ok, protocol_smoke_protocols = extract_protocol_smoke(preflight_result)
        if args.preflight_output_json:
            preflight_module.write_preflight_result(Path(args.preflight_output_json).expanduser(), preflight_result)
        if not preflight_ok:
            errors.append("preflight failed")
        errors.extend(validate_protocol_smoke(protocol_smoke_protocols))

    queue = QueuePreviewResult(rc=0, argv=(), stdout="")
    if not bool(args.skip_queue):
        queue = run_queue_preview(args)
        if queue.rc != 0:
            errors.append(f"queue preview failed rc={queue.rc}")
    queue_pending_count = parse_queue_pending_count(queue.stdout) if not bool(args.skip_queue) else None
    expected_queue_count = int(args.expected_queue_count)
    if not bool(args.skip_queue) and queue.rc == 0:
        if queue_pending_count is None:
            errors.append(f"queue pending count missing: expected {expected_queue_count}")
        elif queue_pending_count != expected_queue_count:
            errors.append(
                f"queue pending count mismatch: expected {expected_queue_count}, got {queue_pending_count}"
            )

    summary = None
    try:
        summary = run_summary(args)
        if args.summary_output_json:
            from src.bin import summarize_infer_swap_eval

            summarize_infer_swap_eval.write_summary(Path(args.summary_output_json).expanduser(), summary)
    except Exception as exc:  # noqa: BLE001 - audit should report all failing evidence sources.
        errors.append(f"summary failed: {exc}")

    probe_payload = None
    probe_path = str(args.probe_json or "") or None
    if probe_path:
        probe_file = Path(probe_path).expanduser()
        if probe_file.exists():
            from src.bin import summarize_infer_swap_eval

            probe_payload = summarize_infer_swap_eval.load_probe_payload(probe_file)
            errors.extend(
                validate_probe_payload(
                    probe_payload,
                    args,
                    expected_workers=expected_workers,
                    expected_batch_size=expected_batch_size,
                )
            )
        elif not bool(args.allow_missing_probe):
            errors.append(f"probe json missing: {probe_file}")
    elif not bool(args.allow_missing_probe):
        errors.append("probe json not configured")

    if summary is not None and args.evidence_output_md:
        from src.bin import summarize_infer_swap_eval

        summarize_infer_swap_eval.write_markdown_report(
            Path(args.evidence_output_md).expanduser(),
            summary,
            probe_payload=probe_payload,
        )

    return InferenceSwapReadinessAudit(
        preflight_ok=preflight_ok,
        queue_rc=queue.rc if not bool(args.skip_queue) else None,
        queue_argv=queue.argv,
        queue_stdout=queue.stdout,
        queue_pending_count=queue_pending_count,
        expected_queue_count=expected_queue_count,
        summary_all_scored=summary.all_scored if summary is not None else None,
        summary_task_count=summary.task_count if summary is not None else None,
        summary_scored_count=summary.scored_count if summary is not None else None,
        summary_total_count=summary.total_count if summary is not None else None,
        protocol_smoke_ok=protocol_smoke_ok,
        protocol_smoke_protocols=protocol_smoke_protocols,
        probe_path=probe_path,
        probe_loaded=probe_payload is not None,
        probe_model=_optional_str_from_mapping(probe_payload, "model"),
        probe_protocol=_optional_str_from_mapping(probe_payload, "protocol"),
        probe_gpu_full_concurrency=_optional_int_from_mapping(probe_payload, "gpu_full_concurrency"),
        probe_throughput_best_concurrency=_optional_int_from_mapping(probe_payload, "throughput_best_concurrency"),
        probe_largest_successful_concurrency=_optional_int_from_mapping(probe_payload, "largest_successful_concurrency"),
        expected_infer_max_workers=expected_workers,
        expected_remote_batch_size=expected_batch_size,
        errors=tuple(errors),
    )


def run_queue_preview(args: argparse.Namespace) -> QueuePreviewResult:
    queue_args = argparse.Namespace(**vars(args))
    queue_args.action = "queue"
    scheduler_args = tuple(run_infer_swap_eval.build_scheduler_args(queue_args))
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        rc = int(run_infer_swap_eval.run_scheduler_cli(scheduler_args))
    return QueuePreviewResult(rc=rc, argv=scheduler_args, stdout=stdout.getvalue())


def parse_queue_pending_count(stdout: str) -> int | None:
    patterns = (
        re.compile(r"待调度任务\s*[：:]\s*(\d+)"),
        re.compile(r"pending\s+tasks?\s*[：:=]\s*(\d+)", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.search(stdout)
        if match:
            return int(match.group(1))
    return None


def extract_protocol_smoke(preflight_result: Any) -> tuple[bool | None, tuple[dict[str, Any], ...]]:
    try:
        payload = preflight_result.to_dict()
    except AttributeError:
        return None, ()
    if not isinstance(payload, Mapping):
        return None, ()
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return None, ()
    for check in checks:
        if not isinstance(check, Mapping) or check.get("name") != "protocol_smoke":
            continue
        details = check.get("details")
        if not isinstance(details, Mapping):
            return bool(check.get("ok") is True), ()
        protocols = details.get("protocols")
        if not isinstance(protocols, list):
            return bool(check.get("ok") is True), ()
        return (
            bool(check.get("ok") is True),
            tuple(_protocol_smoke_item(item) for item in protocols if isinstance(item, Mapping)),
        )
    return None, ()


def validate_protocol_smoke(protocols: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    by_protocol = {str(item.get("protocol")): item for item in protocols if item.get("protocol") is not None}
    errors: list[str] = []
    for protocol in ("openai", "nano-vllm-contents"):
        item = by_protocol.get(protocol)
        if item is None:
            errors.append(f"protocol smoke missing: {protocol}")
            continue
        if item.get("ok") is not True:
            errors.append(f"protocol smoke failed: {protocol}")
        request_count = _optional_int_from_mapping(item, "request_count")
        nonempty_count = _optional_int_from_mapping(item, "nonempty_output_count")
        if request_count is None or request_count < run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE:
            errors.append(
                "protocol smoke request_count below batched smoke requirement: "
                f"{protocol}={request_count}/{run_infer_swap_eval.DEFAULT_PROTOCOL_SMOKE_BATCH_SIZE}"
            )
        elif nonempty_count != request_count:
            errors.append(
                f"protocol smoke nonempty output mismatch: {protocol}={nonempty_count}/{request_count}"
            )
    return tuple(errors)


def _protocol_smoke_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": item.get("protocol"),
        "ok": bool(item.get("ok") is True),
        "status": item.get("status"),
        "request_count": _optional_int_from_mapping(item, "request_count"),
        "output_count": _optional_int_from_mapping(item, "output_count"),
        "nonempty_output_count": _optional_int_from_mapping(item, "nonempty_output_count"),
        "output_chars": _optional_int_from_mapping(item, "output_chars"),
        "error": item.get("error"),
    }


def run_summary(args: argparse.Namespace) -> "summarize_infer_swap_eval.InferSwapEvalSummary":
    from src.bin import summarize_infer_swap_eval

    summary_args = summarize_infer_swap_eval.parse_args(
        [
            "--model",
            str(args.infer_model),
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
            "--datasets",
            *[str(item) for item in args.only_datasets],
        ]
    )
    return summarize_infer_swap_eval.run_summary_once(summary_args)


def write_audit(path: Path, audit: InferenceSwapReadinessAudit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_audit_summary(audit: InferenceSwapReadinessAudit) -> str:
    lines = [
        (
            f"ready_to_dispatch={str(audit.ready_to_dispatch).lower()} "
            f"preflight_ok={str(audit.preflight_ok).lower()} "
            f"queue_rc={audit.queue_rc} "
            f"queue_pending={_count_pair(audit.queue_pending_count, audit.expected_queue_count)} "
            f"workers={audit.expected_infer_max_workers} "
            f"remote_batch={audit.expected_remote_batch_size} "
            f"protocol_smoke_ok={str(audit.protocol_smoke_ok).lower()} "
            f"summary_tasks={_count_pair(audit.summary_task_count, audit.summary_total_count)} "
            f"summary_scored={_count_pair(audit.summary_scored_count, audit.summary_total_count)} "
            f"probe_loaded={str(audit.probe_loaded).lower()}"
        )
    ]
    if audit.probe_loaded:
        lines.append(
            "probe="
            f"model={audit.probe_model} "
            f"protocol={audit.probe_protocol} "
            f"gpu_full={audit.probe_gpu_full_concurrency} "
            f"throughput_best={audit.probe_throughput_best_concurrency} "
            f"largest_successful={audit.probe_largest_successful_concurrency}"
        )
    if audit.protocol_smoke_protocols:
        lines.append(
            "protocol_smoke="
            + ",".join(
                f"{item.get('protocol')}:{str(item.get('ok') is True).lower()}"
                for item in audit.protocol_smoke_protocols
            )
        )
    for error in audit.errors:
        lines.append(f"error={error}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = run_audit(args)
    if args.output_json:
        write_audit(Path(args.output_json).expanduser(), audit)
    if args.stdout == "json":
        print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2), flush=True)
    elif args.stdout == "summary":
        print(format_audit_summary(audit), flush=True)
    return 0 if audit.ready_to_dispatch else 1


def _optional_int_from_mapping(payload: Mapping[str, Any] | None, key: str) -> int | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _optional_str_from_mapping(payload: Mapping[str, Any] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


def validate_probe_payload(
    payload: Mapping[str, Any] | None,
    args: argparse.Namespace,
    *,
    expected_workers: int,
    expected_batch_size: int,
) -> tuple[str, ...]:
    if payload is None:
        return ("probe json did not contain an object",)
    errors: list[str] = []
    probe_model = _optional_str_from_mapping(payload, "model")
    if probe_model != str(args.infer_model):
        errors.append(f"probe model mismatch: expected {args.infer_model}, got {probe_model}")
    probe_protocol = _optional_str_from_mapping(payload, "protocol")
    if probe_protocol != "nano-vllm-contents":
        errors.append(f"probe protocol mismatch: expected nano-vllm-contents, got {probe_protocol}")
    required_concurrency = max(int(expected_workers), int(expected_batch_size))
    largest_successful = _optional_int_from_mapping(payload, "largest_successful_concurrency")
    if largest_successful is None or largest_successful < required_concurrency:
        errors.append(
            "probe largest_successful_concurrency insufficient: "
            f"expected >= {required_concurrency}, got {largest_successful}"
        )
    gpu_full = _optional_int_from_mapping(payload, "gpu_full_concurrency")
    if gpu_full is None or gpu_full < required_concurrency:
        errors.append(f"probe gpu_full_concurrency insufficient: expected >= {required_concurrency}, got {gpu_full}")
    return tuple(errors)


def _count_pair(value: int | None, total: int | None) -> str:
    if value is None or total is None:
        return "unknown"
    return f"{value}/{total}"


__all__ = [
    "DEFAULT_AUDIT_JSON",
    "DEFAULT_EVIDENCE_MD",
    "DEFAULT_PROBE_JSON",
    "DEFAULT_SUMMARY_JSON",
    "InferenceSwapReadinessAudit",
    "QueuePreviewResult",
    "format_audit_summary",
    "main",
    "parse_args",
    "parse_queue_pending_count",
    "run_audit",
    "run_queue_preview",
    "run_summary",
    "extract_protocol_smoke",
    "validate_protocol_smoke",
    "validate_probe_payload",
    "write_audit",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
