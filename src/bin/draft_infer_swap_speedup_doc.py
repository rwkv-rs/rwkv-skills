from __future__ import annotations

"""Draft the post-benchmark inference-swap speedup design document from evidence."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.bin import run_infer_swap_eval


DEFAULT_SUMMARY_JSON = "/tmp/rwkv-skills-infer-swap-eval-summary.json"
DEFAULT_PROBE_JSON = "/tmp/rwkv-skills-infer-swap-probe-19082-gpu0-lightweight-full-20260607.json"
DEFAULT_READINESS_JSON = "/tmp/rwkv-skills-infer-swap-readiness-audit.json"
DEFAULT_LAUNCH_BUNDLE_JSON = "/tmp/rwkv-skills-infer-swap-launch-bundle.json"
DEFAULT_OUTPUT_MD = "/tmp/rwkv-skills-infer-swap-speedup-design.md"
MIN_PROTOCOL_SMOKE_BATCH_SIZE = 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft speedup design doc after inference-swap scores are complete")
    parser.add_argument("--summary-json", default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--probe-json", default=DEFAULT_PROBE_JSON)
    parser.add_argument("--readiness-json", default=DEFAULT_READINESS_JSON)
    parser.add_argument(
        "--launch-bundle-json",
        default="",
        help="Optional launch bundle JSON with profile/model/workers/dataset identity",
    )
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a draft even when summary all_scored=false; default refuses to avoid premature conclusions",
    )
    parser.add_argument("--stdout", choices=("summary", "json", "none"), default="summary")
    return parser.parse_args(argv)


def load_json_object(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_summary_complete(summary: Mapping[str, Any]) -> tuple[bool, str]:
    total = _as_int(summary.get("total_count"))
    scored = _as_int(summary.get("scored_count"))
    all_scored = bool(summary.get("all_scored") is True)
    if not all_scored:
        return False, f"summary all_scored=false scored={scored}/{total}"
    if total is None or scored is None or total <= 0 or scored != total:
        return False, f"summary scored_count mismatch scored={scored}/{total}"
    task_count = _as_int(summary.get("task_count"))
    if task_count != total:
        return False, f"summary task_count mismatch tasks={task_count}/{total}"
    datasets = summary.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != total:
        return False, "summary datasets length does not match total_count"
    dataset_names = [str(item.get("dataset")) for item in datasets if isinstance(item, Mapping) and item.get("dataset")]
    if len(dataset_names) != total:
        return False, "summary datasets missing dataset name"
    if len(set(dataset_names)) != len(dataset_names):
        return False, "summary datasets contain duplicate names"
    invalid_status = [
        str(item.get("dataset"))
        for item in datasets
        if isinstance(item, Mapping) and item.get("status") != "scored"
    ]
    if invalid_status:
        return False, "summary datasets not in scored status: " + ",".join(invalid_status)
    missing_task = [
        str(item.get("dataset"))
        for item in datasets
        if isinstance(item, Mapping) and _as_int(item.get("task_id")) is None
    ]
    if missing_task:
        return False, "summary datasets missing task_id: " + ",".join(missing_task)
    incomplete_task = [
        str(item.get("dataset"))
        for item in datasets
        if isinstance(item, Mapping) and item.get("task_status") != "Completed"
    ]
    if incomplete_task:
        return False, "summary datasets task not completed: " + ",".join(incomplete_task)
    missing_score_id = [
        str(item.get("dataset"))
        for item in datasets
        if isinstance(item, Mapping) and _as_int(item.get("score_id")) is None
    ]
    if missing_score_id:
        return False, "summary datasets missing score_id: " + ",".join(missing_score_id)
    missing = [str(item.get("dataset")) for item in datasets if isinstance(item, Mapping) and not item.get("metrics")]
    if missing:
        return False, "summary datasets missing metrics: " + ",".join(missing)
    return True, "complete"


def validate_launch_bundle(
    *,
    summary: Mapping[str, Any],
    probe: Mapping[str, Any],
    launch_bundle: Mapping[str, Any],
) -> tuple[bool, str]:
    if launch_bundle.get("ok") is not True:
        return False, "launch bundle ok is not true"
    if launch_bundle.get("readiness_errors") or launch_bundle.get("phase_gate_errors"):
        return False, "launch bundle contains readiness or phase gate errors"
    params = launch_bundle.get("launch_parameters")
    if not isinstance(params, Mapping):
        return False, "launch bundle missing launch_parameters"
    if params.get("infer_model") != summary.get("model"):
        return False, "launch bundle model does not match summary"
    if params.get("infer_model") != probe.get("model"):
        return False, "launch bundle model does not match probe"
    if params.get("db_target") != summary.get("db_target"):
        return False, "launch bundle db_target does not match summary"
    dataset_names = _summary_dataset_names(summary)
    bundle_datasets = params.get("only_datasets")
    if not isinstance(bundle_datasets, list):
        return False, "launch bundle only_datasets missing"
    if tuple(str(item) for item in bundle_datasets) != dataset_names:
        return False, "launch bundle datasets do not match summary"
    dataset_count = _as_int(params.get("dataset_count"))
    if dataset_count != len(dataset_names):
        return False, f"launch bundle dataset_count mismatch: {dataset_count}/{len(dataset_names)}"
    expected_queue_count = _as_int(params.get("expected_queue_count"))
    if expected_queue_count != len(dataset_names):
        return False, f"launch bundle expected_queue_count mismatch: {expected_queue_count}/{len(dataset_names)}"
    phase_timeout_s = _as_float(params.get("phase_timeout_s"))
    if phase_timeout_s is None or phase_timeout_s <= 0:
        return False, "launch bundle phase_timeout_s missing or invalid"
    summary_watch_interval_s = _as_float(params.get("summary_watch_interval_s"))
    if summary_watch_interval_s is None or summary_watch_interval_s <= 0:
        return False, "launch bundle summary_watch_interval_s missing or invalid"
    for key in (
        "tunnel_command",
        "dispatch_command",
        "summary_command",
        "summary_watch_command",
        "evidence_command",
        "speedup_doc_command",
    ):
        value = launch_bundle.get(key)
        if not isinstance(value, str) or not value:
            return False, f"launch bundle {key} missing"
    speedup_md = launch_bundle.get("speedup_md")
    if not isinstance(speedup_md, str) or not speedup_md:
        return False, "launch bundle speedup_md missing"
    return True, "complete"


def validate_readiness_evidence(
    *,
    summary: Mapping[str, Any],
    probe: Mapping[str, Any],
    launch_bundle: Mapping[str, Any] | None,
    readiness: Mapping[str, Any],
) -> tuple[bool, str]:
    if readiness.get("ready_to_dispatch") is not True:
        return False, "readiness ready_to_dispatch is not true"
    if readiness.get("errors"):
        return False, "readiness contains errors"
    total = _as_int(summary.get("total_count"))
    queue_pending = _as_int(readiness.get("queue_pending_count"))
    expected_queue = _as_int(readiness.get("expected_queue_count"))
    if total is None or queue_pending != total or expected_queue != total:
        return False, f"readiness queue count mismatch: {queue_pending}/{expected_queue}/{total}"
    if readiness.get("probe_model") != summary.get("model"):
        return False, "readiness probe_model does not match summary"
    if readiness.get("probe_model") != probe.get("model"):
        return False, "readiness probe_model does not match probe"
    if readiness.get("probe_protocol") != run_infer_swap_eval.DEFAULT_INFER_PROTOCOL:
        return False, f"readiness probe_protocol is not {run_infer_swap_eval.DEFAULT_INFER_PROTOCOL}"

    expected_workers = _as_int(readiness.get("expected_infer_max_workers"))
    expected_batch = _as_int(readiness.get("expected_remote_batch_size"))
    if launch_bundle is not None:
        params = launch_bundle.get("launch_parameters")
        if not isinstance(params, Mapping):
            return False, "launch bundle missing launch_parameters"
        if expected_workers != _as_int(params.get("infer_max_workers")):
            return False, "readiness workers do not match launch bundle"
        if expected_batch != _as_int(params.get("remote_batch_size")):
            return False, "readiness remote batch does not match launch bundle"
    required_concurrency = max(expected_workers or 0, expected_batch or 0)
    if required_concurrency <= 0:
        return False, "readiness expected workers/batch missing"
    readiness_gpu_full = _as_int(readiness.get("probe_gpu_full_concurrency"))
    if readiness_gpu_full is None or readiness_gpu_full < required_concurrency:
        return False, f"readiness gpu_full_concurrency insufficient: {readiness_gpu_full}/{required_concurrency}"
    readiness_largest = _as_int(readiness.get("probe_largest_successful_concurrency"))
    if readiness_largest is None or readiness_largest < required_concurrency:
        return False, f"readiness largest_successful_concurrency insufficient: {readiness_largest}/{required_concurrency}"
    probe_gpu_full = _as_int(probe.get("gpu_full_concurrency"))
    if probe_gpu_full is None or probe_gpu_full < required_concurrency:
        return False, f"probe gpu_full_concurrency insufficient: {probe_gpu_full}/{required_concurrency}"
    probe_largest = _as_int(probe.get("largest_successful_concurrency"))
    if probe_largest is None or probe_largest < required_concurrency:
        return False, f"probe largest_successful_concurrency insufficient: {probe_largest}/{required_concurrency}"
    return validate_protocol_smoke(readiness.get("protocol_smoke_protocols"))


def validate_protocol_smoke(protocols: Any) -> tuple[bool, str]:
    if not isinstance(protocols, list):
        return False, "readiness protocol_smoke_protocols missing"
    by_protocol = {str(item.get("protocol")): item for item in protocols if isinstance(item, Mapping)}
    for protocol in (run_infer_swap_eval.DEFAULT_INFER_PROTOCOL,):
        item = by_protocol.get(protocol)
        if item is None:
            return False, f"readiness protocol smoke missing: {protocol}"
        if item.get("ok") is not True:
            return False, f"readiness protocol smoke failed: {protocol}"
        request_count = _as_int(item.get("request_count"))
        nonempty_count = _as_int(item.get("nonempty_output_count"))
        if request_count is None or request_count < MIN_PROTOCOL_SMOKE_BATCH_SIZE:
            return False, (
                "readiness protocol smoke request_count below batched smoke requirement: "
                f"{protocol}={request_count}/{MIN_PROTOCOL_SMOKE_BATCH_SIZE}"
            )
        if nonempty_count != request_count:
            return False, f"readiness protocol smoke nonempty output mismatch: {protocol}={nonempty_count}/{request_count}"
    return True, "complete"


def build_speedup_design_doc(
    *,
    summary: Mapping[str, Any],
    probe: Mapping[str, Any],
    readiness: Mapping[str, Any] | None = None,
    launch_bundle: Mapping[str, Any] | None = None,
    allow_incomplete: bool = False,
) -> str:
    complete, reason = validate_summary_complete(summary)
    if not complete and not allow_incomplete:
        raise ValueError(reason)
    launch_bundle_ok = None
    launch_bundle_reason = "not provided"
    launch_parameters: Mapping[str, Any] = {}
    launch_bundle_generated_at = None
    launch_dispatch_command = None
    launch_tunnel_command = None
    launch_summary_command = None
    launch_summary_watch_command = None
    launch_evidence_command = None
    launch_speedup_doc_command = None
    if launch_bundle is not None:
        launch_bundle_ok, launch_bundle_reason = validate_launch_bundle(
            summary=summary,
            probe=probe,
            launch_bundle=launch_bundle,
        )
        if not launch_bundle_ok:
            raise ValueError(launch_bundle_reason)
        params = launch_bundle.get("launch_parameters")
        if isinstance(params, Mapping):
            launch_parameters = params
        launch_bundle_generated_at = launch_bundle.get("generated_at_utc")
        command = launch_bundle.get("dispatch_command")
        if isinstance(command, str) and command:
            launch_dispatch_command = command
        tunnel_command = launch_bundle.get("tunnel_command")
        if isinstance(tunnel_command, str) and tunnel_command:
            launch_tunnel_command = tunnel_command
        summary_command = launch_bundle.get("summary_command")
        if isinstance(summary_command, str) and summary_command:
            launch_summary_command = summary_command
        summary_watch_command = launch_bundle.get("summary_watch_command")
        if isinstance(summary_watch_command, str) and summary_watch_command:
            launch_summary_watch_command = summary_watch_command
        evidence_command = launch_bundle.get("evidence_command")
        if isinstance(evidence_command, str) and evidence_command:
            launch_evidence_command = evidence_command
        speedup_doc_command = launch_bundle.get("speedup_doc_command")
        if isinstance(speedup_doc_command, str) and speedup_doc_command:
            launch_speedup_doc_command = speedup_doc_command
    readiness_ok = None
    readiness_reason = "not provided"
    if readiness is not None:
        readiness_ok, readiness_reason = validate_readiness_evidence(
            summary=summary,
            probe=probe,
            launch_bundle=launch_bundle,
            readiness=readiness,
        )
        if not readiness_ok:
            raise ValueError(readiness_reason)

    lines = [
        "# Inference Swap Speedup Design",
        "",
        "## Evidence Gate",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Summary complete | {str(complete).lower()} |",
        f"| Completion reason | {_md(reason)} |",
        f"| Model | {_md(summary.get('model'))} |",
        f"| DB target | {_md(summary.get('db_target'))} |",
        f"| Scored datasets | {_md(_count_pair(summary.get('scored_count'), summary.get('total_count')))} |",
        f"| Launch bundle complete | {_md(str(launch_bundle_ok).lower() if launch_bundle_ok is not None else launch_bundle_reason)} |",
        f"| Readiness complete | {_md(str(readiness_ok).lower() if readiness_ok is not None else readiness_reason)} |",
    ]
    if launch_parameters:
        lines.extend(
            [
                f"| Launch profile | {_md(launch_parameters.get('profile'))} |",
                f"| Launch base URL | {_md(launch_parameters.get('infer_base_url'))} |",
                f"| Launch workers | {_md(launch_parameters.get('infer_max_workers'))} |",
                f"| Launch remote batch | {_md(launch_parameters.get('remote_batch_size'))} |",
                f"| Launch dataset count | {_md(launch_parameters.get('dataset_count'))} |",
                f"| Launch job count | {_md(launch_parameters.get('job_count'))} |",
                f"| Launch phase timeout | {_md(launch_parameters.get('phase_timeout_s'))} |",
                f"| Launch summary watch interval | {_md(launch_parameters.get('summary_watch_interval_s'))} |",
                f"| Launch bundle generated | {_md(launch_bundle_generated_at)} |",
            ]
        )
        if launch_tunnel_command:
            lines.append(f"| Launch tunnel command | `{_md(launch_tunnel_command)}` |")
        if launch_dispatch_command:
            lines.append(f"| Launch dispatch command | `{_md(launch_dispatch_command)}` |")
        if launch_summary_command:
            lines.append(f"| Launch summary command | `{_md(launch_summary_command)}` |")
        if launch_summary_watch_command:
            lines.append(f"| Launch summary watch command | `{_md(launch_summary_watch_command)}` |")
        if launch_evidence_command:
            lines.append(f"| Launch evidence command | `{_md(launch_evidence_command)}` |")
        if launch_speedup_doc_command:
            lines.append(f"| Launch speedup doc command | `{_md(launch_speedup_doc_command)}` |")
    if readiness is not None:
        lines.extend(
            [
                f"| Readiness ready_to_dispatch | {str(readiness.get('ready_to_dispatch') is True).lower()} |",
                f"| Readiness workers | {_md(readiness.get('expected_infer_max_workers'))} |",
                f"| Readiness remote batch | {_md(readiness.get('expected_remote_batch_size'))} |",
                f"| Readiness protocol smoke | {_md(_protocol_smoke_summary(readiness.get('protocol_smoke_protocols')))} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Remote Inference Probe",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Probe model | {_md(probe.get('model'))} |",
            f"| Probe protocol | {_md(probe.get('protocol'))} |",
            f"| GPU full concurrency | {_md(probe.get('gpu_full_concurrency'))} |",
            f"| Throughput best concurrency | {_md(probe.get('throughput_best_concurrency'))} |",
            f"| Largest successful concurrency | {_md(probe.get('largest_successful_concurrency'))} |",
            f"| Suggested workers | {_md(probe.get('suggested_infer_max_workers'))} |",
            f"| Suggested remote batch | {_md(probe.get('suggested_remote_batch_size'))} |",
            "",
            "## Formal Scores",
            "",
            "| Dataset | Status | Task | Metrics |",
            "| --- | --- | ---: | --- |",
        ]
    )
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    for item in datasets:
        if not isinstance(item, Mapping):
            continue
        metrics = item.get("metrics")
        metrics_text = json.dumps(metrics, ensure_ascii=False, sort_keys=True, separators=(",", ":")) if metrics else "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(item.get("dataset")),
                    _md(item.get("status")),
                    _md(item.get("task_id")),
                    _md(metrics_text),
                ]
            )
            + " |"
        )

    throughput_best = _as_int(probe.get("throughput_best_concurrency"))
    gpu_full = _as_int(probe.get("gpu_full_concurrency"))
    throughput_point = _probe_point(probe, throughput_best)
    gpu_full_point = _probe_point(probe, gpu_full)
    lines.extend(
        [
            "",
            "## Speedup Design",
            "",
            "### Current Evidence",
            "",
            f"- Launch profile: `{_md(launch_parameters.get('profile'))}` with `workers={_md(launch_parameters.get('infer_max_workers'))}` and `remote_batch={_md(launch_parameters.get('remote_batch_size'))}`.",
            f"- `gpu_full_concurrency={gpu_full}` is the highest concurrency that met the GPU-full criterion.",
            f"- `throughput_best_concurrency={throughput_best}` is the best measured output-throughput point.",
            f"- At throughput best: {_point_sentence(throughput_point)}",
            f"- At GPU-full profile: {_point_sentence(gpu_full_point)}",
            "",
            "### Proposed Optimization Tracks",
            "",
            "1. Keep `vllm` as the scheduler remote protocol for standard vLLM OpenAI-compatible serving.",
            "2. Use the full-load profile for maximum GPU occupancy and the throughput-peak profile for a controlled score/speed comparison.",
            "3. Compare completed benchmark scores before changing prompt, routing, or evaluator semantics.",
            "4. If scores are stable, tune only runner-side request concurrency first: `infer_max_workers` and `remote_batch_size`.",
            "5. Use task-level `completions`, `eval`, `scores`, and run logs to identify whether bottlenecks are generation, evaluator, DB writes, or external tool/sandbox calls.",
            "",
            "### Safety Rules",
            "",
            "- Do not claim a speedup from probe throughput alone; benchmark score rows must remain valid.",
            "- Do not compare partial completion progress as a final score.",
            "- Treat `scores.metrics` as the only score source for formal conclusions.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_speedup_doc(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = load_json_object(args.summary_json)
    probe = load_json_object(args.probe_json)
    readiness = None
    readiness_path = Path(str(args.readiness_json)).expanduser()
    if readiness_path.exists():
        readiness = load_json_object(readiness_path)
    else:
        message = f"readiness json missing: {readiness_path}"
        if args.stdout == "json":
            print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), flush=True)
        elif args.stdout == "summary":
            print(f"refusing to write speedup doc: {message}", flush=True)
        return 1
    launch_bundle = None
    if args.launch_bundle_json:
        launch_bundle_path = Path(str(args.launch_bundle_json)).expanduser()
        if launch_bundle_path.exists():
            launch_bundle = load_json_object(launch_bundle_path)
        else:
            message = f"launch bundle json missing: {launch_bundle_path}"
            if args.stdout == "json":
                print(json.dumps({"ok": False, "error": message}, ensure_ascii=False), flush=True)
            elif args.stdout == "summary":
                print(f"refusing to write speedup doc: {message}", flush=True)
            return 1
    try:
        content = build_speedup_design_doc(
            summary=summary,
            probe=probe,
            readiness=readiness,
            launch_bundle=launch_bundle,
            allow_incomplete=bool(args.allow_incomplete),
        )
    except ValueError as exc:
        if args.stdout == "json":
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        elif args.stdout == "summary":
            print(f"refusing to write speedup doc: {exc}", flush=True)
        return 1
    write_speedup_doc(Path(args.output_md).expanduser(), content)
    if args.stdout == "json":
        print(json.dumps({"ok": True, "output_md": str(args.output_md)}, ensure_ascii=False), flush=True)
    elif args.stdout == "summary":
        print(f"wrote speedup design doc: {args.output_md}", flush=True)
    return 0


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_pair(value: Any, total: Any) -> str:
    parsed = _as_int(value)
    parsed_total = _as_int(total)
    if parsed is None or parsed_total is None:
        return "unknown"
    return f"{parsed}/{parsed_total}"


def _summary_dataset_names(summary: Mapping[str, Any]) -> tuple[str, ...]:
    datasets = summary.get("datasets")
    if not isinstance(datasets, list):
        return ()
    return tuple(str(item.get("dataset")) for item in datasets if isinstance(item, Mapping) and item.get("dataset"))


def _protocol_smoke_summary(protocols: Any) -> str:
    if not isinstance(protocols, list):
        return "none"
    parts: list[str] = []
    for item in protocols:
        if not isinstance(item, Mapping):
            continue
        protocol = item.get("protocol")
        nonempty = _as_int(item.get("nonempty_output_count"))
        request_count = _as_int(item.get("request_count"))
        status = "ok" if item.get("ok") is True else "failed"
        parts.append(f"{protocol}:{status}:{_count_pair(nonempty, request_count)}")
    return ",".join(parts) if parts else "none"


def _probe_point(probe: Mapping[str, Any], concurrency: int | None) -> Mapping[str, Any] | None:
    points = probe.get("points")
    if concurrency is None or not isinstance(points, list):
        return None
    for point in points:
        if isinstance(point, Mapping) and _as_int(point.get("concurrency")) == concurrency:
            return point
    return None


def _point_sentence(point: Mapping[str, Any] | None) -> str:
    if point is None:
        return "point not present in probe JSON."
    return (
        f"concurrency={point.get('concurrency')}, "
        f"rps={_fmt_number(point.get('rps'))}, "
        f"output_chars_per_s={_fmt_number(point.get('output_chars_per_s'))}, "
        f"avg_gpu={_fmt_number(point.get('avg_gpu_utilization'))}, "
        f"peak_gpu={_fmt_number(point.get('peak_gpu_utilization'))}."
    )


def _fmt_number(value: Any) -> str:
    if value is None:
        return "none"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _md(value: Any) -> str:
    if value is None:
        return "none"
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "DEFAULT_LAUNCH_BUNDLE_JSON",
    "DEFAULT_OUTPUT_MD",
    "DEFAULT_PROBE_JSON",
    "DEFAULT_READINESS_JSON",
    "DEFAULT_SUMMARY_JSON",
    "MIN_PROTOCOL_SMOKE_BATCH_SIZE",
    "build_speedup_design_doc",
    "load_json_object",
    "main",
    "parse_args",
    "validate_launch_bundle",
    "validate_protocol_smoke",
    "validate_readiness_evidence",
    "validate_summary_complete",
    "write_speedup_doc",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
