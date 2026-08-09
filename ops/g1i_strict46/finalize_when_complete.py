#!/usr/bin/env python3
"""Build the strict-46 delivery only after the audited 184-cell gate passes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

from ops.g1i_strict46.audit_current import TARGETS


TARGET_CELLS = 184
MODEL_CELLS = 46
MODEL_NAMES = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
)
EXPECTED_CELLS = frozenset(
    (model_name, benchmark_name, benchmark_split)
    for model_name in MODEL_NAMES
    for benchmark_name, benchmark_split in TARGETS
)
REQUIRED_DELIVERY_ARTIFACTS = {
    "frontend_scores.json",
    "coverage.csv",
    "task_provenance.json",
    "completion_gate.json",
    "anomaly_audit.json",
    "score_replays.json",
    "knowledge_replay_diagnostics.json",
    "protocol_repair_history.md",
    "truncation_matrix.json",
    "truncation_vs_parameter_size.csv",
    "truncation_vs_g1x.csv",
    "truncation_vs_parameter_size.md",
    "truncation_vs_g1x.md",
}
PROTOCOL_ZERO_COUNTERS = (
    "stored_replay_mismatches",
    "sampler_argmax_mismatches",
    "blank_primary_generation_count",
    "missing_prediction_count",
    "leading_orphan_close_count",
    "truncated_blank_evaluator_answer",
)


def _remaining_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return -1


def _audit_traceability_reasons(rows: object) -> list[str]:
    if not isinstance(rows, list):
        return ["valid_task_rows=missing"]
    reasons: list[str] = []
    task_ids: set[int] = set()
    score_ids: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            reasons.append(f"valid_task_rows[{index}]=invalid")
            continue
        task_id = row.get("task_id")
        score_id = row.get("score_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id <= 0:
            reasons.append(f"valid_task_rows[{index}].task_id={task_id}")
        elif task_id in task_ids:
            reasons.append(f"duplicate_task_id={task_id}")
        else:
            task_ids.add(task_id)
        if not isinstance(score_id, int) or isinstance(score_id, bool) or score_id <= 0:
            reasons.append(f"valid_task_rows[{index}].score_id={score_id}")
        elif score_id in score_ids:
            reasons.append(f"duplicate_score_id={score_id}")
        else:
            score_ids.add(score_id)
        expected = row.get("expected_completion_count")
        observed = (
            row.get("completion_count"),
            row.get("distinct_completion_coordinates"),
            row.get("eval_count"),
        )
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            reasons.append(
                f"valid_task_rows[{index}].expected_completion_count={expected}"
            )
        elif observed != (expected, expected, expected):
            reasons.append(
                f"valid_task_rows[{index}].counts={observed},expected={expected}"
            )
        if str(row.get("status") or "") != "Completed":
            reasons.append(f"valid_task_rows[{index}].status={row.get('status')}")
    return reasons


def completion_gate_reasons(audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(audit.get("target_cells") or 0) != TARGET_CELLS:
        reasons.append(f"target_cells={audit.get('target_cells')}")
    if int(audit.get("valid_complete") or 0) != TARGET_CELLS:
        reasons.append(f"valid_complete={audit.get('valid_complete')}")
    remaining = _remaining_count(audit.get("remaining"))
    if remaining != 0:
        reasons.append(f"remaining={audit.get('remaining')}")
    for key in (
        "active_protocol_issues",
        "unresolved_active_target_tasks",
        "unresolved_failed_target_tasks",
    ):
        if audit.get(key):
            reasons.append(f"{key}={len(audit[key])}")

    models = audit.get("models")
    if not isinstance(models, dict):
        reasons.append("models=missing")
    else:
        for model in MODEL_NAMES:
            row = models.get(model)
            if not isinstance(row, dict):
                reasons.append(f"model_missing={model}")
                continue
            if int(row.get("complete") or 0) != MODEL_CELLS:
                reasons.append(f"{model}:complete={row.get('complete')}")
            if int(row.get("missing") or 0) != 0:
                reasons.append(f"{model}:missing={row.get('missing')}")

    valid_rows = audit.get("valid_task_rows")
    if not isinstance(valid_rows, list) or len(valid_rows) != TARGET_CELLS:
        reasons.append(
            f"valid_task_rows={len(valid_rows) if isinstance(valid_rows, list) else 'missing'}"
        )
    else:
        reasons.extend(_audit_traceability_reasons(valid_rows))
    return reasons


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _run(repo: Path, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _delivery_cell(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("model_name") or ""),
        str(row.get("benchmark_name") or ""),
        str(row.get("benchmark_split") or ""),
    )


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    parsed = [_positive_int(item) for item in value]
    if any(item is None for item in parsed):
        return None
    result = [int(item) for item in parsed if item is not None]
    return result if len(result) == len(set(result)) else None


def _csv_positive_int_list(value: Any) -> list[int] | None:
    if value in (None, ""):
        return []
    return _positive_int_list(str(value).split(";"))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _audit_row_cell(row: dict[str, Any]) -> tuple[str, str, str]:
    benchmark = str(row.get("benchmark") or "")
    if "__" in benchmark:
        benchmark_name, benchmark_split = benchmark.split("__", 1)
    else:
        benchmark_name = str(row.get("benchmark_name") or "")
        benchmark_split = str(row.get("benchmark_split") or "")
    return str(row.get("model_name") or ""), benchmark_name, benchmark_split


def verify_delivery_artifacts(output_dir: Path) -> list[str]:
    """Verify the self-contained final delivery, not only its file hashes."""

    reasons: list[str] = []
    manifest_path = output_dir / "delivery_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest_unavailable={exc}"]
    if not isinstance(manifest, dict):
        return ["manifest=invalid"]

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return ["manifest_artifacts=missing"]
    artifact_names = [
        str(row.get("name"))
        for row in artifacts
        if isinstance(row, dict) and row.get("name")
    ]
    if len(artifact_names) != len(artifacts):
        reasons.append("manifest_artifacts=invalid_rows")
    if len(artifact_names) != len(set(artifact_names)):
        reasons.append("manifest_artifacts=duplicate_names")
    by_name = {
        str(row.get("name")): row
        for row in artifacts
        if isinstance(row, dict) and row.get("name")
    }
    missing = sorted(REQUIRED_DELIVERY_ARTIFACTS - set(by_name))
    if missing:
        reasons.append(f"manifest_missing={missing}")
    for name, row in by_name.items():
        if Path(name).name != name or Path(name).is_absolute():
            reasons.append(f"artifact_name_unsafe={name}")
            continue
        path = output_dir / name
        if not path.is_file():
            reasons.append(f"artifact_missing={name}")
            continue
        if path.stat().st_size != int(row.get("size_bytes") or -1):
            reasons.append(f"artifact_size_mismatch={name}")
        if _sha256(path) != row.get("sha256"):
            reasons.append(f"artifact_hash_mismatch={name}")

    database = str(manifest.get("database") or "")
    source_audit_generated_at = str(
        manifest.get("source_audit_generated_at") or ""
    )
    if not database:
        reasons.append("manifest_database=missing")
    if not source_audit_generated_at:
        reasons.append("manifest_source_audit_generated_at=missing")
    if int(manifest.get("target_cells") or 0) != TARGET_CELLS:
        reasons.append(f"manifest_target_cells={manifest.get('target_cells')}")
    if manifest.get("models") != list(MODEL_NAMES):
        reasons.append("manifest_models=mismatch")

    try:
        frontend = json.loads(
            (output_dir / "frontend_scores.json").read_text(encoding="utf-8")
        )
        provenance = json.loads(
            (output_dir / "task_provenance.json").read_text(encoding="utf-8")
        )
        with (output_dir / "coverage.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            coverage = list(csv.DictReader(handle))
        gate = json.loads(
            (output_dir / "completion_gate.json").read_text(encoding="utf-8")
        )
        anomaly = json.loads(
            (output_dir / "anomaly_audit.json").read_text(encoding="utf-8")
        )
        score_replays = json.loads(
            (output_dir / "score_replays.json").read_text(encoding="utf-8")
        )
        knowledge_replay = json.loads(
            (output_dir / "knowledge_replay_diagnostics.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        reasons.append(f"delivery_views_unavailable={exc}")
        return reasons

    if not isinstance(gate, dict):
        reasons.append("completion_gate=invalid")
    else:
        expected_gate_values = {
            "complete": True,
            "target_cells": TARGET_CELLS,
            "valid_cells": TARGET_CELLS,
            "missing_cells": 0,
            "remaining_cells": 0,
            "unresolved_running": 0,
            "unresolved_failed": 0,
            "active_protocol_issues": 0,
        }
        for field, expected_value in expected_gate_values.items():
            if gate.get(field) != expected_value:
                reasons.append(f"completion_gate_{field}={gate.get(field)}")
        if gate.get("traceability_issues") != []:
            reasons.append(
                f"completion_gate_traceability_issues={gate.get('traceability_issues')}"
            )
        if str(gate.get("source_audit_generated_at") or "") != source_audit_generated_at:
            reasons.append("completion_gate_source_audit=mismatch")

    if not isinstance(frontend, dict):
        reasons.append("frontend=invalid")
        frontend_rows = None
    else:
        frontend_rows = frontend.get("rows")
        expected_frontend_values = {
            "database": database,
            "source_audit_generated_at": source_audit_generated_at,
            "target_cells": TARGET_CELLS,
            "valid_cells": TARGET_CELLS,
            "complete": True,
        }
        for field, expected_value in expected_frontend_values.items():
            if frontend.get(field) != expected_value:
                reasons.append(f"frontend_{field}={frontend.get(field)}")

    if not isinstance(anomaly, dict):
        reasons.append("anomaly_audit=invalid")
    else:
        if str(anomaly.get("database") or "") != database:
            reasons.append("anomaly_database=mismatch")
        if str(anomaly.get("source_audit_generated_at") or "") != source_audit_generated_at:
            reasons.append("anomaly_source_audit=mismatch")
        if int(anomaly.get("target_cells") or 0) != TARGET_CELLS:
            reasons.append(f"anomaly_target_cells={anomaly.get('target_cells')}")
        if int(anomaly.get("valid_cells") or 0) != TARGET_CELLS:
            reasons.append(f"anomaly_valid_cells={anomaly.get('valid_cells')}")
        if _remaining_count(anomaly.get("remaining_cells")) != 0:
            reasons.append(f"anomaly_remaining_cells={anomaly.get('remaining_cells')}")
        for field in (
            "active_protocol_issues",
            "unresolved_active_target_tasks",
            "unresolved_failed_target_tasks",
        ):
            if anomaly.get(field) != []:
                reasons.append(f"anomaly_{field}={anomaly.get(field)}")
        for field in (
            "active_target_tasks",
            "superseded_active_target_tasks",
            "failed_target_tasks",
            "superseded_failed_target_tasks",
            "superseded_valid_target_tasks",
            "invalid_scored_tasks",
            "choice_bias_signals",
            "curve_comparisons",
            "curve_inversions_over_5pp",
            "reference_comparisons",
            "reference_differences_over_5pp",
        ):
            if not isinstance(anomaly.get(field), list):
                reasons.append(f"anomaly_{field}=missing")
        for field in (
            "task_status_counts",
            "invalid_reason_counts",
            "truncation_examples_by_task",
        ):
            if not isinstance(anomaly.get(field), dict):
                reasons.append(f"anomaly_{field}=missing")
        quality_evidence = anomaly.get("quality_evidence")
        if not isinstance(quality_evidence, list) or not quality_evidence:
            reasons.append("anomaly_quality_evidence=missing")
        elif any(
            not isinstance(row, dict)
            or not str(row.get("evidence_file") or "")
            or not isinstance(row.get("payload"), dict)
            for row in quality_evidence
        ):
            reasons.append("anomaly_quality_evidence=invalid")
        else:
            for index, row in enumerate(quality_evidence):
                payload = row["payload"]
                aggregate = payload.get("aggregate")
                if isinstance(aggregate, dict):
                    unresolved = {
                        field: _number(aggregate[field])
                        for field in PROTOCOL_ZERO_COUNTERS
                        if field in aggregate and _number(aggregate[field]) != 0.0
                    }
                    if unresolved:
                        reasons.append(
                            f"anomaly_quality_evidence[{index}]_protocol="
                            f"{unresolved}"
                        )
                if "decision" in payload:
                    decision = payload.get("decision")
                    if (
                        not isinstance(decision, dict)
                        or decision.get("accept_tasks") is not True
                        or decision.get("retest_required") is not False
                    ):
                        reasons.append(
                            f"anomaly_quality_evidence[{index}]_decision=unresolved"
                        )

    if not isinstance(knowledge_replay, dict):
        reasons.append("knowledge_replay=invalid")
    else:
        if str(knowledge_replay.get("database") or "") != database:
            reasons.append("knowledge_replay_database=mismatch")
        knowledge_tasks = knowledge_replay.get("tasks")
        diagnostic_ids = _positive_int_list(
            knowledge_replay.get("diagnostic_only_task_ids")
        )
        if not isinstance(knowledge_tasks, list):
            reasons.append("knowledge_replay_tasks=missing")
        if diagnostic_ids is None:
            reasons.append("knowledge_replay_diagnostic_ids=invalid")
        elif isinstance(knowledge_tasks, list):
            actual_diagnostic_ids = sorted(
                int(row["task_id"])
                for row in knowledge_tasks
                if isinstance(row, dict)
                and row.get("diagnostic_only") is True
                and _positive_int(row.get("task_id")) is not None
            )
            if sorted(diagnostic_ids) != actual_diagnostic_ids:
                reasons.append("knowledge_replay_diagnostic_ids=mismatch")

    for name, rows in (
        ("frontend", frontend_rows),
        ("provenance", provenance),
        ("coverage", coverage),
    ):
        if not isinstance(rows, list) or len(rows) != TARGET_CELLS:
            reasons.append(
                f"{name}_rows={len(rows) if isinstance(rows, list) else 'missing'}"
            )
    if not all(isinstance(rows, list) for rows in (frontend_rows, provenance, coverage)):
        return reasons

    views = {
        "frontend": {_delivery_cell(row): row for row in frontend_rows},
        "provenance": {_delivery_cell(row): row for row in provenance},
        "coverage": {_delivery_cell(row): row for row in coverage},
    }
    for name, mapping in views.items():
        if len(mapping) != TARGET_CELLS:
            reasons.append(f"{name}_unique_cells={len(mapping)}")
        if set(mapping) != EXPECTED_CELLS:
            reasons.append(
                f"{name}_strict46_cells=mismatch:"
                f"missing={len(EXPECTED_CELLS - set(mapping))},"
                f"extra={len(set(mapping) - EXPECTED_CELLS)}"
            )
    if not (
        set(views["frontend"])
        == set(views["provenance"])
        == set(views["coverage"])
        == EXPECTED_CELLS
    ):
        reasons.append("delivery_cell_sets=mismatch")
        return reasons

    task_ids: set[int] = set()
    score_ids: set[int] = set()
    history_by_field: dict[str, dict[tuple[str, str, str], set[int]]] = {
        field: {}
        for field in (
            "invalid_historical_task_ids",
            "failed_task_ids",
            "superseded_failed_task_ids",
            "superseded_active_task_ids",
            "superseded_valid_task_ids",
        )
    }
    for cell in sorted(EXPECTED_CELLS):
        frontend_row = views["frontend"][cell]
        provenance_row = views["provenance"][cell]
        coverage_row = views["coverage"][cell]
        task_values = (
            _positive_int(frontend_row.get("task_id")),
            _positive_int(provenance_row.get("final_task_id")),
            _positive_int(coverage_row.get("task_id")),
        )
        score_values = (
            _positive_int(frontend_row.get("score_id")),
            _positive_int(provenance_row.get("final_score_id")),
            _positive_int(coverage_row.get("score_id")),
        )
        if None in task_values or len(set(task_values)) != 1:
            reasons.append(f"task_provenance_mismatch={cell}:{task_values}")
        elif task_values[0] in task_ids:
            reasons.append(f"duplicate_final_task_id={task_values[0]}")
        else:
            task_ids.add(int(task_values[0]))
        if None in score_values or len(set(score_values)) != 1:
            reasons.append(f"score_provenance_mismatch={cell}:{score_values}")
        elif score_values[0] in score_ids:
            reasons.append(f"duplicate_final_score_id={score_values[0]}")
        else:
            score_ids.add(int(score_values[0]))

        for row_name, row, status_field in (
            ("frontend", frontend_row, "status"),
            ("provenance", provenance_row, "final_task_status"),
            ("coverage", coverage_row, "status"),
        ):
            if str(row.get("source_database") or "") != database:
                reasons.append(f"{row_name}_source_database_mismatch={cell}")
            if str(row.get(status_field) or "") != "Completed":
                reasons.append(f"{row_name}_status_mismatch={cell}")
        if frontend_row.get("coverage_status") != "valid":
            reasons.append(f"frontend_coverage_status={cell}")
        if provenance_row.get("coverage_status") != "valid":
            reasons.append(f"provenance_coverage_status={cell}")
        if coverage_row.get("coverage_status") != "valid":
            reasons.append(f"coverage_status={cell}")

        expected_values = tuple(
            _positive_int(row.get("expected_completion_count"))
            for row in (frontend_row, provenance_row, coverage_row)
        )
        observed_values = tuple(
            tuple(
                _positive_int(row.get(field))
                for field in (
                    "completion_count",
                    "distinct_completion_coordinates",
                    "eval_count",
                )
            )
            for row in (frontend_row, provenance_row, coverage_row)
        )
        if (
            None in expected_values
            or len(set(expected_values)) != 1
            or any(observed != (expected_values[0],) * 3 for observed in observed_values)
        ):
            reasons.append(
                f"completion_eval_provenance_mismatch={cell}:"
                f"observed={observed_values},expected={expected_values}"
            )

        metric_names = tuple(
            str(row.get("metric_name") or "")
            for row in (frontend_row, provenance_row, coverage_row)
        )
        metric_values = tuple(
            _number(row.get("metric_value"))
            for row in (frontend_row, provenance_row, coverage_row)
        )
        if not metric_names[0] or len(set(metric_names)) != 1:
            reasons.append(f"metric_name_mismatch={cell}:{metric_names}")
        if None in metric_values or len(set(metric_values)) != 1:
            reasons.append(f"metric_value_mismatch={cell}:{metric_values}")
        if not isinstance(frontend_row.get("metrics"), dict) or (
            frontend_row.get("metrics") != provenance_row.get("metrics")
        ):
            reasons.append(f"metrics_payload_mismatch={cell}")
        for field in ("evaluator", "cot_mode", "score_created_at"):
            values = tuple(
                str(row.get(field) or "")
                for row in (frontend_row, provenance_row, coverage_row)
            )
            if not values[0] or len(set(values)) != 1:
                reasons.append(f"{field}_mismatch={cell}:{values}")
        if (
            not isinstance(frontend_row.get("sampling_config"), dict)
            or not frontend_row.get("sampling_config")
            or frontend_row.get("sampling_config")
            != provenance_row.get("sampling_config")
        ):
            reasons.append(f"sampling_config_mismatch={cell}")

        for field in ("active_task_ids", "unresolved_failed_task_ids"):
            values = (
                _positive_int_list(frontend_row.get(field)),
                _positive_int_list(provenance_row.get(field)),
                _csv_positive_int_list(coverage_row.get(field)),
            )
            if values != ([], [], []):
                reasons.append(f"unresolved_history={cell}:{field}:{values}")
        final_task_id = task_values[0]
        for field in history_by_field:
            values = (
                _positive_int_list(frontend_row.get(field)),
                _positive_int_list(provenance_row.get(field)),
                _csv_positive_int_list(coverage_row.get(field)),
            )
            if None in values or not (values[0] == values[1] == values[2]):
                reasons.append(f"historical_task_ids_mismatch={cell}:{field}")
                continue
            ids = set(values[0] or [])
            if final_task_id in ids:
                reasons.append(f"historical_task_reuses_final={cell}:{field}")
            history_by_field[field][cell] = ids
        failed_ids = history_by_field["failed_task_ids"].get(cell, set())
        superseded_failed_ids = history_by_field[
            "superseded_failed_task_ids"
        ].get(cell, set())
        if not superseded_failed_ids.issubset(failed_ids):
            reasons.append(f"superseded_failed_not_in_failed={cell}")

    if isinstance(anomaly, dict):
        anomaly_links = {
            "invalid_scored_tasks": "invalid_historical_task_ids",
            "failed_target_tasks": "failed_task_ids",
            "superseded_failed_target_tasks": "superseded_failed_task_ids",
            "superseded_active_target_tasks": "superseded_active_task_ids",
            "superseded_valid_target_tasks": "superseded_valid_task_ids",
        }
        all_historical_ids: set[int] = set()
        for anomaly_field, provenance_field in anomaly_links.items():
            rows = anomaly.get(anomaly_field)
            if not isinstance(rows, list):
                continue
            expected_by_cell: dict[tuple[str, str, str], set[int]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    reasons.append(f"anomaly_{anomaly_field}=invalid_row")
                    continue
                task_id = _positive_int(row.get("task_id"))
                cell = _audit_row_cell(row)
                if task_id is None or cell not in EXPECTED_CELLS:
                    reasons.append(
                        f"anomaly_{anomaly_field}=invalid_mapping:{cell}:{task_id}"
                    )
                    continue
                expected_by_cell.setdefault(cell, set()).add(task_id)
                all_historical_ids.add(task_id)
            actual_by_cell = history_by_field[provenance_field]
            for cell in EXPECTED_CELLS:
                if expected_by_cell.get(cell, set()) != actual_by_cell.get(cell, set()):
                    reasons.append(
                        f"historical_mapping_mismatch={cell}:{anomaly_field}"
                    )
        active_ids = {
            _positive_int(row.get("task_id"))
            for row in anomaly.get("active_target_tasks", [])
            if isinstance(row, dict)
        }
        superseded_active_ids = {
            _positive_int(row.get("task_id"))
            for row in anomaly.get("superseded_active_target_tasks", [])
            if isinstance(row, dict)
        }
        failed_ids = {
            _positive_int(row.get("task_id"))
            for row in anomaly.get("failed_target_tasks", [])
            if isinstance(row, dict)
        }
        superseded_failed_ids = {
            _positive_int(row.get("task_id"))
            for row in anomaly.get("superseded_failed_target_tasks", [])
            if isinstance(row, dict)
        }
        if active_ids != superseded_active_ids:
            reasons.append("anomaly_active_tasks=not_fully_superseded")
        if failed_ids != superseded_failed_ids:
            reasons.append("anomaly_failed_tasks=not_fully_superseded")
    else:
        all_historical_ids = set()

    if not isinstance(score_replays, list):
        reasons.append("score_replays=invalid")
    else:
        replay_pairs: set[tuple[int, int]] = set()
        known_task_ids = task_ids | all_historical_ids
        for index, row in enumerate(score_replays):
            if not isinstance(row, dict):
                reasons.append(f"score_replays[{index}]=invalid")
                continue
            source_task_id = _positive_int(row.get("source_task_id"))
            replay_task_id = _positive_int(row.get("replay_task_id"))
            pair = (source_task_id, replay_task_id)
            if None in pair or source_task_id == replay_task_id:
                reasons.append(f"score_replays[{index}]=invalid_ids:{pair}")
                continue
            normalized_pair = (int(source_task_id), int(replay_task_id))
            if normalized_pair in replay_pairs:
                reasons.append(f"score_replays[{index}]=duplicate:{normalized_pair}")
            replay_pairs.add(normalized_pair)
            if source_task_id not in all_historical_ids:
                reasons.append(f"score_replays[{index}]=unknown_source:{source_task_id}")
            if replay_task_id not in known_task_ids:
                reasons.append(f"score_replays[{index}]=unknown_replay:{replay_task_id}")
            if not str(row.get("evidence_file") or ""):
                reasons.append(f"score_replays[{index}]=missing_evidence_file")
    return reasons


def _commands(
    repo: Path,
    audit: Path,
    truncation: Path,
    output_dir: Path,
) -> list[tuple[str, list[str]]]:
    python = str(repo / ".venv/bin/python")
    return [
        (
            "refresh_audit",
            [python, "ops/g1i_strict46/audit_current.py", "--output", str(audit)],
        ),
        (
            "truncation_matrix",
            [
                python,
                "ops/g1i_strict46/report_final_truncation_matrix.py",
                "--output",
                str(truncation),
                "--size-table",
                str(output_dir / "truncation_vs_parameter_size.csv"),
                "--family-table",
                str(output_dir / "truncation_vs_g1x.csv"),
            ],
        ),
        (
            "delivery",
            [
                python,
                "ops/g1i_strict46/build_final_artifacts.py",
                "--audit",
                str(audit),
                "--output-dir",
                str(output_dir),
                "--truncation-matrix",
                str(truncation),
            ],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument(
        "--audit", type=Path,
        default=Path("logs/audits/g1i_strict46_current.json"),
    )
    parser.add_argument(
        "--truncation", type=Path,
        default=Path("logs/audits/g1ghi_final_truncation_matrix_20260807.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("logs/audits/g1i_strict46_final_20260807"),
    )
    parser.add_argument(
        "--state", type=Path,
        default=Path("logs/audits/g1i_strict46_finalizer_state.json"),
    )
    parser.add_argument(
        "--lock", type=Path,
        default=Path("logs/audits/g1i_strict46_finalizer.lock"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    audit_path = args.audit if args.audit.is_absolute() else repo / args.audit
    truncation_path = args.truncation if args.truncation.is_absolute() else repo / args.truncation
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    state_path = args.state if args.state.is_absolute() else repo / args.state
    lock_path = args.lock if args.lock.is_absolute() else repo / args.lock
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                reasons = completion_gate_reasons(audit)
            except (OSError, json.JSONDecodeError) as exc:
                reasons = [f"audit_unavailable={exc}"]

            if reasons:
                _write_json(
                    state_path,
                    {
                        "status": "waiting",
                        "checked_at": datetime.now().astimezone(),
                        "reasons": reasons,
                    },
                )
                if args.once:
                    return 3
                time.sleep(max(1.0, args.interval_s))
                continue

            failed = False
            for name, command in _commands(repo, audit_path, truncation_path, output_dir):
                completed = _run(repo, command)
                if completed.returncode:
                    _write_json(
                        state_path,
                        {
                            "status": "failed",
                            "checked_at": datetime.now().astimezone(),
                            "step": name,
                            "returncode": completed.returncode,
                            "stdout_tail": completed.stdout[-4000:],
                            "stderr_tail": completed.stderr[-4000:],
                        },
                    )
                    failed = True
                    break
                if name == "refresh_audit":
                    refreshed = json.loads(audit_path.read_text(encoding="utf-8"))
                    refreshed_reasons = completion_gate_reasons(refreshed)
                    if refreshed_reasons:
                        _write_json(
                            state_path,
                            {
                                "status": "waiting",
                                "checked_at": datetime.now().astimezone(),
                                "reasons": refreshed_reasons,
                            },
                        )
                        failed = True
                        break

            if failed:
                if args.once:
                    return 4
                time.sleep(max(1.0, args.interval_s))
                continue

            gate = json.loads((output_dir / "completion_gate.json").read_text(encoding="utf-8"))
            if not gate.get("complete"):
                _write_json(
                    state_path,
                    {
                        "status": "failed",
                        "checked_at": datetime.now().astimezone(),
                        "step": "verify_delivery_gate",
                        "gate": gate,
                    },
                )
                if args.once:
                    return 5
                time.sleep(max(1.0, args.interval_s))
                continue

            verification_reasons = verify_delivery_artifacts(output_dir)
            if verification_reasons:
                _write_json(
                    state_path,
                    {
                        "status": "failed",
                        "checked_at": datetime.now().astimezone(),
                        "step": "verify_delivery_artifacts",
                        "reasons": verification_reasons,
                    },
                )
                if args.once:
                    return 6
                time.sleep(max(1.0, args.interval_s))
                continue

            _write_json(
                state_path,
                {
                    "status": "complete",
                    "completed_at": datetime.now().astimezone(),
                    "output_dir": str(output_dir),
                    "gate": gate,
                },
            )
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
