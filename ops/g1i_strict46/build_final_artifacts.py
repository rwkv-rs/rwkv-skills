#!/usr/bin/env python3
"""Build traceable strict-46 delivery artifacts from the authoritative audit.

The default mode is a completion gate: it writes diagnostic artifacts but
returns non-zero until all 184 cells are valid.  ``--allow-incomplete`` is for
previewing the artifact shape while evaluation is still running.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from ops.g1i_strict46.audit_current import MODELS, MODEL_SIZE_BY_NAME, TARGETS
from ops.g1i_strict46.report_final_truncation_matrix import (
    write_summary_csv,
    write_summary_markdown,
)


PROTOCOL_ZERO_COUNTERS = (
    "stored_replay_mismatches",
    "sampler_argmax_mismatches",
    "blank_primary_generation_count",
    "missing_prediction_count",
    "leading_orphan_close_count",
    "truncated_blank_evaluator_answer",
)


def _cell_key(row: dict[str, Any]) -> tuple[str, str, str]:
    benchmark = str(row.get("benchmark") or "")
    if "__" in benchmark:
        name, split = benchmark.split("__", 1)
    else:
        name = str(row.get("benchmark_name") or "")
        split = str(row.get("benchmark_split") or "")
    return str(row.get("model_name") or ""), name, split


def _primary_metric(metrics: object) -> tuple[str | None, float | None]:
    if not isinstance(metrics, dict):
        return None, None
    scalar = {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    avg = sorted(
        ((int(key.split("@", 1)[1]), key, value) for key, value in scalar.items() if key.startswith("avg@") and key.split("@", 1)[1].isdigit()),
        reverse=True,
    )
    if avg:
        _k, key, value = avg[0]
        return key, value
    for preferred in ("accuracy", "exact_accuracy", "prompt_accuracy", "pass@1"):
        if preferred in scalar:
            return preferred, scalar[preferred]
    if scalar:
        key = sorted(scalar)[0]
        return key, scalar[key]
    return None, None


def build_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    valid_by_key = {_cell_key(row): row for row in audit.get("valid_task_rows", [])}
    active_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in audit.get("unresolved_active_target_tasks", []):
        active_by_key.setdefault(_cell_key(row), []).append(int(row["task_id"]))
    superseded_active_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in audit.get("superseded_active_target_tasks", []):
        superseded_active_by_key.setdefault(_cell_key(row), []).append(
            int(row["task_id"])
        )
    superseded_valid_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in audit.get("superseded_valid_target_tasks", []):
        superseded_valid_by_key.setdefault(_cell_key(row), []).append(
            int(row["task_id"])
        )
    failed_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in audit.get("unresolved_failed_target_tasks", []):
        failed_by_key.setdefault(_cell_key(row), []).append(int(row["task_id"]))
    historical_failed_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in audit.get("failed_target_tasks", []):
        historical_failed_by_key.setdefault(_cell_key(row), []).append(
            int(row["task_id"])
        )
    superseded_failed_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in audit.get("superseded_failed_target_tasks", []):
        superseded_failed_by_key.setdefault(_cell_key(row), []).append(
            int(row["task_id"])
        )
    invalid_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in audit.get("invalid_scored_tasks", []):
        invalid_by_key.setdefault(_cell_key(row), []).append(int(row["task_id"]))

    rows: list[dict[str, Any]] = []
    domain_order = {"knowledge": 0, "math": 1, "coding": 2, "instruction_following": 3}
    targets = sorted(TARGETS.items(), key=lambda item: (domain_order[item[1][0]], item[0]))
    for model_name in MODELS:
        for (benchmark_name, benchmark_split), (domain, required_mode) in targets:
            key = (model_name, benchmark_name, benchmark_split)
            valid = valid_by_key.get(key)
            metric_name, metric_value = _primary_metric(valid.get("metrics") if valid else None)
            rows.append(
                {
                    "source_database": audit.get("database"),
                    "model_name": model_name,
                    "model_size": MODEL_SIZE_BY_NAME[model_name],
                    "domain": domain,
                    "benchmark_name": benchmark_name,
                    "benchmark_split": benchmark_split,
                    "source_benchmark_name": (
                        valid.get("source_benchmark_name")
                        if valid
                        else None
                    ) or benchmark_name,
                    "source_benchmark_split": (
                        valid.get("source_benchmark_split")
                        if valid
                        else None
                    ) or benchmark_split,
                    "required_mode": required_mode,
                    "coverage_status": "valid" if valid else "missing",
                    "task_id": int(valid["task_id"]) if valid else None,
                    "score_id": valid.get("score_id") if valid else None,
                    "score_created_at": valid.get("score_created_at") if valid else None,
                    "task_created_at": valid.get("task_created_at") if valid else None,
                    "status": valid.get("status") if valid else None,
                    "evaluator": valid.get("evaluator") if valid else None,
                    "cot_mode": valid.get("cot_mode") if valid else None,
                    "sampling_config": valid.get("sampling_config") if valid else None,
                    "representative_prompt_tail": valid.get("representative_prompt_tail") if valid else None,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "metrics": valid.get("metrics") if valid else None,
                    "completion_count": valid.get("completion_count") if valid else None,
                    "expected_completion_count": valid.get("expected_completion_count") if valid else None,
                    "distinct_completion_coordinates": valid.get("distinct_completion_coordinates") if valid else None,
                    "eval_count": valid.get("eval_count") if valid else None,
                    "eval_pass_rate": valid.get("eval_pass_rate") if valid else None,
                    "blank_primary_generation_count": valid.get("blank_primary_generation_count") if valid else None,
                    "missing_prediction_count": valid.get("missing_prediction_count") if valid else None,
                    "leading_orphan_close_count": valid.get("leading_orphan_close_count") if valid else None,
                    "overall_truncation_count": valid.get("overall_truncation_count") if valid else None,
                    "overall_truncation_rate": valid.get("overall_truncation_rate") if valid else None,
                    "initial_generation_truncation_count": valid.get("initial_generation_truncation_count") if valid else None,
                    "final_stage_truncation_count": valid.get("final_stage_truncation_count") if valid else None,
                    "active_task_ids": sorted(active_by_key.get(key, [])),
                    "superseded_active_task_ids": sorted(
                        superseded_active_by_key.get(key, [])
                    ),
                    "superseded_valid_task_ids": sorted(
                        superseded_valid_by_key.get(key, [])
                    ),
                    "unresolved_failed_task_ids": sorted(
                        failed_by_key.get(key, [])
                    ),
                    "failed_task_ids": sorted(
                        historical_failed_by_key.get(key, [])
                    ),
                    "superseded_failed_task_ids": sorted(
                        superseded_failed_by_key.get(key, [])
                    ),
                    "invalid_historical_task_ids": sorted(invalid_by_key.get(key, [])),
                }
            )
    return rows


def traceability_reasons(rows: list[dict[str, Any]]) -> list[str]:
    """Prove that every valid cell resolves to task/completion/eval/score rows."""

    reasons: list[str] = []
    seen_task_ids: set[int] = set()
    seen_score_ids: set[int] = set()
    for row in rows:
        if row.get("coverage_status") != "valid":
            continue
        cell = "/".join(
            str(row.get(key) or "")
            for key in ("model_name", "benchmark_name", "benchmark_split")
        )
        task_id = row.get("task_id")
        score_id = row.get("score_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id <= 0:
            reasons.append(f"{cell}:missing_task_id")
        elif task_id in seen_task_ids:
            reasons.append(f"{cell}:duplicate_task_id:{task_id}")
        else:
            seen_task_ids.add(task_id)
        if not isinstance(score_id, int) or isinstance(score_id, bool) or score_id <= 0:
            reasons.append(f"{cell}:missing_score_id")
        elif score_id in seen_score_ids:
            reasons.append(f"{cell}:duplicate_score_id:{score_id}")
        else:
            seen_score_ids.add(score_id)
        if str(row.get("status") or "") != "Completed":
            reasons.append(f"{cell}:status:{row.get('status')}")

        expected = row.get("expected_completion_count")
        completion = row.get("completion_count")
        coordinates = row.get("distinct_completion_coordinates")
        eval_count = row.get("eval_count")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            reasons.append(f"{cell}:invalid_expected_completion_count:{expected}")
        elif (completion, coordinates, eval_count) != (expected, expected, expected):
            reasons.append(
                f"{cell}:row_counts:completion={completion},coordinates={coordinates},"
                f"eval={eval_count},expected={expected}"
            )
        if row.get("metric_name") is None or row.get("metric_value") is None:
            reasons.append(f"{cell}:missing_primary_score_metric")
        if not row.get("source_database"):
            reasons.append(f"{cell}:missing_source_database")
    return reasons


def build_completion_gate(audit: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the authoritative completion gate from the audit payload.

    ``audit_current.py`` represents ``remaining`` as the concrete list of
    uncovered cells.  Older preview artifacts used an integer, so accept both
    shapes without weakening the gate.  A full set of valid rows is not enough
    while a target runner is still unresolved: the delivery contract also
    requires zero active/failed target tasks and zero protocol issues.
    """

    remaining = audit.get("remaining", [])
    if isinstance(remaining, list):
        remaining_count = len(remaining)
    elif isinstance(remaining, int) and not isinstance(remaining, bool):
        remaining_count = remaining
    else:
        remaining_count = -1

    valid = sum(row["coverage_status"] == "valid" for row in rows)
    unresolved_running = len(audit.get("unresolved_active_target_tasks", []))
    unresolved_failed = len(audit.get("unresolved_failed_target_tasks", []))
    protocol_issues = len(audit.get("active_protocol_issues", []))
    traceability = traceability_reasons(rows)
    complete = (
        len(rows) == 184
        and valid == 184
        and remaining_count == 0
        and unresolved_running == 0
        and unresolved_failed == 0
        and protocol_issues == 0
        and not traceability
    )
    return {
        "source_audit_generated_at": audit.get("generated_at"),
        "complete": complete,
        "target_cells": 184,
        "valid_cells": valid,
        "missing_cells": 184 - valid,
        "remaining_cells": remaining_count,
        "unresolved_running": unresolved_running,
        "unresolved_failed": unresolved_failed,
        "active_protocol_issues": protocol_issues,
        "traceability_issues": traceability,
    }


def build_anomaly_audit(
    audit: dict[str, Any],
    quality_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract the evidence needed to review every non-coverage warning."""

    keys = (
        "task_status_counts",
        "invalid_reason_counts",
        "knowledge_replay_report",
        "diagnostic_knowledge_replay_task_ids",
        "superseded_valid_target_tasks",
        "active_target_tasks",
        "active_protocol_issues",
        "unresolved_active_target_tasks",
        "superseded_active_target_tasks",
        "failed_target_tasks",
        "unresolved_failed_target_tasks",
        "superseded_failed_target_tasks",
        "invalid_scored_tasks",
        "choice_bias_signals",
        "curve_comparisons",
        "curve_inversions_over_5pp",
        "reference_comparisons",
        "reference_differences_over_5pp",
        "truncation_examples_by_task",
    )
    return {
        "source_audit_generated_at": audit.get("generated_at"),
        "database": audit.get("database"),
        "target_cells": 184,
        "valid_cells": audit.get("valid_complete"),
        "remaining_cells": audit.get("remaining"),
        "quality_evidence": quality_evidence or [],
        **{key: audit.get(key, [] if key != "invalid_reason_counts" else {}) for key in keys},
    }


def load_quality_evidence(
    paths: list[Path],
    *,
    required: bool = False,
) -> list[dict[str, Any]]:
    """Load independent protocol/anomaly investigations into the delivery."""

    evidence: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"final delivery requires quality evidence: {path}"
                )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and isinstance(payload.get("probes"), list)
            and isinstance(payload.get("aggregate"), dict)
            and "choice" in str(payload.get("purpose") or "").lower()
        ):
            _validate_choice_sampling_evidence(path, payload)
        if isinstance(payload, dict) and "decision" in payload:
            _validate_resolution_evidence(path, payload)
        evidence.append(
            {
                "evidence_file": str(path),
                "payload": payload,
            }
        )
    return evidence


def validate_quality_evidence_coverage(
    rows: list[dict[str, Any]],
    quality_evidence: list[dict[str, Any]],
) -> None:
    """Require resolution evidence for every evaluator-facing truncation.

    The strict-46 audit deliberately exposes two different truncation
    counters.  Math has a recovery stage, so only a non-zero *final-stage*
    count reaches the evaluator and needs a resolution.  Coding and
    instruction-following have no such recovery stage, so their non-zero
    overall count needs a resolution.  Knowledge is constrained-choice: its
    transport telemetry is reviewed by the choice-sampling evidence and must
    not be mistaken for an evaluator-facing semantic truncation.

    The two resolution artifacts have intentionally distinct schemas:
    Math lists ``tasks[].task_id`` while non-Math lists ``task_ids``.  At the
    final gate both lists must be an exact, duplicate-free partition of the
    canonical tasks that require resolution.  This rejects stale historical
    task IDs as well as accidentally cross-wired evidence files.
    """

    canonical_by_task_id: dict[int, dict[str, Any]] = {}
    expected: dict[str, set[int]] = {"math": set(), "nonmath": set()}
    issues: list[str] = []

    for row in rows:
        if row.get("coverage_status") != "valid":
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id <= 0:
            continue
        if task_id in canonical_by_task_id:
            issues.append(f"duplicate canonical task ID: {task_id}")
            continue
        canonical_by_task_id[task_id] = row
        domain = str(row.get("domain") or "")
        if domain == "math":
            counter = row.get("final_stage_truncation_count")
            if isinstance(counter, (int, float)) and not isinstance(counter, bool) and counter > 0:
                expected["math"].add(task_id)
        elif domain in {"coding", "instruction_following"}:
            counter = row.get("overall_truncation_count")
            if isinstance(counter, (int, float)) and not isinstance(counter, bool) and counter > 0:
                expected["nonmath"].add(task_id)

    provided: dict[str, list[int]] = {"math": [], "nonmath": []}
    for evidence in quality_evidence:
        payload = evidence.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("decision"), dict):
            continue
        source = str(evidence.get("evidence_file") or "<unknown>")
        has_math_schema = "tasks" in payload
        has_nonmath_schema = "task_ids" in payload
        if has_math_schema and has_nonmath_schema:
            issues.append(
                "ambiguous truncation resolution schema contains both tasks and "
                f"task_ids: {source}"
            )
            continue
        if has_math_schema:
            task_rows = payload.get("tasks")
            if not isinstance(task_rows, list):
                issues.append(f"invalid math resolution tasks list: {source}")
                continue
            for index, task in enumerate(task_rows):
                if not isinstance(task, dict):
                    issues.append(
                        f"invalid math resolution task row: {source} index={index}"
                    )
                    continue
                task_id = task.get("task_id")
                if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id <= 0:
                    issues.append(
                        "invalid math resolution task ID: "
                        f"{source} index={index} value={task_id!r}"
                    )
                    continue
                provided["math"].append(task_id)
        elif has_nonmath_schema:
            task_ids = payload.get("task_ids")
            if not isinstance(task_ids, list):
                issues.append(f"invalid nonmath resolution task_ids list: {source}")
                continue
            for index, task_id in enumerate(task_ids):
                if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id <= 0:
                    issues.append(
                        "invalid nonmath resolution task ID: "
                        f"{source} index={index} value={task_id!r}"
                    )
                    continue
                provided["nonmath"].append(task_id)

    allowed_domains = {
        "math": {"math"},
        "nonmath": {"coding", "instruction_following"},
    }
    for scope in ("math", "nonmath"):
        task_ids = provided[scope]
        duplicates = sorted(
            task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1
        )
        if duplicates:
            issues.append(f"duplicate {scope} resolution task IDs: {duplicates}")

        unique_ids = set(task_ids)
        missing = sorted(expected[scope] - unique_ids)
        if missing:
            issues.append(f"missing {scope} resolution task IDs: {missing}")

        wrong_domain: list[str] = []
        extraneous: list[int] = []
        for task_id in sorted(unique_ids - expected[scope]):
            row = canonical_by_task_id.get(task_id)
            if row is None:
                extraneous.append(task_id)
                continue
            domain = str(row.get("domain") or "")
            if domain not in allowed_domains[scope]:
                wrong_domain.append(f"{task_id}({domain or 'missing'})")
            else:
                # The task is in the right domain but its evaluator-facing
                # truncation counter is zero, so this evidence is stale.
                extraneous.append(task_id)
        if wrong_domain:
            issues.append(
                f"wrong-domain {scope} resolution task IDs: {wrong_domain}"
            )
        if extraneous:
            issues.append(f"extraneous {scope} resolution task IDs: {extraneous}")

    if issues:
        raise ValueError(
            "quality evidence does not exactly cover canonical evaluator-facing "
            "truncations: " + "; ".join(issues)
        )


def load_knowledge_replay_diagnostics(
    path: Path,
    *,
    expected_database: str | None,
    required: bool,
) -> dict[str, Any] | None:
    """Normalize legacy/new replay labels without promoting them to scores.

    Historical reports called these rows ``replay_eligible_except_cutoff``.
    The stricter protocol renamed the concept to diagnostic-only because an
    answer-adapter replay cannot repair the prompt used for generation.  Keep
    both schemas readable, but emit one unambiguous final classification.
    """

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"final delivery requires Knowledge replay diagnostics: {path}"
            )
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError(f"invalid Knowledge replay diagnostics: {path}")
    database = str(payload.get("database") or "")
    if expected_database and database != expected_database:
        raise ValueError(
            "Knowledge replay diagnostics database mismatch: "
            f"expected={expected_database} actual={database or 'missing'}"
        )

    normalized_tasks: list[dict[str, Any]] = []
    diagnostic_ids: list[int] = []
    for source in payload["tasks"]:
        if not isinstance(source, dict):
            raise ValueError(f"invalid Knowledge replay task row: {source!r}")
        row = dict(source)
        diagnostic_only = bool(
            row.get("diagnostic_only")
            or row.get("knowledge_replay_diagnostic_evidence")
            or row.get("replay_eligible_except_cutoff")
        )
        row["diagnostic_only"] = diagnostic_only
        row["classification"] = (
            "diagnostic_only"
            if diagnostic_only
            else "strict_reuse_eligible"
            if row.get("strict_reuse_eligible")
            else "ineligible"
        )
        if diagnostic_only:
            task_id = row.get("task_id")
            if not isinstance(task_id, int) or isinstance(task_id, bool):
                raise ValueError(
                    f"diagnostic replay row has invalid task_id: {task_id!r}"
                )
            diagnostic_ids.append(task_id)
        normalized_tasks.append(row)
    return {
        "source_file": str(path),
        "database": database,
        "adapter": payload.get("adapter"),
        "read_only": payload.get("read_only"),
        "diagnostic_only_task_ids": sorted(set(diagnostic_ids)),
        "tasks": normalized_tasks,
    }


def _validate_choice_sampling_evidence(
    summary_path: Path, payload: dict[str, Any]
) -> None:
    """Prove that a choice-sampling summary matches its replay artifacts."""

    totals = {
        "rows_replayed": 0,
        "stored_replay_mismatches": 0,
        "sampler_argmax_mismatches": 0,
    }
    seen_task_ids: set[int] = set()
    for row in payload["probes"]:
        task_id = int(row["task_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"duplicate choice-sampling task id: {task_id}")
        seen_task_ids.add(task_id)
        evidence_path = summary_path.parent / str(row["evidence"])
        if not evidence_path.exists():
            raise FileNotFoundError(
                f"choice-sampling evidence is missing: {evidence_path}"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        expected = {
            "task_id": task_id,
            "model": str(row["model"]),
            "rows_probed": int(row["rows"]),
            "stored_replay_mismatches": int(row["stored_replay_mismatches"]),
            "sampler_argmax_mismatches": int(row["sampler_argmax_mismatches"]),
        }
        actual = {key: evidence.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                "choice-sampling summary does not match evidence: "
                f"task={task_id} expected={expected} actual={actual}"
            )
        # Keep the final anomaly artifact self-contained.  A reviewer should
        # not need access to this server-side relative path to inspect the
        # per-sample logits and replay decisions behind the summary.
        row["evidence_payload"] = evidence
        totals["rows_replayed"] += expected["rows_probed"]
        totals["stored_replay_mismatches"] += expected[
            "stored_replay_mismatches"
        ]
        totals["sampler_argmax_mismatches"] += expected[
            "sampler_argmax_mismatches"
        ]

    aggregate = payload["aggregate"]
    actual_totals = {key: int(aggregate.get(key, -1)) for key in totals}
    if actual_totals != totals:
        raise ValueError(
            "choice-sampling aggregate does not match evidence: "
            f"expected={totals} actual={actual_totals}"
        )
    for field in ("stored_replay_mismatches", "sampler_argmax_mismatches"):
        if totals[field] != 0:
            raise ValueError(
                f"choice-sampling protocol issue is unresolved: {field}={totals[field]}"
            )


def _validate_resolution_evidence(path: Path, payload: dict[str, Any]) -> None:
    """Require an explicit clean decision for final protocol evidence."""

    aggregate = payload.get("aggregate")
    decision = payload.get("decision")
    if not isinstance(aggregate, dict) or not isinstance(decision, dict):
        raise ValueError(f"invalid protocol resolution evidence: {path}")
    if decision.get("accept_tasks") is not True or decision.get("retest_required") is not False:
        raise ValueError(f"protocol resolution is not final: {path}")
    unresolved = {
        field: int(aggregate[field])
        for field in PROTOCOL_ZERO_COUNTERS
        if field in aggregate and int(aggregate[field]) != 0
    }
    if unresolved:
        raise ValueError(f"protocol resolution has unresolved counters: {unresolved}")


def load_replay_provenance(replay_dir: Path | None) -> list[dict[str, Any]]:
    """Collect source-to-replay task mappings emitted by the replay service."""

    if replay_dir is None:
        return []
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    evidence_paths = list(replay_dir.glob("*.json")) if replay_dir.exists() else []
    initial_commit = replay_dir.parent / "g1i_math_replay_commit_20260807.json"
    if initial_commit.exists():
        evidence_paths.append(initial_commit)
    for path in sorted(set(evidence_paths)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task in payload.get("tasks", []):
            source = task.get("task_id")
            replay = task.get("replay_task_id")
            if not isinstance(source, int) or not isinstance(replay, int):
                continue
            key = (source, replay)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "source_task_id": source,
                    "replay_task_id": replay,
                    "strategy_task_ids": task.get("strategy_task_ids", {}),
                    "model": task.get("model"),
                    "benchmark": task.get("benchmark"),
                    "judge_mode": task.get("judge_mode"),
                    "stored_primary_metric": task.get("stored_primary_metric"),
                    "stored_primary_score": task.get("stored_primary_score"),
                    "replayed_primary_score": task.get("replayed_primary_score"),
                    "delta_pp": task.get("delta_pp"),
                    "stored_answers_changed_by_replay": task.get(
                        "stored_answers_changed_by_replay"
                    ),
                    "stored_answers_with_synthetic_suffix": task.get(
                        "stored_answers_with_synthetic_suffix"
                    ),
                    "evidence_file": str(path),
                }
            )
    return sorted(rows, key=lambda row: (row["source_task_id"], row["replay_task_id"]))


def load_truncation_tables(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tables = payload.get("summary_tables")
    if not isinstance(tables, dict):
        raise ValueError(f"truncation matrix has no summary_tables: {path}")
    required = ("truncation_vs_parameter_size", "truncation_vs_g1x")
    for key in required:
        if not isinstance(tables.get(key), list):
            raise ValueError(f"truncation matrix has no list {key}: {path}")
    return payload, tables


def validate_final_truncation_tables(
    tables: dict[str, list[dict[str, Any]]],
    *,
    expected_g1i_task_ids: set[int],
) -> None:
    """Reject stale or incomplete truncation tables at the final 184-cell gate."""

    size_rows = tables["truncation_vs_parameter_size"]
    expected_size_keys = {
        (family, size)
        for family in ("G1g", "G1h", "G1i")
        for size in ("1.5B", "2.9B", "7.2B", "13.3B")
    }
    actual_size_keys = {
        (str(row.get("family")), str(row.get("parameter_size")))
        for row in size_rows
    }
    if actual_size_keys != expected_size_keys:
        raise ValueError(
            "truncation size table is incomplete: "
            f"missing={sorted(expected_size_keys - actual_size_keys)} "
            f"extra={sorted(actual_size_keys - expected_size_keys)}"
        )
    for row in size_rows:
        if int(row.get("cells") or 0) != 46:
            raise ValueError(
                "truncation size row does not cover 46 cells: "
                f"{row.get('family')}/{row.get('parameter_size')}={row.get('cells')}"
            )
        if int(row.get("complete_cells") or 0) != 46 or int(row.get("running_cells") or 0) != 0:
            raise ValueError(
                "truncation size row is not final: "
                f"{row.get('family')}/{row.get('parameter_size')} "
                f"complete={row.get('complete_cells')} running={row.get('running_cells')}"
            )
        if row.get("family") == "G1i":
            if int(row.get("protocol_compatible_cells") or 0) != 46:
                raise ValueError(
                    "final G1i truncation row is protocol-incompatible: "
                    f"{row.get('parameter_size')}="
                    f"{row.get('protocol_compatible_cells')}"
                )
            completion_count = int(row.get("completion_count") or 0)
            observable_count = int(row.get("observable_final_output_count") or 0)
            if completion_count <= 0 or observable_count != completion_count:
                raise ValueError(
                    "final G1i truncation row lacks evaluator-facing telemetry: "
                    f"{row.get('parameter_size')} observable={observable_count} "
                    f"completions={completion_count}"
                )
    table_g1i_ids = {
        int(task_id)
        for row in size_rows
        if row.get("family") == "G1i"
        for task_id in row.get("task_ids", [])
    }
    if table_g1i_ids != expected_g1i_task_ids:
        raise ValueError(
            "truncation table task IDs do not match final G1i audit: "
            f"missing={sorted(expected_g1i_task_ids - table_g1i_ids)} "
            f"extra={sorted(table_g1i_ids - expected_g1i_task_ids)}"
        )

    family_rows = tables["truncation_vs_g1x"]
    domain_cells = {
        "all": 184,
        "knowledge": 84,
        "math": 64,
        "coding": 28,
        "instruction_following": 8,
    }
    expected_family_keys = {
        (family, domain)
        for family in ("G1g", "G1h", "G1i")
        for domain in domain_cells
    }
    actual_family_keys = {
        (str(row.get("family")), str(row.get("domain")))
        for row in family_rows
    }
    if actual_family_keys != expected_family_keys:
        raise ValueError(
            "truncation G1x table is incomplete: "
            f"missing={sorted(expected_family_keys - actual_family_keys)} "
            f"extra={sorted(actual_family_keys - expected_family_keys)}"
        )
    for row in family_rows:
        family = str(row.get("family"))
        domain = str(row.get("domain"))
        expected_cells = domain_cells[domain]
        if int(row.get("cells") or 0) != expected_cells:
            raise ValueError(
                "truncation G1x row has incorrect coverage: "
                f"{family}/{domain}={row.get('cells')} expected={expected_cells}"
            )
        if (
            int(row.get("complete_cells") or 0) != expected_cells
            or int(row.get("running_cells") or 0) != 0
        ):
            raise ValueError(
                "truncation G1x row is not final: "
                f"{family}/{domain} complete={row.get('complete_cells')} "
                f"running={row.get('running_cells')}"
            )

    g1i_all = next(
        row
        for row in family_rows
        if row.get("family") == "G1i" and row.get("domain") == "all"
    )
    g1i_all_ids = {int(task_id) for task_id in g1i_all.get("task_ids", [])}
    if g1i_all_ids != expected_g1i_task_ids:
        raise ValueError(
            "truncation G1x task IDs do not match final G1i audit: "
            f"missing={sorted(expected_g1i_task_ids - g1i_all_ids)} "
            f"extra={sorted(g1i_all_ids - expected_g1i_task_ids)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    audit: dict[str, Any], output_dir: Path, artifact_names: list[str]
) -> dict[str, Any]:
    """Create a reproducible inventory without hashing the manifest itself."""

    return {
        "database": audit.get("database"),
        "source_audit_generated_at": audit.get("generated_at"),
        "target_cells": 184,
        "models": list(MODELS),
        "artifacts": [
            {
                "name": name,
                "size_bytes": (output_dir / name).stat().st_size,
                "sha256": _sha256(output_dir / name),
            }
            for name in artifact_names
        ],
    }


def copy_protocol_repair_history(
    source: Path,
    output_dir: Path,
    *,
    required: bool,
) -> str | None:
    """Make the final delivery self-contained with the repair/failure history."""

    if not source.exists():
        if required:
            raise FileNotFoundError(
                f"final delivery requires protocol repair audit: {source}"
            )
        return None
    destination = output_dir / "protocol_repair_history.md"
    destination.write_bytes(source.read_bytes())
    return destination.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=Path("logs/audits/g1i_math_replays"),
    )
    parser.add_argument(
        "--truncation-matrix",
        type=Path,
        default=Path("logs/audits/g1ghi_final_truncation_matrix_20260807.json"),
    )
    parser.add_argument(
        "--knowledge-replay-report",
        type=Path,
        default=Path(
            "logs/audits/g1h_g1i_knowledge_replay_frontend46_20260806.json"
        ),
    )
    parser.add_argument(
        "--quality-evidence",
        action="append",
        type=Path,
        default=[
            Path(
                "logs/audits/g1i_choice_bias/"
                "choice_sampling_audit_20260807.json"
            ),
            Path(
                "logs/audits/"
                "g1i_math_final_truncation_resolution_20260807.json"
            ),
            Path(
                "logs/audits/"
                "g1i_nonmath_truncation_resolution_20260807.json"
            ),
        ],
    )
    parser.add_argument(
        "--repair-audit",
        type=Path,
        default=Path("logs/audits/g1i_strict46_protocol_fix_20260805.md"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rows = build_rows(audit)
    gate = build_completion_gate(audit, rows)
    valid = int(gate["valid_cells"])
    complete = bool(gate["complete"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "frontend_scores.json"
    json_path.write_text(
        json.dumps(
            {
                "database": audit.get("database"),
                "source_audit_generated_at": audit.get("generated_at"),
                "target_cells": 184,
                "valid_cells": valid,
                "complete": complete,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = args.output_dir / "coverage.csv"
    csv_fields = [
        "source_database", "model_name", "model_size", "domain",
        "benchmark_name", "benchmark_split", "source_benchmark_name",
        "source_benchmark_split",
        "required_mode", "coverage_status", "task_id", "score_id", "score_created_at",
        "task_created_at", "status", "evaluator", "cot_mode", "sampling_config",
        "representative_prompt_tail", "metric_name", "metric_value", "completion_count",
        "expected_completion_count", "distinct_completion_coordinates", "eval_count",
        "eval_pass_rate", "blank_primary_generation_count", "missing_prediction_count",
        "leading_orphan_close_count", "overall_truncation_count", "overall_truncation_rate",
        "initial_generation_truncation_count", "final_stage_truncation_count",
        "active_task_ids", "superseded_active_task_ids",
        "superseded_valid_task_ids",
        "unresolved_failed_task_ids", "failed_task_ids",
        "superseded_failed_task_ids",
        "invalid_historical_task_ids",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["sampling_config"] = json.dumps(
                row["sampling_config"], ensure_ascii=False, sort_keys=True
            ) if row["sampling_config"] is not None else ""
            for field in (
                "active_task_ids",
                "superseded_active_task_ids",
                "superseded_valid_task_ids",
                "unresolved_failed_task_ids",
                "failed_task_ids",
                "superseded_failed_task_ids",
                "invalid_historical_task_ids",
            ):
                rendered[field] = ";".join(str(value) for value in row[field])
            writer.writerow(rendered)

    provenance_path = args.output_dir / "task_provenance.json"
    provenance_path.write_text(
        json.dumps(
            [
                {
                    "source_database": row["source_database"],
                    "model_name": row["model_name"],
                    "benchmark_name": row["benchmark_name"],
                    "benchmark_split": row["benchmark_split"],
                    "source_benchmark_name": row["source_benchmark_name"],
                    "source_benchmark_split": row["source_benchmark_split"],
                    "final_task_id": row["task_id"],
                    "final_score_id": row["score_id"],
                    "final_task_status": row["status"],
                    "coverage_status": row["coverage_status"],
                    "score_created_at": row["score_created_at"],
                    "metric_name": row["metric_name"],
                    "metric_value": row["metric_value"],
                    "metrics": row["metrics"],
                    "evaluator": row["evaluator"],
                    "cot_mode": row["cot_mode"],
                    "sampling_config": row["sampling_config"],
                    "completion_count": row["completion_count"],
                    "expected_completion_count": row["expected_completion_count"],
                    "distinct_completion_coordinates": row["distinct_completion_coordinates"],
                    "eval_count": row["eval_count"],
                    "active_task_ids": row["active_task_ids"],
                    "superseded_active_task_ids": row[
                        "superseded_active_task_ids"
                    ],
                    "superseded_valid_task_ids": row[
                        "superseded_valid_task_ids"
                    ],
                    "invalid_historical_task_ids": row["invalid_historical_task_ids"],
                    "unresolved_failed_task_ids": row[
                        "unresolved_failed_task_ids"
                    ],
                    "failed_task_ids": row["failed_task_ids"],
                    "superseded_failed_task_ids": row[
                        "superseded_failed_task_ids"
                    ],
                }
                for row in rows
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (args.output_dir / "completion_gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    quality_evidence = load_quality_evidence(
        args.quality_evidence,
        required=complete,
    )
    if complete:
        validate_quality_evidence_coverage(rows, quality_evidence)

    anomaly_path = args.output_dir / "anomaly_audit.json"
    anomaly_path.write_text(
        json.dumps(
            build_anomaly_audit(
                audit,
                quality_evidence,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    replay_path = args.output_dir / "score_replays.json"
    replay_path.write_text(
        json.dumps(
            load_replay_provenance(args.replay_dir),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    knowledge_replay = load_knowledge_replay_diagnostics(
        args.knowledge_replay_report,
        expected_database=str(audit.get("database") or "") or None,
        required=complete,
    )
    knowledge_replay_artifact: str | None = None
    if knowledge_replay is not None:
        knowledge_replay_artifact = "knowledge_replay_diagnostics.json"
        (args.output_dir / knowledge_replay_artifact).write_text(
            json.dumps(
                knowledge_replay,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    truncation_artifacts: list[str] = []
    if args.truncation_matrix.exists():
        truncation_payload, truncation_tables = load_truncation_tables(
            args.truncation_matrix
        )
        if complete:
            validate_final_truncation_tables(
                truncation_tables,
                expected_g1i_task_ids={
                    int(row["task_id"])
                    for row in rows
                    if row["coverage_status"] == "valid" and row["task_id"] is not None
                },
            )
        truncation_json_path = args.output_dir / "truncation_matrix.json"
        truncation_json_path.write_text(
            json.dumps(
                truncation_payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        write_summary_csv(
            args.output_dir / "truncation_vs_parameter_size.csv",
            truncation_tables["truncation_vs_parameter_size"],
        )
        write_summary_csv(
            args.output_dir / "truncation_vs_g1x.csv",
            truncation_tables["truncation_vs_g1x"],
        )
        write_summary_markdown(
            args.output_dir / "truncation_vs_parameter_size.md",
            truncation_tables["truncation_vs_parameter_size"],
            dimension="parameter_size",
        )
        write_summary_markdown(
            args.output_dir / "truncation_vs_g1x.md",
            truncation_tables["truncation_vs_g1x"],
            dimension="domain",
        )
        truncation_artifacts = [
            "truncation_matrix.json",
            "truncation_vs_parameter_size.csv",
            "truncation_vs_g1x.csv",
            "truncation_vs_parameter_size.md",
            "truncation_vs_g1x.md",
        ]
    elif complete:
        raise ValueError(
            f"final delivery requires truncation matrix: {args.truncation_matrix}"
        )

    repair_artifact = copy_protocol_repair_history(
        args.repair_audit,
        args.output_dir,
        required=complete,
    )

    artifact_names = [
        "frontend_scores.json",
        "coverage.csv",
        "task_provenance.json",
        "completion_gate.json",
        "anomaly_audit.json",
        "score_replays.json",
        *([knowledge_replay_artifact] if knowledge_replay_artifact else []),
        *([repair_artifact] if repair_artifact else []),
        *truncation_artifacts,
    ]
    (args.output_dir / "delivery_manifest.json").write_text(
        json.dumps(
            build_manifest(audit, args.output_dir, artifact_names),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate, ensure_ascii=False, sort_keys=True))
    return 0 if complete or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
