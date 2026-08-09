from __future__ import annotations

import csv
import hashlib
import json

from ops.g1i_strict46.audit_current import TARGETS
from ops.g1i_strict46.build_final_artifacts import build_manifest
from ops.g1i_strict46.finalize_when_complete import (
    MODEL_NAMES,
    REQUIRED_DELIVERY_ARTIFACTS,
    completion_gate_reasons,
    verify_delivery_artifacts,
)


def test_verify_delivery_artifacts_rejects_missing_manifest(tmp_path) -> None:
    reasons = verify_delivery_artifacts(tmp_path)

    assert len(reasons) == 1
    assert reasons[0].startswith("manifest_unavailable=")


def _complete_audit() -> dict[str, object]:
    return {
        "target_cells": 184,
        "valid_complete": 184,
        "remaining": 0,
        "active_protocol_issues": [],
        "unresolved_active_target_tasks": [],
        "unresolved_failed_target_tasks": [],
        "models": {
            model: {"complete": 46, "missing": 0}
            for model in MODEL_NAMES
        },
        "valid_task_rows": [
            {
                "task_id": task_id + 1,
                "score_id": task_id + 1001,
                "status": "Completed",
                "expected_completion_count": 8,
                "completion_count": 8,
                "distinct_completion_coordinates": 8,
                "eval_count": 8,
            }
            for task_id in range(184)
        ],
    }


def test_finalizer_gate_accepts_only_full_strict46_audit() -> None:
    assert completion_gate_reasons(_complete_audit()) == []


def test_finalizer_gate_rejects_running_failed_and_model_gaps() -> None:
    audit = _complete_audit()
    audit["valid_complete"] = 183
    audit["remaining"] = 1
    audit["unresolved_active_target_tasks"] = [{"task_id": 99}]
    audit["unresolved_failed_target_tasks"] = [{"task_id": 100}]
    audit["models"][MODEL_NAMES[-1]] = {"complete": 45, "missing": 1}
    audit["valid_task_rows"] = audit["valid_task_rows"][:-1]

    reasons = completion_gate_reasons(audit)

    assert "valid_complete=183" in reasons
    assert "remaining=1" in reasons
    assert "unresolved_active_target_tasks=1" in reasons
    assert "unresolved_failed_target_tasks=1" in reasons
    assert f"{MODEL_NAMES[-1]}:complete=45" in reasons
    assert f"{MODEL_NAMES[-1]}:missing=1" in reasons
    assert "valid_task_rows=183" in reasons


def test_finalizer_gate_rejects_broken_task_completion_eval_score_provenance() -> None:
    audit = _complete_audit()
    audit["valid_task_rows"][0] = {
        **audit["valid_task_rows"][0],
        "score_id": None,
        "eval_count": 7,
    }

    reasons = completion_gate_reasons(audit)

    assert "valid_task_rows[0].score_id=None" in reasons
    assert any("valid_task_rows[0].counts=" in reason for reason in reasons)


def _write_complete_delivery(output_dir) -> None:
    database = "strict46"
    generated_at = "2026-08-07T08:30:00+08:00"
    frontend_rows = []
    provenance_rows = []
    coverage_rows = []
    task_id = 1
    for model_name in MODEL_NAMES:
        for (benchmark_name, benchmark_split), (domain, required_mode) in TARGETS.items():
            score_id = task_id + 1000
            cot_mode = "CoT" if required_mode == "cot" else "NoCoT"
            common = {
                "source_database": database,
                "model_name": model_name,
                "benchmark_name": benchmark_name,
                "benchmark_split": benchmark_split,
                "source_benchmark_name": benchmark_name,
                "source_benchmark_split": benchmark_split,
                "domain": domain,
                "required_mode": required_mode,
                "coverage_status": "valid",
                "task_id": task_id,
                "score_id": score_id,
                "score_created_at": "2026-08-07T08:00:00+08:00",
                "status": "Completed",
                "evaluator": "strict46_naive",
                "cot_mode": cot_mode,
                "sampling_config": {"prompt_profile": "naive", "avg_k": 8},
                "metric_name": "avg@8",
                "metric_value": 0.5,
                "metrics": {"avg@8": 0.5},
                "expected_completion_count": 8,
                "completion_count": 8,
                "distinct_completion_coordinates": 8,
                "eval_count": 8,
                "active_task_ids": [],
                "superseded_active_task_ids": [],
                "superseded_valid_task_ids": [],
                "unresolved_failed_task_ids": [],
                "failed_task_ids": [],
                "superseded_failed_task_ids": [],
                "invalid_historical_task_ids": [],
            }
            frontend_rows.append(common)
            provenance_rows.append(
                {
                    **common,
                    "final_task_id": task_id,
                    "final_score_id": score_id,
                    "final_task_status": "Completed",
                }
            )
            coverage_row = dict(common)
            coverage_row["sampling_config"] = json.dumps(
                common["sampling_config"], sort_keys=True
            )
            for field in (
                "active_task_ids",
                "superseded_active_task_ids",
                "superseded_valid_task_ids",
                "unresolved_failed_task_ids",
                "failed_task_ids",
                "superseded_failed_task_ids",
                "invalid_historical_task_ids",
            ):
                coverage_row[field] = ""
            coverage_rows.append(coverage_row)
            task_id += 1

    (output_dir / "frontend_scores.json").write_text(
        json.dumps(
            {
                "database": database,
                "source_audit_generated_at": generated_at,
                "target_cells": 184,
                "valid_cells": 184,
                "complete": True,
                "rows": frontend_rows,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "task_provenance.json").write_text(
        json.dumps(provenance_rows), encoding="utf-8"
    )
    with (output_dir / "coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0]))
        writer.writeheader()
        writer.writerows(coverage_rows)
    (output_dir / "completion_gate.json").write_text(
        json.dumps(
            {
                "source_audit_generated_at": generated_at,
                "complete": True,
                "target_cells": 184,
                "valid_cells": 184,
                "missing_cells": 0,
                "remaining_cells": 0,
                "unresolved_running": 0,
                "unresolved_failed": 0,
                "active_protocol_issues": 0,
                "traceability_issues": [],
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "anomaly_audit.json").write_text(
        json.dumps(
            {
                "source_audit_generated_at": generated_at,
                "database": database,
                "target_cells": 184,
                "valid_cells": 184,
                "remaining_cells": 0,
                "quality_evidence": [
                    {"evidence_file": "probe.json", "payload": {"ok": True}}
                ],
                "task_status_counts": {"Completed": 184},
                "invalid_reason_counts": {},
                "active_target_tasks": [],
                "active_protocol_issues": [],
                "unresolved_active_target_tasks": [],
                "superseded_active_target_tasks": [],
                "superseded_valid_target_tasks": [],
                "failed_target_tasks": [],
                "unresolved_failed_target_tasks": [],
                "superseded_failed_target_tasks": [],
                "invalid_scored_tasks": [],
                "choice_bias_signals": [],
                "curve_comparisons": [],
                "curve_inversions_over_5pp": [],
                "reference_comparisons": [],
                "reference_differences_over_5pp": [],
                "truncation_examples_by_task": {},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "score_replays.json").write_text("[]\n", encoding="utf-8")
    (output_dir / "knowledge_replay_diagnostics.json").write_text(
        json.dumps(
            {
                "database": database,
                "diagnostic_only_task_ids": [],
                "tasks": [],
            }
        ),
        encoding="utf-8",
    )
    for name in REQUIRED_DELIVERY_ARTIFACTS:
        path = output_dir / name
        if path.exists():
            continue
        if path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        elif path.suffix == ".csv":
            path.write_text("field,value\n", encoding="utf-8")
        else:
            path.write_text("# Evidence\n", encoding="utf-8")
    manifest = build_manifest(
        {"database": database, "generated_at": generated_at},
        output_dir,
        sorted(REQUIRED_DELIVERY_ARTIFACTS),
    )
    (output_dir / "delivery_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _rewrite_json_and_rehash(output_dir, name: str, payload: object) -> None:
    path = output_dir / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = output_dir / "delivery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact = next(row for row in manifest["artifacts"] if row["name"] == name)
    artifact["size_bytes"] = path.stat().st_size
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_delivery_artifacts_accepts_complete_traceable_bundle(tmp_path) -> None:
    _write_complete_delivery(tmp_path)

    assert verify_delivery_artifacts(tmp_path) == []


def test_verify_delivery_artifacts_rejects_hashed_nonzero_protocol_gate(
    tmp_path,
) -> None:
    _write_complete_delivery(tmp_path)
    gate_path = tmp_path / "completion_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["active_protocol_issues"] = 1
    _rewrite_json_and_rehash(tmp_path, "completion_gate.json", gate)

    reasons = verify_delivery_artifacts(tmp_path)

    assert "completion_gate_active_protocol_issues=1" in reasons


def test_verify_delivery_artifacts_rejects_unmapped_historical_task(tmp_path) -> None:
    _write_complete_delivery(tmp_path)
    anomaly_path = tmp_path / "anomaly_audit.json"
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
    anomaly["invalid_scored_tasks"] = [
        {
            "model_name": MODEL_NAMES[0],
            "benchmark_name": "mmlu",
            "benchmark_split": "test",
            "task_id": 9001,
        }
    ]
    _rewrite_json_and_rehash(tmp_path, "anomaly_audit.json", anomaly)

    reasons = verify_delivery_artifacts(tmp_path)

    assert any(reason.startswith("historical_mapping_mismatch=") for reason in reasons)


def test_verify_delivery_artifacts_rejects_184_rows_with_wrong_cell_set(
    tmp_path,
) -> None:
    _write_complete_delivery(tmp_path)
    frontend_path = tmp_path / "frontend_scores.json"
    frontend = json.loads(frontend_path.read_text(encoding="utf-8"))
    frontend["rows"][0]["benchmark_name"] = frontend["rows"][1][
        "benchmark_name"
    ]
    frontend["rows"][0]["benchmark_split"] = frontend["rows"][1][
        "benchmark_split"
    ]
    _rewrite_json_and_rehash(tmp_path, "frontend_scores.json", frontend)

    reasons = verify_delivery_artifacts(tmp_path)

    assert any(reason.startswith("frontend_strict46_cells=mismatch") for reason in reasons)


def test_verify_delivery_artifacts_rejects_anomaly_protocol_issue(tmp_path) -> None:
    _write_complete_delivery(tmp_path)
    anomaly_path = tmp_path / "anomaly_audit.json"
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
    anomaly["active_protocol_issues"] = [{"task_id": 9001}]
    _rewrite_json_and_rehash(tmp_path, "anomaly_audit.json", anomaly)

    reasons = verify_delivery_artifacts(tmp_path)

    assert any(reason.startswith("anomaly_active_protocol_issues=") for reason in reasons)
