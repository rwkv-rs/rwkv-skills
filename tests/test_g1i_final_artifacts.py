from __future__ import annotations

import json
import sys

import pytest

from ops.g1i_strict46.audit_current import MODELS
from ops.g1i_strict46.build_final_artifacts import (
    _primary_metric,
    build_anomaly_audit,
    build_completion_gate,
    build_manifest,
    build_rows,
    copy_protocol_repair_history,
    load_knowledge_replay_diagnostics,
    load_quality_evidence,
    load_truncation_tables,
    load_replay_provenance,
    main,
    validate_quality_evidence_coverage,
    validate_final_truncation_tables,
)


def test_copy_protocol_repair_history_is_self_contained(tmp_path) -> None:
    source = tmp_path / "source.md"
    source.write_text("# Repairs\n\n- task 10 -> task 20\n", encoding="utf-8")
    output_dir = tmp_path / "delivery"
    output_dir.mkdir()

    name = copy_protocol_repair_history(source, output_dir, required=True)

    assert name == "protocol_repair_history.md"
    assert (output_dir / name).read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )


def test_copy_protocol_repair_history_is_required_for_final(tmp_path) -> None:
    try:
        copy_protocol_repair_history(
            tmp_path / "missing.md",
            tmp_path,
            required=True,
        )
    except FileNotFoundError as exc:
        assert "requires protocol repair audit" in str(exc)
    else:
        raise AssertionError("missing final repair audit should fail the delivery")


def test_primary_metric_prefers_reported_avg_k() -> None:
    assert _primary_metric({"accuracy": 0.4, "avg@1": 0.5, "avg@8": 0.6}) == ("avg@8", 0.6)


def test_build_rows_always_materializes_strict_184_cells() -> None:
    rows = build_rows({"valid_task_rows": []})

    assert len(rows) == 184
    assert {row["model_name"] for row in rows} == set(MODELS)
    assert all(row["coverage_status"] == "missing" for row in rows)


def test_build_rows_links_valid_and_historical_task_ids() -> None:
    model = MODELS[0]
    audit = {
        "database": "strict46",
        "valid_task_rows": [
            {
                "model_name": model,
                "benchmark": "mmlu__test",
                "source_benchmark_name": "mmlu",
                "source_benchmark_split": "test",
                "task_id": 20,
                "score_id": 30,
                "metrics": {"avg@1": 0.5},
                "score_created_at": "now",
                "task_created_at": "before",
                "status": "Completed",
                "evaluator": "multi_choice_plain_naive",
                "cot_mode": "NoCoT",
                "sampling_config": {"avg_k": 1, "temperature": 0.8},
                "representative_prompt_tail": "Assistant: <think></think>\n",
                "completion_count": 10,
                "expected_completion_count": 10,
                "distinct_completion_coordinates": 10,
                "eval_count": 10,
            }
        ],
        "invalid_scored_tasks": [
            {
                "model_name": model,
                "benchmark_name": "mmlu",
                "benchmark_split": "test",
                "task_id": 10,
            }
        ],
        "failed_target_tasks": [
            {
                "model_name": model,
                "benchmark_name": "mmlu",
                "benchmark_split": "test",
                "task_id": 11,
            },
            {
                "model_name": model,
                "benchmark_name": "mmlu",
                "benchmark_split": "test",
                "task_id": 12,
            },
        ],
        "unresolved_failed_target_tasks": [],
        "superseded_active_target_tasks": [
            {
                "model_name": model,
                "benchmark_name": "mmlu",
                "benchmark_split": "test",
                "task_id": 13,
            }
        ],
        "superseded_valid_target_tasks": [
            {
                "model_name": model,
                "benchmark_name": "mmlu",
                "benchmark_split": "test",
                "task_id": 9,
            }
        ],
        "superseded_failed_target_tasks": [
            {
                "model_name": model,
                "benchmark_name": "mmlu",
                "benchmark_split": "test",
                "task_id": 11,
            },
            {
                "model_name": model,
                "benchmark_name": "mmlu",
                "benchmark_split": "test",
                "task_id": 12,
            },
        ],
    }

    row = next(
        item
        for item in build_rows(audit)
        if item["model_name"] == model and item["benchmark_name"] == "mmlu"
    )

    assert row["coverage_status"] == "valid"
    assert row["source_database"] == "strict46"
    assert row["source_benchmark_name"] == "mmlu"
    assert row["source_benchmark_split"] == "test"
    assert row["task_id"] == 20
    assert row["score_id"] == 30
    assert row["evaluator"] == "multi_choice_plain_naive"
    assert row["cot_mode"] == "NoCoT"
    assert row["sampling_config"]["temperature"] == 0.8
    assert row["completion_count"] == row["expected_completion_count"] == 10
    assert row["distinct_completion_coordinates"] == row["eval_count"] == 10
    assert row["invalid_historical_task_ids"] == [10]
    assert row["superseded_active_task_ids"] == [13]
    assert row["superseded_valid_task_ids"] == [9]
    assert row["unresolved_failed_task_ids"] == []
    assert row["failed_task_ids"] == [11, 12]
    assert row["superseded_failed_task_ids"] == [11, 12]


def _traceable_rows() -> list[dict[str, object]]:
    return [
        {
            "coverage_status": "valid",
            "model_name": f"model-{index}",
            "benchmark_name": "benchmark",
            "benchmark_split": "test",
            "task_id": index + 1,
            "score_id": index + 1001,
            "status": "Completed",
            "expected_completion_count": 8,
            "completion_count": 8,
            "distinct_completion_coordinates": 8,
            "eval_count": 8,
            "metric_name": "avg@8",
            "metric_value": 0.5,
            "source_database": "strict46",
        }
        for index in range(184)
    ]


def test_completion_gate_accepts_concrete_empty_remaining_list() -> None:
    rows = _traceable_rows()

    gate = build_completion_gate(
        {
            "generated_at": "2026-08-07T08:30:00+08:00",
            "remaining": [],
            "unresolved_active_target_tasks": [],
            "unresolved_failed_target_tasks": [],
            "active_protocol_issues": [],
        },
        rows,
    )

    assert gate["complete"] is True
    assert gate["remaining_cells"] == 0
    assert gate["source_audit_generated_at"] == "2026-08-07T08:30:00+08:00"


def test_completion_gate_rejects_unresolved_runner_despite_full_scores() -> None:
    rows = _traceable_rows()

    gate = build_completion_gate(
        {
            "remaining": [],
            "unresolved_active_target_tasks": [{"task_id": 123}],
            "unresolved_failed_target_tasks": [],
            "active_protocol_issues": [],
        },
        rows,
    )

    assert gate["complete"] is False
    assert gate["unresolved_running"] == 1


def test_completion_gate_rejects_untraceable_completion_eval_score_chain() -> None:
    rows = _traceable_rows()
    rows[0] = {**rows[0], "eval_count": 7, "score_id": None}

    gate = build_completion_gate(
        {
            "remaining": [],
            "unresolved_active_target_tasks": [],
            "unresolved_failed_target_tasks": [],
            "active_protocol_issues": [],
        },
        rows,
    )

    assert gate["complete"] is False
    assert any("missing_score_id" in reason for reason in gate["traceability_issues"])
    assert any("row_counts" in reason for reason in gate["traceability_issues"])


def test_anomaly_audit_preserves_review_evidence() -> None:
    audit = {
        "generated_at": "2026-08-07T08:30:00+08:00",
        "database": "strict46",
        "valid_complete": 183,
        "remaining": 1,
        "invalid_reason_counts": {"bad_prompt": 2},
        "superseded_valid_target_tasks": [{"task_id": 9}],
        "superseded_active_target_tasks": [{"task_id": 10}],
        "superseded_failed_target_tasks": [{"task_id": 11}],
        "curve_inversions_over_5pp": [{"benchmark": "mmlu"}],
        "reference_differences_over_5pp": [{"benchmark": "aime25"}],
        "truncation_examples_by_task": {"12": [{"sample_index": 1}]},
    }

    quality = [{"evidence_file": "probe.json", "payload": {"ok": True}}]
    result = build_anomaly_audit(audit, quality)

    assert result["database"] == "strict46"
    assert result["source_audit_generated_at"] == "2026-08-07T08:30:00+08:00"
    assert result["valid_cells"] == 183
    assert result["invalid_reason_counts"] == {"bad_prompt": 2}
    assert result["superseded_valid_target_tasks"] == [{"task_id": 9}]
    assert result["superseded_active_target_tasks"] == [{"task_id": 10}]
    assert result["superseded_failed_target_tasks"] == [{"task_id": 11}]
    assert result["curve_inversions_over_5pp"] == [{"benchmark": "mmlu"}]
    assert result["truncation_examples_by_task"]["12"][0]["sample_index"] == 1
    assert result["quality_evidence"] == quality


def test_quality_evidence_loader_skips_missing_and_preserves_payload(
    tmp_path,
) -> None:
    present = tmp_path / "probe.json"
    present.write_text('{"rows": 112, "mismatches": 0}\n', encoding="utf-8")

    evidence = load_quality_evidence([tmp_path / "missing.json", present])

    assert evidence == [
        {
            "evidence_file": str(present),
            "payload": {"rows": 112, "mismatches": 0},
        }
    ]


def test_quality_evidence_loader_requires_declared_final_inputs(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="requires quality evidence"):
        load_quality_evidence([tmp_path / "missing.json"], required=True)


def _canonical_truncation_rows() -> list[dict[str, object]]:
    return [
        {
            "coverage_status": "valid",
            "task_id": 1,
            "domain": "math",
            "final_stage_truncation_count": 2,
            "overall_truncation_count": 20,
        },
        {
            "coverage_status": "valid",
            "task_id": 2,
            "domain": "coding",
            "overall_truncation_count": 1,
        },
        {
            "coverage_status": "valid",
            "task_id": 3,
            "domain": "instruction_following",
            "overall_truncation_count": 3,
        },
        {
            "coverage_status": "valid",
            "task_id": 4,
            "domain": "knowledge",
            # Constrained-choice transport telemetry is not an
            # evaluator-facing semantic truncation.
            "overall_truncation_count": 999,
        },
        {
            "coverage_status": "valid",
            "task_id": 5,
            "domain": "math",
            # Stage-0 recovery activity alone must not require evidence.
            "initial_generation_truncation_count": 10,
            "overall_truncation_count": 10,
            "final_stage_truncation_count": 0,
        },
    ]


def _truncation_resolution_evidence(
    *,
    math_ids: list[int] | None = None,
    nonmath_ids: list[int] | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "evidence_file": "math_resolution.json",
            "payload": {
                "decision": {"accept_tasks": True, "retest_required": False},
                "tasks": [
                    {"task_id": task_id}
                    for task_id in ([1] if math_ids is None else math_ids)
                ],
            },
        },
        {
            "evidence_file": "nonmath_resolution.json",
            "payload": {
                "decision": {"accept_tasks": True, "retest_required": False},
                "task_ids": [2, 3] if nonmath_ids is None else nonmath_ids,
            },
        },
    ]


def test_quality_evidence_coverage_matches_only_evaluator_facing_truncations() -> None:
    validate_quality_evidence_coverage(
        _canonical_truncation_rows(),
        _truncation_resolution_evidence(),
    )


def test_quality_evidence_coverage_reports_missing_task_ids() -> None:
    with pytest.raises(ValueError, match=r"missing math resolution task IDs: \[1\]"):
        validate_quality_evidence_coverage(
            _canonical_truncation_rows(),
            _truncation_resolution_evidence(math_ids=[]),
        )


def test_quality_evidence_coverage_reports_duplicate_task_ids() -> None:
    with pytest.raises(ValueError, match=r"duplicate nonmath resolution task IDs: \[2\]"):
        validate_quality_evidence_coverage(
            _canonical_truncation_rows(),
            _truncation_resolution_evidence(nonmath_ids=[2, 2, 3]),
        )


def test_quality_evidence_coverage_reports_extraneous_task_ids() -> None:
    with pytest.raises(ValueError, match=r"extraneous math resolution task IDs: \[5, 999\]"):
        validate_quality_evidence_coverage(
            _canonical_truncation_rows(),
            _truncation_resolution_evidence(math_ids=[1, 5, 999]),
        )


@pytest.mark.parametrize(
    ("math_ids", "nonmath_ids", "message"),
    (
        ([1, 2], [2, 3], r"wrong-domain math resolution task IDs: \['2\(coding\)'\]"),
        ([1], [2, 3, 4], r"wrong-domain nonmath resolution task IDs: \['4\(knowledge\)'\]"),
        ([1], [1, 2, 3], r"wrong-domain nonmath resolution task IDs: \['1\(math\)'\]"),
    ),
)
def test_quality_evidence_coverage_reports_wrong_domain_task_ids(
    math_ids,
    nonmath_ids,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_quality_evidence_coverage(
            _canonical_truncation_rows(),
            _truncation_resolution_evidence(
                math_ids=math_ids,
                nonmath_ids=nonmath_ids,
            ),
        )


def test_incomplete_preview_does_not_require_quality_evidence_coverage(
    tmp_path,
    monkeypatch,
) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "database": "strict46-preview",
                "generated_at": "2026-08-07T16:00:00+08:00",
                "valid_task_rows": [],
                "remaining": [],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "delivery"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_final_artifacts.py",
            "--audit",
            str(audit_path),
            "--output-dir",
            str(output_dir),
            "--allow-incomplete",
        ],
    )

    assert main() == 0
    anomaly = json.loads(
        (output_dir / "anomaly_audit.json").read_text(encoding="utf-8")
    )
    assert anomaly["quality_evidence"] == []


@pytest.mark.parametrize(
    "flag",
    ("replay_eligible_except_cutoff", "diagnostic_only"),
)
def test_knowledge_replay_diagnostic_field_rename_is_backward_compatible(
    tmp_path,
    flag,
) -> None:
    path = tmp_path / "knowledge_replay.json"
    path.write_text(
        json.dumps(
            {
                "database": "strict46",
                "adapter": "current_raw_multiple_choice_adapter",
                "read_only": True,
                "tasks": [
                    {
                        "task_id": 12,
                        "strict_reuse_eligible": False,
                        flag: True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = load_knowledge_replay_diagnostics(
        path,
        expected_database="strict46",
        required=True,
    )

    assert payload is not None
    assert payload["diagnostic_only_task_ids"] == [12]
    assert payload["tasks"][0]["diagnostic_only"] is True
    assert payload["tasks"][0]["classification"] == "diagnostic_only"


def test_knowledge_replay_diagnostic_rejects_wrong_database(tmp_path) -> None:
    path = tmp_path / "knowledge_replay.json"
    path.write_text(
        '{"database":"old","tasks":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="database mismatch"):
        load_knowledge_replay_diagnostics(
            path,
            expected_database="strict46",
            required=True,
        )


def test_choice_sampling_quality_evidence_is_verified_against_probes(
    tmp_path,
) -> None:
    probe = {
        "task_id": 12,
        "model": "model",
        "rows_probed": 8,
        "stored_replay_mismatches": 0,
        "sampler_argmax_mismatches": 0,
    }
    (tmp_path / "task_12_probe.json").write_text(
        json.dumps(probe), encoding="utf-8"
    )
    summary = {
        "purpose": "choice transport audit",
        "probes": [
            {
                "task_id": 12,
                "model": "model",
                "rows": 8,
                "stored_replay_mismatches": 0,
                "sampler_argmax_mismatches": 0,
                "evidence": "task_12_probe.json",
            }
        ],
        "aggregate": {
            "rows_replayed": 8,
            "stored_replay_mismatches": 0,
            "sampler_argmax_mismatches": 0,
        },
    }
    summary_path = tmp_path / "choice_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    payload = load_quality_evidence([summary_path])[0]["payload"]
    assert payload["aggregate"] == summary["aggregate"]
    assert payload["probes"][0]["evidence_payload"] == probe


def test_choice_sampling_quality_evidence_rejects_stale_aggregate(tmp_path) -> None:
    (tmp_path / "task_12_probe.json").write_text(
        json.dumps(
            {
                "task_id": 12,
                "model": "model",
                "rows_probed": 8,
                "stored_replay_mismatches": 0,
                "sampler_argmax_mismatches": 0,
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "choice_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "purpose": "choice transport audit",
                "probes": [
                    {
                        "task_id": 12,
                        "model": "model",
                        "rows": 8,
                        "stored_replay_mismatches": 0,
                        "sampler_argmax_mismatches": 0,
                        "evidence": "task_12_probe.json",
                    }
                ],
                "aggregate": {
                    "rows_replayed": 7,
                    "stored_replay_mismatches": 0,
                    "sampler_argmax_mismatches": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="aggregate does not match"):
        load_quality_evidence([summary_path])


def test_choice_sampling_quality_evidence_rejects_confirmed_protocol_mismatch(
    tmp_path,
) -> None:
    probe = {
        "task_id": 12,
        "model": "model",
        "rows_probed": 8,
        "stored_replay_mismatches": 1,
        "sampler_argmax_mismatches": 0,
    }
    (tmp_path / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "purpose": "choice transport audit",
                "probes": [
                    {
                        "task_id": 12,
                        "model": "model",
                        "rows": 8,
                        "stored_replay_mismatches": 1,
                        "sampler_argmax_mismatches": 0,
                        "evidence": "probe.json",
                    }
                ],
                "aggregate": {
                    "rows_replayed": 8,
                    "stored_replay_mismatches": 1,
                    "sampler_argmax_mismatches": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="protocol issue is unresolved"):
        load_quality_evidence([summary_path])


def test_delivery_manifest_hashes_every_declared_artifact(tmp_path) -> None:
    (tmp_path / "one.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "two.csv").write_text("a,b\n", encoding="utf-8")

    manifest = build_manifest(
        {"database": "strict46"}, tmp_path, ["one.json", "two.csv"]
    )

    assert manifest["database"] == "strict46"
    assert [row["name"] for row in manifest["artifacts"]] == [
        "one.json",
        "two.csv",
    ]
    assert all(len(row["sha256"]) == 64 for row in manifest["artifacts"])


def test_replay_provenance_preserves_source_and_new_task_ids(tmp_path) -> None:
    (tmp_path / "source_10.json").write_text(
        """{
          "tasks": [{
            "task_id": 10,
            "replay_task_id": 20,
            "strategy_task_ids": {"strategy_c": 20},
            "model": "g1i",
            "benchmark": "aime25_test",
            "stored_primary_score": 0.1,
            "replayed_primary_score": 0.2,
            "delta_pp": 10.0
          }]
        }""",
        encoding="utf-8",
    )

    rows = load_replay_provenance(tmp_path)

    assert rows == [
        {
            "source_task_id": 10,
            "replay_task_id": 20,
            "strategy_task_ids": {"strategy_c": 20},
            "model": "g1i",
            "benchmark": "aime25_test",
            "judge_mode": None,
            "stored_primary_metric": None,
            "stored_primary_score": 0.1,
            "replayed_primary_score": 0.2,
            "delta_pp": 10.0,
            "stored_answers_changed_by_replay": None,
            "stored_answers_with_synthetic_suffix": None,
            "evidence_file": str(tmp_path / "source_10.json"),
        }
    ]


def test_replay_provenance_reads_initial_commit_without_replay_directory(
    tmp_path,
) -> None:
    replay_dir = tmp_path / "g1i_math_replays"
    (tmp_path / "g1i_math_replay_commit_20260807.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": 10,
                        "replay_task_id": 20,
                        "model": "g1i",
                        "benchmark": "aime25_test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = load_replay_provenance(replay_dir)

    assert [(row["source_task_id"], row["replay_task_id"]) for row in rows] == [
        (10, 20)
    ]


def _final_truncation_tables() -> dict[str, list[dict[str, object]]]:
    size_rows = []
    family_task_ids: dict[str, list[int]] = {}
    task_id = 1
    for family in ("G1g", "G1h", "G1i"):
        family_task_ids[family] = []
        for size in ("1.5B", "2.9B", "7.2B", "13.3B"):
            ids = list(range(task_id, task_id + 46))
            task_id += 46
            family_task_ids[family].extend(ids)
            size_rows.append(
                {
                    "family": family,
                    "parameter_size": size,
                    "cells": 46,
                    "complete_cells": 46,
                    "running_cells": 0,
                    "protocol_compatible_cells": 46,
                    "completion_count": 46,
                    "observable_final_output_count": 46,
                    "task_ids": ids,
                }
            )
    domain_cells = {
        "all": 184,
        "knowledge": 84,
        "math": 64,
        "coding": 28,
        "instruction_following": 8,
    }
    family_rows = []
    for family in ("G1g", "G1h", "G1i"):
        ids = family_task_ids[family]
        offset = 0
        for domain, cells in domain_cells.items():
            domain_ids = ids if domain == "all" else ids[offset : offset + cells]
            if domain != "all":
                offset += cells
            family_rows.append(
                {
                    "family": family,
                    "domain": domain,
                    "cells": cells,
                    "complete_cells": cells,
                    "running_cells": 0,
                    "protocol_compatible_cells": cells,
                    "completion_count": cells,
                    "observable_final_output_count": cells,
                    "task_ids": domain_ids,
                }
            )
    return {
        "truncation_vs_parameter_size": size_rows,
        "truncation_vs_g1x": family_rows,
    }


def test_final_truncation_gate_accepts_exact_12_rows_and_g1i_ids() -> None:
    tables = _final_truncation_tables()
    expected = {
        int(task_id)
        for row in tables["truncation_vs_parameter_size"]
        if row["family"] == "G1i"
        for task_id in row["task_ids"]
    }

    validate_final_truncation_tables(tables, expected_g1i_task_ids=expected)


def test_final_truncation_gate_rejects_stale_task_ids() -> None:
    tables = _final_truncation_tables()

    try:
        validate_final_truncation_tables(tables, expected_g1i_task_ids={999})
    except ValueError as exc:
        assert "do not match final G1i audit" in str(exc)
    else:
        raise AssertionError("stale truncation task IDs were accepted")


def test_final_truncation_gate_rejects_empty_g1x_table() -> None:
    tables = _final_truncation_tables()
    expected = {
        int(task_id)
        for row in tables["truncation_vs_parameter_size"]
        if row["family"] == "G1i"
        for task_id in row["task_ids"]
    }
    tables["truncation_vs_g1x"] = []

    try:
        validate_final_truncation_tables(tables, expected_g1i_task_ids=expected)
    except ValueError as exc:
        assert "G1x table is incomplete" in str(exc)
    else:
        raise AssertionError("empty G1x truncation table was accepted")


def test_final_truncation_gate_rejects_nonfinal_g1x_domain_row() -> None:
    tables = _final_truncation_tables()
    expected = {
        int(task_id)
        for row in tables["truncation_vs_parameter_size"]
        if row["family"] == "G1i"
        for task_id in row["task_ids"]
    }
    row = next(
        item
        for item in tables["truncation_vs_g1x"]
        if item["family"] == "G1i" and item["domain"] == "math"
    )
    row["complete_cells"] = 63

    try:
        validate_final_truncation_tables(tables, expected_g1i_task_ids=expected)
    except ValueError as exc:
        assert "G1x row is not final" in str(exc)
    else:
        raise AssertionError("non-final G1x truncation row was accepted")


def test_final_truncation_gate_rejects_g1i_missing_final_output_telemetry() -> None:
    tables = _final_truncation_tables()
    expected = {
        int(task_id)
        for row in tables["truncation_vs_parameter_size"]
        if row["family"] == "G1i"
        for task_id in row["task_ids"]
    }
    row = next(
        item
        for item in tables["truncation_vs_parameter_size"]
        if item["family"] == "G1i" and item["parameter_size"] == "7.2B"
    )
    row["observable_final_output_count"] = 45

    with pytest.raises(ValueError, match="lacks evaluator-facing telemetry"):
        validate_final_truncation_tables(
            tables,
            expected_g1i_task_ids=expected,
        )


def test_final_truncation_gate_rejects_g1i_protocol_incompatible_row() -> None:
    tables = _final_truncation_tables()
    expected = {
        int(task_id)
        for row in tables["truncation_vs_parameter_size"]
        if row["family"] == "G1i"
        for task_id in row["task_ids"]
    }
    row = next(
        item
        for item in tables["truncation_vs_parameter_size"]
        if item["family"] == "G1i" and item["parameter_size"] == "13.3B"
    )
    row["protocol_compatible_cells"] = 45

    with pytest.raises(ValueError, match="protocol-incompatible"):
        validate_final_truncation_tables(
            tables,
            expected_g1i_task_ids=expected,
        )


def test_load_truncation_tables_requires_both_tables(tmp_path) -> None:
    path = tmp_path / "matrix.json"
    path.write_text('{"summary_tables": {"truncation_vs_parameter_size": []}}')

    try:
        load_truncation_tables(path)
    except ValueError as exc:
        assert "truncation_vs_g1x" in str(exc)
    else:
        raise AssertionError("incomplete truncation tables were accepted")
