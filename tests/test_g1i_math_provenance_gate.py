from __future__ import annotations

from ops.g1i_strict46.audit_current import _math_final_provenance_reasons
from ops.g1i_strict46.math_replay_provenance import (
    ATTESTATION_SCHEMA_VERSION,
    PROVENANCE_VERSION,
    FinalMathReplayContract,
    canonical_json_sha256,
    parse_task_desc,
)
from ops.g1i_strict46.judge_transcript import TRANSCRIPT_SCHEMA_VERSION


MODEL = "rwkv7-g1i-7.2b-20260805-ctx16384"


def _contract() -> FinalMathReplayContract:
    return FinalMathReplayContract.from_values(
        extractor_lineage_sha256="a" * 64,
        imported_free_response_sha256="b" * 64,
        comparator_implementation_sha256="c" * 64,
        math_verify_version="0.9.0",
        replay_git_hash="d" * 40,
    )


def _source() -> dict[str, object]:
    return {
        "task_id": 29000,
        "status": "Completed",
        "evaluator": "free_response_naive",
        "task_git_hash": "f" * 40,
        "task_desc": "original_generation=true",
        "task_is_tmp": False,
        "task_is_param_search": False,
        "model_name": MODEL,
        "benchmark_name": "math_500",
        "benchmark_split": "test",
    }


def _replay() -> dict[str, object]:
    contract = _contract()
    attestation_sha = "e" * 64
    result_sha = "9" * 64
    source_evidence = {
        "source_task_id": 29000,
        "source_git_hash": "f" * 40,
        "source_evaluator": "free_response_naive",
        "model_name": MODEL,
        "dataset_slug": "math_500_test",
    }
    source_evidence_sha = canonical_json_sha256(source_evidence)
    fields = {
        "provenance_version": PROVENANCE_VERSION,
        "replay_source_task_id": "29000",
        "reason_tag": contract.reason_tag,
        "extractor_lineage_sha256": contract.extractor_lineage_sha256,
        "imported_free_response_sha256": contract.imported_free_response_sha256,
        "comparator_implementation_sha256": (
            contract.comparator_implementation_sha256
        ),
        "math_verify_version": contract.math_verify_version,
        "determinism_attestation_sha256": attestation_sha,
        "pythonhashseed": "42",
        "replay_git_hash": contract.replay_git_hash,
        "source_git_hash": "f" * 40,
        "source_evidence_sha256": source_evidence_sha,
    }
    return {
        "task_id": 30000,
        "task_git_hash": contract.replay_git_hash,
        "task_desc": ";".join(f"{key}={value}" for key, value in fields.items()),
        "model_name": MODEL,
        "benchmark_name": "math_500",
        "benchmark_split": "test",
        "evaluator": "free_response_naive",
        "eval_count": 4000,
        "judgement_output_source_row_count": 0,
        "judgement_reference_row_count": 0,
        "judgement_output_source_mismatch_count": 0,
        "metrics": {
            "replay_provenance": {
                "source_task_id": 29000,
                "source_evidence": source_evidence,
                "source_evidence_sha256": source_evidence_sha,
                "contract": contract.as_dict(),
                "runtime": {
                    "imported_free_response_sha256": (
                        contract.imported_free_response_sha256
                    ),
                    "comparator_implementation_sha256": (
                        contract.comparator_implementation_sha256
                    ),
                    "math_verify_version": contract.math_verify_version,
                    "replay_git_hash": contract.replay_git_hash,
                    "pythonhashseed": "42",
                },
                "determinism_attestation": {
                    "schema_version": ATTESTATION_SCHEMA_VERSION,
                    "passed": True,
                    "sha256": attestation_sha,
                    "seeds": ["0", "1", "42", "999"],
                    "task_result_sha256": {"29000": result_sha},
                    "source_evidence_sha256_by_task": {
                        "29000": source_evidence_sha
                    },
                    "judge_transcript_sha256_by_task": {},
                },
                "evaluation_result_sha256": result_sha,
            }
        },
    }


def test_math_final_gate_accepts_only_frozen_append_only_replay() -> None:
    row = _replay()

    reasons = _math_final_provenance_reasons(
        row,
        contract=_contract(),
        source_tasks={29000: _source()},
    )

    assert reasons == []
    assert row["math_provenance_gate"]["passed"] is True
    assert row["math_provenance_gate"]["source_task_id"] == 29000


def test_math_final_gate_requires_attested_external_judge_transcript() -> None:
    row = _replay()
    source = _source()
    source["evaluator"] = "free_response_judge_naive"
    provenance = row["metrics"]["replay_provenance"]
    evidence = provenance["source_evidence"]
    evidence["source_evaluator"] = "free_response_judge_naive"
    evidence_sha = canonical_json_sha256(evidence)
    provenance["source_evidence_sha256"] = evidence_sha
    provenance["determinism_attestation"][
        "source_evidence_sha256_by_task"
    ] = {"29000": evidence_sha}
    transcript_sha = "8" * 64
    provenance["determinism_attestation"][
        "judge_transcript_sha256_by_task"
    ] = {"29000": transcript_sha}
    provenance["judge_transcript"] = {
        "schema_version": TRANSCRIPT_SCHEMA_VERSION,
        "sha256": transcript_sha,
        "protocol_fingerprint_sha256": ["7" * 64],
        "statistics": {
            "scope_count": 1,
            "protocol_count": 1,
            "unique_input_count": 12,
            "coordinate_count": 12,
            "actual_judge_call_count": 12,
            "true_coordinate_count": 7,
            "false_coordinate_count": 5,
        },
    }
    fields = parse_task_desc(row["task_desc"])
    fields["source_evidence_sha256"] = evidence_sha
    fields["judge_transcript_sha256"] = transcript_sha
    row["task_desc"] = ";".join(
        f"{key}={value}" for key, value in fields.items()
    )

    assert _math_final_provenance_reasons(
        row,
        contract=_contract(),
        source_tasks={29000: source},
    ) == []

    del provenance["judge_transcript"]
    reasons = _math_final_provenance_reasons(
        row,
        contract=_contract(),
        source_tasks={29000: source},
    )
    assert "math_replay_score_judge_transcript_missing" in reasons


def test_math_final_gate_never_accepts_explicitly_disallowed_rows() -> None:
    for task_id in (28869, 28872):
        row = _replay()
        row["task_id"] = task_id

        reasons = _math_final_provenance_reasons(
            row,
            contract=_contract(),
            source_tasks={29000: _source()},
        )

        assert reasons == [f"explicitly_disallowed_final_math_task:{task_id}"]


def test_math_final_gate_rejects_replay_chains() -> None:
    source = _source()
    source["task_desc"] = "replay_source_task_id=28000"

    reasons = _math_final_provenance_reasons(
        _replay(),
        contract=_contract(),
        source_tasks={29000: source},
    )

    assert "math_replay_source_is_replay_chain" in reasons


def test_math_final_gate_rejects_noncanonical_commit_seed() -> None:
    row = _replay()
    row["task_desc"] = str(row["task_desc"]).replace(
        "pythonhashseed=42", "pythonhashseed=999"
    )
    row["metrics"]["replay_provenance"]["runtime"]["pythonhashseed"] = "999"

    reasons = _math_final_provenance_reasons(
        row,
        contract=_contract(),
        source_tasks={29000: _source()},
    )

    assert "math_replay_pythonhashseed:999!=expected:42" in reasons


def test_answer_judge_bypass_requires_all_rows_to_be_judgement_references() -> None:
    row = {
        "task_id": 30001,
        "benchmark_name": "answer_judge",
        "benchmark_split": "test",
        "evaluator": "free_response_naive",
        "eval_count": 1600,
        "judgement_output_source_row_count": 1600,
        "judgement_reference_row_count": 1600,
        "judgement_output_source_mismatch_count": 0,
    }

    assert (
        _math_final_provenance_reasons(
            row,
            contract=_contract(),
            source_tasks={},
        )
        == []
    )
    assert row["math_provenance_gate"]["mode"] == (
        "answer_judge_reference_short_circuit"
    )

    row["judgement_reference_row_count"] = 1599
    reasons = _math_final_provenance_reasons(
        row,
        contract=_contract(),
        source_tasks={},
    )
    assert "math_replay_source_task_id_missing_or_invalid" in reasons
