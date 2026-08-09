from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

from ops.g1i_strict46.math_replay_provenance import (
    ATTESTATION_SCHEMA_VERSION,
    FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
    FINAL_IMPORTED_FREE_RESPONSE_SHA256,
    FINAL_MATH_VERIFY_VERSION,
    FINAL_REPLAY_GIT_HASH,
    DeterminismAttestation,
    FinalMathReplayContract,
    RuntimeMathProvenance,
    build_replay_task_desc,
    collect_runtime_math_provenance,
    evaluation_result_sha256,
    load_determinism_attestation,
    parse_task_desc,
    runtime_contract_reasons,
)
from src.eval.metrics import free_response


def _contract() -> FinalMathReplayContract:
    return FinalMathReplayContract.from_values(
        extractor_lineage_sha256="a" * 64,
        imported_free_response_sha256="b" * 64,
        comparator_implementation_sha256="c" * 64,
        math_verify_version="0.9.0",
        replay_git_hash="d" * 40,
    )


def _runtime() -> RuntimeMathProvenance:
    return RuntimeMathProvenance(
        imported_module_path="/repo/src/eval/metrics/free_response.py",
        imported_free_response_sha256="b" * 64,
        comparator_implementation=(
            "src.eval.metrics.free_response._deterministic_math_verify"
        ),
        comparator_implementation_sha256="c" * 64,
        math_verify_version="0.9.0",
        pythonhashseed="42",
        replay_git_hash="d" * 40,
        blockers=(),
    )


def _write_attestation(path: Path, *, result_overrides: dict[str, str] | None = None) -> None:
    contract = _contract()
    results = {"123": "e" * 64, **(result_overrides or {})}
    payload = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "passed": True,
        "source_task_ids": [123],
        "extractor_lineage_sha256": contract.extractor_lineage_sha256,
        "imported_free_response_sha256": contract.imported_free_response_sha256,
        "comparator_implementation_sha256": (
            contract.comparator_implementation_sha256
        ),
        "math_verify_version": contract.math_verify_version,
        "replay_git_hash": contract.replay_git_hash,
        "reason_tag": contract.reason_tag,
        "source_evidence_sha256_by_task": {"123": "d" * 64},
        "judge_transcript_sha256_by_task": {},
        "pythonhashseed_runs": [
            {
                "seed": seed,
                "task_result_sha256": results,
                "source_evidence_sha256_by_task": {"123": "d" * 64},
                "judge_transcript_sha256_by_task": {},
            }
            for seed in ("0", "1", "42", "999")
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_frozen_runtime_hashes_match_and_git_drift_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "42")

    runtime = collect_runtime_math_provenance()
    expected_function_sha = hashlib.sha256(
        inspect.getsource(free_response._deterministic_math_verify).encode("utf-8")
    ).hexdigest()

    assert runtime.imported_free_response_sha256 == FINAL_IMPORTED_FREE_RESPONSE_SHA256
    assert expected_function_sha == FINAL_COMPARATOR_IMPLEMENTATION_SHA256
    assert runtime.comparator_implementation_sha256 == expected_function_sha
    assert runtime.math_verify_version == FINAL_MATH_VERIFY_VERSION
    assert runtime.blockers == ()
    expected_runtime_reasons = (
        []
        if runtime.replay_git_hash == FINAL_REPLAY_GIT_HASH
        else [
            f"replay_git_hash:{runtime.replay_git_hash}"
            f"!=expected:{FINAL_REPLAY_GIT_HASH}"
        ]
    )
    assert runtime_contract_reasons(
        runtime, FinalMathReplayContract.from_values()
    ) == expected_runtime_reasons

    drifted_git = "0" * 40
    if drifted_git == FINAL_REPLAY_GIT_HASH:
        drifted_git = "1" * 40
    drifted = replace(runtime, replay_git_hash=drifted_git)
    reasons = runtime_contract_reasons(
        drifted, FinalMathReplayContract.from_values()
    )
    assert reasons == [
        f"replay_git_hash:{drifted_git}"
        f"!=expected:{FINAL_REPLAY_GIT_HASH}"
    ]


def test_attestation_requires_exact_tasks_and_four_equal_seed_runs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attestation.json"
    _write_attestation(path)

    attestation, reasons = load_determinism_attestation(
        path,
        contract=_contract(),
        runtime=_runtime(),
        source_task_ids=[123],
    )

    assert reasons == []
    assert attestation is not None
    assert attestation.seeds == ("0", "1", "42", "999")
    assert attestation.task_result_sha256 == {"123": "e" * 64}


def test_attestation_rejects_non_numeric_task_keys(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    _write_attestation(path, result_overrides={"not-a-task": "f" * 64})

    _attestation, reasons = load_determinism_attestation(
        path,
        contract=_contract(),
        runtime=_runtime(),
        source_task_ids=[123],
    )

    assert "determinism_attestation_run_0_task_id_key_invalid" in reasons
    assert "determinism_attestation_run_0_task_ids_mismatch" in reasons


def test_attestation_rejects_duplicate_source_task_ids(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    _write_attestation(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_task_ids"] = [123, 123]
    path.write_text(json.dumps(payload), encoding="utf-8")

    _attestation, reasons = load_determinism_attestation(
        path,
        contract=_contract(),
        runtime=_runtime(),
        source_task_ids=[123],
    )

    assert "determinism_attestation_source_task_ids_duplicate" in reasons


def test_replay_description_round_trips_every_provenance_field() -> None:
    attestation = DeterminismAttestation(
        path="/tmp/attestation.json",
        sha256="e" * 64,
        seeds=("0", "1", "42", "999"),
        task_result_sha256={"123": "f" * 64},
        source_evidence_sha256_by_task={"123": "d" * 64},
        judge_transcript_sha256_by_task={},
        payload={"schema_version": ATTESTATION_SCHEMA_VERSION, "passed": True},
    )

    description = build_replay_task_desc(
        source_task_id=123,
        source_git_hash="f" * 40,
        contract=_contract(),
        runtime=_runtime(),
        attestation=attestation,
    )
    parsed = parse_task_desc(description)

    assert parsed["replay_source_task_id"] == "123"
    assert parsed["source_git_hash"] == "f" * 40
    assert parsed["determinism_attestation_sha256"] == "e" * 64
    assert parsed["pythonhashseed"] == "42"
    assert parsed["source_evidence_sha256"] == "d" * 64
    assert parsed["reason_tag"] == _contract().reason_tag


def test_replay_description_includes_attested_judge_transcript() -> None:
    attestation = DeterminismAttestation(
        path="/tmp/attestation.json",
        sha256="e" * 64,
        seeds=("0", "1", "42", "999"),
        task_result_sha256={"123": "f" * 64},
        source_evidence_sha256_by_task={"123": "d" * 64},
        judge_transcript_sha256_by_task={"123": "9" * 64},
        payload={"schema_version": ATTESTATION_SCHEMA_VERSION, "passed": True},
    )

    description = build_replay_task_desc(
        source_task_id=123,
        source_git_hash="f" * 40,
        contract=_contract(),
        runtime=_runtime(),
        attestation=attestation,
    )

    assert parse_task_desc(description)["judge_transcript_sha256"] == "9" * 64


def test_runtime_contract_gate_is_fail_closed_on_any_hash_drift() -> None:
    drifted = replace(_runtime(), comparator_implementation_sha256="f" * 64)

    reasons = runtime_contract_reasons(drifted, _contract())

    assert reasons == [
        "comparator_implementation_sha256:"
        f"{'f' * 64}!=expected:{'c' * 64}"
    ]


def test_evaluation_result_digest_ignores_dict_insertion_order() -> None:
    first = evaluation_result_sha256(
        rows_by_group={"strategy_c": [(0, 0, True)]},
        payloads_by_group={
            "strategy_c": [
                {
                    "sample_index": 0,
                    "repeat_index": 0,
                    "answer": "7",
                    "ref_answer": "7",
                    "is_passed": True,
                }
            ]
        },
    )
    second = evaluation_result_sha256(
        rows_by_group={"strategy_c": [(0, 0, True)]},
        payloads_by_group={
            "strategy_c": [
                {
                    "is_passed": True,
                    "ref_answer": "7",
                    "answer": "7",
                    "repeat_index": 0,
                    "sample_index": 0,
                }
            ]
        },
    )

    assert first == second
