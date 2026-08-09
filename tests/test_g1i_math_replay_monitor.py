from __future__ import annotations

from pathlib import Path

from ops.g1i_strict46.monitor_math_replays import (
    REPLAY_CONTRACT,
    REASON_TAG,
    SOURCE_QUERY,
    _build_attestation_command,
    _build_replay_command,
    _final_replays_by_source,
    _score_judge_transcript_matches,
)
from ops.g1i_strict46.math_replay_provenance import (
    ATTESTATION_SCHEMA_VERSION,
    canonical_json_sha256,
)


def test_replay_command_explicitly_overrides_low_environment_judge_concurrency() -> None:
    repo = Path("/srv/rwkv-skills")
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    output = repo / "logs/audits/g1i_math_replays/source_123.json"

    command = _build_replay_command(
        repo=repo,
        replay_script=replay_script,
        source_task_id=123,
        dbname="strict46",
        output=output,
        judge_max_workers=32,
    )

    worker_flag = command.index("--judge-max-workers")
    assert command[worker_flag + 1] == "32"
    assert command[command.index("--judge-mode") + 1] == "exact"
    assert command[command.index("--dbname") + 1] == "strict46"
    assert command[command.index("--output") + 1] == str(output)
    assert command[command.index("--reason-tag") + 1] == REASON_TAG
    assert command[command.index("--final-comparator-sha256") + 1] == (
        REPLAY_CONTRACT.comparator_implementation_sha256
    )
    assert "--determinism-attestation" in command
    assert "--advisory-lock-held-by-caller" in command
    assert "--commit" in command


def test_math_source_query_filters_root_naive_cot_before_selection() -> None:
    assert "t.evaluator IN ('free_response_naive', 'free_response_judge_naive')" in SOURCE_QUERY
    assert "sampling_config->>'prompt_profile'" in SOURCE_QUERY
    assert "sampling_config->>'cot_mode'" in SOURCE_QUERY
    assert "t.git_hash AS task_git_hash" in SOURCE_QUERY


def test_llm_judge_commands_share_one_per_source_transcript() -> None:
    repo = Path("/srv/rwkv-skills")
    transcript = Path("/evidence/source_123.judge-transcript.json")
    attestation = Path("/evidence/source_123.json")
    replay_output = Path("/evidence/source_123.replay.json")

    attest = _build_attestation_command(
        repo=repo,
        source_task_id=123,
        dbname="strict46",
        output=attestation,
        judge_max_workers=16,
        seeds=["0", "1", "42", "999"],
        judge_mode="llm",
        judge_transcript=transcript,
    )
    replay = _build_replay_command(
        repo=repo,
        replay_script=(
            repo / "ops/g1i_strict46/recompute_math_from_completions.py"
        ),
        source_task_id=123,
        dbname="strict46",
        output=replay_output,
        judge_max_workers=16,
        determinism_attestation=attestation,
        judge_mode="llm",
        judge_transcript=transcript,
    )

    for command in (attest, replay):
        assert command[command.index("--judge-mode") + 1] == "llm"
        assert command[command.index("--judge-transcript") + 1] == str(
            transcript
        )


def test_monitor_requires_consistent_judge_transcript_summary() -> None:
    transcript_sha = "8" * 64
    attestation = {
        "judge_transcript_sha256_by_task": {"123": transcript_sha}
    }
    provenance = {
        "judge_transcript": {
            "schema_version": "rwkv.strict46.judge_transcript.v1",
            "sha256": transcript_sha,
            "protocol_fingerprint_sha256": ["7" * 64],
            "statistics": {
                "protocol_count": 1,
                "unique_input_count": 3,
                "actual_judge_call_count": 3,
                "coordinate_count": 4,
                "true_coordinate_count": 2,
                "false_coordinate_count": 2,
                "scope_count": 1,
            },
        }
    }

    assert _score_judge_transcript_matches(
        provenance,
        score_attestation=attestation,
        source_task_id=123,
        parsed={"judge_transcript_sha256": transcript_sha},
        judge_mode="llm",
    )

    provenance["judge_transcript"]["statistics"][
        "true_coordinate_count"
    ] = 3
    assert not _score_judge_transcript_matches(
        provenance,
        score_attestation=attestation,
        source_task_id=123,
        parsed={"judge_transcript_sha256": transcript_sha},
        judge_mode="llm",
    )


def test_existing_replay_matcher_requires_full_frozen_description() -> None:
    source = {
        "task_id": 123,
        "task_git_hash": "f" * 40,
        "model_name": "g1i",
        "benchmark_name": "math_500",
        "benchmark_split": "test",
    }
    source_evidence = {"source_task_id": 123}
    source_evidence_sha = canonical_json_sha256(source_evidence)
    fields = {
        "provenance_version": "g1i_math_replay_v1",
        "replay_source_task_id": "123",
        "reason_tag": REPLAY_CONTRACT.reason_tag,
        "extractor_lineage_sha256": REPLAY_CONTRACT.extractor_lineage_sha256,
        "imported_free_response_sha256": (
            REPLAY_CONTRACT.imported_free_response_sha256
        ),
        "comparator_implementation_sha256": (
            REPLAY_CONTRACT.comparator_implementation_sha256
        ),
        "math_verify_version": REPLAY_CONTRACT.math_verify_version,
        "determinism_attestation_sha256": "e" * 64,
        "pythonhashseed": "42",
        "replay_git_hash": REPLAY_CONTRACT.replay_git_hash,
        "source_git_hash": "f" * 40,
        "source_evidence_sha256": source_evidence_sha,
    }
    replay = {
        **source,
        "task_id": 456,
        "task_git_hash": REPLAY_CONTRACT.replay_git_hash,
        "task_desc": ";".join(f"{key}={value}" for key, value in fields.items()),
        "metrics": {
            "replay_provenance": {
                "source_task_id": 123,
                "source_evidence": source_evidence,
                "source_evidence_sha256": source_evidence_sha,
                "contract": REPLAY_CONTRACT.as_dict(),
                "runtime": {
                    "imported_free_response_sha256": (
                        REPLAY_CONTRACT.imported_free_response_sha256
                    ),
                    "comparator_implementation_sha256": (
                        REPLAY_CONTRACT.comparator_implementation_sha256
                    ),
                    "math_verify_version": REPLAY_CONTRACT.math_verify_version,
                    "replay_git_hash": REPLAY_CONTRACT.replay_git_hash,
                    "pythonhashseed": "42",
                },
                "determinism_attestation": {
                    "schema_version": ATTESTATION_SCHEMA_VERSION,
                    "passed": True,
                    "sha256": "e" * 64,
                    "seeds": ["0", "1", "42", "999"],
                    "task_result_sha256": {"123": "9" * 64},
                    "source_evidence_sha256_by_task": {
                        "123": source_evidence_sha
                    },
                    "judge_transcript_sha256_by_task": {},
                },
                "evaluation_result_sha256": "9" * 64,
            }
        },
    }
    sources = {("g1i", "math_500", "test"): source}

    assert _final_replays_by_source(
        [source, replay],
        sources,
        contract=REPLAY_CONTRACT,
    ) == {123: [replay]}

    replay["task_desc"] = replay["task_desc"].replace(
        REPLAY_CONTRACT.comparator_implementation_sha256,
        "0" * 64,
    )
    assert _final_replays_by_source(
        [source, replay],
        sources,
        contract=REPLAY_CONTRACT,
    ) == {123: []}


def test_existing_replay_matcher_rejects_noncanonical_commit_seed() -> None:
    source = {
        "task_id": 123,
        "task_git_hash": "f" * 40,
        "model_name": "g1i",
        "benchmark_name": "math_500",
        "benchmark_split": "test",
    }
    source_evidence = {"source_task_id": 123}
    source_evidence_sha = canonical_json_sha256(source_evidence)
    fields = {
        "provenance_version": "g1i_math_replay_v1",
        "replay_source_task_id": "123",
        "reason_tag": REPLAY_CONTRACT.reason_tag,
        "extractor_lineage_sha256": REPLAY_CONTRACT.extractor_lineage_sha256,
        "imported_free_response_sha256": REPLAY_CONTRACT.imported_free_response_sha256,
        "comparator_implementation_sha256": REPLAY_CONTRACT.comparator_implementation_sha256,
        "math_verify_version": REPLAY_CONTRACT.math_verify_version,
        "determinism_attestation_sha256": "e" * 64,
        "pythonhashseed": "999",
        "replay_git_hash": REPLAY_CONTRACT.replay_git_hash,
        "source_git_hash": "f" * 40,
        "source_evidence_sha256": source_evidence_sha,
    }
    replay = {
        **source,
        "task_id": 456,
        "task_git_hash": REPLAY_CONTRACT.replay_git_hash,
        "task_desc": ";".join(f"{key}={value}" for key, value in fields.items()),
        "metrics": {
            "replay_provenance": {
                "source_task_id": 123,
                "source_evidence": source_evidence,
                "source_evidence_sha256": source_evidence_sha,
                "contract": REPLAY_CONTRACT.as_dict(),
                "runtime": {
                    "imported_free_response_sha256": REPLAY_CONTRACT.imported_free_response_sha256,
                    "comparator_implementation_sha256": REPLAY_CONTRACT.comparator_implementation_sha256,
                    "math_verify_version": REPLAY_CONTRACT.math_verify_version,
                    "replay_git_hash": REPLAY_CONTRACT.replay_git_hash,
                    "pythonhashseed": "999",
                },
                "determinism_attestation": {
                    "schema_version": ATTESTATION_SCHEMA_VERSION,
                    "passed": True,
                    "sha256": "e" * 64,
                    "seeds": ["0", "1", "42", "999"],
                    "task_result_sha256": {"123": "9" * 64},
                    "source_evidence_sha256_by_task": {
                        "123": source_evidence_sha
                    },
                    "judge_transcript_sha256_by_task": {},
                },
                "evaluation_result_sha256": "9" * 64,
            }
        },
    }

    assert _final_replays_by_source(
        [source, replay],
        {("g1i", "math_500", "test"): source},
        contract=REPLAY_CONTRACT,
    ) == {123: []}
