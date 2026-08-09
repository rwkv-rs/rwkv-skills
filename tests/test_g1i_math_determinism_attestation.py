from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops.g1i_strict46.attest_math_replay_determinism import (
    _atomic_write_json,
    _child_command,
    _validated_run,
)
from ops.g1i_strict46.math_replay_provenance import FinalMathReplayContract


def _contract() -> FinalMathReplayContract:
    return FinalMathReplayContract.from_values(
        extractor_lineage_sha256="a" * 64,
        imported_free_response_sha256="b" * 64,
        comparator_implementation_sha256="c" * 64,
        math_verify_version="0.9.0",
        replay_git_hash="d" * 40,
    )


def test_attestation_child_is_explicitly_read_only_and_fully_pinned() -> None:
    repo = Path("/srv/rwkv-skills")
    command = _child_command(
        repo=repo,
        task_ids=[123, 456],
        dbname="strict46",
        output=Path("/tmp/seed-42.json"),
        contract=_contract(),
        judge_mode="exact",
        judge_max_workers=8,
    )

    assert "--emit-determinism-run" in command
    assert "--commit" not in command
    assert command[command.index("--final-comparator-sha256") + 1] == "c" * 64
    assert command[command.index("--final-git-hash") + 1] == "d" * 40
    assert command[command.index("--judge-max-workers") + 1] == "8"


def test_judge_record_and_seed_children_are_mutually_exclusive() -> None:
    repo = Path("/srv/rwkv-skills")
    transcript = Path("/tmp/source-123.judge-transcript.json")
    common = {
        "repo": repo,
        "task_ids": [123],
        "dbname": "strict46",
        "contract": _contract(),
        "judge_mode": "llm",
        "judge_max_workers": 8,
    }

    record = _child_command(
        **common,
        output=Path("/tmp/record.json"),
        record_judge_transcript=transcript,
    )
    assert "--record-judge-transcript" in record
    assert record[record.index("--record-judge-transcript") + 1] == str(
        transcript
    )
    assert "--emit-determinism-run" not in record
    assert "--judge-transcript" not in record

    seed = _child_command(
        **common,
        output=Path("/tmp/seed.json"),
        judge_transcript=transcript,
    )
    assert "--emit-determinism-run" in seed
    assert seed[seed.index("--judge-transcript") + 1] == str(transcript)
    assert "--record-judge-transcript" not in seed


def test_validated_run_requires_every_source_digest(tmp_path: Path) -> None:
    contract = _contract()
    artifact = tmp_path / "seed.json"
    artifact.write_text(
        json.dumps(
            {
                "blocked": False,
                "provenance_preflight": {
                    "passed": True,
                    "contract": contract.as_dict(),
                    "runtime": {"pythonhashseed": "42"},
                },
                "attestation_run": {
                    "seed": "42",
                    "source_task_ids": [123, 456],
                    "task_result_sha256": {"123": "e" * 64},
                    "source_evidence_sha256_by_task": {
                        "123": "f" * 64,
                        "456": "0" * 64,
                    },
                    "judge_transcript_sha256_by_task": {},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="result task IDs incomplete"):
        _validated_run(
            artifact,
            seed="42",
            expected_task_ids=[123, 456],
            contract=contract,
        )


def test_attestation_artifact_is_idempotent_but_never_replaced(
    tmp_path: Path,
) -> None:
    path = tmp_path / "attestation.json"
    first = {
        "schema_version": "v1",
        "generated_at": "first observation",
        "source_task_ids": [123],
        "passed": True,
    }
    _atomic_write_json(path, first)
    original = path.read_bytes()

    _atomic_write_json(path, {**first, "generated_at": "retry observation"})
    assert path.read_bytes() == original

    with pytest.raises(RuntimeError, match="refusing to replace different"):
        _atomic_write_json(path, {**first, "source_task_ids": [456]})
    assert path.read_bytes() == original
