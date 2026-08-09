#!/usr/bin/env python3
"""Build a fail-closed multi-PYTHONHASHSEED Math replay attestation.

Each child is a read-only invocation of ``recompute_math_from_completions.py``.
The final JSON is written atomically only when every requested task produces
the identical evaluation digest under at least four distinct seeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from ops.g1i_strict46.math_replay_provenance import (
    ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
    ATTESTATION_SCHEMA_VERSION,
    FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
    FINAL_IMPORTED_FREE_RESPONSE_SHA256,
    FINAL_MATH_VERIFY_VERSION,
    FINAL_REPLAY_GIT_HASH,
    FinalMathReplayContract,
)
from ops.g1i_strict46.judge_transcript import load_judge_transcript


DEFAULT_DB_NAME = "chase_rwkv_skills_frontend46_20260804"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _substantive_attestation(payload: object) -> object:
    """Drop only observational metadata when comparing immutable artifacts."""

    if not isinstance(payload, dict):
        return payload
    return {key: value for key, value in payload.items() if key != "generated_at"}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Create an attestation atomically, or reuse an equivalent artifact.

    A final attestation is evidence, not a cache entry.  It must never be
    silently replaced after another process has consumed its SHA.  Retrying
    the builder is idempotent when the substantive payload is identical;
    contract, source, seed, or result drift fails closed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"refusing to replace unreadable attestation: {path}"
            ) from exc
        if _substantive_attestation(existing) == _substantive_attestation(payload):
            return
        raise RuntimeError(f"refusing to replace different attestation: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-linking a completed temp file gives create-if-absent
            # semantics without exposing a partially-written destination.
            os.link(temporary_path, path)
        except FileExistsError:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"refusing to replace unreadable attestation: {path}"
                ) from exc
            if _substantive_attestation(existing) != _substantive_attestation(
                payload
            ):
                raise RuntimeError(
                    f"refusing to replace different attestation: {path}"
                )
    finally:
        temporary_path.unlink(missing_ok=True)


def _child_command(
    *,
    repo: Path,
    task_ids: list[int],
    dbname: str,
    output: Path,
    contract: FinalMathReplayContract,
    judge_mode: str,
    judge_max_workers: int | None,
    judge_transcript: Path | None = None,
    record_judge_transcript: Path | None = None,
) -> list[str]:
    command = [
        str(repo / ".venv/bin/python"),
        str(repo / "ops/g1i_strict46/recompute_math_from_completions.py"),
        *(str(task_id) for task_id in task_ids),
        "--dbname",
        dbname,
        "--reason-tag",
        contract.reason_tag,
        "--final-extractor-lineage-sha256",
        contract.extractor_lineage_sha256,
        "--final-imported-free-response-sha256",
        contract.imported_free_response_sha256,
        "--final-comparator-sha256",
        contract.comparator_implementation_sha256,
        "--final-math-verify-version",
        contract.math_verify_version,
        "--final-git-hash",
        contract.replay_git_hash,
        "--judge-mode",
        judge_mode,
        "--output",
        str(output),
        "--summary",
    ]
    if record_judge_transcript is not None:
        command.extend(
            ["--record-judge-transcript", str(record_judge_transcript)]
        )
    else:
        command.append("--emit-determinism-run")
    if judge_transcript is not None:
        command.extend(["--judge-transcript", str(judge_transcript)])
    if judge_max_workers is not None:
        command.extend(["--judge-max-workers", str(judge_max_workers)])
    return command


def _validated_run(
    path: Path,
    *,
    seed: str,
    expected_task_ids: list[int],
    contract: FinalMathReplayContract,
    expected_judge_transcript_sha256_by_task: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"seed {seed}: invalid child artifact: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("blocked") is not False:
        raise RuntimeError(f"seed {seed}: child replay was blocked")
    provenance = payload.get("provenance_preflight")
    if not isinstance(provenance, dict) or provenance.get("passed") is not True:
        raise RuntimeError(f"seed {seed}: runtime provenance preflight failed")
    if provenance.get("contract") != contract.as_dict():
        raise RuntimeError(f"seed {seed}: child contract drift")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, dict) or str(runtime.get("pythonhashseed")) != seed:
        raise RuntimeError(f"seed {seed}: child PYTHONHASHSEED attestation drift")
    run = payload.get("attestation_run")
    if not isinstance(run, dict) or str(run.get("seed")) != seed:
        raise RuntimeError(f"seed {seed}: determinism run metadata missing")
    try:
        observed_ids = sorted(int(task_id) for task_id in run["source_task_ids"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"seed {seed}: invalid source task IDs") from exc
    if observed_ids != expected_task_ids:
        raise RuntimeError(
            f"seed {seed}: source task IDs {observed_ids} != {expected_task_ids}"
        )
    results = run.get("task_result_sha256")
    if not isinstance(results, dict):
        raise RuntimeError(f"seed {seed}: result digest map missing")
    if sorted(int(key) for key in results if str(key).isdigit()) != expected_task_ids:
        raise RuntimeError(f"seed {seed}: result task IDs incomplete")
    if len(results) != len(expected_task_ids) or any(
        not SHA256_RE.fullmatch(str(value)) for value in results.values()
    ):
        raise RuntimeError(f"seed {seed}: invalid result digests")
    source_evidence = run.get("source_evidence_sha256_by_task")
    if not isinstance(source_evidence, dict):
        raise RuntimeError(f"seed {seed}: source evidence digest map missing")
    if sorted(
        int(key) for key in source_evidence if str(key).isdigit()
    ) != expected_task_ids:
        raise RuntimeError(f"seed {seed}: source evidence task IDs incomplete")
    if len(source_evidence) != len(expected_task_ids) or any(
        not SHA256_RE.fullmatch(str(value)) for value in source_evidence.values()
    ):
        raise RuntimeError(f"seed {seed}: invalid source evidence digests")
    judge_transcripts = run.get("judge_transcript_sha256_by_task")
    expected_judge_transcripts = (
        expected_judge_transcript_sha256_by_task or {}
    )
    if judge_transcripts != expected_judge_transcripts:
        raise RuntimeError(
            f"seed {seed}: Judge transcript digest map mismatch"
        )
    return {
        "seed": seed,
        "task_result_sha256": results,
        "source_evidence_sha256_by_task": source_evidence,
        "judge_transcript_sha256_by_task": judge_transcripts,
        "runtime": runtime,
    }


def _ensure_judge_transcript(
    *,
    path: Path,
    repo: Path,
    task_ids: list[int],
    dbname: str,
    output: Path,
    contract: FinalMathReplayContract,
    judge_max_workers: int | None,
) -> str:
    """Record one immutable transcript while excluding concurrent recorders."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.record.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not path.exists():
            record_command = _child_command(
                repo=repo,
                task_ids=task_ids,
                dbname=dbname,
                output=output,
                contract=contract,
                judge_mode="llm",
                judge_max_workers=judge_max_workers,
                record_judge_transcript=path,
            )
            record_environment = dict(os.environ)
            record_environment["PYTHONHASHSEED"] = "42"
            recorded = subprocess.run(
                record_command,
                cwd=repo,
                env=record_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if recorded.returncode != 0:
                raise RuntimeError(
                    "Judge transcript recording failed with "
                    f"{recorded.returncode}; "
                    f"stderr={recorded.stderr[-2000:]!r}"
                )
            try:
                record_payload = json.loads(output.read_text(encoding="utf-8"))
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise RuntimeError(
                    "Judge transcript recording output is invalid"
                ) from exc
            if (
                not isinstance(record_payload, dict)
                or record_payload.get("blocked") is not False
                or sorted(
                    int(row["task_id"])
                    for row in record_payload.get("tasks", [])
                    if isinstance(row, dict) and row.get("replayable")
                )
                != task_ids
            ):
                raise RuntimeError(
                    "Judge transcript recording did not validate every task"
                )
        return load_judge_transcript(path).sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="+", type=int)
    parser.add_argument("--dbname", default=DEFAULT_DB_NAME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "42", "999"])
    parser.add_argument(
        "--judge-mode",
        choices=("exact", "llm"),
        required=True,
        help=(
            "Frozen evaluator family. LLM mode records one temperature-zero "
            "transcript if needed, then all seed runs are network-free."
        ),
    )
    parser.add_argument(
        "--judge-transcript",
        type=Path,
        help=(
            "Immutable per-source Judge transcript. Required for llm mode; "
            "created once under PYTHONHASHSEED=42 when absent."
        ),
    )
    parser.add_argument("--judge-max-workers", type=int)
    parser.add_argument("--reason-tag", default="")
    parser.add_argument(
        "--final-extractor-lineage-sha256",
        default=ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
    )
    parser.add_argument(
        "--final-imported-free-response-sha256",
        default=FINAL_IMPORTED_FREE_RESPONSE_SHA256,
    )
    parser.add_argument(
        "--final-comparator-sha256",
        default=FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
    )
    parser.add_argument("--final-math-verify-version", default=FINAL_MATH_VERIFY_VERSION)
    parser.add_argument("--final-git-hash", default=FINAL_REPLAY_GIT_HASH)
    args = parser.parse_args()

    task_ids = sorted(set(args.task_ids))
    seeds = list(dict.fromkeys(str(seed).strip() for seed in args.seeds))
    if len(seeds) < 4 or any(not seed.isdigit() for seed in seeds):
        parser.error("--seeds requires at least four distinct numeric values")
    for required in ("0", "1", "42"):
        if required not in seeds:
            parser.error(f"--seeds must include {required}")
    if args.judge_max_workers is not None and args.judge_max_workers < 1:
        parser.error("--judge-max-workers must be at least 1")
    if args.judge_mode == "llm" and args.judge_transcript is None:
        parser.error("--judge-mode=llm requires --judge-transcript")
    if args.judge_mode == "llm" and len(task_ids) != 1:
        parser.error(
            "--judge-mode=llm requires exactly one source task so its "
            "transcript can be fully consumed before append-only commit"
        )
    if args.judge_mode == "exact" and args.judge_transcript is not None:
        parser.error("--judge-transcript is forbidden with --judge-mode=exact")

    contract = FinalMathReplayContract.from_values(
        extractor_lineage_sha256=args.final_extractor_lineage_sha256,
        imported_free_response_sha256=args.final_imported_free_response_sha256,
        comparator_implementation_sha256=args.final_comparator_sha256,
        math_verify_version=args.final_math_verify_version,
        replay_git_hash=args.final_git_hash,
        reason_tag=args.reason_tag,
    )
    if contract.blockers():
        parser.error("invalid final contract: " + ", ".join(contract.blockers()))

    repo = Path(__file__).resolve().parents[2]
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="g1i-math-attestation-") as directory:
        artifact_root = Path(directory)
        judge_transcript_sha256_by_task: dict[str, str] = {}
        if args.judge_mode == "llm":
            assert args.judge_transcript is not None
            transcript_sha256 = _ensure_judge_transcript(
                path=args.judge_transcript,
                repo=repo,
                task_ids=task_ids,
                dbname=args.dbname,
                output=artifact_root / "judge_record.json",
                contract=contract,
                judge_max_workers=args.judge_max_workers,
            )
            judge_transcript_sha256_by_task = {
                str(task_id): transcript_sha256
                for task_id in task_ids
            }
        for seed in seeds:
            artifact = artifact_root / f"seed_{seed}.json"
            command = _child_command(
                repo=repo,
                task_ids=task_ids,
                dbname=args.dbname,
                output=artifact,
                contract=contract,
                judge_mode=args.judge_mode,
                judge_max_workers=args.judge_max_workers,
                judge_transcript=args.judge_transcript,
            )
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"seed {seed}: child returned {completed.returncode}; "
                    f"stderr={completed.stderr[-2000:]!r}"
                )
            runs.append(
                _validated_run(
                    artifact,
                    seed=seed,
                    expected_task_ids=task_ids,
                    contract=contract,
                    expected_judge_transcript_sha256_by_task=(
                        judge_transcript_sha256_by_task
                    ),
                )
            )

    canonical_results = runs[0]["task_result_sha256"]
    if any(run["task_result_sha256"] != canonical_results for run in runs[1:]):
        raise RuntimeError("cross-PYTHONHASHSEED replay result mismatch")
    canonical_source_evidence = runs[0]["source_evidence_sha256_by_task"]
    if any(
        run["source_evidence_sha256_by_task"] != canonical_source_evidence
        for run in runs[1:]
    ):
        raise RuntimeError("cross-PYTHONHASHSEED source evidence mismatch")
    canonical_judge_transcripts = runs[0][
        "judge_transcript_sha256_by_task"
    ]
    if any(
        run["judge_transcript_sha256_by_task"]
        != canonical_judge_transcripts
        for run in runs[1:]
    ):
        raise RuntimeError("cross-PYTHONHASHSEED Judge transcript mismatch")
    runtime_keys = (
        "imported_free_response_sha256",
        "comparator_implementation_sha256",
        "math_verify_version",
        "replay_git_hash",
    )
    canonical_runtime = runs[0]["runtime"]
    for run in runs[1:]:
        if any(run["runtime"].get(key) != canonical_runtime.get(key) for key in runtime_keys):
            raise RuntimeError("cross-PYTHONHASHSEED runtime provenance mismatch")

    attestation = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "passed": True,
        "generated_at": datetime.now().astimezone(),
        "source_task_ids": task_ids,
        "extractor_lineage_sha256": contract.extractor_lineage_sha256,
        "imported_free_response_sha256": contract.imported_free_response_sha256,
        "comparator_implementation_sha256": (
            contract.comparator_implementation_sha256
        ),
        "math_verify_version": contract.math_verify_version,
        "replay_git_hash": contract.replay_git_hash,
        "reason_tag": contract.reason_tag,
        "source_evidence_sha256_by_task": canonical_source_evidence,
        "judge_transcript_sha256_by_task": canonical_judge_transcripts,
        "pythonhashseed_runs": [
            {
                "seed": run["seed"],
                "task_result_sha256": run["task_result_sha256"],
                "source_evidence_sha256_by_task": run[
                    "source_evidence_sha256_by_task"
                ],
                "judge_transcript_sha256_by_task": run[
                    "judge_transcript_sha256_by_task"
                ],
            }
            for run in runs
        ],
    }
    _atomic_write_json(args.output, attestation)
    print(json.dumps(attestation, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
