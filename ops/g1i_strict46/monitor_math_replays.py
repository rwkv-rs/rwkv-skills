#!/usr/bin/env python3
"""Replay scores for pre-fix G1i Math tasks after generation completes.

Only source tasks created before the stage-boundary adapter deployment are
eligible.  Replays are idempotent: a completed score whose task description
records ``replay_source_task_id=<id>`` suppresses any further replay.  The
source task, completions, eval rows, and score remain untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import MATH, MODELS, STRICT_CONFIG_ROOT
from ops.g1i_strict46.monitor_blank_recovery_replays import (
    _filter_source_candidates,
)
from ops.g1i_strict46.monitor_judge_determinism_replays import (
    TASK_QUERY,
    _eligible_source,
    _once_exit_code,
    _plan_ids,
    _plan_replays,
    _replay_artifact,
    _select_latest_valid_sources,
    _split_post_candidates,
    _terminal_action,
)
from ops.g1i_strict46.math_replay_provenance import (
    ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
    ATTESTATION_SCHEMA_VERSION,
    FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
    FINAL_IMPORTED_FREE_RESPONSE_SHA256,
    FINAL_MATH_VERIFY_VERSION,
    FINAL_REPLAY_GIT_HASH,
    FINAL_REPLAY_PYTHONHASHSEED,
    PROVENANCE_VERSION,
    FinalMathReplayContract,
    canonical_json_sha256,
    parse_task_desc,
)
from ops.g1i_strict46.judge_transcript import TRANSCRIPT_SCHEMA_VERSION
from ops.g1i_strict46.replay_lock import (
    held_replay_advisory_locks,
    replay_advisory_lock_keys,
)
from src.db.pool import _build_conninfo
from src.eval.env_config import load_env_file
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
CURRENT_WAVE_STARTED_AT = datetime(2026, 8, 6, 12, 54, 0)
ADAPTER_DEPLOYED_AT = datetime(2026, 8, 7, 1, 15, 0)
REPLAY_CONTRACT = FinalMathReplayContract.from_values()
REASON_TAG = REPLAY_CONTRACT.reason_tag
SOURCE_QUERY = TASK_QUERY.replace(
    't."desc" AS task_desc,',
    't."desc" AS task_desc,\n    t.git_hash AS task_git_hash,',
).replace(
    "ORDER BY t.task_id",
    """
  AND t.evaluator IN ('free_response_naive', 'free_response_judge_naive')
  AND LOWER(COALESCE(t.sampling_config->>'prompt_profile', '')) = 'naive'
  AND REGEXP_REPLACE(
        LOWER(COALESCE(t.sampling_config->>'cot_mode', '')),
        '[^a-z]', '', 'g'
      ) = 'cot'
ORDER BY t.task_id
""",
)


def _judge_mode_for_source(source: dict[str, Any]) -> str:
    return (
        "llm"
        if "judge" in str(source.get("evaluator") or "").lower()
        else "exact"
    )


def _score_judge_transcript_matches(
    score_provenance: dict[str, Any],
    *,
    score_attestation: dict[str, Any],
    source_task_id: int,
    parsed: dict[str, str],
    judge_mode: str,
) -> bool:
    transcript = score_provenance.get("judge_transcript")
    attested_map = score_attestation.get(
        "judge_transcript_sha256_by_task"
    )
    if not isinstance(attested_map, dict):
        return False
    if judge_mode == "exact":
        return (
            "judge_transcript_sha256" not in parsed
            and transcript is None
            and not attested_map
        )
    expected_sha = str(parsed.get("judge_transcript_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        return False
    if not isinstance(transcript, dict):
        return False
    protocols = transcript.get("protocol_fingerprint_sha256")
    statistics = transcript.get("statistics")
    statistic_fields = {
        "protocol_count",
        "unique_input_count",
        "actual_judge_call_count",
        "coordinate_count",
        "true_coordinate_count",
        "false_coordinate_count",
        "scope_count",
    }
    return (
        transcript.get("schema_version") == TRANSCRIPT_SCHEMA_VERSION
        and str(transcript.get("sha256") or "").lower() == expected_sha
        and isinstance(protocols, list)
        and bool(protocols)
        and all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in protocols)
        and isinstance(statistics, dict)
        and statistic_fields.issubset(statistics)
        and all(
            not isinstance(statistics.get(key), bool)
            and isinstance(statistics.get(key), int)
            and int(statistics[key]) >= 0
            for key in statistic_fields
        )
        and int(statistics["scope_count"]) == 1
        and int(statistics["protocol_count"]) >= 1
        and int(statistics["actual_judge_call_count"])
        == int(statistics["unique_input_count"])
        and int(statistics["true_coordinate_count"])
        + int(statistics["false_coordinate_count"])
        == int(statistics["coordinate_count"])
        and int(statistics["actual_judge_call_count"])
        <= int(statistics["coordinate_count"])
        and str(attested_map.get(str(source_task_id)) or "").lower()
        == expected_sha
        and set(str(key) for key in attested_map) == {str(source_task_id)}
    )


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _eligible_sources(connection: psycopg.Connection[Any]) -> list[dict[str, Any]]:
    rows = connection.execute(
        SOURCE_QUERY,
        (
            list(MODELS),
            CURRENT_WAVE_STARTED_AT,
            sorted({name for name, _split in MATH}),
        ),
    ).fetchall()
    candidates = _filter_source_candidates(
        [
            dict(row)
            for row in rows
            if isinstance(row.get("task_created_at"), datetime)
            and row["task_created_at"] < ADAPTER_DEPLOYED_AT
        ]
    )
    selected, _invalid = _select_latest_valid_sources(candidates)
    return list(selected.values())


def _scan(
    connection: psycopg.Connection[Any],
    *,
    contract: FinalMathReplayContract = REPLAY_CONTRACT,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    all_rows = [
        dict(row)
        for row in connection.execute(
            SOURCE_QUERY,
            (
                list(MODELS),
                CURRENT_WAVE_STARTED_AT,
                sorted({name for name, _split in MATH}),
            ),
        ).fetchall()
    ]
    candidates = _filter_source_candidates(all_rows)
    pre = [
        row
        for row in candidates
        if isinstance(row.get("task_created_at"), datetime)
        and row["task_created_at"] < ADAPTER_DEPLOYED_AT
    ]
    post = [
        row
        for row in candidates
        if isinstance(row.get("task_created_at"), datetime)
        and row["task_created_at"] >= ADAPTER_DEPLOYED_AT
    ]
    sources, invalid_sources = _select_latest_valid_sources(pre)
    resolved, pending = _split_post_candidates(post)
    existing = _final_replays_by_source(all_rows, sources, contract=contract)
    return all_rows, _plan_replays(
        sources,
        resolved,
        pending,
        existing,
        invalid_sources,
        deployed_at=ADAPTER_DEPLOYED_AT,
    )


def _final_replays_by_source(
    rows: list[dict[str, Any]],
    sources: dict[tuple[str, str, str], dict[str, Any]],
    *,
    contract: FinalMathReplayContract,
) -> dict[int, list[dict[str, Any]]]:
    """Match only exact final-provenance replays for each immutable source."""

    matches: dict[int, list[dict[str, Any]]] = {}
    for source in sources.values():
        source_task_id = int(source["task_id"])
        source_git_hash = str(source.get("task_git_hash") or "").lower()
        accepted: list[dict[str, Any]] = []
        for row in rows:
            if (
                str(row.get("model_name") or "") != str(source.get("model_name") or "")
                or str(row.get("benchmark_name") or "")
                != str(source.get("benchmark_name") or "")
                or str(row.get("benchmark_split") or "")
                != str(source.get("benchmark_split") or "")
            ):
                continue
            parsed = parse_task_desc(row.get("task_desc"))
            expected = {
                "provenance_version": PROVENANCE_VERSION,
                "replay_source_task_id": str(source_task_id),
                "reason_tag": contract.reason_tag,
                "extractor_lineage_sha256": contract.extractor_lineage_sha256,
                "imported_free_response_sha256": (
                    contract.imported_free_response_sha256
                ),
                "comparator_implementation_sha256": (
                    contract.comparator_implementation_sha256
                ),
                "math_verify_version": contract.math_verify_version,
                "replay_git_hash": contract.replay_git_hash,
                "source_git_hash": source_git_hash,
            }
            if any(str(parsed.get(key) or "").lower() != value.lower() for key, value in expected.items()):
                continue
            if str(row.get("task_git_hash") or "").lower() != contract.replay_git_hash:
                continue
            if not re.fullmatch(
                r"[0-9a-f]{64}",
                str(parsed.get("determinism_attestation_sha256") or "").lower(),
            ):
                continue
            source_evidence_sha = str(
                parsed.get("source_evidence_sha256") or ""
            ).lower()
            if not re.fullmatch(r"[0-9a-f]{64}", source_evidence_sha):
                continue
            if (
                str(parsed.get("pythonhashseed") or "")
                != FINAL_REPLAY_PYTHONHASHSEED
            ):
                continue
            metrics = row.get("metrics")
            score_provenance = (
                metrics.get("replay_provenance")
                if isinstance(metrics, dict)
                else None
            )
            if not isinstance(score_provenance, dict):
                continue
            if int(score_provenance.get("source_task_id") or 0) != source_task_id:
                continue
            score_source_evidence = score_provenance.get("source_evidence")
            if (
                not isinstance(score_source_evidence, dict)
                or canonical_json_sha256(score_source_evidence)
                != source_evidence_sha
                or str(
                    score_provenance.get("source_evidence_sha256") or ""
                ).lower()
                != source_evidence_sha
            ):
                continue
            if score_provenance.get("contract") != contract.as_dict():
                continue
            runtime = score_provenance.get("runtime")
            if not isinstance(runtime, dict):
                continue
            runtime_expected = {
                "imported_free_response_sha256": (
                    contract.imported_free_response_sha256
                ),
                "comparator_implementation_sha256": (
                    contract.comparator_implementation_sha256
                ),
                "math_verify_version": contract.math_verify_version,
                "replay_git_hash": contract.replay_git_hash,
                "pythonhashseed": str(parsed["pythonhashseed"]),
            }
            if any(
                str(runtime.get(key) or "").lower() != value.lower()
                for key, value in runtime_expected.items()
            ):
                continue
            score_attestation = score_provenance.get("determinism_attestation")
            attested_seeds = (
                score_attestation.get("seeds")
                if isinstance(score_attestation, dict)
                else None
            )
            normalized_attested_seeds = (
                {str(seed) for seed in attested_seeds}
                if isinstance(attested_seeds, list)
                else set()
            )
            result_sha = str(
                score_provenance.get("evaluation_result_sha256") or ""
            ).lower()
            if (
                not isinstance(score_attestation, dict)
                or score_attestation.get("schema_version")
                != ATTESTATION_SCHEMA_VERSION
                or score_attestation.get("passed") is not True
                or len(normalized_attested_seeds) < 4
                or not {"0", "1", "42"}.issubset(
                    normalized_attested_seeds
                )
                or any(not seed.isdigit() for seed in normalized_attested_seeds)
                or str(score_attestation.get("sha256") or "").lower()
                != str(parsed["determinism_attestation_sha256"]).lower()
                or not re.fullmatch(r"[0-9a-f]{64}", result_sha)
                or str(
                    (
                        score_attestation.get("task_result_sha256")
                        if isinstance(
                            score_attestation.get("task_result_sha256"), dict
                        )
                        else {}
                    ).get(str(source_task_id))
                    or ""
                ).lower()
                != result_sha
                or str(
                    (
                        score_attestation.get(
                            "source_evidence_sha256_by_task"
                        )
                        if isinstance(
                            score_attestation.get(
                                "source_evidence_sha256_by_task"
                            ),
                            dict,
                        )
                        else {}
                    ).get(str(source_task_id))
                    or ""
                ).lower()
                != source_evidence_sha
            ):
                continue
            if not _score_judge_transcript_matches(
                score_provenance,
                score_attestation=score_attestation,
                source_task_id=source_task_id,
                parsed=parsed,
                judge_mode=_judge_mode_for_source(source),
            ):
                continue
            accepted.append(dict(row))
        matches[source_task_id] = accepted
    return matches


def _build_replay_command(
    *,
    repo: Path,
    replay_script: Path,
    source_task_id: int,
    dbname: str,
    output: Path,
    judge_max_workers: int,
    contract: FinalMathReplayContract = REPLAY_CONTRACT,
    determinism_attestation: Path | None = None,
    judge_mode: str = "exact",
    judge_transcript: Path | None = None,
) -> list[str]:
    attestation = determinism_attestation or output.with_suffix(".attestation.json")
    command = [
        str(repo / ".venv/bin/python"),
        str(replay_script),
        str(source_task_id),
        "--dbname",
        dbname,
        "--judge-mode",
        judge_mode,
        "--judge-max-workers",
        str(judge_max_workers),
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
        "--determinism-attestation",
        str(attestation),
        "--commit",
        "--advisory-lock-held-by-caller",
        "--output",
        str(output),
        "--summary",
    ]
    if judge_transcript is not None:
        command.extend(["--judge-transcript", str(judge_transcript)])
    return command


def _build_attestation_command(
    *,
    repo: Path,
    source_task_id: int,
    dbname: str,
    output: Path,
    judge_max_workers: int,
    seeds: list[str],
    contract: FinalMathReplayContract = REPLAY_CONTRACT,
    judge_mode: str = "exact",
    judge_transcript: Path | None = None,
) -> list[str]:
    command = [
        str(repo / ".venv/bin/python"),
        str(repo / "ops/g1i_strict46/attest_math_replay_determinism.py"),
        str(source_task_id),
        "--dbname",
        dbname,
        "--output",
        str(output),
        "--seeds",
        *seeds,
        "--judge-mode",
        judge_mode,
        "--judge-max-workers",
        str(judge_max_workers),
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
    ]
    if judge_transcript is not None:
        command.extend(["--judge-transcript", str(judge_transcript)])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument(
        "--judge-max-workers",
        type=int,
        default=32,
        help="Explicit LLM-judge concurrency passed to every replay subprocess.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/g1i_math_replay_monitor_events.jsonl"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("logs/audits/g1i_math_replay_monitor.lock"),
    )
    parser.add_argument(
        "--attestation-dir",
        type=Path,
        default=Path("logs/audits/g1i_math_replay_attestations"),
    )
    parser.add_argument(
        "--attestation-seeds",
        nargs="+",
        default=["0", "1", "42", "999"],
    )
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
    if args.judge_max_workers < 1:
        parser.error("--judge-max-workers must be at least 1")
    attestation_seeds = list(
        dict.fromkeys(str(seed).strip() for seed in args.attestation_seeds)
    )
    if len(attestation_seeds) < 4 or any(
        not seed.isdigit() for seed in attestation_seeds
    ):
        parser.error("--attestation-seeds requires four distinct numeric values")
    if any(seed not in attestation_seeds for seed in ("0", "1", "42")):
        parser.error("--attestation-seeds must include 0, 1 and 42")
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
    load_env_file(repo / ".env")
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    output_dir = repo / "logs/audits/g1i_math_replays"
    output_dir.mkdir(parents=True, exist_ok=True)
    args.attestation_dir.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    conninfo = _build_conninfo(config)
    replay_env = dict(os.environ)
    replay_env["RWKV_BENCHMARK_CONFIG_ROOT"] = str(STRICT_CONFIG_ROOT)
    replay_env["PYTHONHASHSEED"] = FINAL_REPLAY_PYTHONHASHSEED
    locally_blocked: dict[int, str] = {}
    last_state: dict[str, list[int]] | None = None

    with args.lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            with psycopg.connect(
                conninfo, row_factory=dict_row
            ) as connection:
                rows, plan = _scan(connection, contract=contract)

            state = _plan_ids(plan)
            if state != last_state:
                _append_event(
                    args.events,
                    {
                        "event": "math_replay_monitor_state",
                        "observed_at": datetime.now().astimezone(),
                        "candidate_task_count": len(rows),
                        "task_ids_by_state": state,
                    },
                )
                last_state = state

            eligible_ids = {
                int(row["task_id"]) for row in plan["eligible_to_replay"]
            }
            locally_blocked = {
                task_id: reason
                for task_id, reason in locally_blocked.items()
                if task_id in eligible_ids
            }

            replay_lock_busy = False
            replay_failed = False
            replay_succeeded = False
            state_changed_under_lock = False
            for row in plan["eligible_to_replay"]:
                source_task_id = int(row["task_id"])
                if source_task_id in locally_blocked:
                    continue
                lock_keys = replay_advisory_lock_keys(
                    dbname=args.dbname,
                    source_task_id=source_task_id,
                    model_name=str(row["model_name"]),
                    benchmark_name=str(row["benchmark_name"]),
                    benchmark_split=str(row["benchmark_split"]),
                )
                with psycopg.connect(
                    conninfo,
                    row_factory=dict_row,
                    autocommit=True,
                ) as lock_connection:
                    with held_replay_advisory_locks(
                        lock_connection, lock_keys
                    ) as acquired:
                        if not acquired:
                            replay_lock_busy = True
                            _append_event(
                                args.events,
                                {
                                    "event": "math_replay_lock_busy",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "lock_keys": lock_keys,
                                },
                            )
                            continue
                        _fresh_rows, fresh_plan = _scan(
                            lock_connection,
                            contract=contract,
                        )
                        fresh = _eligible_source(fresh_plan, source_task_id)
                        if fresh is None:
                            state_changed_under_lock = True
                            _append_event(
                                args.events,
                                {
                                    "event": "math_replay_cancelled",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "reason": "database_state_reconciled_under_lock",
                                },
                            )
                            continue
                        attempt_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                        output = output_dir / (
                            f"source_{source_task_id}_{contract.reason_tag}_"
                            f"{attempt_id}.json"
                        )
                        attestation_output = args.attestation_dir / (
                            f"source_{source_task_id}.json"
                        )
                        judge_mode = _judge_mode_for_source(fresh)
                        judge_transcript = (
                            args.attestation_dir
                            / f"source_{source_task_id}.judge-transcript.json"
                            if judge_mode == "llm"
                            else None
                        )
                        attestation_command = _build_attestation_command(
                            repo=repo,
                            source_task_id=source_task_id,
                            dbname=args.dbname,
                            output=attestation_output,
                            judge_max_workers=args.judge_max_workers,
                            seeds=attestation_seeds,
                            contract=contract,
                            judge_mode=judge_mode,
                            judge_transcript=judge_transcript,
                        )
                        attestation_completed = subprocess.run(
                            attestation_command,
                            cwd=repo,
                            env=replay_env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False,
                        )
                        if attestation_completed.returncode != 0:
                            failure_reason = (
                                "determinism_attestation_returncode:"
                                f"{attestation_completed.returncode}"
                            )
                            locally_blocked[source_task_id] = failure_reason
                            replay_failed = True
                            _append_event(
                                args.events,
                                {
                                    "event": "math_replay_attestation_failed",
                                    "observed_at": datetime.now().astimezone(),
                                    "source": fresh,
                                    "returncode": attestation_completed.returncode,
                                    "failure_reason": failure_reason,
                                    "stdout_tail": attestation_completed.stdout[-4000:],
                                    "stderr_tail": attestation_completed.stderr[-4000:],
                                    "output": str(attestation_output),
                                },
                            )
                            continue
                        command = _build_replay_command(
                            repo=repo,
                            replay_script=replay_script,
                            source_task_id=source_task_id,
                            dbname=args.dbname,
                            output=output,
                            judge_max_workers=args.judge_max_workers,
                            contract=contract,
                            determinism_attestation=attestation_output,
                            judge_mode=judge_mode,
                            judge_transcript=judge_transcript,
                        )
                        started = datetime.now().astimezone()
                        try:
                            completed = subprocess.run(
                                command,
                                cwd=repo,
                                env=replay_env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False,
                            )
                        except OSError as error:
                            returncode: int | None = None
                            stdout_tail = ""
                            stderr_tail = repr(error)
                            replay = None
                            failure_reason = (
                                "subprocess_launch_error:"
                                f"{type(error).__name__}:{error}"
                            )
                        else:
                            returncode = completed.returncode
                            stdout_tail = completed.stdout[-4000:]
                            stderr_tail = completed.stderr[-4000:]
                            replay, artifact_error = (
                                _replay_artifact(output, source_task_id)
                                if completed.returncode == 0
                                else (None, None)
                            )
                            failure_reason = (
                                f"subprocess_returncode:{completed.returncode}"
                                if completed.returncode != 0
                                else artifact_error
                            )
                        event: dict[str, Any] = {
                            "event": (
                                "math_replay_completed"
                                if failure_reason is None
                                else "math_replay_failed"
                            ),
                            "observed_at": datetime.now().astimezone(),
                            "started_at": started,
                            "source": fresh,
                            "returncode": returncode,
                            "failure_reason": failure_reason,
                            "stdout_tail": stdout_tail,
                            "stderr_tail": stderr_tail,
                            "output": str(output),
                        }
                        if replay is not None:
                            event["replay"] = replay
                        _append_event(args.events, event)
                        if failure_reason is None:
                            replay_succeeded = True
                        else:
                            replay_failed = True
                            locally_blocked[source_task_id] = str(failure_reason)

            if args.once:
                return _once_exit_code(replay_failed=replay_failed, plan=plan)
            if replay_succeeded or state_changed_under_lock:
                continue
            if replay_lock_busy:
                time.sleep(max(1.0, args.interval_s))
                continue
            action = _terminal_action(plan, set(locally_blocked))
            if action == "blocked":
                _append_event(
                    args.events,
                    {
                        "event": "math_replay_monitor_blocked",
                        "observed_at": datetime.now().astimezone(),
                        "task_ids_by_state": state,
                        "locally_blocked": locally_blocked,
                    },
                )
                return 2
            if action == "complete":
                _append_event(
                    args.events,
                    {
                        "event": "math_replay_monitor_completed",
                        "observed_at": datetime.now().astimezone(),
                        "task_ids_by_state": state,
                    },
                )
                return 0
            time.sleep(max(1.0, args.interval_s))


if __name__ == "__main__":
    raise SystemExit(main())
