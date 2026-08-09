#!/usr/bin/env python3
"""Replay Math scores from persisted completions with the current adapter.

The default mode is read-only.  ``--commit`` creates a fresh traceable task,
copies the source completions, writes newly evaluated rows and records the
recomputed score; it never mutates or deletes the source task.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import re
import psycopg
from psycopg.rows import dict_row

from src.db.database import init_db
from src.db.eval_db_service import EvalDbService
from src.db.pool import _build_conninfo
from src.eval.benchmark_config import resolve_benchmark_model_config
from src.eval.env_config import load_env_file
from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k
from src.eval.metrics.free_response import (
    STRATEGY_GROUPS,
    STRATEGY_C,
    FreeResponseEvaluation,
    attach_strategy_task_ids,
    build_grouped_metrics_payload,
    evaluate_free_response,
)
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import make_dataset_slug
from src.eval.tasks.maths.common import build_llm_judge

from ops.g1i_strict46.math_replay_provenance import (
    ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
    FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
    FINAL_IMPORTED_FREE_RESPONSE_SHA256,
    FINAL_MATH_VERIFY_VERSION,
    FINAL_REPLAY_GIT_HASH,
    FINAL_REPLAY_PYTHONHASHSEED,
    FinalMathReplayContract,
    build_replay_task_desc,
    canonical_json_sha256,
    collect_runtime_math_provenance,
    evaluation_result_sha256,
    load_determinism_attestation,
    parse_task_desc,
    runtime_contract_reasons,
    sha256_file,
)
from ops.g1i_strict46.judge_transcript import (
    JudgeTranscriptArtifact,
    JudgeTranscriptError,
    JudgeTranscriptRecorder,
    JudgeTranscriptReplayer,
    load_judge_transcript,
)
from ops.g1i_strict46.replay_lock import (
    held_replay_advisory_locks,
    replay_advisory_lock_keys,
)


DEFAULT_DB_NAME = "chase_rwkv_skills_frontend46_20260804"
MATH_VERIFY_RETRY_TIMEOUT_S = 15.0
REPLAY_SCORE_PERSISTENCE_GRACE = timedelta(minutes=10)


EXISTING_REPLAY_QUERY = """
SELECT
    t.task_id,
    t.status,
    t.created_at AS task_created_at,
    t.git_hash AS task_git_hash,
    t."desc" AS task_desc,
    latest_score.score_id,
    COALESCE(stats.completion_count, 0) AS completion_count,
    COALESCE(stats.eval_count, 0) AS eval_count
FROM task t
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN LATERAL (
    SELECT s.score_id
    FROM scores s
    WHERE s.task_id = t.task_id
    ORDER BY s.score_id DESC
    LIMIT 1
) AS latest_score ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE c.status = 'Completed') AS completion_count,
        COUNT(e.eval_id) FILTER (WHERE c.status = 'Completed') AS eval_count
    FROM completions c
    LEFT JOIN eval e ON e.completions_id = c.completions_id
    WHERE c.task_id = t.task_id
) AS stats ON TRUE
WHERE t."desc" = %s
  AND m.model_name = %s
  AND b.benchmark_name = %s
  AND b.benchmark_split = %s
  AND t.is_param_search = FALSE
  AND t.is_tmp = FALSE
ORDER BY t.task_id
"""


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _judge_endpoint_url(judge: object) -> object | None:
    """Return the endpoint actually bound to the live Judge client.

    ``LLMJudgeConfig.base_url`` can differ textually from the URL normalized by
    the OpenAI client (most commonly by a trailing slash).  Transcript identity
    must follow the client that would make the request, even during network-free
    replay.
    """

    client = getattr(judge, "client", None)
    return getattr(client, "base_url", None) or getattr(
        getattr(judge, "config", None), "base_url", None
    )


def _render_output(
    payload: dict[str, object], *, output_path: Path | None, summary: bool
) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if not summary:
        print(rendered)


def _source_completion_preflight(
    *,
    task: dict[str, Any],
    benchmark: dict[str, Any],
    sampling_config: dict[str, Any],
    payloads: list[dict[str, Any]],
) -> dict[str, object]:
    """Require one immutable, complete original-generation coordinate grid."""

    blockers: list[str] = []
    status = str(task.get("status") or "")
    if status.lower() != "completed":
        blockers.append(f"source_status:{status or 'empty'}!=Completed")
    if bool(task.get("is_tmp")) or bool(task.get("is_param_search")):
        blockers.append("source_task_is_auxiliary")
    if parse_task_desc(task.get("desc")).get("replay_source_task_id"):
        blockers.append("source_task_is_replay_chain")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", str(task.get("git_hash") or "")):
        blockers.append("source_git_hash_missing_or_invalid")

    try:
        expected_rows = int(sampling_config.get("effective_sample_count") or 0)
    except (TypeError, ValueError):
        expected_rows = 0
    if expected_rows <= 0:
        blockers.append("source_effective_sample_count_missing_or_invalid")
    if len(payloads) != expected_rows:
        blockers.append(f"source_rows:{len(payloads)}!=expected:{expected_rows}")

    coordinate_payloads: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for index, payload in enumerate(payloads):
        try:
            coordinate = (
                int(payload["sample_index"]),
                int(payload["repeat_index"]),
                int(payload.get("pass_index", 0)),
            )
        except (KeyError, TypeError, ValueError):
            blockers.append(f"source_coordinate_invalid:{index}")
            continue
        if min(coordinate) < 0:
            blockers.append(f"source_coordinate_negative:{index}")
        coordinate_payloads.append((coordinate, payload))
    coordinates = [coordinate for coordinate, _payload in coordinate_payloads]
    distinct_coordinates = set(coordinates)
    if len(distinct_coordinates) != len(coordinates):
        blockers.append(
            f"source_duplicate_coordinates:{len(coordinates) - len(distinct_coordinates)}"
        )

    try:
        benchmark_samples = int(benchmark.get("num_samples") or 0)
    except (TypeError, ValueError):
        benchmark_samples = 0
    try:
        avg_k = int(float(sampling_config.get("avg_k") or 0))
    except (TypeError, ValueError):
        avg_k = 0
    sample_indices = {coordinate[0] for coordinate in distinct_coordinates}
    repeat_indices = {coordinate[1] for coordinate in distinct_coordinates}
    pass_indices = {coordinate[2] for coordinate in distinct_coordinates}
    if benchmark_samples <= 0:
        blockers.append("source_benchmark_num_samples_missing_or_invalid")
    elif sample_indices != set(range(benchmark_samples)):
        blockers.append("source_sample_index_domain_mismatch")
    if avg_k <= 0:
        blockers.append("source_avg_k_missing_or_invalid")
    elif repeat_indices != set(range(avg_k)):
        blockers.append("source_repeat_index_domain_mismatch")
    if pass_indices != {0}:
        blockers.append(f"source_pass_index_domain:{sorted(pass_indices)}!=expected:[0]")

    fingerprint_payload = [
        {
            "sample_index": coordinate[0],
            "repeat_index": coordinate[1],
            "pass_index": coordinate[2],
            "prompt": str(payload.get("prompt") or ""),
            "completion": str(payload.get("completion") or ""),
            "prompt2": str(payload.get("prompt2") or ""),
            "completion2": str(payload.get("completion2") or ""),
            "ref_answer": str(payload.get("ref_answer") or ""),
            "context": payload.get("context") if isinstance(payload.get("context"), dict) else {},
        }
        for coordinate, payload in sorted(
            coordinate_payloads, key=lambda item: item[0]
        )
    ]
    prompt_fingerprint_payload = [
        {
            "sample_index": coordinate[0],
            "repeat_index": coordinate[1],
            "pass_index": coordinate[2],
            "prompt": str(payload.get("prompt") or ""),
            "prompt2": str(payload.get("prompt2") or ""),
            "strategy_a_prompt": str(payload.get("strategy_a_prompt") or ""),
        }
        for coordinate, payload in sorted(
            coordinate_payloads, key=lambda item: item[0]
        )
    ]
    return {
        "passed": not blockers,
        "blockers": blockers,
        "expected_rows": expected_rows,
        "rows": len(payloads),
        "distinct_coordinates": len(distinct_coordinates),
        "distinct_sample_indices": len(sample_indices),
        "distinct_repeat_indices": len(repeat_indices),
        "pass_indices": sorted(pass_indices),
        "ordered_payload_sha256": canonical_json_sha256(fingerprint_payload),
        "ordered_prompt_sha256": canonical_json_sha256(
            prompt_fingerprint_payload
        ),
    }


def _source_strategy_preflight(
    service: EvalDbService,
    *,
    source_task_id: int,
    model_name: str,
    benchmark: dict[str, Any],
    source_payloads: list[dict[str, Any]],
    stored_metrics: dict[str, Any],
) -> dict[str, object]:
    """Prove and fingerprint the source task's persisted A/B/C grids."""

    blockers: list[str] = []
    raw_task_ids = stored_metrics.get("strategy_task_ids")
    task_ids: dict[str, int] = {}
    if not isinstance(raw_task_ids, dict):
        blockers.append("source_strategy_task_ids_missing")
    else:
        for group in STRATEGY_GROUPS:
            try:
                task_id = int(raw_task_ids[group])
            except (KeyError, TypeError, ValueError):
                blockers.append(f"source_{group}_task_id_missing_or_invalid")
                continue
            if task_id <= 0:
                blockers.append(f"source_{group}_task_id_missing_or_invalid")
                continue
            task_ids[group] = task_id
        if set(raw_task_ids) != set(STRATEGY_GROUPS):
            blockers.append("source_strategy_task_id_keys_mismatch")
    if task_ids.get(STRATEGY_C) != int(source_task_id):
        blockers.append("source_strategy_c_is_not_root_task")
    if len(set(task_ids.values())) != len(task_ids):
        blockers.append("source_strategy_task_ids_not_distinct")

    expected_coordinates = {
        (
            int(payload["sample_index"]),
            int(payload["repeat_index"]),
            int(payload.get("pass_index", 0)),
        )
        for payload in source_payloads
    }
    source_payload_sha = str(
        _source_completion_preflight(
            task={
                "status": "Completed",
                "is_tmp": False,
                "is_param_search": False,
                "desc": "original_source=true",
                "git_hash": "0" * 40,
            },
            benchmark=benchmark,
            sampling_config={
                "effective_sample_count": len(source_payloads),
                "avg_k": len(
                    {int(payload["repeat_index"]) for payload in source_payloads}
                ),
            },
            payloads=source_payloads,
        )["ordered_payload_sha256"]
    )
    grid_sha256_by_group: dict[str, str] = {}
    reference_sha256_by_group: dict[str, str] = {}
    for group in STRATEGY_GROUPS:
        strategy_task_id = task_ids.get(group)
        if strategy_task_id is None:
            continue
        bundle = service.get_task_bundle(task_id=str(strategy_task_id))
        if not bundle or not bundle.get("task"):
            blockers.append(f"source_{group}_task_bundle_missing")
            continue
        task = bundle["task"]
        group_model = bundle.get("model") or {}
        group_benchmark = bundle.get("benchmark") or {}
        if str(task.get("status") or "").lower() != "completed":
            blockers.append(f"source_{group}_status_not_completed")
        if str(group_model.get("model_name") or "") != model_name:
            blockers.append(f"source_{group}_model_mismatch")
        if (
            str(group_benchmark.get("benchmark_name") or "")
            != str(benchmark.get("benchmark_name") or "")
            or str(group_benchmark.get("benchmark_split") or "")
            != str(benchmark.get("benchmark_split") or "")
        ):
            blockers.append(f"source_{group}_benchmark_mismatch")

        group_payloads = service.list_completion_payloads(
            task_id=str(strategy_task_id), status="Completed"
        )
        group_coordinates = service.list_completion_keys(
            task_id=str(strategy_task_id), status="Completed"
        )
        if group_coordinates != expected_coordinates:
            blockers.append(f"source_{group}_completion_coordinate_grid_mismatch")
        group_payload_fingerprint = [
            {
                "sample_index": int(payload["sample_index"]),
                "repeat_index": int(payload["repeat_index"]),
                "pass_index": int(payload.get("pass_index", 0)),
                "prompt": str(payload.get("prompt") or ""),
                "completion": str(payload.get("completion") or ""),
                "prompt2": str(payload.get("prompt2") or ""),
                "completion2": str(payload.get("completion2") or ""),
                "ref_answer": str(payload.get("ref_answer") or ""),
                "context": (
                    payload.get("context")
                    if isinstance(payload.get("context"), dict)
                    else {}
                ),
            }
            for payload in sorted(
                group_payloads,
                key=lambda item: (
                    int(item["sample_index"]),
                    int(item["repeat_index"]),
                    int(item.get("pass_index", 0)),
                ),
            )
        ]
        if canonical_json_sha256(group_payload_fingerprint) != source_payload_sha:
            blockers.append(f"source_{group}_completion_payload_mismatch")

        eval_rows = service.list_eval_rows(task_id=str(strategy_task_id))
        if len(eval_rows) != len(expected_coordinates):
            blockers.append(
                f"source_{group}_eval_rows:{len(eval_rows)}"
                f"!=expected:{len(expected_coordinates)}"
            )
        normalized_eval = [
            {
                "answer": str(row.get("answer") or ""),
                "ref_answer": str(row.get("ref_answer") or ""),
                "is_passed": bool(row.get("is_passed", False)),
                "fail_reason": str(row.get("fail_reason") or ""),
            }
            for row in eval_rows
        ]
        grid_sha256_by_group[group] = canonical_json_sha256(normalized_eval)
        reference_sha256_by_group[group] = canonical_json_sha256(
            [row["ref_answer"] for row in normalized_eval]
        )

    if len(set(reference_sha256_by_group.values())) > 1:
        blockers.append("source_strategy_reference_grid_mismatch")
    return {
        "passed": not blockers,
        "blockers": blockers,
        "task_ids": task_ids,
        "expected_rows": len(expected_coordinates),
        "completion_payload_sha256": source_payload_sha,
        "eval_grid_sha256_by_group": grid_sha256_by_group,
        "reference_sha256_by_group": reference_sha256_by_group,
    }


def _existing_replays(
    connection: psycopg.Connection[Any],
    *,
    task_desc: str,
    model_name: str,
    benchmark_name: str,
    benchmark_split: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        EXISTING_REPLAY_QUERY,
        (task_desc, model_name, benchmark_name, benchmark_split),
    ).fetchall()
    return [dict(row) for row in rows]


def _classify_existing_replays(
    rows: list[dict[str, Any]],
    *,
    expected_rows: int,
    expected_git_hash: str,
    now: datetime | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Return valid, pending, or retry without mutating an earlier attempt."""

    valid = [
        row
        for row in rows
        if str(row.get("status") or "").lower() == "completed"
        and row.get("score_id") is not None
        and int(row.get("completion_count") or 0) == expected_rows
        and int(row.get("eval_count") or 0) == expected_rows
        and str(row.get("task_git_hash") or "").lower() == expected_git_hash
    ]
    if valid:
        return "valid", max(valid, key=lambda row: int(row["task_id"]))

    observed_at = now or datetime.now()
    pending: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "").lower()
        created_at = row.get("task_created_at")
        if status == "completed" and row.get("score_id") is None:
            pass
        elif status != "running":
            continue
        if not isinstance(created_at, datetime):
            continue
        comparable_now = observed_at
        if comparable_now.tzinfo is None and created_at.tzinfo is not None:
            comparable_now = comparable_now.replace(tzinfo=created_at.tzinfo)
        elif comparable_now.tzinfo is not None and created_at.tzinfo is None:
            comparable_now = comparable_now.replace(tzinfo=None)
        if comparable_now - created_at <= REPLAY_SCORE_PERSISTENCE_GRACE:
            pending.append(row)
    if pending:
        return "pending", max(pending, key=lambda row: int(row["task_id"]))
    return "retry", None


def _validate_created_replay_task(
    service: EvalDbService,
    *,
    replay_task_id: int,
    expected_desc: str,
    expected_git_hash: str,
) -> None:
    bundle = service.get_task_bundle(task_id=str(replay_task_id))
    task = bundle.get("task") if isinstance(bundle, dict) else None
    if not isinstance(task, dict):
        raise RuntimeError(f"replay task {replay_task_id}: task bundle missing")
    actual_desc = str(task.get("desc") or "")
    actual_git_hash = str(task.get("git_hash") or "").lower()
    if actual_desc != expected_desc:
        raise RuntimeError(
            f"replay task {replay_task_id}: desc drift after create"
        )
    if actual_git_hash != expected_git_hash:
        raise RuntimeError(
            f"replay task {replay_task_id}: git_hash {actual_git_hash!r}"
            f" != expected {expected_git_hash!r}"
        )


def _numeric_metric_keys(metrics: object) -> list[str]:
    if not isinstance(metrics, dict):
        return []
    return sorted(
        key
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and (key.startswith("avg@") or key.startswith("pass@") or key == "exact_accuracy")
    )


def _replayed_metrics(
    rows: list[tuple[int, int, bool]],
    stored_metrics: dict[str, Any],
    *,
    exact_accuracy: float,
    sampling_config: dict[str, Any] | None = None,
) -> dict[str, float]:
    result: dict[str, float] = {"exact_accuracy": exact_accuracy}
    pass_ks, avg_ks = _metric_ks(stored_metrics, sampling_config)
    result.update(compute_avg_at_k(rows, avg_ks))
    result.update(compute_pass_at_k(rows, pass_ks))
    return result


def _metric_ks(
    metrics: dict[str, Any],
    sampling_config: dict[str, Any] | None = None,
) -> tuple[tuple[int, ...], tuple[float | int, ...]]:
    """Recover the score protocol from score metrics or task metadata.

    Failed source tasks can have complete stored completions but no score row.
    Their task sampling configuration remains authoritative for ``avg@k`` and
    ``pass@k``.  Prefer explicit stored metric keys when present, then fall
    back only for a missing family so an append-only replay preserves the
    original task protocol instead of degrading to bare ``exact_accuracy``.
    """

    pass_ks: list[int] = []
    avg_ks: list[float | int] = []
    for key in _numeric_metric_keys(metrics):
        if key.startswith("pass@"):
            try:
                pass_ks.append(int(key.removeprefix("pass@")))
            except ValueError:
                continue
        elif key.startswith("avg@"):
            try:
                number = float(key.removeprefix("avg@"))
            except ValueError:
                continue
            avg_ks.append(int(number) if number.is_integer() else number)

    config = sampling_config if isinstance(sampling_config, dict) else {}
    if not pass_ks:
        configured_pass_ks = config.get("pass_ks")
        if not isinstance(configured_pass_ks, (list, tuple, set)):
            configured_pass_ks = (
                [configured_pass_ks] if configured_pass_ks is not None else []
            )
        for value in configured_pass_ks:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                pass_ks.append(number)
    if not avg_ks:
        configured_avg_ks = config.get("avg_k")
        if not isinstance(configured_avg_ks, (list, tuple, set)):
            configured_avg_ks = (
                [configured_avg_ks] if configured_avg_ks is not None else []
            )
        for value in configured_avg_ks:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number <= 0:
                continue
            avg_ks.append(int(number) if number.is_integer() else number)
    return tuple(sorted(set(pass_ks))), tuple(sorted(set(avg_ks)))


def _primary_score(metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    for prefix in ("avg@", "pass@"):
        keys = sorted(key for key in metrics if key.startswith(prefix))
        if keys:
            value = metrics.get(keys[-1])
            return keys[-1], float(value) if isinstance(value, (int, float)) else None
    value = metrics.get("exact_accuracy")
    return "exact_accuracy", float(value) if isinstance(value, (int, float)) else None


def _evaluation_preflight(
    evaluation: FreeResponseEvaluation,
    *,
    expected_rows: int,
) -> dict[str, object]:
    """Prove that every strategy row is present and no timeout is hidden.

    This gate is intentionally evaluated before any replay task is created.
    A verifier timeout is not an incorrect answer: after the isolated 15-second
    retry it remains an unresolved evaluation error and therefore blocks the
    complete source replay.
    """

    blockers: list[str] = []
    groups: dict[str, dict[str, object]] = {}
    if evaluation.samples != expected_rows:
        blockers.append(
            f"samples:{evaluation.samples}!=expected_rows:{expected_rows}"
        )

    expected_coordinates: set[tuple[int, int]] | None = None
    for group in STRATEGY_GROUPS:
        rows = evaluation.rows_by_group.get(group)
        payloads = evaluation.payloads_by_group.get(group)
        retry_stats = evaluation.math_verify_retry_stats_by_group.get(group)
        row_count = len(rows) if isinstance(rows, list) else 0
        payload_count = len(payloads) if isinstance(payloads, list) else 0
        coordinates = (
            {(int(row[0]), int(row[1])) for row in rows}
            if isinstance(rows, list)
            else set()
        )
        duplicate_coordinates = max(0, row_count - len(coordinates))
        attempted = (
            int(retry_stats.get("attempted_count") or 0)
            if isinstance(retry_stats, dict)
            else -1
        )
        resolved = (
            int(retry_stats.get("resolved_count") or 0)
            if isinstance(retry_stats, dict)
            else -1
        )
        unresolved = (
            int(retry_stats.get("unresolved_count") or 0)
            if isinstance(retry_stats, dict)
            else -1
        )
        retry_rows = (
            retry_stats.get("rows") if isinstance(retry_stats, dict) else None
        )
        retry_row_count = len(retry_rows) if isinstance(retry_rows, list) else -1

        if row_count != expected_rows:
            blockers.append(f"{group}.rows:{row_count}!=expected:{expected_rows}")
        if payload_count != expected_rows:
            blockers.append(
                f"{group}.payloads:{payload_count}!=expected:{expected_rows}"
            )
        if duplicate_coordinates:
            blockers.append(
                f"{group}.duplicate_coordinates:{duplicate_coordinates}"
            )
        if expected_coordinates is None:
            expected_coordinates = coordinates
        elif coordinates != expected_coordinates:
            blockers.append(f"{group}.coordinate_set_mismatch")
        if not isinstance(retry_stats, dict):
            blockers.append(f"{group}.missing_math_verify_retry_stats")
        else:
            if attempted != resolved + unresolved:
                blockers.append(
                    f"{group}.retry_counts:{attempted}!={resolved}+{unresolved}"
                )
            if retry_row_count != attempted:
                blockers.append(
                    f"{group}.retry_rows:{retry_row_count}!=attempted:{attempted}"
                )
            if unresolved:
                blockers.append(f"{group}.unresolved_math_verify_timeouts:{unresolved}")

        groups[group] = {
            "rows": row_count,
            "payloads": payload_count,
            "distinct_coordinates": len(coordinates),
            "duplicate_coordinates": duplicate_coordinates,
            "math_verify_retry": retry_stats or {},
        }

    if len(evaluation.payloads) != expected_rows:
        blockers.append(
            f"primary_payloads:{len(evaluation.payloads)}!=expected:{expected_rows}"
        )
    return {
        "expected_rows": expected_rows,
        "samples": evaluation.samples,
        "groups": groups,
        "blockers": blockers,
        "passed": not blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="+", type=int)
    parser.add_argument("--dbname", default=DEFAULT_DB_NAME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--reason-tag",
        default="",
        help=(
            "Stable final provenance tag. It must equal the tag derived from "
            "the accepted extractor lineage and comparator implementation SHA."
        ),
    )
    parser.add_argument(
        "--final-extractor-lineage-sha256",
        default=os.environ.get(
            "RWKV_FINAL_MATH_EXTRACTOR_LINEAGE_SHA256",
            ACCEPTED_EXTRACTOR_LINEAGE_SHA256,
        ),
    )
    parser.add_argument(
        "--final-imported-free-response-sha256",
        default=os.environ.get(
            "RWKV_FINAL_MATH_IMPORTED_FREE_RESPONSE_SHA256",
            FINAL_IMPORTED_FREE_RESPONSE_SHA256,
        ),
    )
    parser.add_argument(
        "--final-comparator-sha256",
        default=os.environ.get(
            "RWKV_FINAL_MATH_COMPARATOR_SHA256",
            FINAL_COMPARATOR_IMPLEMENTATION_SHA256,
        ),
    )
    parser.add_argument(
        "--final-math-verify-version",
        default=os.environ.get(
            "RWKV_FINAL_MATH_VERIFY_VERSION",
            FINAL_MATH_VERIFY_VERSION,
        ),
    )
    parser.add_argument(
        "--final-git-hash",
        default=os.environ.get("RWKV_FINAL_MATH_REPLAY_GIT_HASH", FINAL_REPLAY_GIT_HASH),
    )
    parser.add_argument(
        "--determinism-attestation",
        type=Path,
        help=(
            "Cross-PYTHONHASHSEED attestation covering exactly these source task "
            "IDs and this frozen runtime. Required for preflight and commit."
        ),
    )
    transcript_group = parser.add_mutually_exclusive_group()
    transcript_group.add_argument(
        "--record-judge-transcript",
        type=Path,
        help=(
            "Read-only bootstrap: call the temperature-zero external Judge "
            "once and publish an immutable transcript. Requires "
            "PYTHONHASHSEED=42 and --judge-mode=llm."
        ),
    )
    transcript_group.add_argument(
        "--judge-transcript",
        type=Path,
        help=(
            "Replay a previously frozen Judge transcript without network "
            "calls. Required for every LLM-judge attestation/final replay."
        ),
    )
    parser.add_argument(
        "--judge-mode",
        choices=("auto", "exact", "llm"),
        default="auto",
        help="auto reuses LLM judge for source evaluators whose name contains 'judge'",
    )
    parser.add_argument(
        "--judge-max-workers",
        type=int,
        default=None,
        help=(
            "Explicit LLM-judge concurrency for replay. When omitted, the normal "
            "JUDGE_MAX_WORKERS resolution is used."
        ),
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help=(
            "Create a fresh, non-temporary replay task through EvalDbService, "
            "copy the original completions, ingest the recomputed eval rows, "
            "and record the score. The source task remains unchanged."
        ),
    )
    parser.add_argument(
        "--advisory-lock-held-by-caller",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--emit-determinism-run",
        action="store_true",
        help=(
            "Read-only bootstrap mode for the multi-PYTHONHASHSEED attestation "
            "builder. It validates the frozen runtime and emits per-task result "
            "digests, but cannot commit and does not waive the final attestation."
        ),
    )
    args = parser.parse_args()
    if args.emit_determinism_run and args.commit:
        parser.error("--emit-determinism-run cannot be combined with --commit")
    if args.record_judge_transcript and (
        args.commit or args.emit_determinism_run
    ):
        parser.error(
            "--record-judge-transcript is a read-only bootstrap mode and "
            "cannot be combined with --commit/--emit-determinism-run"
        )
    if args.record_judge_transcript and args.judge_mode != "llm":
        parser.error("--record-judge-transcript requires --judge-mode=llm")
    if args.judge_transcript and args.judge_mode == "exact":
        parser.error("--judge-transcript cannot be used with --judge-mode=exact")
    if args.commit and len(set(args.task_ids)) != 1:
        parser.error("--commit requires exactly one source task per invocation")
    if args.reason_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.reason_tag):
        parser.error("--reason-tag may contain only letters, digits, dot, underscore, and dash")

    contract = FinalMathReplayContract.from_values(
        extractor_lineage_sha256=args.final_extractor_lineage_sha256,
        imported_free_response_sha256=args.final_imported_free_response_sha256,
        comparator_implementation_sha256=args.final_comparator_sha256,
        math_verify_version=args.final_math_verify_version,
        replay_git_hash=args.final_git_hash,
        reason_tag=args.reason_tag,
    )
    runtime = collect_runtime_math_provenance()
    runtime_reasons = runtime_contract_reasons(runtime, contract)
    # The attestation deliberately exercises several interpreter hash seeds,
    # but the durable replay must be minted by one canonical seed.  Otherwise
    # identical attested outputs can still produce multiple task descriptions
    # and defeat idempotent resume after a crash.
    if (
        not args.emit_determinism_run
        and runtime.pythonhashseed != FINAL_REPLAY_PYTHONHASHSEED
    ):
        runtime_reasons.append(
            "final_replay_pythonhashseed:"
            f"{runtime.pythonhashseed or 'empty'}"
            f"!=expected:{FINAL_REPLAY_PYTHONHASHSEED}"
        )
    if args.emit_determinism_run or args.record_judge_transcript:
        attestation = None
        attestation_reasons: list[str] = []
    else:
        attestation, attestation_reasons = load_determinism_attestation(
            args.determinism_attestation,
            contract=contract,
            runtime=runtime,
            source_task_ids=sorted(set(args.task_ids)),
        )
    provenance_reasons = list(dict.fromkeys(runtime_reasons + attestation_reasons))
    if provenance_reasons or (
        attestation is None
        and not args.emit_determinism_run
        and not args.record_judge_transcript
    ):
        output = {
            "database": args.dbname,
            "adapter": "current_global_free_response_adapter",
            "primary_group": STRATEGY_C,
            "read_only": True,
            "commit_requested": bool(args.commit),
            "committed": False,
            "blocked": True,
            "provenance_preflight": {
                "passed": False,
                "blockers": provenance_reasons,
                "contract": contract.as_dict(),
                "runtime": runtime.as_dict(),
                "determinism_attestation": (
                    attestation.as_dict() if attestation is not None else None
                ),
            },
            "tasks": [],
        }
        _render_output(output, output_path=args.output, summary=args.summary)
        if args.summary:
            print(
                json.dumps(
                    {
                        "database": args.dbname,
                        "tasks": 0,
                        "replayable": 0,
                        "blocked": 1,
                        "provenance_blockers": provenance_reasons,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
        return 2

    judge_transcript_artifact: JudgeTranscriptArtifact | None = None
    judge_transcript_recorder: JudgeTranscriptRecorder | None = None
    judge_transcript_replayer: JudgeTranscriptReplayer | None = None
    try:
        if args.record_judge_transcript:
            judge_transcript_recorder = JudgeTranscriptRecorder(
                args.record_judge_transcript
            )
        elif args.judge_transcript:
            judge_transcript_artifact = load_judge_transcript(
                args.judge_transcript
            )
            judge_transcript_replayer = JudgeTranscriptReplayer(
                judge_transcript_artifact
            )
    except JudgeTranscriptError as exc:
        output = {
            "database": args.dbname,
            "adapter": "current_global_free_response_adapter",
            "read_only": True,
            "commit_requested": bool(args.commit),
            "committed": False,
            "blocked": True,
            "provenance_preflight": {
                "passed": False,
                "blockers": [
                    f"judge_transcript_load_failed:{type(exc).__name__}:{exc}"
                ],
                "contract": contract.as_dict(),
                "runtime": runtime.as_dict(),
            },
            "tasks": [],
        }
        _render_output(output, output_path=args.output, summary=args.summary)
        return 2

    # Load service credentials only after recording the interpreter-start
    # PYTHONHASHSEED.  Loading a seed from .env here could not change Python's
    # already-initialized hash secret and would create a false attestation.
    load_env_file(Path(".env"))
    db_config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    conninfo = _build_conninfo(db_config)
    init_db(db_config)
    service = EvalDbService()
    report: list[dict[str, object]] = []
    blocked = False
    for task_id in sorted(set(args.task_ids)):
        bundle = service.get_task_bundle(task_id=str(task_id))
        if not bundle or not bundle.get("model") or not bundle.get("benchmark"):
            report.append({"task_id": task_id, "replayable": False, "reason": "task_bundle_missing"})
            blocked = True
            continue
        task = bundle["task"]
        model = bundle["model"]
        benchmark = bundle["benchmark"]
        sampling_config = _json_object(task.get("sampling_config"))
        payloads = service.list_completion_payloads(task_id=str(task_id), status="Completed")
        if not payloads:
            report.append({"task_id": task_id, "replayable": False, "reason": "no_completions"})
            blocked = True
            continue

        source_preflight = _source_completion_preflight(
            task=task,
            benchmark=benchmark,
            sampling_config=sampling_config,
            payloads=payloads,
        )
        if not bool(source_preflight["passed"]):
            report.append(
                {
                    "task_id": task_id,
                    "replayable": False,
                    "blocked": True,
                    "reason": "source_completion_preflight_failed",
                    "blocking_reasons": source_preflight["blockers"],
                    "source_preflight": source_preflight,
                    "provenance_preflight": {
                        "passed": True,
                        "contract": contract.as_dict(),
                        "runtime": runtime.as_dict(),
                        "determinism_attestation": (
                            attestation.as_dict() if attestation is not None else None
                        ),
                    },
                    "model": model["model_name"],
                    "benchmark": (
                        f"{benchmark['benchmark_name']}__"
                        f"{benchmark.get('benchmark_split') or ''}"
                    ),
                    "rows": len(payloads),
                }
            )
            blocked = True
            continue

        stored_score = service.get_score_payload(task_id=str(task_id)) or {}
        stored_metrics = _json_object(stored_score.get("metrics"))
        source_strategy_preflight = _source_strategy_preflight(
            service,
            source_task_id=task_id,
            model_name=str(model["model_name"]),
            benchmark=benchmark,
            source_payloads=payloads,
            stored_metrics=stored_metrics,
        )
        if not bool(source_strategy_preflight["passed"]):
            report.append(
                {
                    "task_id": task_id,
                    "replayable": False,
                    "blocked": True,
                    "reason": "source_strategy_preflight_failed",
                    "blocking_reasons": source_strategy_preflight["blockers"],
                    "source_preflight": source_preflight,
                    "source_strategy_preflight": source_strategy_preflight,
                    "model": model["model_name"],
                    "benchmark": (
                        f"{benchmark['benchmark_name']}__"
                        f"{benchmark.get('benchmark_split') or ''}"
                    ),
                    "rows": len(payloads),
                }
            )
            blocked = True
            continue

        slug = make_dataset_slug(
            str(benchmark["benchmark_name"]),
            str(benchmark.get("benchmark_split") or ""),
        )
        dataset_path = resolve_or_prepare_dataset(slug, verbose=False)
        evaluator = str(task.get("evaluator") or "")
        source_evidence = {
            "schema_version": "g1i-math-source-evidence.v1",
            "source_task_id": task_id,
            "source_git_hash": str(task.get("git_hash") or "").lower(),
            "source_evaluator": evaluator,
            "model_name": str(model["model_name"]),
            "dataset_slug": slug,
            "dataset_file_sha256": sha256_file(Path(dataset_path)),
            "ordered_completion_payload_sha256": source_preflight[
                "ordered_payload_sha256"
            ],
            "ordered_prompt_sha256": source_preflight[
                "ordered_prompt_sha256"
            ],
            "sampling_config_sha256": canonical_json_sha256(sampling_config),
            "source_strategy_task_ids": source_strategy_preflight["task_ids"],
            "source_strategy_eval_grid_sha256": source_strategy_preflight[
                "eval_grid_sha256_by_group"
            ],
            "source_strategy_reference_sha256": source_strategy_preflight[
                "reference_sha256_by_group"
            ],
        }
        source_evidence_sha256 = canonical_json_sha256(source_evidence)
        expected_judge_mode = (
            "llm" if "judge" in evaluator.lower() else "exact"
        )
        judge_mode = args.judge_mode
        if judge_mode == "auto":
            judge_mode = expected_judge_mode
        elif judge_mode != expected_judge_mode:
            report.append(
                {
                    "task_id": task_id,
                    "replayable": False,
                    "blocked": True,
                    "reason": "judge_mode_source_evaluator_mismatch",
                    "blocking_reasons": [
                        f"judge_mode:{judge_mode}"
                        f"!=source_evaluator_mode:{expected_judge_mode}"
                    ],
                    "judge_mode": judge_mode,
                    "source_evaluator": evaluator,
                    "model": model["model_name"],
                    "benchmark": slug,
                    "rows": len(payloads),
                }
            )
            blocked = True
            continue
        judge = None
        if judge_mode == "llm":
            live_judge = build_llm_judge(
                judge_model=(
                    str(sampling_config.get("judger_model_name"))
                    if sampling_config.get("judger_model_name")
                    else None
                ),
                judge_max_workers=args.judge_max_workers,
                required=True,
            )
            root_config = resolve_benchmark_model_config(
                slug,
                str(model["model_name"]),
                stage=None,
            )
            if root_config is not None and root_config.judge_prompt_template:
                live_judge.config.prompt_template = root_config.judge_prompt_template
            scope = f"task:{task_id}"
            endpoint_url = _judge_endpoint_url(live_judge)
            if judge_transcript_recorder is not None:
                judge = judge_transcript_recorder.wrap(
                    live_judge,
                    scope=scope,
                    endpoint_url=endpoint_url,
                )
            elif judge_transcript_replayer is not None:
                judge = judge_transcript_replayer.wrap(
                    live_judge.config,
                    scope=scope,
                    endpoint_url=endpoint_url,
                )
            else:
                report.append(
                    {
                        "task_id": task_id,
                        "replayable": False,
                        "blocked": True,
                        "reason": "judge_transcript_required",
                        "blocking_reasons": [
                            "llm_judge_requires_record_or_replay_transcript"
                        ],
                        "judge_mode": judge_mode,
                        "model": model["model_name"],
                        "benchmark": slug,
                        "rows": len(payloads),
                    }
                )
                blocked = True
                continue
        elif judge_transcript_artifact is not None:
            report.append(
                {
                    "task_id": task_id,
                    "replayable": False,
                    "blocked": True,
                    "reason": "unexpected_judge_transcript_for_exact_replay",
                    "judge_mode": judge_mode,
                    "model": model["model_name"],
                    "benchmark": slug,
                    "rows": len(payloads),
                }
            )
            blocked = True
            continue
        try:
            evaluation = evaluate_free_response(
                payloads,
                dataset_path=str(dataset_path),
                judge=judge,
                primary_group=STRATEGY_C,
                math_verify_retry_timeout_s=MATH_VERIFY_RETRY_TIMEOUT_S,
            )
        except JudgeTranscriptError as exc:
            report.append(
                {
                    "task_id": task_id,
                    "replayable": False,
                    "blocked": True,
                    "reason": "judge_transcript_replay_failed",
                    "blocking_reasons": [f"{type(exc).__name__}:{exc}"],
                    "judge_mode": judge_mode,
                    "model": model["model_name"],
                    "benchmark": slug,
                    "rows": len(payloads),
                }
            )
            blocked = True
            continue
        judge_errors = {
            group: stats
            for group, stats in evaluation.judge_stats_by_group.items()
            if int(stats.get("invalid_output_count") or 0)
            or int(stats.get("request_error_count") or 0)
        }
        preflight = _evaluation_preflight(evaluation, expected_rows=len(payloads))
        result_sha256 = evaluation_result_sha256(
            rows_by_group=evaluation.rows_by_group,
            payloads_by_group=evaluation.payloads_by_group,
        )
        attested_result_sha256 = (
            attestation.task_result_sha256.get(str(task_id), "")
            if attestation is not None
            else ""
        )
        attested_source_evidence_sha256 = (
            attestation.source_evidence_sha256_by_task.get(str(task_id), "")
            if attestation is not None
            else ""
        )
        judge_transcript_sha256 = (
            judge_transcript_artifact.sha256
            if judge_transcript_artifact is not None and judge_mode == "llm"
            else ""
        )
        attested_judge_transcript_sha256 = (
            attestation.judge_transcript_sha256_by_task.get(str(task_id), "")
            if attestation is not None
            else ""
        )
        if attestation is not None and result_sha256 != attested_result_sha256:
            preflight["blockers"].append(
                "determinism_attestation_result_sha256:"
                f"{attested_result_sha256 or 'missing'}!=actual:{result_sha256}"
            )
            preflight["passed"] = False
        if attestation is not None:
            if judge_mode == "llm":
                if not judge_transcript_sha256:
                    preflight["blockers"].append(
                        "judge_transcript_missing_for_llm_replay"
                    )
                    preflight["passed"] = False
                elif (
                    judge_transcript_sha256
                    != attested_judge_transcript_sha256
                ):
                    preflight["blockers"].append(
                        "determinism_attestation_judge_transcript_sha256:"
                        f"{attested_judge_transcript_sha256 or 'missing'}"
                        f"!=actual:{judge_transcript_sha256}"
                    )
                    preflight["passed"] = False
            elif attested_judge_transcript_sha256:
                preflight["blockers"].append(
                    "unexpected_attested_judge_transcript_for_exact_replay"
                )
                preflight["passed"] = False
        if (
            attestation is not None
            and source_evidence_sha256 != attested_source_evidence_sha256
        ):
            preflight["blockers"].append(
                "determinism_attestation_source_evidence_sha256:"
                f"{attested_source_evidence_sha256 or 'missing'}"
                f"!=actual:{source_evidence_sha256}"
            )
            preflight["passed"] = False
        preflight["source"] = source_preflight
        preflight["source_strategy"] = source_strategy_preflight
        preflight["provenance"] = {
            "contract": contract.as_dict(),
            "runtime": runtime.as_dict(),
            "determinism_attestation": (
                attestation.as_dict() if attestation is not None else None
            ),
            "evaluation_result_sha256": result_sha256,
            "attested_result_sha256": attested_result_sha256,
            "source_evidence": source_evidence,
            "source_evidence_sha256": source_evidence_sha256,
            "attested_source_evidence_sha256": (
                attested_source_evidence_sha256
            ),
            "judge_transcript_sha256": judge_transcript_sha256 or None,
            "attested_judge_transcript_sha256": (
                attested_judge_transcript_sha256 or None
            ),
        }
        if judge_errors or not bool(preflight["passed"]):
            reasons = list(preflight["blockers"])
            if judge_errors:
                reasons.append("judge_request_or_format_errors")
            report.append(
                {
                    "task_id": task_id,
                    "replayable": False,
                    "blocked": True,
                    "reason": "evaluation_preflight_failed",
                    "blocking_reasons": reasons,
                    "judge_mode": judge_mode,
                    "judge_stats_by_group": evaluation.judge_stats_by_group,
                    "judge_errors": judge_errors,
                    "math_verify_retry_timeout_s": MATH_VERIFY_RETRY_TIMEOUT_S,
                    "evaluation_preflight": preflight,
                    "source_preflight": source_preflight,
                    "source_strategy_preflight": source_strategy_preflight,
                    "runtime_provenance": runtime.as_dict(),
                    "determinism_attestation": (
                        attestation.as_dict() if attestation is not None else None
                    ),
                    "model": model["model_name"],
                    "benchmark": slug,
                    "rows": len(payloads),
                }
            )
            blocked = True
            continue
        rows = evaluation.rows_by_group[STRATEGY_C]
        exact_accuracy = float(evaluation.metrics_by_group[STRATEGY_C]["exact_accuracy"])
        replayed_metrics = _replayed_metrics(
            rows,
            stored_metrics,
            exact_accuracy=exact_accuracy,
            sampling_config=sampling_config,
        )
        stored_key, stored_value = _primary_score(stored_metrics)
        replay_key, replay_value = _primary_score(replayed_metrics)
        if stored_key != replay_key:
            replay_value = replayed_metrics.get(stored_key) if stored_key else replay_value
            replay_value = float(replay_value) if isinstance(replay_value, (int, float)) else None

        stored_eval_rows = service.list_eval_answers_for_tasks(task_ids=[task_id])
        stored_answers = [str(row.get("answer") or "") for row in stored_eval_rows]
        replayed_answers = [str(row.get("answer") or "") for row in evaluation.payloads]
        changed_answers = sum(
            old != new
            for old, new in zip(stored_answers, replayed_answers, strict=False)
        )
        contaminated_stored_answers = sum(
            "</think>" in answer or "Therefore, the final answer is" in answer
            for answer in stored_answers
        )

        replay_task_id: int | None = None
        strategy_task_ids: dict[str, int] = {}
        idempotency_action = "dryrun"
        post_commit_validated = False
        replay_task_desc = (
            build_replay_task_desc(
                source_task_id=task_id,
                source_git_hash=str(task.get("git_hash") or ""),
                contract=contract,
                runtime=runtime,
                attestation=attestation,
            )
            if attestation is not None
            else None
        )
        if args.commit:
            if replay_task_desc is None or attestation is None:
                raise RuntimeError(
                    "commit reached without a validated determinism attestation"
                )
            if judge_transcript_replayer is not None:
                # ``--commit`` is intentionally limited to one source task.
                # Consume the full immutable transcript before the first
                # append-only database write; a partial/drifted ledger must
                # never mint a replay task.
                judge_transcript_replayer.assert_consumed()
            lock_keys = replay_advisory_lock_keys(
                dbname=args.dbname,
                source_task_id=task_id,
                model_name=str(model["model_name"]),
                benchmark_name=str(benchmark["benchmark_name"]),
                benchmark_split=str(benchmark.get("benchmark_split") or ""),
            )
            lock_connection_context = (
                nullcontext(None)
                if args.advisory_lock_held_by_caller
                else psycopg.connect(
                    conninfo,
                    row_factory=dict_row,
                    autocommit=True,
                )
            )
            with lock_connection_context as lock_connection:
                lock_context = (
                    nullcontext(True)
                    if args.advisory_lock_held_by_caller
                    else held_replay_advisory_locks(lock_connection, lock_keys)
                )
                with lock_context as acquired:
                    if not acquired:
                        idempotency_action = "advisory_lock_busy"
                        blocked = True
                    else:
                        # Re-read the immutable source under the shared replay
                        # lock. Any drift aborts before a replay task is created.
                        locked_payloads = service.list_completion_payloads(
                            task_id=str(task_id), status="Completed"
                        )
                        locked_source_preflight = _source_completion_preflight(
                            task=task,
                            benchmark=benchmark,
                            sampling_config=sampling_config,
                            payloads=locked_payloads,
                        )
                        if (
                            not bool(locked_source_preflight["passed"])
                            or locked_source_preflight["ordered_payload_sha256"]
                            != source_preflight["ordered_payload_sha256"]
                        ):
                            idempotency_action = "source_drift_under_lock"
                            blocked = True
                        else:
                            with psycopg.connect(
                                conninfo,
                                row_factory=dict_row,
                            ) as state_connection:
                                existing_rows = _existing_replays(
                                    state_connection,
                                    task_desc=replay_task_desc,
                                    model_name=str(model["model_name"]),
                                    benchmark_name=str(benchmark["benchmark_name"]),
                                    benchmark_split=str(
                                        benchmark.get("benchmark_split") or ""
                                    ),
                                )
                            existing_state, existing_row = _classify_existing_replays(
                                existing_rows,
                                expected_rows=len(payloads),
                                expected_git_hash=contract.replay_git_hash,
                            )
                            if existing_state == "valid" and existing_row is not None:
                                replay_task_id = int(existing_row["task_id"])
                                idempotency_action = "reuse_completed_replay"
                                post_commit_validated = True
                            elif existing_state == "pending" and existing_row is not None:
                                replay_task_id = int(existing_row["task_id"])
                                idempotency_action = "wait_for_existing_replay"
                                blocked = True
                            else:
                                idempotency_action = "append_new_replay"
                                old_desc = os.environ.get("RWKV_TASK_DESC")
                                old_tmp = os.environ.get("RWKV_TASK_IS_TMP")
                                os.environ["RWKV_TASK_DESC"] = replay_task_desc
                                os.environ["RWKV_TASK_IS_TMP"] = "0"
                                try:
                                    replay_task_id = int(
                                        service.get_or_create_task(
                                            job_name=str(
                                                task.get("evaluator") or "math_eval"
                                            ),
                                            job_id=None,
                                            dataset=slug,
                                            model=str(model["model_name"]),
                                            is_param_search=False,
                                            sampling_config=sampling_config,
                                            allow_resume=False,
                                        )
                                    )
                                    _validate_created_replay_task(
                                        service,
                                        replay_task_id=replay_task_id,
                                        expected_desc=replay_task_desc,
                                        expected_git_hash=contract.replay_git_hash,
                                    )
                                    inserted = service.insert_completion_payloads_batch(
                                        payloads=payloads,
                                        task_id=str(replay_task_id),
                                    )
                                    if inserted != len(payloads):
                                        raise RuntimeError(
                                            f"replay task {replay_task_id}: inserted "
                                            f"{inserted}/{len(payloads)} completions"
                                        )
                                    strategy_task_ids = (
                                        service.ingest_eval_payload_groups(
                                            task_id=str(replay_task_id),
                                            completion_payloads=payloads,
                                            payloads_by_group=(
                                                evaluation.payloads_by_group
                                            ),
                                            primary_group=evaluation.primary_group,
                                        )
                                    )
                                    if set(strategy_task_ids) != set(STRATEGY_GROUPS):
                                        raise RuntimeError(
                                            f"replay task {replay_task_id}: strategy "
                                            f"tasks {sorted(strategy_task_ids)}"
                                            f" != expected {sorted(STRATEGY_GROUPS)}"
                                        )
                                    pass_ks, avg_ks = _metric_ks(
                                        stored_metrics, sampling_config
                                    )
                                    metrics_payload, _metric_details = (
                                        build_grouped_metrics_payload(
                                            evaluation,
                                            pass_k=pass_ks,
                                            avg_k=avg_ks,
                                            report_pass_k=pass_ks,
                                            report_avg_k=avg_ks,
                                        )
                                    )
                                    attach_strategy_task_ids(
                                        metrics_payload, strategy_task_ids
                                    )
                                    metrics_payload["replay_provenance"] = {
                                        "source_task_id": task_id,
                                        "source_evidence": source_evidence,
                                        "source_evidence_sha256": (
                                            source_evidence_sha256
                                        ),
                                        "contract": contract.as_dict(),
                                        "runtime": runtime.as_dict(),
                                        "determinism_attestation": (
                                            attestation.as_dict()
                                        ),
                                        "evaluation_result_sha256": result_sha256,
                                    }
                                    if judge_transcript_artifact is not None:
                                        metrics_payload["replay_provenance"][
                                            "judge_transcript"
                                        ] = judge_transcript_artifact.provenance()
                                    service.record_score_payload(
                                        payload={
                                            "cot_mode": "CoT",
                                            "metrics": metrics_payload,
                                        },
                                        task_id=str(replay_task_id),
                                    )
                                    with psycopg.connect(
                                        conninfo,
                                        row_factory=dict_row,
                                    ) as state_connection:
                                        persisted_rows = _existing_replays(
                                            state_connection,
                                            task_desc=replay_task_desc,
                                            model_name=str(model["model_name"]),
                                            benchmark_name=str(
                                                benchmark["benchmark_name"]
                                            ),
                                            benchmark_split=str(
                                                benchmark.get("benchmark_split")
                                                or ""
                                            ),
                                        )
                                    persisted_state, persisted_row = (
                                        _classify_existing_replays(
                                            persisted_rows,
                                            expected_rows=len(payloads),
                                            expected_git_hash=(
                                                contract.replay_git_hash
                                            ),
                                        )
                                    )
                                    if (
                                        persisted_state != "valid"
                                        or persisted_row is None
                                        or int(persisted_row["task_id"])
                                        != replay_task_id
                                    ):
                                        raise RuntimeError(
                                            f"replay task {replay_task_id}: "
                                            "post-commit provenance/count validation "
                                            f"failed ({persisted_state})"
                                        )
                                    post_commit_validated = True
                                except Exception:
                                    if replay_task_id is not None:
                                        service.update_task_status(
                                            task_id=str(replay_task_id),
                                            status="failed",
                                        )
                                    raise
                                finally:
                                    if old_desc is None:
                                        os.environ.pop("RWKV_TASK_DESC", None)
                                    else:
                                        os.environ["RWKV_TASK_DESC"] = old_desc
                                    if old_tmp is None:
                                        os.environ.pop("RWKV_TASK_IS_TMP", None)
                                    else:
                                        os.environ["RWKV_TASK_IS_TMP"] = old_tmp

        staged = 0
        adapter_affected = 0
        repaired_box_prefix = 0
        for payload, evaluated in zip(payloads, evaluation.payloads, strict=True):
            prompt = str(payload.get("prompt2") or "")
            completion = str(payload.get("completion2") or "")
            if not prompt:
                continue
            staged += 1
            adapter_affected += 1
            if "\\boxed{" in prompt and completion and bool(evaluated.get("answer")):
                repaired_box_prefix += 1

        report.append(
            {
                "task_id": task_id,
                "replayable": True,
                "blocked": idempotency_action
                in {
                    "advisory_lock_busy",
                    "source_drift_under_lock",
                    "wait_for_existing_replay",
                },
                "judge_mode": judge_mode,
                "judge_stats_by_group": evaluation.judge_stats_by_group,
                "math_verify_retry_timeout_s": MATH_VERIFY_RETRY_TIMEOUT_S,
                "evaluation_preflight": preflight,
                "source_preflight": source_preflight,
                "source_strategy_preflight": source_strategy_preflight,
                "source_evidence": source_evidence,
                "source_evidence_sha256": source_evidence_sha256,
                "runtime_provenance": runtime.as_dict(),
                "determinism_attestation": (
                    attestation.as_dict() if attestation is not None else None
                ),
                "evaluation_result_sha256": result_sha256,
                "judge_transcript": (
                    judge_transcript_artifact.provenance()
                    if judge_transcript_artifact is not None
                    and judge_mode == "llm"
                    else None
                ),
                "model": model["model_name"],
                "benchmark": slug,
                "rows": len(payloads),
                "staged_rows": staged,
                "rows_using_stage2_think_close": adapter_affected,
                "rows_with_repaired_box_prefix": repaired_box_prefix,
                "stored_eval_rows": len(stored_answers),
                "stored_answers_changed_by_replay": changed_answers,
                "stored_answers_with_synthetic_suffix": contaminated_stored_answers,
                "replay_task_id": replay_task_id,
                "replay_task_desc": replay_task_desc,
                "idempotency_action": idempotency_action,
                "post_commit_validated": post_commit_validated,
                "strategy_task_ids": strategy_task_ids,
                "stored_primary_metric": stored_key,
                "stored_primary_score": stored_value,
                "replayed_primary_score": replay_value,
                "delta_pp": (
                    None
                    if stored_value is None or replay_value is None
                    else (replay_value - stored_value) * 100.0
                ),
                "stored_metrics": {
                    key: stored_metrics[key]
                    for key in _numeric_metric_keys(stored_metrics)
                },
                "replayed_metrics": replayed_metrics,
                "replayed_passed_rows": sum(bool(row[2]) for row in rows),
                "replayed_missing_answers": sum(
                    not bool(payload.get("answer")) for payload in evaluation.payloads
                ),
            }
        )

    transcript_finalization_blockers: list[str] = []
    try:
        if judge_transcript_recorder is not None:
            expected_ids = sorted(set(args.task_ids))
            replayable_ids = sorted(
                int(row["task_id"])
                for row in report
                if row.get("replayable") and not row.get("blocked")
            )
            if blocked or replayable_ids != expected_ids:
                transcript_finalization_blockers.append(
                    "judge_transcript_not_published_after_incomplete_preflight"
                )
            else:
                judge_transcript_artifact = judge_transcript_recorder.persist()
                for row in report:
                    if row.get("judge_mode") == "llm":
                        row["judge_transcript"] = (
                            judge_transcript_artifact.provenance()
                        )
                        provenance = row.get("evaluation_preflight")
                        if isinstance(provenance, dict):
                            nested = provenance.get("provenance")
                            if isinstance(nested, dict):
                                nested["judge_transcript_sha256"] = (
                                    judge_transcript_artifact.sha256
                                )
        elif judge_transcript_replayer is not None:
            judge_transcript_replayer.assert_consumed()
    except JudgeTranscriptError as exc:
        transcript_finalization_blockers.append(
            f"judge_transcript_finalization_failed:{type(exc).__name__}:{exc}"
        )

    if transcript_finalization_blockers:
        blocked = True
        for row in report:
            if row.get("replayable"):
                row["blocked"] = True
                row.setdefault("blocking_reasons", []).extend(
                    transcript_finalization_blockers
                )

    output = {
        "database": args.dbname,
        "adapter": "current_global_free_response_adapter",
        "primary_group": STRATEGY_C,
        "read_only": not args.commit,
        "commit_requested": bool(args.commit),
        "committed": any(
            row.get("replay_task_id")
            and row.get("idempotency_action") == "append_new_replay"
            for row in report
        ),
        "blocked": blocked,
        "provenance_preflight": {
            "passed": not transcript_finalization_blockers,
            "blockers": transcript_finalization_blockers,
            "mode": (
                "judge_transcript_recording"
                if args.record_judge_transcript
                else (
                    "determinism_attestation_bootstrap"
                    if args.emit_determinism_run
                    else "final_replay"
                )
            ),
            "contract": contract.as_dict(),
            "runtime": runtime.as_dict(),
            "determinism_attestation": (
                attestation.as_dict() if attestation is not None else None
            ),
            "judge_transcript": (
                judge_transcript_artifact.provenance()
                if judge_transcript_artifact is not None
                else None
            ),
        },
        "tasks": report,
    }
    if args.emit_determinism_run:
        output["attestation_run"] = {
            "seed": runtime.pythonhashseed,
            "source_task_ids": sorted(set(args.task_ids)),
            "task_result_sha256": {
                str(row["task_id"]): str(row["evaluation_result_sha256"])
                for row in report
                if row.get("replayable")
                and row.get("evaluation_result_sha256")
                and not row.get("blocked")
            },
            "source_evidence_sha256_by_task": {
                str(row["task_id"]): str(row["source_evidence_sha256"])
                for row in report
                if row.get("replayable")
                and row.get("source_evidence_sha256")
                and not row.get("blocked")
            },
            "judge_transcript_sha256_by_task": {
                str(row["task_id"]): str(
                    judge_transcript_artifact.sha256
                )
                for row in report
                if row.get("replayable")
                and row.get("judge_mode") == "llm"
                and not row.get("blocked")
                and judge_transcript_artifact is not None
            },
        }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary:
        print(
            json.dumps(
                {
                    "database": args.dbname,
                    "tasks": len(report),
                    "replayable": sum(bool(row.get("replayable")) for row in report),
                    "blocked": sum(bool(row.get("blocked")) for row in report),
                    "rows": sum(int(row.get("rows") or 0) for row in report),
                    "provenance_preflight": output["provenance_preflight"],
                    "idempotency_actions": {
                        str(row.get("task_id")): row.get("idempotency_action")
                        for row in report
                        if row.get("idempotency_action")
                    },
                    "stage2_think_close_rows": sum(
                        int(row.get("rows_using_stage2_think_close") or 0) for row in report
                    ),
                    "score_changes": [
                        {
                            "task_id": row.get("task_id"),
                            "stored": row.get("stored_primary_score"),
                            "replayed": row.get("replayed_primary_score"),
                            "delta_pp": row.get("delta_pp"),
                        }
                        for row in report
                        if row.get("replayable")
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(rendered)
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
