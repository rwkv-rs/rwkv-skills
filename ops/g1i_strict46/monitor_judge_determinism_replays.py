#!/usr/bin/env python3
"""Append-only replay of pre-fix strict-46 external-Judge cells.

The original external LLM Judge sampled at temperature 0.8.  The global
free-response adapter now sends deterministic Judge requests at temperature
0.0.  This monitor reconciles only the four strict-46 Math families that use
the external Judge and never regenerates model completions:

* one newest pre-deployment source is selected per model x benchmark cell;
* a complete, scored post-deployment root task resolves the whole cell, even
  when it was created by another global repair;
* otherwise a settled source with a complete coordinate grid is replayed once
  through :mod:`recompute_math_from_completions`;
* an exact source/reason provenance marker provides a second idempotency gate.

Source tasks, completions, eval rows, and scores remain immutable.  ``--once``
performs one reconciliation pass; the default mode waits for running sources
and exits only when every selected pre-cutoff source is resolved or blocked.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import (
    JUDGE_DETERMINISM_DEPLOYED_AT,
    MODELS,
    RAW_COMPLETIONS_PROTOCOL_DEPLOYED_AT,
    STRICT_CONFIG_ROOT,
    _expected_avg_k,
    _expected_evaluator,
    _sampling_protocol_reasons,
    canonical_target_benchmark,
)
from ops.g1i_strict46.replay_lock import (
    held_replay_advisory_locks,
    replay_advisory_lock_keys,
)
from src.db.pool import _build_conninfo
from src.eval.benchmark_config import resolve_benchmark_model_config
from src.eval.env_config import load_env_file, resolve_judge_max_tokens
from src.eval.metrics.free_response import (
    DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE,
    llm_judge_protocol_stats_reasons,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
CURRENT_WAVE_STARTED_AT = datetime(2026, 8, 6, 12, 54, 0)
ROOT_EVALUATOR = "free_response_judge_naive"
# ``v2`` is intentionally a new provenance namespace.  Earlier deterministic
# temperature-0 replays did not persist the protocol fingerprint, so their
# marker must not suppress the fingerprinted replay required by this monitor.
REASON_TAG = "g1i_llm_judge_protocol_fingerprint_v2_20260807"
JUDGE_TARGETS = frozenset(
    {
        ("amc23", "test"),
        ("comp_math_24_25", "test"),
        ("gaokao2023en", "test"),
        ("minerva_math", "test"),
    }
)
TERMINAL_FAILURE_STATUSES = frozenset(
    {"failed", "cancelled", "canceled", "stopped"}
)
SCORE_PERSISTENCE_GRACE = timedelta(minutes=10)

TASK_QUERY = """
SELECT
    t.task_id,
    t.status,
    t.created_at AS task_created_at,
    t.evaluator,
    t.sampling_config,
    t."desc" AS task_desc,
    t.is_param_search,
    t.is_tmp,
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    b.num_samples AS benchmark_num_samples,
    latest_score.score_id,
    latest_score.cot_mode,
    latest_score.metrics,
    latest_score.created_at AS score_created_at,
    COALESCE(completion_stats.completion_count, 0) AS completion_count,
    COALESCE(completion_stats.total_completion_count, 0)
        AS total_completion_count,
    COALESCE(completion_stats.non_completed_completion_count, 0)
        AS non_completed_completion_count,
    COALESCE(completion_stats.distinct_completion_coordinates, 0)
        AS distinct_completion_coordinates,
    COALESCE(completion_stats.distinct_sample_repeat_coordinates, 0)
        AS distinct_sample_repeat_coordinates,
    COALESCE(completion_stats.distinct_sample_indices, 0)
        AS distinct_sample_indices,
    completion_stats.min_sample_index,
    completion_stats.max_sample_index,
    COALESCE(completion_stats.distinct_avg_repeat_indices, 0)
        AS distinct_avg_repeat_indices,
    completion_stats.min_avg_repeat_index,
    completion_stats.max_avg_repeat_index,
    COALESCE(completion_stats.distinct_pass_indices, 0)
        AS distinct_pass_indices,
    completion_stats.min_pass_index,
    completion_stats.max_pass_index,
    COALESCE(completion_stats.eval_count, 0) AS eval_count,
    COALESCE(completion_stats.passed_eval_count, 0) AS passed_eval_count
FROM task t
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN LATERAL (
    SELECT s.score_id, s.cot_mode, s.metrics, s.created_at
    FROM scores s
    WHERE s.task_id = t.task_id
    ORDER BY s.score_id DESC
    LIMIT 1
) AS latest_score ON TRUE
LEFT JOIN LATERAL (
    SELECT
        COUNT(*) FILTER (WHERE c.status = 'Completed') AS completion_count,
        COUNT(*) AS total_completion_count,
        COUNT(*) FILTER (WHERE c.status <> 'Completed')
            AS non_completed_completion_count,
        COUNT(DISTINCT (c.sample_index, c.avg_repeat_index, c.pass_index))
            FILTER (WHERE c.status = 'Completed')
            AS distinct_completion_coordinates,
        COUNT(DISTINCT (c.sample_index, c.avg_repeat_index))
            FILTER (WHERE c.status = 'Completed')
            AS distinct_sample_repeat_coordinates,
        COUNT(DISTINCT c.sample_index) FILTER (WHERE c.status = 'Completed')
            AS distinct_sample_indices,
        MIN(c.sample_index) FILTER (WHERE c.status = 'Completed')
            AS min_sample_index,
        MAX(c.sample_index) FILTER (WHERE c.status = 'Completed')
            AS max_sample_index,
        COUNT(DISTINCT c.avg_repeat_index) FILTER (WHERE c.status = 'Completed')
            AS distinct_avg_repeat_indices,
        MIN(c.avg_repeat_index) FILTER (WHERE c.status = 'Completed')
            AS min_avg_repeat_index,
        MAX(c.avg_repeat_index) FILTER (WHERE c.status = 'Completed')
            AS max_avg_repeat_index,
        COUNT(DISTINCT c.pass_index) FILTER (WHERE c.status = 'Completed')
            AS distinct_pass_indices,
        MIN(c.pass_index) FILTER (WHERE c.status = 'Completed')
            AS min_pass_index,
        MAX(c.pass_index) FILTER (WHERE c.status = 'Completed')
            AS max_pass_index,
        COUNT(e.eval_id) FILTER (WHERE c.status = 'Completed') AS eval_count,
        COUNT(e.eval_id) FILTER (
            WHERE c.status = 'Completed' AND e.is_passed IS TRUE
        ) AS passed_eval_count
    FROM completions c
    LEFT JOIN eval e ON e.completions_id = c.completions_id
    WHERE c.task_id = t.task_id
) AS completion_stats ON TRUE
WHERE m.model_name = ANY(%s)
  AND t.created_at >= %s
  AND b.benchmark_name = ANY(%s)
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


def _safe_int(value: object) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@contextmanager
def _strict_config_environment() -> Iterator[None]:
    """Resolve expected stages from the same strict G1h/G1i config root."""

    key = "RWKV_BENCHMARK_CONFIG_ROOT"
    previous = os.environ.get(key)
    os.environ[key] = str(STRICT_CONFIG_ROOT)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _strict_protocol_reasons(row: dict[str, Any]) -> list[str]:
    """Validate the persisted strict-46 generation and task protocol."""

    reasons: list[str] = []
    sampling_config = _json_object(row.get("sampling_config"))
    audit_row = {
        **row,
        "domain": "math",
        "sampling_config": sampling_config,
    }
    expected_evaluator = _expected_evaluator(audit_row)
    if expected_evaluator is None:
        reasons.append("missing_expected_evaluator")
    elif str(row.get("evaluator") or "") != expected_evaluator:
        reasons.append(
            f"evaluator:{row.get('evaluator')!r}"
            f"!=expected:{expected_evaluator!r}"
        )
    if str(sampling_config.get("prompt_profile") or "").lower() != "naive":
        reasons.append("prompt_profile_not_naive")
    configured_mode = re.sub(
        r"[^a-z]", "", str(sampling_config.get("cot_mode") or "").lower()
    )
    if configured_mode != "cot":
        reasons.append(f"cot_mode:{configured_mode or 'empty'}!=expected:cot")
    if sampling_config.get("sample_limit") is not None:
        reasons.append(f"sample_limit:{sampling_config.get('sample_limit')}")
    if _safe_int(sampling_config.get("n_shot")) != 0:
        reasons.append(f"n_shot:{sampling_config.get('n_shot')}!=expected:0")
    if bool(row.get("is_param_search")):
        reasons.append("is_param_search")
    if bool(row.get("is_tmp")):
        reasons.append("is_tmp")
    task_created_at = row.get("task_created_at")
    if not isinstance(task_created_at, datetime):
        reasons.append("missing_task_created_at")
    elif task_created_at < RAW_COMPLETIONS_PROTOCOL_DEPLOYED_AT:
        reasons.append("generation_predates_raw_completions_protocol_fix")

    with _strict_config_environment():
        expected_avg_k = _expected_avg_k(audit_row)
        reasons.extend(_sampling_protocol_reasons(audit_row))
    actual_avg_k = _safe_float(sampling_config.get("avg_k"))
    if actual_avg_k is None:
        reasons.append("missing_or_invalid_avg_k")
    elif abs(actual_avg_k - expected_avg_k) > 1e-12:
        reasons.append(f"avg_k:{actual_avg_k}!=expected:{expected_avg_k}")
    return list(dict.fromkeys(reasons))


def _judge_stats_reasons(row: dict[str, Any]) -> list[str]:
    """Require persisted evidence that every Judge request parsed cleanly."""

    metrics = _json_object(row.get("metrics"))
    judge_stats = metrics.get("judge_stats")
    if not isinstance(judge_stats, dict):
        return ["missing_persisted_judge_stats"]
    total = _safe_int(judge_stats.get("total"))
    parsed = _safe_int(judge_stats.get("parsed_count"))
    invalid = _safe_int(judge_stats.get("invalid_output_count"))
    request = _safe_int(judge_stats.get("request_error_count"))
    errors = _safe_int(judge_stats.get("error_count"))
    reasons: list[str] = []
    if parsed != total:
        reasons.append(f"judge_parsed_count:{parsed}!=total:{total}")
    if invalid or request or errors:
        reasons.append(
            f"judge_errors:invalid:{invalid},request:{request},total:{errors}"
        )
    sampling_config = _json_object(row.get("sampling_config"))
    expected_model = str(
        sampling_config.get("judger_model_name") or os.environ.get("JUDGE_MODEL") or ""
    )
    slug = canonical_slug(
        f"{row.get('benchmark_name', '')}_{row.get('benchmark_split', '')}"
    )
    with _strict_config_environment():
        benchmark_config = resolve_benchmark_model_config(
            slug,
            str(row.get("model_name") or ""),
            stage=None,
        )
    expected_prompt = (
        benchmark_config.judge_prompt_template
        if benchmark_config is not None and benchmark_config.judge_prompt_template
        else DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE
    )
    reasons.extend(
        llm_judge_protocol_stats_reasons(
            judge_stats,
            expected_model=expected_model or None,
            expected_prompt_template=expected_prompt,
            expected_max_completion_tokens=resolve_judge_max_tokens(None),
        )
    )
    return reasons


def _score_metric_reasons(row: dict[str, Any]) -> list[str]:
    metrics = _json_object(row.get("metrics"))
    audit_row = {**row, "domain": "math"}
    with _strict_config_environment():
        expected_avg_k = _expected_avg_k(audit_row)
    suffix = (
        str(int(expected_avg_k))
        if float(expected_avg_k).is_integer()
        else str(expected_avg_k)
    )
    metric_name = f"avg@{suffix}"
    metric_value = metrics.get(metric_name)
    if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
        return [f"missing_primary_metric:{metric_name}"]
    eval_count = _safe_int(row.get("eval_count"))
    if eval_count <= 0:
        return ["zero_eval_count_for_primary_metric"]
    eval_pass_rate = _safe_int(row.get("passed_eval_count")) / eval_count
    if abs(float(metric_value) - eval_pass_rate) > 1e-12:
        return [
            f"primary_metric_eval_mismatch:{metric_name}:"
            f"{float(metric_value)}!={eval_pass_rate}"
        ]
    return []


def _provenance_marker(source_task_id: int, reason_tag: str) -> str:
    return f"replay_source_task_id={source_task_id};{reason_tag}"


def _cell(row: dict[str, Any]) -> tuple[str, str, str]:
    benchmark = canonical_target_benchmark(
        str(row.get("benchmark_name") or ""),
        str(row.get("benchmark_split") or ""),
    )
    return str(row.get("model_name") or ""), benchmark[0], benchmark[1]


def _filter_task_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the exact root evaluator and approved Naive/CoT cells."""

    candidates: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        model_name, benchmark_name, benchmark_split = _cell(row)
        sampling_config = _json_object(row.get("sampling_config"))
        configured_mode = str(sampling_config.get("cot_mode") or "").lower()
        score_mode = row.get("cot_mode")
        if model_name not in MODELS:
            continue
        if (benchmark_name, benchmark_split) not in JUDGE_TARGETS:
            continue
        if str(row.get("evaluator") or "") != ROOT_EVALUATOR:
            continue
        if str(sampling_config.get("prompt_profile") or "").lower() != "naive":
            continue
        if configured_mode not in {"cot", "co_t"}:
            continue
        if score_mode is not None and str(score_mode).lower() != "cot":
            continue
        row["benchmark_name"] = benchmark_name
        row["benchmark_split"] = benchmark_split
        candidates.append(row)
    return candidates


def _latest_by_cell(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        cell = _cell(row)
        previous = latest.get(cell)
        current_key = (row.get("task_created_at"), int(row["task_id"]))
        previous_key = (
            previous.get("task_created_at"),
            int(previous["task_id"]),
        ) if previous is not None else None
        if previous_key is None or current_key > previous_key:
            latest[cell] = row
    return latest


def _select_latest_valid_sources(
    rows: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    """Select the newest replayable source without letting drift hide history."""

    observed_at = now or datetime.now()
    sources: dict[tuple[str, str, str], dict[str, Any]] = {}
    invalid_sources: list[dict[str, Any]] = []
    rows_by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_cell.setdefault(_cell(row), []).append(row)
    for cell, cell_rows in rows_by_cell.items():
        for row in sorted(
            cell_rows,
            key=lambda item: (item.get("task_created_at"), int(item["task_id"])),
            reverse=True,
        ):
            protocol_reasons = _strict_protocol_reasons(row)
            status = str(row.get("status") or "").lower()
            waiting = status not in TERMINAL_FAILURE_STATUSES | {"completed"}
            if status == "completed" and row.get("score_id") is None:
                created_at = row.get("task_created_at")
                waiting = bool(
                    isinstance(created_at, datetime)
                    and observed_at - created_at <= SCORE_PERSISTENCE_GRACE
                )
                if not waiting:
                    protocol_reasons.append("score_persistence_timeout")
            contract_reasons = [] if waiting else _completion_contract_reasons(row)
            if protocol_reasons or contract_reasons:
                invalid_sources.append(
                    {
                        **row,
                        "cell": "::".join(cell),
                        "protocol_reasons": protocol_reasons,
                        "completion_contract_reasons": contract_reasons,
                    }
                )
                continue
            sources[cell] = row
            break
    return sources, invalid_sources


def _completion_contract_reasons(row: dict[str, Any]) -> list[str]:
    """Validate the immutable coordinate grid needed for a safe replay."""

    reasons: list[str] = []
    sampling_config = _json_object(row.get("sampling_config"))
    benchmark_samples = _safe_int(row.get("benchmark_num_samples"))
    avg_k = _safe_int(sampling_config.get("avg_k"))
    recorded_expected = _safe_int(sampling_config.get("effective_sample_count"))
    expected = benchmark_samples * avg_k if benchmark_samples > 0 and avg_k > 0 else 0
    completion_count = _safe_int(row.get("completion_count"))
    total_completion_count = _safe_int(row.get("total_completion_count"))
    non_completed_count = _safe_int(row.get("non_completed_completion_count"))

    if expected <= 0:
        reasons.append("missing_expected_completion_count")
    elif recorded_expected != expected:
        reasons.append(f"effective_sample_count:{recorded_expected}!=expected:{expected}")
    if expected > 0 and completion_count != expected:
        reasons.append(f"completion_count:{completion_count}!=expected:{expected}")
    if completion_count <= 0:
        reasons.append("zero_completions")
    if total_completion_count != completion_count:
        reasons.append(
            f"total_completion_count:{total_completion_count}"
            f"!=completed:{completion_count}"
        )
    if non_completed_count:
        reasons.append(f"non_completed_completions:{non_completed_count}")
    distinct_coordinates = _safe_int(row.get("distinct_completion_coordinates"))
    if distinct_coordinates != completion_count:
        reasons.append(
            "distinct_completion_coordinates:"
            f"{distinct_coordinates}!=completions:{completion_count}"
        )
    distinct_sample_repeat = _safe_int(
        row.get("distinct_sample_repeat_coordinates")
    )
    if distinct_sample_repeat != completion_count:
        reasons.append(
            "distinct_sample_repeat_coordinates:"
            f"{distinct_sample_repeat}!=completions:{completion_count}"
        )
    distinct_samples = _safe_int(row.get("distinct_sample_indices"))
    if benchmark_samples > 0 and distinct_samples != benchmark_samples:
        reasons.append(
            f"distinct_sample_indices:{distinct_samples}!=expected:{benchmark_samples}"
        )
    if benchmark_samples > 0 and (
        _safe_int(row.get("min_sample_index")) != 0
        or _safe_int(row.get("max_sample_index")) != benchmark_samples - 1
    ):
        reasons.append(
            "sample_index_range:"
            f"{_safe_int(row.get('min_sample_index'))}.."
            f"{_safe_int(row.get('max_sample_index'))}"
            f"!=expected:0..{benchmark_samples - 1}"
        )
    distinct_repeats = _safe_int(row.get("distinct_avg_repeat_indices"))
    if avg_k > 0 and distinct_repeats != avg_k:
        reasons.append(
            f"distinct_avg_repeat_indices:{distinct_repeats}!=expected:{avg_k}"
        )
    if avg_k > 0 and (
        _safe_int(row.get("min_avg_repeat_index")) != 0
        or _safe_int(row.get("max_avg_repeat_index")) != avg_k - 1
    ):
        reasons.append(
            "avg_repeat_index_range:"
            f"{_safe_int(row.get('min_avg_repeat_index'))}.."
            f"{_safe_int(row.get('max_avg_repeat_index'))}"
            f"!=expected:0..{avg_k - 1}"
        )
    distinct_pass_indices = _safe_int(row.get("distinct_pass_indices"))
    if distinct_pass_indices != 1:
        reasons.append(
            f"distinct_pass_indices:{distinct_pass_indices}!=expected:1"
        )
    if (
        _safe_int(row.get("min_pass_index")) != 0
        or _safe_int(row.get("max_pass_index")) != 0
    ):
        reasons.append(
            "pass_index_range:"
            f"{_safe_int(row.get('min_pass_index'))}.."
            f"{_safe_int(row.get('max_pass_index'))}!=expected:0..0"
        )
    return reasons


def _is_complete_post_cutoff(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "").lower() != "completed":
        return False
    if row.get("score_id") is None or str(row.get("cot_mode") or "").lower() != "cot":
        return False
    if _strict_protocol_reasons(row) or _completion_contract_reasons(row):
        return False
    if _safe_int(row.get("eval_count")) != _safe_int(row.get("completion_count")):
        return False
    judge_reasons = (
        _judge_stats_reasons(row)
        if "judge" in str(row.get("evaluator") or "").lower()
        else []
    )
    return not (judge_reasons or _score_metric_reasons(row))


def _split_post_candidates(
    rows: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    """Return fully-valid and genuinely in-flight post-deployment cells."""

    resolved = _latest_by_cell(
        [row for row in rows if _is_complete_post_cutoff(row)]
    )
    observed_at = now or datetime.now()

    def post_is_pending(row: dict[str, Any]) -> bool:
        if _strict_protocol_reasons(row):
            return False
        status = str(row.get("status") or "").lower()
        if status in TERMINAL_FAILURE_STATUSES:
            return False
        if status != "completed":
            return True
        if row.get("score_id") is not None:
            return False
        created_at = row.get("task_created_at")
        return bool(
            isinstance(created_at, datetime)
            and observed_at - created_at <= SCORE_PERSISTENCE_GRACE
        )

    pending = _latest_by_cell(
        [
            row
            for row in rows
            if _cell(row) not in resolved and post_is_pending(row)
        ]
    )
    return resolved, pending


def _split_candidates(
    rows: list[dict[str, Any]],
    *,
    deployed_at: datetime,
    now: datetime | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    dict[tuple[str, str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    candidates = _filter_task_candidates(rows)
    pre = [
        row
        for row in candidates
        if isinstance(row.get("task_created_at"), datetime)
        and row["task_created_at"] < deployed_at
    ]
    post = [
        row
        for row in candidates
        if isinstance(row.get("task_created_at"), datetime)
        and row["task_created_at"] >= deployed_at
    ]
    observed_at = now or datetime.now()
    sources, invalid_sources = _select_latest_valid_sources(pre, now=observed_at)

    resolved, pending = _split_post_candidates(post, now=observed_at)
    return sources, resolved, pending, invalid_sources


def _marker_replays_by_source(
    rows: list[dict[str, Any]],
    sources: dict[tuple[str, str, str], dict[str, Any]],
    reason_tag: str,
) -> dict[int, list[dict[str, Any]]]:
    matches: dict[int, list[dict[str, Any]]] = {}
    # Multiple marker-identical tasks are possible in historical data.  Build
    # from rows rather than a LIMIT 1 query so any fully valid replay wins.
    for source in sources.values():
        source_task_id = int(source["task_id"])
        marker = _provenance_marker(source_task_id, reason_tag)
        matches[source_task_id] = [
            dict(row)
            for row in rows
            if str(row.get("task_desc") or "") == marker
            and _cell(row) == _cell(source)
        ]
    return matches


def _replay_is_pending(
    row: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    """Return whether an existing replay is both validly configured and live.

    A malformed ``Running`` row must not suppress a new valid replay forever.
    Likewise, ``Completed`` without a persisted score gets only the bounded
    score-persistence grace interval used by the cell-level reconciler.
    """

    if _strict_protocol_reasons(row):
        return False
    status = str(row.get("status") or "").lower()
    if status in TERMINAL_FAILURE_STATUSES:
        return False
    if status != "completed":
        return True
    if row.get("score_id") is not None:
        return False
    created_at = row.get("task_created_at")
    observed_at = now or datetime.now()
    return bool(
        isinstance(created_at, datetime)
        and observed_at - created_at <= SCORE_PERSISTENCE_GRACE
    )


def _classify_existing_replays(
    rows: list[dict[str, Any]],
    *,
    deployed_at: datetime,
    now: datetime | None = None,
) -> tuple[str, dict[str, Any] | None]:
    valid = [
        row
        for row in rows
        if isinstance(row.get("task_created_at"), datetime)
        and row["task_created_at"] >= deployed_at
        and _is_complete_post_cutoff(row)
    ]
    if valid:
        return "valid", max(
            valid,
            key=lambda row: (row.get("task_created_at"), int(row["task_id"])),
        )
    pending = [row for row in rows if _replay_is_pending(row, now=now)]
    if pending:
        return "pending", max(
            pending,
            key=lambda row: (row.get("task_created_at"), int(row["task_id"])),
        )
    if rows:
        return "blocked", max(
            rows,
            key=lambda row: (row.get("task_created_at"), int(row["task_id"])),
        )
    return "missing", None


def _plan_replays(
    sources_by_cell: dict[tuple[str, str, str], dict[str, Any]],
    resolved_by_cell: dict[tuple[str, str, str], dict[str, Any]],
    pending_post_by_cell: dict[tuple[str, str, str], dict[str, Any]],
    existing_replays: dict[int, list[dict[str, Any]]],
    invalid_sources: list[dict[str, Any]] | None = None,
    *,
    deployed_at: datetime = JUDGE_DETERMINISM_DEPLOYED_AT,
    now: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    plan: dict[str, list[dict[str, Any]]] = {
        "eligible_to_replay": [],
        "resolved_post_cutoff": [],
        "waiting_sources": [],
        "pending_post_cutoff": [],
        "already_replayed": [],
        "pending_existing_replay": [],
        "blocked_existing_replay": [],
        "blocked_incomplete_source": [],
        "ignored_invalid_sources": list(invalid_sources or []),
        "blocked_invalid_source_cells": [],
    }
    selected_cells = set(sources_by_cell)
    unresolved_invalid_cells: set[tuple[str, str, str]] = set()
    for invalid in invalid_sources or []:
        invalid_cell = _cell(invalid)
        if invalid_cell in selected_cells or invalid_cell in unresolved_invalid_cells:
            continue
        unresolved_invalid_cells.add(invalid_cell)
        plan["blocked_invalid_source_cells"].append(invalid)
    for cell, source in sorted(sources_by_cell.items()):
        row = dict(source)
        row["cell"] = "::".join(cell)
        post = resolved_by_cell.get(cell)
        if post is not None:
            row["post_cutoff_task_id"] = int(post["task_id"])
            plan["resolved_post_cutoff"].append(row)
            continue
        pending_post = pending_post_by_cell.get(cell)
        if pending_post is not None:
            row["post_cutoff_task_id"] = int(pending_post["task_id"])
            plan["pending_post_cutoff"].append(row)
            continue

        status = str(row.get("status") or "").lower()
        if status not in TERMINAL_FAILURE_STATUSES | {"completed"}:
            plan["waiting_sources"].append(row)
            continue
        if status == "completed" and row.get("score_id") is None:
            plan["waiting_sources"].append(row)
            continue
        protocol_reasons = _strict_protocol_reasons(row)
        contract_reasons = _completion_contract_reasons(row)
        if protocol_reasons:
            row["protocol_reasons"] = protocol_reasons
            plan["blocked_incomplete_source"].append(row)
            continue
        if contract_reasons:
            row["completion_contract_reasons"] = contract_reasons
            plan["blocked_incomplete_source"].append(row)
            continue

        replays = existing_replays.get(int(row["task_id"]), [])
        replay_state, replay = _classify_existing_replays(
            replays,
            deployed_at=deployed_at,
            now=now,
        )
        if replay_state == "missing":
            plan["eligible_to_replay"].append(row)
            continue
        annotated = {**row, "existing_replays": replays}
        if replay_state == "valid":
            annotated["accepted_replay_task_id"] = int(replay["task_id"])
            plan["already_replayed"].append(annotated)
        elif replay_state == "pending":
            plan["pending_existing_replay"].append(annotated)
        else:
            plan["blocked_existing_replay"].append(annotated)
    return plan


def _scan(
    connection: psycopg.Connection[Any],
    *,
    wave_started_at: datetime,
    deployed_at: datetime,
    reason_tag: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows = _fetch_task_rows(
        connection,
        wave_started_at=wave_started_at,
        benchmark_names=sorted({name for name, _split in JUDGE_TARGETS}),
    )
    sources, resolved, pending_post, invalid_sources = _split_candidates(
        rows, deployed_at=deployed_at
    )
    existing = _marker_replays_by_source(rows, sources, reason_tag)
    return rows, _plan_replays(
        sources,
        resolved,
        pending_post,
        existing,
        invalid_sources,
        deployed_at=deployed_at,
    )


def _fetch_task_rows(
    connection: psycopg.Connection[Any],
    *,
    wave_started_at: datetime,
    benchmark_names: list[str],
) -> list[dict[str, Any]]:
    """Fetch root candidate rows with one shared validation projection."""

    return [
        dict(row)
        for row in connection.execute(
            TASK_QUERY,
            (list(MODELS), wave_started_at, benchmark_names),
        ).fetchall()
    ]


def _build_replay_command(
    *,
    repo: Path,
    replay_script: Path,
    source_task_id: int,
    dbname: str,
    reason_tag: str,
    output: Path,
    judge_max_workers: int,
) -> list[str]:
    return [
        str(repo / ".venv/bin/python"),
        str(replay_script),
        str(source_task_id),
        "--dbname",
        dbname,
        "--judge-mode",
        "auto",
        "--judge-max-workers",
        str(judge_max_workers),
        "--reason-tag",
        reason_tag,
        "--commit",
        "--summary",
        "--output",
        str(output),
    ]


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _plan_ids(plan: dict[str, list[dict[str, Any]]]) -> dict[str, list[int]]:
    return {
        state: [int(row["task_id"]) for row in entries]
        for state, entries in plan.items()
    }


def _eligible_source(
    plan: dict[str, list[dict[str, Any]]], source_task_id: int
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in plan["eligible_to_replay"]
            if int(row["task_id"]) == int(source_task_id)
        ),
        None,
    )


def _terminal_action(
    plan: dict[str, list[dict[str, Any]]],
    locally_blocked_source_ids: set[int] | frozenset[int] = frozenset(),
) -> str:
    """Return wait/blocked/complete without abandoning independent waiters."""

    launchable = [
        row
        for row in plan["eligible_to_replay"]
        if int(row["task_id"]) not in locally_blocked_source_ids
    ]
    waiting = (
        launchable
        + plan["waiting_sources"]
        + plan["pending_post_cutoff"]
        + plan["pending_existing_replay"]
    )
    if waiting:
        return "wait"
    blocked = (
        plan.get("blocked_incomplete_source", [])
        + plan.get("blocked_invalid_source_cells", [])
        + plan.get("blocked_existing_replay", [])
        + [
            row
            for row in plan["eligible_to_replay"]
            if int(row["task_id"]) in locally_blocked_source_ids
        ]
    )
    return "blocked" if blocked else "complete"


def _once_exit_code(
    *,
    replay_failed: bool,
    plan: dict[str, list[dict[str, Any]]],
) -> int:
    if replay_failed:
        return 1
    return 2 if _terminal_action(plan) == "blocked" else 0


def _replay_artifact(
    output: Path, source_task_id: int
) -> tuple[dict[str, Any] | None, str | None]:
    if not output.exists():
        return None, "missing_replay_artifact"
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"invalid_replay_artifact:{error}"
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        return None, "replay_artifact_missing_tasks"
    match = next(
        (
            row
            for row in tasks
            if isinstance(row, dict)
            and _safe_int(row.get("task_id")) == source_task_id
        ),
        None,
    )
    if match is None:
        return None, "replay_artifact_missing_source"
    if not bool(match.get("replayable")):
        return match, f"replay_not_replayable:{match.get('reason') or 'unknown'}"
    if _safe_int(match.get("replay_task_id")) <= 0:
        return match, "replay_artifact_missing_committed_task"
    return match, None


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if parsed.tzinfo is not None:
        raise argparse.ArgumentTypeError("timestamps must be timezone-naive database time")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default=DB_NAME)
    parser.add_argument("--wave-started-at", type=_parse_datetime, default=CURRENT_WAVE_STARTED_AT)
    parser.add_argument(
        "--deployed-at",
        type=_parse_datetime,
        default=JUDGE_DETERMINISM_DEPLOYED_AT,
    )
    parser.add_argument("--reason-tag", default=REASON_TAG)
    parser.add_argument("--interval-s", type=float, default=30.0)
    parser.add_argument("--judge-max-workers", type=int, default=32)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("logs/audits/g1i_judge_determinism_replay_events.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/audits/g1i_judge_determinism_replays"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("logs/audits/g1i_judge_determinism_replay_monitor.lock"),
    )
    args = parser.parse_args()
    if args.interval_s < 1:
        parser.error("--interval-s must be at least 1")
    if args.judge_max_workers < 1:
        parser.error("--judge-max-workers must be at least 1")
    if args.wave_started_at >= args.deployed_at:
        parser.error("--wave-started-at must be earlier than --deployed-at")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.reason_tag):
        parser.error(
            "--reason-tag may contain only letters, digits, dot, underscore, and dash"
        )

    repo = Path(__file__).resolve().parents[2]
    load_env_file(repo / ".env")
    replay_script = repo / "ops/g1i_strict46/recompute_math_from_completions.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    conninfo = _build_conninfo(config)
    last_state: dict[str, list[int]] | None = None
    locally_blocked: dict[int, str] = {}
    replay_env = dict(os.environ)
    replay_env["RWKV_BENCHMARK_CONFIG_ROOT"] = str(STRICT_CONFIG_ROOT)

    with args.lock.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _append_event(
            args.events,
            {
                "event": "judge_determinism_monitor_started",
                "observed_at": datetime.now().astimezone(),
                "database": args.dbname,
                "wave_started_at": args.wave_started_at,
                "deployed_at": args.deployed_at,
                "reason_tag": args.reason_tag,
            },
        )
        while True:
            with psycopg.connect(
                conninfo, row_factory=dict_row
            ) as connection:
                rows, plan = _scan(
                    connection,
                    wave_started_at=args.wave_started_at,
                    deployed_at=args.deployed_at,
                    reason_tag=args.reason_tag,
                )

            state = _plan_ids(plan)
            if state != last_state:
                _append_event(
                    args.events,
                    {
                        "event": "judge_determinism_monitor_state",
                        "observed_at": datetime.now().astimezone(),
                        "candidate_task_count": len(rows),
                        "task_ids_by_state": state,
                    },
                )
                last_state = state

            eligible_ids = {
                int(row["task_id"]) for row in plan["eligible_to_replay"]
            }
            # A launch failure is locally terminal only while that exact source
            # remains eligible.  A valid concurrent post-cutoff repair or marker
            # automatically clears it on the next scan.
            locally_blocked = {
                task_id: reason
                for task_id, reason in locally_blocked.items()
                if task_id in eligible_ids
            }
            replay_failed = False
            replay_succeeded = False
            state_changed_under_lock = False
            for source in plan["eligible_to_replay"]:
                source_task_id = int(source["task_id"])
                if source_task_id in locally_blocked:
                    continue
                lock_keys = replay_advisory_lock_keys(
                    dbname=args.dbname,
                    source_task_id=source_task_id,
                    model_name=str(source["model_name"]),
                    benchmark_name=str(source["benchmark_name"]),
                    benchmark_split=str(source["benchmark_split"]),
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
                            _append_event(
                                args.events,
                                {
                                    "event": "judge_determinism_replay_lock_busy",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "lock_keys": lock_keys,
                                },
                            )
                            continue

                        # The lock-holding connection is intentionally separate
                        # from the monitor scan connection and remains open for
                        # the whole subprocess.  Recheck after acquiring it so a
                        # concurrent blank/global replay can cancel this launch.
                        _fresh_rows, fresh_plan = _scan(
                            lock_connection,
                            wave_started_at=args.wave_started_at,
                            deployed_at=args.deployed_at,
                            reason_tag=args.reason_tag,
                        )
                        fresh_source = _eligible_source(
                            fresh_plan, source_task_id
                        )
                        if fresh_source is None:
                            state_changed_under_lock = True
                            _append_event(
                                args.events,
                                {
                                    "event": "judge_determinism_replay_cancelled",
                                    "observed_at": datetime.now().astimezone(),
                                    "source_task_id": source_task_id,
                                    "reason": "database_state_reconciled_under_lock",
                                    "task_ids_by_state": _plan_ids(fresh_plan),
                                },
                            )
                            continue

                        # Keep every attempt append-only.  A unique artifact
                        # path also prevents a stale JSON file from making a
                        # later no-output subprocess look successful.
                        attempt_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
                        output = args.output_dir / (
                            f"source_{source_task_id}_{args.reason_tag}_"
                            f"{attempt_id}.json"
                        )
                        command = _build_replay_command(
                            repo=repo,
                            replay_script=replay_script,
                            source_task_id=source_task_id,
                            dbname=args.dbname,
                            reason_tag=args.reason_tag,
                            output=output,
                            judge_max_workers=args.judge_max_workers,
                        )
                        started_at = datetime.now().astimezone()
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
                                "judge_determinism_replay_completed"
                                if failure_reason is None
                                else "judge_determinism_replay_failed_blocked"
                            ),
                            "observed_at": datetime.now().astimezone(),
                            "started_at": started_at,
                            "source": fresh_source,
                            "returncode": returncode,
                            "failure_reason": failure_reason,
                            "command": command,
                            "output": str(output),
                            "stdout_tail": stdout_tail,
                            "stderr_tail": stderr_tail,
                        }
                        if replay is not None:
                            event["replay"] = replay
                        _append_event(args.events, event)
                        if failure_reason is None:
                            replay_succeeded = True
                        else:
                            replay_failed = True
                            locally_blocked[source_task_id] = str(
                                failure_reason
                            )

            if args.once:
                return _once_exit_code(replay_failed=replay_failed, plan=plan)
            if replay_succeeded or state_changed_under_lock:
                continue
            action = _terminal_action(plan, set(locally_blocked))
            if action == "blocked":
                _append_event(
                    args.events,
                    {
                        "event": "judge_determinism_monitor_blocked",
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
                        "event": "judge_determinism_monitor_completed",
                        "observed_at": datetime.now().astimezone(),
                        "task_ids_by_state": state,
                    },
                )
                return 0
            time.sleep(args.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
