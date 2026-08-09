#!/usr/bin/env python3
"""Compare truncation telemetry across G1f/G1g/G1h/G1i databases.

The report deliberately selects one latest task per
``database × family × size × strict-46 cell`` before aggregating.  This keeps
old reruns and high-Avg@N math tasks from silently dominating the answer.
Both macro (cell averaged) and micro (completion weighted) rates are emitted.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import (
    RAW_COMPLETIONS_PROTOCOL_DEPLOYED_AT,
    TARGETS,
)
from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DEFAULT_DATABASES = (
    "chase_rwkv_skills",
    "rwkv-g1h-fallback-20260720",
    "chase_rwkv_skills_frontend46_20260804",
)

FAMILY_RE = re.compile(r"(?:rwkv7[-_])?g1([fghi])(?:[-_]|$)", re.I)
SIZE_RE = re.compile(r"(?<!\d)(1[.]5|2[.]9|7[.]2|13[.]3)b", re.I)

META_QUERY = r"""
SELECT
    m.model_name,
    b.benchmark_name,
    b.benchmark_split,
    t.task_id,
    t.status,
    t.created_at AS task_created_at,
    t.evaluator,
    t.sampling_config,
    s.cot_mode,
    s.created_at AS score_created_at
FROM model m
JOIN task t ON t.model_id = m.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN scores s ON s.task_id = t.task_id
WHERE m.model_name ~* 'g1[fghi]'
  AND b.benchmark_name = ANY(%s)
  AND (s.task_id IS NOT NULL OR t.status = 'Running')
ORDER BY t.task_id
"""

STATS_QUERY = r"""
SELECT
    c.task_id,
    COUNT(c.completions_id) AS completion_count,
    COUNT(e.eval_id) AS eval_count,
    COUNT(*) FILTER (WHERE e.fail_reason = 'missing_prediction')
        AS missing_prediction_count,
    COUNT(*) FILTER (
        WHERE COALESCE(
            NULLIF(c.context->>'direct_raw_completion', ''),
            NULLIF(c.context #>> '{strategy_a,completion}', ''),
            NULLIF(c.context #>> '{stages,0,completion}', ''),
            ''
        ) = ''
    ) AS blank_primary_count,
    COUNT(*) FILTER (
        WHERE c.completions_id IS NOT NULL
          AND (
              c.context ? 'direct_raw_finish_reason'
              OR c.context #> '{stats,truncated}' IS NOT NULL
              OR c.context #> '{stats,strategy_a,truncated}' IS NOT NULL
              OR c.context #> '{stats,stage1,truncated}' IS NOT NULL
              OR c.context #> '{stats,stage2,truncated}' IS NOT NULL
              OR c.context #> '{stages,0,stop_reason}' IS NOT NULL
          )
    ) AS telemetry_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
           OR COALESCE(c.context->>'direct_raw_finish_reason', '') IN (
               'length', 'max_length', 'max_tokens'
           )
           OR COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN (
               'length', 'max_length', 'max_tokens'
           )
    ) AS overall_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,strategy_a,truncated}')::boolean, FALSE)
           OR COALESCE((c.context #>> '{stats,stage1,truncated}')::boolean, FALSE)
           OR COALESCE(c.context->>'direct_raw_finish_reason', '') IN (
               'length', 'max_length', 'max_tokens'
           )
           OR COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN (
               'length', 'max_length', 'max_tokens'
           )
    ) AS initial_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,stage2,truncated}')::boolean, FALSE)
    ) AS final_stage_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context->>'direct_raw_finish_reason', '') IN (
            'length', 'max_length', 'max_tokens'
        )
    ) AS direct_raw_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,0,stop_reason}', '') IN (
            'length', 'max_length', 'max_tokens'
        )
    ) AS stage0_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{strategy_a,completion}', '') <> ''
    ) AS strategy_a_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,0,completion}', '') <> ''
    ) AS staged_count,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,1,completion}', '') <> ''
    ) AS recovery_count
FROM completions c
LEFT JOIN eval e ON e.completions_id = c.completions_id
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
"""


def _normalize_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"nocot", "no_cot", "false", "0"}:
        return "no_cot"
    if normalized in {"cot", "true", "1"}:
        return "cot"
    return normalized


def _family_size(model_name: str) -> tuple[str, str] | None:
    family_match = FAMILY_RE.search(model_name)
    size_match = SIZE_RE.search(model_name)
    if not family_match or not size_match:
        return None
    return f"G1{family_match.group(1).lower()}", f"{size_match.group(1)}B"


def _domain_mode(row: dict[str, Any]) -> tuple[str, str] | None:
    target = TARGETS.get((str(row["benchmark_name"]), str(row["benchmark_split"])))
    if target is None:
        return None
    domain, expected_mode = target
    actual_mode = _normalize_mode(row.get("cot_mode"))
    # Some old non-math score rows did not persist cot_mode.  Their task
    # evaluator still unambiguously identifies the strict NoCoT family.
    if not actual_mode:
        actual_mode = expected_mode
    if actual_mode != expected_mode:
        return None
    return domain, expected_mode


def _load_database(database: str) -> list[dict[str, Any]]:
    config = replace(DEFAULT_DB_CONFIG, dbname=database)
    benchmark_names = sorted({name for name, _split in TARGETS})
    with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(META_QUERY, (benchmark_names,))
            selected = _select_latest(dict(row) for row in cursor.fetchall())
            task_ids = [int(row["task_id"]) for row in selected]
            if not task_ids:
                return selected
            cursor.execute(STATS_QUERY, (task_ids,))
            stats = {int(row["task_id"]): dict(row) for row in cursor.fetchall()}
            empty_stats = {
                "completion_count": 0,
                "eval_count": 0,
                "missing_prediction_count": 0,
                "blank_primary_count": 0,
                "telemetry_count": 0,
                "overall_truncation_count": 0,
                "initial_truncation_count": 0,
                "final_stage_truncation_count": 0,
                "direct_raw_truncation_count": 0,
                "stage0_truncation_count": 0,
                "strategy_a_count": 0,
                "staged_count": 0,
                "recovery_count": 0,
            }
            for row in selected:
                row.update(stats.get(int(row["task_id"]), empty_stats))
            return selected


def _select_latest(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        parsed = _family_size(str(row["model_name"]))
        target = _domain_mode(row)
        if parsed is None or target is None:
            continue
        family, size = parsed
        # Historical architecture comparisons must use completed scored tasks.
        # Only the live G1i campaign intentionally includes partial Running
        # telemetry, because those tasks are the post-fix source of truth.
        if family != "G1i" and row.get("score_created_at") is None:
            continue
        if (
            family == "G1i"
            and row.get("task_created_at") is not None
            and row["task_created_at"] < RAW_COMPLETIONS_PROTOCOL_DEPLOYED_AT
        ):
            continue
        domain, expected_mode = target
        row["family"] = family
        row["size"] = size
        row["domain"] = domain
        row["expected_mode"] = expected_mode
        key = (
            family,
            size,
            str(row["benchmark_name"]),
            str(row["benchmark_split"]),
            expected_mode,
        )
        current = selected.get(key)
        ordering = (row.get("task_created_at"), int(row["task_id"]))
        if current is None:
            selected[key] = row
            continue
        current_ordering = (
            current.get("task_created_at"),
            int(current["task_id"]),
        )
        if ordering > current_ordering:
            selected[key] = row
    return list(selected.values())


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = [row for row in rows if int(row["completion_count"] or 0) > 0]
    count_fields = (
        "overall_truncation_count",
        "initial_truncation_count",
        "final_stage_truncation_count",
        "direct_raw_truncation_count",
        "stage0_truncation_count",
    )
    completions = sum(int(row["completion_count"] or 0) for row in materialized)
    telemetry = sum(int(row["telemetry_count"] or 0) for row in materialized)
    eval_count = sum(int(row["eval_count"] or 0) for row in materialized)
    missing_prediction = sum(
        int(row["missing_prediction_count"] or 0) for row in materialized
    )
    blank_primary = sum(int(row["blank_primary_count"] or 0) for row in materialized)
    totals = {
        field: sum(int(row[field] or 0) for row in materialized)
        for field in count_fields
    }
    macro: dict[str, float | None] = {}
    for field in count_fields:
        rates = [
            int(row[field] or 0) / int(row["completion_count"])
            for row in materialized
            if int(row["completion_count"] or 0)
        ]
        macro[field.removesuffix("_count") + "_rate"] = fmean(rates) if rates else None
    missing_rates = [
        int(row["missing_prediction_count"] or 0) / int(row["eval_count"])
        for row in materialized
        if int(row["eval_count"] or 0)
    ]
    blank_rates = [
        int(row["blank_primary_count"] or 0) / int(row["completion_count"])
        for row in materialized
        if int(row["completion_count"] or 0)
    ]
    return {
        "cells": len(materialized),
        "scored_cells": sum(row.get("score_created_at") is not None for row in materialized),
        "active_cells": sum(str(row.get("status")) == "Running" for row in materialized),
        "completions": completions,
        "telemetry_coverage": _rate(telemetry, completions),
        "eval_coverage": _rate(eval_count, completions),
        "missing_prediction_micro_rate": _rate(missing_prediction, eval_count),
        "blank_primary_micro_rate": _rate(blank_primary, completions),
        "macro_missing_prediction_rate": fmean(missing_rates) if missing_rates else None,
        "macro_blank_primary_rate": fmean(blank_rates) if blank_rates else None,
        **{
            field.removesuffix("_count") + "_micro_rate": _rate(total, completions)
            for field, total in totals.items()
        },
        **{"macro_" + key: value for key, value in macro.items()},
    }


def _group_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["size"])].append(row)
        groups[(row["family"], row["size"], row["domain"])].append(row)
    return {
        "by_family_size": {
            "/".join(key): _aggregate(value)
            for key, value in sorted(groups.items())
            if len(key) == 2
        },
        "by_family_size_domain": {
            "/".join(key): _aggregate(value)
            for key, value in sorted(groups.items())
            if len(key) == 3
        },
    }


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):6.2f}%"


def _print_summary(database: str, report: dict[str, Any]) -> None:
    print(f"\n[{database}]")
    print(
        "family size cells scored active completions telemetry "
        "initial(macro/micro) final(macro/micro) missing blank"
    )
    for key, item in report["by_family_size"].items():
        family, size = key.split("/")
        print(
            f"{family:4s} {size:5s} {item['cells']:5d} {item['scored_cells']:6d} "
            f"{item['active_cells']:6d} {item['completions']:11d} "
            f"{_pct(item['telemetry_coverage'])} "
            f"{_pct(item['macro_initial_truncation_rate'])}/"
            f"{_pct(item['initial_truncation_micro_rate'])} "
            f"{_pct(item['macro_final_stage_truncation_rate'])}/"
            f"{_pct(item['final_stage_truncation_micro_rate'])} "
            f"{_pct(item['macro_missing_prediction_rate'])} "
            f"{_pct(item['macro_blank_primary_rate'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--databases", nargs="+", default=list(DEFAULT_DATABASES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output: dict[str, Any] = {
        "selection": (
            "latest scored task per database/family/size/strict46 cell; "
            "post-fix G1i additionally uses latest Running telemetry"
        ),
        "databases": {},
    }
    for database in args.databases:
        try:
            rows = _load_database(database)
        except psycopg.Error as exc:
            output["databases"][database] = {"error": str(exc)}
            print(f"\n[{database}] ERROR {exc}")
            continue
        report = {"selected_rows": rows, **_group_report(rows)}
        output["databases"][database] = report
        _print_summary(database, report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
