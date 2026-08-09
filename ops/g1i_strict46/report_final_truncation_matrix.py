#!/usr/bin/env python3
"""Read-only final-output truncation matrix for G1g/G1h/G1i.

Math intentionally excludes the first reasoning generation.  Its denominator
uses the generation actually submitted to the evaluator: stage 2 when present,
otherwise an accepted strategy-A response or a legacy single-stage final
response.  Other domains use the same evaluator-facing rule.  Missing
finish-reason telemetry is reported as missing coverage, never as a clean stop.

All rates come exclusively from the latest matching task group persisted in
the evaluation database (task/completions/context).  Backend-wide vLLM request
metrics are deliberately not an input because they mix tasks and stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from ops.g1i_strict46.audit_current import (
    CODING,
    INSTRUCTION,
    KNOWLEDGE,
    MATH,
    REFERENCE_MODELS,
)
from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


G1I_MODELS = {
    "1.5B": "rwkv7-g1i-1.5b-20260805-ctx16384",
    "2.9B": "rwkv7-g1i-2.9b-20260805-ctx16384",
    "7.2B": "rwkv7-g1i-7.2b-20260805-ctx16384",
    "13.3B": "rwkv7-g1i-13.3b-20260805-ctx16384",
}
MODELS = {
    **{name: (family, size) for family, rows in REFERENCE_MODELS.items() for size, name in rows.items()},
    **{name: ("G1i", size) for size, name in G1I_MODELS.items()},
}
TARGET_DOMAIN = {
    **{key: "knowledge" for key in KNOWLEDGE},
    **{key: "math" for key in MATH},
    **{key: "coding" for key in CODING},
    **{key: "instruction_following" for key in INSTRUCTION},
}
TARGET_NAMES = sorted({name for name, _split in TARGET_DOMAIN})
TARGET_ALIASES = {
    # Historical G1g/G1h used the dataset's physical split name.  Strict-46
    # calls the same SimpleQA Verified rows ``test`` in its logical target.
    ("simpleqa", "verified"): ("simpleqa", "test"),
}
TRUNC_REASONS = ("length", "max_length", "max_tokens")
STATS_FIELDS = (
    "completion_count",
    "any_output_telemetry_count",
    "math_final_output_count",
    "math_final_truncated_count",
    "ordinary_final_output_count",
    "ordinary_final_truncated_count",
    "knowledge_final_output_count",
    "knowledge_final_truncated_count",
)
SIZE_ORDER = {"1.5B": 0, "2.9B": 1, "7.2B": 2, "13.3B": 3}
FAMILY_ORDER = {"G1g": 0, "G1h": 1, "G1i": 2}
# Increment whenever STATS_SQL or the evaluator-facing final-output precedence
# changes.  Exact-task caches are immutable only within the same definition.
STATS_SCHEMA_VERSION = "final-evaluator-facing-v2"

META_SQL = """
SELECT m.model_name, b.benchmark_name, b.benchmark_split,
       t.task_id, t.status, t.created_at AS task_created_at,
       t.evaluator, t.sampling_config, s.cot_mode,
       s.created_at AS score_created_at
FROM task t
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN scores s ON s.task_id = t.task_id
WHERE m.model_name = ANY(%s)
  AND b.benchmark_name = ANY(%s)
  AND t.evaluator NOT LIKE '%%:strategy_%%'
ORDER BY t.task_id
"""

STATS_SQL = r"""
SELECT c.task_id,
       COUNT(*) AS completion_count,
       COUNT(*) FILTER (
         WHERE c.context #> '{stages,1}' IS NOT NULL
            OR c.context ? 'direct_raw_finish_reason'
            OR c.context #> '{stages,0,stop_reason}' IS NOT NULL
            OR c.context #> '{stats,truncated}' IS NOT NULL
       ) AS any_output_telemetry_count,
       COUNT(*) FILTER (
         WHERE c.context #> '{stages,1}' IS NOT NULL
            OR (
              c.context #> '{stages,1}' IS NULL
              AND (
                c.context ? 'strategy_a_stop_reason'
                OR c.context #> '{strategy_a,stop_reason}' IS NOT NULL
                OR c.context ? 'direct_raw_finish_reason'
                OR c.context #> '{stages,0,stop_reason}' IS NOT NULL
                OR c.context #> '{stats,truncated}' IS NOT NULL
              )
            )
       ) AS math_final_output_count,
       COUNT(*) FILTER (
         WHERE CASE
           -- The evaluator consumes stage 2 when it exists.  Stage 1 is the
           -- reasoning generation and must never enter the final-only rate.
           WHEN c.context #> '{stages,1}' IS NOT NULL THEN
             COALESCE((c.context #>> '{stats,stage2,truncated}')::boolean, FALSE)
             OR COALESCE(c.context #>> '{stages,1,stop_reason}', '') = ANY(%s)
           -- With strategy-A filtering, a correct full response is accepted
           -- directly and no two-stage record is created.  That response is
           -- evaluator-facing final output, not the discarded reasoning stage.
           WHEN c.context ? 'strategy_a_stop_reason' THEN
             COALESCE((c.context #>> '{stats,strategy_a,truncated}')::boolean, FALSE)
             OR COALESCE(c.context->>'strategy_a_stop_reason', '') = ANY(%s)
           WHEN c.context #> '{strategy_a,stop_reason}' IS NOT NULL THEN
             COALESCE((c.context #>> '{stats,strategy_a,truncated}')::boolean, FALSE)
             OR COALESCE(c.context #>> '{strategy_a,stop_reason}', '') = ANY(%s)
           -- Preserve the actual final generation for legacy single-stage
           -- tasks, but only as a fallback when no stage-2/strategy-A output
           -- exists.  This does not count stage 1 of a two-stage task.
           WHEN c.context ? 'direct_raw_finish_reason' THEN
             COALESCE(c.context->>'direct_raw_finish_reason', '') = ANY(%s)
           WHEN c.context #> '{stages,0,stop_reason}' IS NOT NULL THEN
             COALESCE((c.context #>> '{stats,stage1,truncated}')::boolean, FALSE)
             OR COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
             OR COALESCE(c.context #>> '{stages,0,stop_reason}', '') = ANY(%s)
           ELSE COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
         END
       ) AS math_final_truncated_count,
       COUNT(*) FILTER (
         WHERE c.context ? 'direct_raw_finish_reason'
            OR c.context #> '{stages,0,stop_reason}' IS NOT NULL
            OR c.context #> '{stats,truncated}' IS NOT NULL
       ) AS ordinary_final_output_count,
       COUNT(*) FILTER (
         WHERE CASE
           WHEN c.context ? 'direct_raw_finish_reason' THEN
             COALESCE(c.context->>'direct_raw_finish_reason', '') = ANY(%s)
           WHEN c.context #> '{stages,0,stop_reason}' IS NOT NULL THEN
             COALESCE(c.context #>> '{stages,0,stop_reason}', '') = ANY(%s)
           ELSE COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
         END
       ) AS ordinary_final_truncated_count,
       COUNT(*) FILTER (
         WHERE c.context ? 'direct_raw_finish_reason'
            OR c.context #> '{format_bridges,answer_stage_raw_stop_reason}' IS NOT NULL
            OR c.context #> '{format_bridges,strategy_b_final_raw_stop_reason}' IS NOT NULL
            OR c.context ? 'strategy_a_stop_reason'
            OR c.context #> '{strategy_a,stop_reason}' IS NOT NULL
       ) AS knowledge_final_output_count,
       COUNT(*) FILTER (
         WHERE CASE
           WHEN c.context #> '{format_bridges,answer_stage_raw_stop_reason}' IS NOT NULL THEN
             COALESCE(c.context #>> '{format_bridges,answer_stage_raw_stop_reason}', '') = ANY(%s)
             AND COALESCE(c.context #>> '{format_bridges,answer_stage_raw_completion}', '')
                 !~ '^[[:space:]]*[\[(]?[A-Z][\])]?[[:space:]]*$'
           WHEN c.context #> '{format_bridges,strategy_b_final_raw_stop_reason}' IS NOT NULL THEN
             COALESCE(c.context #>> '{format_bridges,strategy_b_final_raw_stop_reason}', '') = ANY(%s)
             AND COALESCE(c.context #>> '{format_bridges,strategy_b_final_raw_completion}', '')
                 !~ '^[[:space:]]*[\[(]?[A-Z][\])]?[[:space:]]*$'
           WHEN c.context ? 'direct_raw_finish_reason' THEN
             COALESCE(c.context->>'direct_raw_finish_reason', '') = ANY(%s)
             AND COALESCE(c.context->>'direct_raw_completion', '')
                 !~ '^[[:space:]]*[\[(]?[A-Z][\])]?[[:space:]]*$'
           WHEN c.context ? 'strategy_a_stop_reason' THEN
             COALESCE(c.context->>'strategy_a_stop_reason', '') = ANY(%s)
           ELSE
             COALESCE(c.context #>> '{strategy_a,stop_reason}', '') = ANY(%s)
         END
       ) AS knowledge_final_truncated_count
FROM completions c
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
"""


def normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    if mode in {"nocot", "no_cot", "false", "0"}:
        return "no_cot"
    if mode in {"cot", "true", "1"}:
        return "cot"
    return mode


def infer_mode(row: dict[str, Any], domain: str) -> str:
    mode = normalize_mode(row.get("cot_mode"))
    if mode:
        return mode
    config = row.get("sampling_config") or {}
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            config = {}
    if isinstance(config, dict):
        for key in ("cot_mode", "mode"):
            mode = normalize_mode(config.get(key))
            if mode:
                return mode
    evaluator = str(row.get("evaluator") or "").lower()
    if "nocot" in evaluator or "no_cot" in evaluator:
        return "no_cot"
    if re.search(r"(^|_)cot($|_)", evaluator):
        return "cot"
    return "cot" if domain == "math" else "no_cot"


def select_latest(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        model_name = str(row["model_name"])
        model = MODELS.get(model_name)
        source_key = (str(row["benchmark_name"]), str(row["benchmark_split"]))
        target_key = TARGET_ALIASES.get(source_key, source_key)
        target = TARGET_DOMAIN.get(target_key)
        if model is None or target is None:
            continue
        # Strategy A/B rows are auxiliary members of a Math evaluation group.
        # The unsuffixed evaluator is the group root that owns the score and
        # provenance; auxiliary rows must never shadow it merely because their
        # task ids were allocated later.
        if ":strategy_" in str(row.get("evaluator") or ""):
            continue
        family, size = model
        mode = infer_mode(row, target)
        expected = "cot" if target == "math" else "no_cot"
        protocol_compatible = mode == expected
        row.update(
            family=family,
            size=size,
            domain=target,
            mode=mode,
            required_mode=expected,
            protocol_compatible=protocol_compatible,
            source_benchmark_name=source_key[0],
            source_benchmark_split=source_key[1],
            benchmark_name=target_key[0],
            benchmark_split=target_key[1],
        )
        key = (family, size, target, f"{target_key[0]}/{target_key[1]}")
        current = selected.get(key)
        # Select the absolute newest persisted task group first.  Protocol
        # compatibility is metadata on that newest row, never a preference
        # that silently falls back to an older task.  If the newest task is
        # incompatible, aggregate() leaves it out of the strictly comparable
        # rate and reports the cell as incompatible.
        ordering = (row.get("task_created_at"), int(row["task_id"]))
        current_ordering = (
            current.get("task_created_at"),
            int(current["task_id"]),
        ) if current is not None else None
        if current is None or ordering > current_ordering:
            selected[key] = row
    return list(selected.values())


def load_database(
    database: str,
    *,
    cached_rows: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    config = replace(DEFAULT_DB_CONFIG, dbname=database)
    with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(META_SQL, (list(MODELS), TARGET_NAMES))
            rows = select_latest(dict(row) for row in cur.fetchall())
            if not rows:
                return rows
            cached_rows = cached_rows or {}
            refresh_ids: list[int] = []
            for row in rows:
                task_id = int(row["task_id"])
                cached = cached_rows.get(task_id)
                refresh = (
                    cached is None
                    or str(row.get("status")) == "Running"
                    or str(cached.get("status")) != str(row.get("status"))
                    or str(cached.get("score_created_at") or "")
                    != str(row.get("score_created_at") or "")
                )
                if refresh:
                    refresh_ids.append(task_id)
                else:
                    row.update({field: cached.get(field) for field in STATS_FIELDS})
            if refresh_ids:
                cur.execute(
                    STATS_SQL,
                    (
                        list(TRUNC_REASONS), list(TRUNC_REASONS), list(TRUNC_REASONS),
                        list(TRUNC_REASONS), list(TRUNC_REASONS), list(TRUNC_REASONS),
                        list(TRUNC_REASONS), list(TRUNC_REASONS), list(TRUNC_REASONS),
                        list(TRUNC_REASONS), list(TRUNC_REASONS), list(TRUNC_REASONS),
                        refresh_ids,
                    ),
                )
                stats = {int(row["task_id"]): dict(row) for row in cur.fetchall()}
                for row in rows:
                    task_id = int(row["task_id"])
                    if task_id in refresh_ids:
                        row.update(stats.get(task_id, {}))
            return rows


def load_exact_task_stats_cache(path: Path | None) -> dict[str, dict[int, dict[str, Any]]]:
    """Load statistics keyed by exact database and task id.

    This cache can only avoid recomputing immutable completion statistics for
    the *same* task.  It cannot select a task, substitute an older task, or
    bridge databases. Running tasks and any task whose status/score timestamp
    changed are always refreshed from PostgreSQL.
    """

    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if payload.get("stats_schema_version") != STATS_SCHEMA_VERSION:
        return {}
    cache: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in payload.get("selected_rows") or []:
        database = str(row.get("database") or "")
        task_id = row.get("task_id")
        if database and task_id is not None:
            cache[database][int(task_id)] = row
    return dict(cache)


def merge_databases(database_rows: list[tuple[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    """Prefer the first database containing a given architecture cell."""
    selected: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for database, rows in database_rows:
        for row in rows:
            key = (
                row["family"], row["size"], row["domain"], row["mode"],
                f"{row['benchmark_name']}/{row['benchmark_split']}",
            )
            if key not in selected:
                row["database"] = database
                selected[key] = row
    return list(selected.values())


def _final_counts(row: dict[str, Any]) -> tuple[int, int, int]:
    """Return persisted completions, observable final outputs, and truncations."""

    total = int(row.get("completion_count") or 0)
    domain = str(row.get("domain") or "")
    if domain == "math":
        observable = int(row.get("math_final_output_count") or 0)
        truncated = int(row.get("math_final_truncated_count") or 0)
    elif domain == "knowledge":
        observable = int(row.get("knowledge_final_output_count") or 0)
        truncated = int(row.get("knowledge_final_truncated_count") or 0)
    else:
        observable = int(row.get("ordinary_final_output_count") or 0)
        truncated = int(row.get("ordinary_final_truncated_count") or 0)
    return total, observable, truncated


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate final-output telemetry, including across mixed domains."""

    total = sum(int(row.get("completion_count") or 0) for row in rows)
    counts = [_final_counts(row) for row in rows]
    observable = sum(item[1] for item in counts)
    truncated = sum(item[2] for item in counts)
    compatible_rows = [
        row for row in rows if bool(row.get("protocol_compatible", True))
    ]
    compatible_counts = [_final_counts(row) for row in compatible_rows]
    compatible_total = sum(
        int(row.get("completion_count") or 0) for row in compatible_rows
    )
    compatible_observable = sum(item[1] for item in compatible_counts)
    compatible_truncated = sum(item[2] for item in compatible_counts)
    return {
        "cells": len(rows),
        "protocol_compatible_cells": len(compatible_rows),
        "protocol_incompatible_cells": sum(
            not bool(row.get("protocol_compatible", True)) for row in rows
        ),
        "complete_cells": sum(
            str(row.get("status")) == "Completed"
            and row.get("score_created_at") is not None
            for row in rows
        ),
        "failed_cells": sum(str(row.get("status")) == "Failed" for row in rows),
        "running_cells": sum(str(row.get("status")) == "Running" for row in rows),
        "completion_count": total,
        "observable_final_output_count": observable,
        "telemetry_coverage": observable / total if total else None,
        "final_truncated_count": truncated,
        "final_truncation_rate_all_completions": truncated / total if total else None,
        "conditional_final_truncation_rate": truncated / observable if observable else None,
        "protocol_compatible_completion_count": compatible_total,
        "protocol_compatible_observable_final_output_count": compatible_observable,
        "protocol_compatible_final_truncated_count": compatible_truncated,
        "protocol_compatible_final_truncation_rate_all_completions": (
            compatible_truncated / compatible_total if compatible_total else None
        ),
        "protocol_compatible_conditional_final_truncation_rate": (
            compatible_truncated / compatible_observable
            if compatible_observable else None
        ),
        "task_ids": [int(row["task_id"]) for row in rows],
    }


def build_summary_tables(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the two quantitative tables requested for final delivery.

    ``truncation_vs_parameter_size`` retains architecture so that a size trend
    is never created by mixing G1g/G1h/G1i. ``truncation_vs_g1x`` contains an
    overall row and one row per domain for every architecture family.
    """

    by_size: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_family_domain: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row["family"])
        size = str(row["size"])
        domain = str(row["domain"])
        by_size[(family, size)].append(row)
        by_family_domain[(family, domain)].append(row)
        by_family[family].append(row)

    size_rows = [
        {"family": family, "parameter_size": size, **aggregate(group)}
        for (family, size), group in sorted(
            by_size.items(),
            key=lambda item: (
                FAMILY_ORDER.get(item[0][0], 99),
                SIZE_ORDER.get(item[0][1], 99),
            ),
        )
    ]
    family_rows: list[dict[str, Any]] = []
    for family in sorted(by_family, key=lambda value: FAMILY_ORDER.get(value, 99)):
        family_rows.append({"family": family, "domain": "all", **aggregate(by_family[family])})
        for domain in ("knowledge", "math", "coding", "instruction_following"):
            group = by_family_domain.get((family, domain), [])
            if group:
                family_rows.append({"family": family, "domain": domain, **aggregate(group)})
    return {
        "truncation_vs_parameter_size": size_rows,
        "truncation_vs_g1x": family_rows,
    }


SUMMARY_FIELDS = [
    "family", "parameter_size", "domain", "cells",
    "protocol_compatible_cells", "protocol_incompatible_cells",
    "complete_cells", "running_cells", "failed_cells", "completion_count", "observable_final_output_count",
    "final_truncated_count",
    "final_truncation_rate_all_completions",
    "conditional_final_truncation_rate",
    "protocol_compatible_completion_count",
    "protocol_compatible_observable_final_output_count",
    "protocol_compatible_final_truncated_count",
    "protocol_compatible_final_truncation_rate_all_completions",
    "protocol_compatible_conditional_final_truncation_rate",
    "task_ids",
]


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["task_ids"] = ";".join(str(value) for value in row["task_ids"])
            writer.writerow(rendered)


def _format_rate(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{float(value) * 100:.4f}%"


def write_summary_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    dimension: str,
) -> None:
    """Write a compact, researcher-facing rendering of a truncation summary.

    The overall rate keeps every completion in the denominator. The
    conditional rate only uses rows whose final-stage stop telemetry is
    observable, so missing telemetry is never silently treated as clean.
    """

    if dimension not in {"parameter_size", "domain"}:
        raise ValueError(f"unsupported truncation dimension: {dimension}")
    dimension_label = "参数量" if dimension == "parameter_size" else "领域"
    header = [
        "架构",
        dimension_label,
        "单元",
        "完成",
        "协议一致",
        "completions",
        "可观测最终输出",
        "遥测覆盖率",
        "最终截断数",
        "总体截断率",
        "条件截断率",
        "可比总体截断率",
        "可比条件截断率",
    ]
    # The public table intentionally omits the telemetry-coverage column; the
    # underlying counts remain in the JSON audit artifact.
    del header[7]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("family") or "—"),
            str(row.get(dimension) or "—"),
            str(row.get("cells") or 0),
            str(row.get("complete_cells") or 0),
            str(row.get("protocol_compatible_cells") or 0),
            str(row.get("completion_count") or 0),
            str(row.get("observable_final_output_count") or 0),
            str(row.get("final_truncated_count") or 0),
            _format_rate(row.get("final_truncation_rate_all_completions")),
            _format_rate(row.get("conditional_final_truncation_rate")),
            _format_rate(
                row.get(
                    "protocol_compatible_final_truncation_rate_all_completions"
                )
            ),
            _format_rate(
                row.get(
                    "protocol_compatible_conditional_final_truncation_rate"
                )
            ),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--databases", nargs="+",
        default=["chase_rwkv_skills_frontend46_20260804"],
        help=(
            "source databases in priority order; defaults to the current strict-46 "
            "database only so historical fallback tasks never enter the table"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size-table", type=Path)
    parser.add_argument("--family-table", type=Path)
    args = parser.parse_args()
    stats_cache = load_exact_task_stats_cache(args.output)
    loaded: list[tuple[str, list[dict[str, Any]]]] = []
    errors: dict[str, str] = {}
    for database in args.databases:
        try:
            loaded.append(
                (
                    database,
                    load_database(database, cached_rows=stats_cache.get(database)),
                )
            )
        except psycopg.Error as exc:
            errors[database] = str(exc)
    rows = merge_databases(loaded)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["size"], row["domain"], row["mode"])].append(row)
    matrix = {"/".join(key): aggregate(value) for key, value in sorted(groups.items())}
    summary_tables = build_summary_tables(rows)
    result = {
        "stats_schema_version": STATS_SCHEMA_VERSION,
        "definition": {
            "source": (
                "latest matching task group persisted in the evaluation database; "
                "vLLM backend metrics excluded"
            ),
            "selection": (
                "per architecture, parameter size, benchmark and required mode: "
                "absolute latest task by task_created_at then task_id; "
                "an incompatible latest task is reported as incompatible and "
                "never replaced by an older compatible task"
            ),
            "math": "final answer stage only; first reasoning stage excluded",
            "other_domains": "actual evaluator-facing generation",
            "primary_rate": "truncated final outputs / all persisted completions",
            "conditional_rate": "truncated final outputs / outputs with observable final finish telemetry",
            "required_modes": "Knowledge/Coding/Instruction Following=NoCoT; Math=CoT",
            "reference_exception": (
                "G1g/G1h LiveCodeBench may use the displayed historical CoT task when no "
                "NoCoT task exists; such cells are counted as protocol-incompatible"
            ),
        },
        "database_errors": errors,
        "matrix": matrix,
        "summary_tables": summary_tables,
        "selected_rows": rows,
    }
    for key, item in matrix.items():
        rate = item["final_truncation_rate_all_completions"]
        conditional = item["conditional_final_truncation_rate"]
        pct = "n/a" if rate is None else f"{100 * rate:.4f}%"
        conditional_pct = "n/a" if conditional is None else f"{100 * conditional:.4f}%"
        print(
            f"{key:45s} all={pct:>10s} conditional={conditional_pct:>10s} "
            f"({item['final_truncated_count']}/{item['observable_final_output_count']}) "
            f"coverage={item['telemetry_coverage']!s} cells={item['cells']} "
            f"complete={item['complete_cells']} running={item['running_cells']}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.size_table:
        write_summary_csv(args.size_table, summary_tables["truncation_vs_parameter_size"])
    if args.family_table:
        write_summary_csv(args.family_table, summary_tables["truncation_vs_g1x"])


if __name__ == "__main__":
    main()
