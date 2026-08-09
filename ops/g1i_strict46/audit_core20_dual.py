#!/usr/bin/env python3
"""Audit the Core20 score matrix for G1g/G1i CoT/NoCoT coverage.

This is deliberately read-only.  It selects the newest scored task for each
model/benchmark/cot-mode cell, then checks the persisted evidence cardinality
and answer artifacts before a cell is considered usable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


G1I_MODELS = (
    "rwkv7-g1i-1.5b-20260805-ctx16384",
    "rwkv7-g1i-2.9b-20260805-ctx16384",
    "rwkv7-g1i-7.2b-20260805-ctx16384",
    "rwkv7-g1i-13.3b-20260805-ctx16384",
)

G1G_MODELS = (
    "rwkv7-g1g-1.5b-20260526-ctx8192",
    "rwkv7-g1g-2.9b-20260526-ctx8192",
    "rwkv7-g1g-7.2b-20260523-ctx8192",
    "rwkv7-g1g-13.3b-20260523-ctx8192",
)

MODEL_FAMILIES = {
    "g1i": G1I_MODELS,
    "g1g": G1G_MODELS,
    "all": G1G_MODELS + G1I_MODELS,
}

TARGETS = (
    ("mmlu", "test", "knowledge"),
    ("mmlu_pro", "test", "knowledge"),
    ("gpqa", "diamond", "knowledge"),
    ("arc_challenge", "test", "knowledge"),
    ("hellaswag", "validation", "knowledge"),
    ("bbh_mcq", "test", "knowledge"),
    ("truthfulqa_mc1", "validation", "knowledge"),
    ("ceval", "test", "knowledge"),
    ("gsm8k", "test", "math"),
    ("math_500", "test", "math"),
    ("aime24", "test", "math"),
    ("aime25", "test", "math"),
    ("amc23", "test", "math"),
    ("olympiadbench", "test", "math"),
    ("human_eval", "test", "coding"),
    ("human_eval_plus", "test", "coding"),
    ("mbpp_plus", "test", "coding"),
    ("livecodebench", "test", "coding"),
    ("ifeval", "test", "instruction_following"),
    ("ifbench", "test", "instruction_following"),
)

TASK_QUERY = """
SELECT
    t.task_id, t.status, t.evaluator, t.created_at AS task_created_at,
    t.sampling_config, m.model_name, b.benchmark_name, b.benchmark_split,
    s.score_id, s.cot_mode, s.metrics, s.created_at AS score_created_at
FROM public.task t
JOIN public.model m ON m.model_id = t.model_id
JOIN public.benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN public.scores s ON s.task_id = t.task_id
WHERE t.is_tmp = FALSE
  AND t.is_param_search = FALSE
  AND m.model_name = ANY(%s)
  AND (b.benchmark_name, b.benchmark_split) IN (
      SELECT * FROM unnest(%s::text[], %s::text[])
  )
ORDER BY t.task_id
"""

STATS_QUERY = """
SELECT
    c.task_id,
    COUNT(*)::bigint AS completion_count,
    COUNT(*) FILTER (WHERE c.status = 'Completed')::bigint AS completed_completion_count,
    COUNT(DISTINCT (c.sample_index, c.avg_repeat_index, c.pass_index))::bigint
        AS distinct_coordinates,
    COUNT(e.eval_id)::bigint AS eval_count,
    COUNT(*) FILTER (WHERE e.fail_reason = 'missing_prediction')::bigint
        AS missing_prediction_count,
    COUNT(*) FILTER (
        WHERE COALESCE(
            NULLIF(BTRIM(c.context->>'direct_raw_completion'), ''),
            NULLIF(BTRIM(c.context #>> '{strategy_a,completion}'), ''),
            NULLIF(BTRIM(c.context #>> '{stages,0,completion}'), ''),
            ''
        ) = ''
    )::bigint AS blank_primary_count,
    COUNT(*) FILTER (
        WHERE COALESCE((c.context #>> '{stats,truncated}')::boolean, FALSE)
           OR COALESCE(c.context->>'direct_raw_finish_reason', '')
                IN ('length', 'max_tokens')
           OR (
               t.evaluator = ANY(%s)
               AND COALESCE(c.context #>> '{stages,0,stop_reason}', '')
                    IN ('length', 'max_length', 'max_tokens')
           )
    )::bigint AS overall_truncation_count,
    COUNT(*) FILTER (
        WHERE COALESCE(
            (c.context #>> '{stats,stage2,truncated}')::boolean,
            FALSE
        )
    )::bigint AS final_stage_truncation_count,
    COUNT(*) FILTER (WHERE c.status <> 'Completed')::bigint
        AS noncompleted_completion_count
    ,COUNT(*) FILTER (
        WHERE COALESCE(
            NULLIF(c.context->>'direct_raw_completion', ''),
            NULLIF(c.context #>> '{strategy_a,completion}', ''),
            NULLIF(c.context #>> '{stages,0,completion}', ''),
            ''
        ) ~ '^[[:space:]]*>?</think>'
    )::bigint AS leading_orphan_close_count
    ,COUNT(*) FILTER (
        WHERE NULLIF(BTRIM(e.answer), '') IS NULL
    )::bigint AS blank_eval_answer_count
FROM public.completions c
JOIN public.task t ON t.task_id = c.task_id
LEFT JOIN public.eval e ON e.completions_id = c.completions_id
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id
"""


def _mode(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    return "CoT" if raw in {"cot", "true", "1"} else "NoCoT"


def _json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return str(value)


def _expected_modes(domain: str) -> tuple[str, ...]:
    return ("NoCoT", "CoT") if domain in {"knowledge", "math"} else ("NoCoT",)


def _expected_evaluators(domain: str, mode: str, benchmark: str) -> tuple[str, ...]:
    if domain == "knowledge":
        return ("multi_choice_cot_naive",) if mode == "CoT" else ("multi_choice_plain_naive",)
    if domain == "math":
        if mode == "CoT":
            return ("free_response_naive", "free_response_judge_naive")
        return ("free_response_plain_naive", "free_response_judge_plain_naive")
    if domain == "coding":
        if benchmark in {"human_eval", "human_eval_plus"}:
            return ("code_human_eval_naive",)
        if benchmark == "mbpp_plus":
            return ("code_mbpp_naive",)
        return ("code_livecodebench_plain_naive",)
    return ("instruction_following_naive",)


def _quality_reasons(
    row: dict[str, Any],
    stats: dict[str, Any] | None,
    *,
    domain: str,
    benchmark: str,
    mode: str,
) -> list[str]:
    reasons: list[str] = []
    config = row.get("sampling_config") or {}
    expected = int(config.get("effective_sample_count") or 0)
    expected_evaluators = _expected_evaluators(domain, mode, benchmark)
    if str(row.get("evaluator") or "") not in expected_evaluators:
        reasons.append(
            f"evaluator:{row.get('evaluator')}!=expected:{','.join(expected_evaluators)}"
        )
    if _mode(row.get("cot_mode")) != mode:
        reasons.append(f"cot_mode:{row.get('cot_mode')}!=expected:{mode}")
    if row.get("status") != "Completed":
        reasons.append(f"task_status:{row.get('status')}")
    if str(config.get("prompt_profile") or "").lower() != "naive":
        reasons.append("not_naive_prompt_profile")
    if stats is None:
        reasons.append("no_completion_stats")
        return reasons
    for field, label in (
        ("completion_count", "completions"),
        ("completed_completion_count", "completed_completions"),
        ("eval_count", "evals"),
        ("distinct_coordinates", "distinct_coordinates"),
    ):
        value = int(stats.get(field) or 0)
        if value != expected and field != "eval_count":
            reasons.append(f"{label}:{value}!=expected:{expected}")
    completion_count = int(stats.get("completion_count") or 0)
    if int(stats.get("eval_count") or 0) != completion_count:
        reasons.append(f"evals:{int(stats.get('eval_count') or 0)}!=completions:{completion_count}")
    # A missing final answer is a valid wrong model outcome for free-response
    # and instruction-following tasks. Multiple-choice tasks are different:
    # their constrained answer stage must emit one legal option, so a missing
    # prediction indicates broken generation/extraction evidence.
    if domain == "knowledge" and int(stats.get("missing_prediction_count") or 0):
        reasons.append(f"missing_prediction:{int(stats['missing_prediction_count'])}")
    if int(stats.get("blank_primary_count") or 0):
        reasons.append(f"blank_primary:{int(stats['blank_primary_count'])}")
    if int(stats.get("noncompleted_completion_count") or 0):
        reasons.append(f"noncompleted_completions:{int(stats['noncompleted_completion_count'])}")
    if int(stats.get("leading_orphan_close_count") or 0):
        reasons.append(
            f"leading_orphan_close:{int(stats['leading_orphan_close_count'])}"
        )
    if domain == "knowledge" and int(stats.get("blank_eval_answer_count") or 0):
        reasons.append(f"blank_eval_answers:{int(stats['blank_eval_answer_count'])}")

    metrics = row.get("metrics")
    numeric_metrics = 0
    if not isinstance(metrics, dict) or not metrics:
        reasons.append("missing_score_metrics")
    else:
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric_metrics += 1
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                reasons.append(f"score_metric_out_of_range:{key}={value}")
        if numeric_metrics == 0:
            reasons.append("no_numeric_score_metrics")
    return reasons


def _quality_warnings(stats: dict[str, Any] | None, *, domain: str) -> list[str]:
    if stats is None:
        return []
    truncation_field = (
        "final_stage_truncation_count"
        if domain == "math"
        else "overall_truncation_count"
    )
    warnings: list[str] = []
    count = int(stats.get(truncation_field) or 0)
    if count:
        warnings.append(f"{truncation_field}:{count}")
    if domain != "knowledge":
        missing = int(stats.get("missing_prediction_count") or 0)
        blank_eval = int(stats.get("blank_eval_answer_count") or 0)
        if missing:
            warnings.append(f"model_missing_prediction:{missing}")
        if blank_eval:
            warnings.append(f"model_blank_answer:{blank_eval}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dbname", default="chase_rwkv_skills_frontend46_20260804")
    parser.add_argument(
        "--family",
        choices=tuple(MODEL_FAMILIES),
        default="g1i",
        help="model family to audit; 'all' covers the eight G1g/G1i models",
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    config = replace(DEFAULT_DB_CONFIG, dbname=args.dbname)
    models = MODEL_FAMILIES[args.family]
    target_names = [name for name, _split, _domain in TARGETS]
    target_splits = [split for _name, split, _domain in TARGETS]
    with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as connection:
        rows = [dict(row) for row in connection.execute(
            TASK_QUERY, (list(models), target_names, target_splits)
        )]
        scored = [row for row in rows if row.get("score_id") is not None]
        selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in scored:
            key = (
                str(row["model_name"]), str(row["benchmark_name"]),
                str(row["benchmark_split"]), str(row["cot_mode"]),
            )
            old = selected.get(key)
            ordering = (str(row.get("score_created_at") or ""), int(row["score_id"]))
            old_ordering = (str(old.get("score_created_at") or ""), int(old["score_id"])) if old else None
            if old is None or ordering > old_ordering:
                selected[key] = row
        active: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in rows:
            if str(row.get("status")) != "Running" or row.get("score_id") is not None:
                continue
            config_mode = _mode((row.get("sampling_config") or {}).get("cot_mode"))
            key = (str(row["model_name"]), str(row["benchmark_name"]), str(row["benchmark_split"]), config_mode)
            active[key] = row

        ids = [int(row["task_id"]) for row in selected.values()]
        stats_by_id: dict[int, dict[str, Any]] = {}
        if ids:
            stats_by_id = {
                int(row["task_id"]): {key: _json(value) for key, value in dict(row).items()}
                for row in connection.execute(
                    STATS_QUERY,
                    (
                        [
                            "code_human_eval_naive",
                            "code_mbpp_naive",
                            "code_livecodebench_plain_naive",
                            "instruction_following_naive",
                        ],
                        ids,
                    ),
                )
            }

    cells: list[dict[str, Any]] = []
    for model in models:
        for benchmark, split, domain in TARGETS:
            for mode in _expected_modes(domain):
                key = (model, benchmark, split, mode)
                row = selected.get(key)
                if row is None:
                    state = "running" if key in active else "missing"
                    cells.append({"model": model, "benchmark": f"{benchmark}__{split}", "domain": domain, "mode": mode, "state": state, "task_id": active.get(key, {}).get("task_id")})
                    continue
                reasons = _quality_reasons(
                    row,
                    stats_by_id.get(int(row["task_id"])),
                    domain=domain,
                    benchmark=benchmark,
                    mode=mode,
                )
                replacement = active.get(key)
                if reasons and replacement is not None:
                    cells.append({
                        "model": model,
                        "benchmark": f"{benchmark}__{split}",
                        "domain": domain,
                        "mode": mode,
                        "state": "running",
                        "task_id": int(replacement["task_id"]),
                        "replaces_invalid_task_id": int(row["task_id"]),
                    })
                    continue
                cells.append({
                    "model": model,
                    "benchmark": f"{benchmark}__{split}",
                    "domain": domain,
                    "mode": mode,
                    "state": "valid" if not reasons else "invalid",
                    "task_id": int(row["task_id"]),
                    "evaluator": row.get("evaluator"),
                    "score_created_at": _json(row.get("score_created_at")),
                    "metrics": _json(row.get("metrics")),
                    "stats": stats_by_id.get(int(row["task_id"])),
                    "reasons": reasons,
                    "warnings": _quality_warnings(
                        stats_by_id.get(int(row["task_id"])), domain=domain
                    ),
                })

    payload = {
        "database": args.dbname,
        "family": args.family,
        "generated_at": datetime.now().astimezone().isoformat(),
        "target_cells": len(cells),
        "valid_cells": sum(cell["state"] == "valid" for cell in cells),
        "invalid_scored_cells": sum(cell["state"] == "invalid" for cell in cells),
        "warning_scored_cells": sum(bool(cell.get("warnings")) for cell in cells),
        "running_cells": sum(cell["state"] == "running" for cell in cells),
        "missing_cells": sum(cell["state"] == "missing" for cell in cells),
        "cells": cells,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(json.dumps({key: payload[key] for key in ("target_cells", "valid_cells", "invalid_scored_cells", "warning_scored_cells", "running_cells", "missing_cells")}, ensure_ascii=False, indent=2))
    for cell in cells:
        if cell["state"] in {"invalid", "missing"}:
            print(json.dumps(cell, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
