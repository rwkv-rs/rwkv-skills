#!/usr/bin/env python3
"""Read-only compact inspection of persisted completions and eval rows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os

import psycopg
from psycopg.rows import dict_row


DB_NAME = "chase_rwkv_skills_frontend46_20260804"


def _classify_fail_reason(reason: str) -> str:
    if not reason:
        return "passed_or_unreported"
    lowered = reason.lower()
    if "wrong answer" in lowered:
        return "wrong_answer"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "syntaxerror" in lowered or "indentationerror" in lowered:
        return "syntax_or_indentation_error"
    return "runtime_or_other_error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_ids", metavar="TASK_ID", type=int, nargs="+")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print one aggregate quality summary per task before sampled rows",
    )
    parser.add_argument(
        "--detailed-fail-reasons",
        action="store_true",
        help="include the full fail-reason histogram in summary output",
    )
    parser.add_argument(
        "--only-final-stage-truncated",
        action="store_true",
        help=(
            "sample only rows whose final-answer recovery stage reached its "
            "length limit"
        ),
    )
    args = parser.parse_args()
    os.environ["RWKV_EVAL_SPACE_DB_DATABASE_NAME"] = DB_NAME

    from src.db.pool import _build_conninfo
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    query = """
        SELECT c.task_id, c.sample_index, c.avg_repeat_index, c.pass_index,
               COALESCE(
                   NULLIF(c.context #>> '{stages,0,prompt}', ''),
                   NULLIF(c.context #>> '{strategy_a,prompt}', ''),
                   ''
               ) AS prompt,
               COALESCE(
                   NULLIF(c.context->>'direct_raw_completion', ''),
                   NULLIF(c.context #>> '{stages,0,completion}', ''),
                   NULLIF(c.context #>> '{strategy_a,completion}', ''),
                   ''
               ) AS completion,
               COALESCE(c.context->>'direct_raw_completion', '')
                   AS direct_raw_completion,
               COALESCE(c.context #>> '{strategy_a,completion}', '')
                   AS strategy_a_completion,
               COALESCE(c.context #>> '{stages,0,completion}', '')
                   AS stage0_completion,
               COALESCE(
                   NULLIF(c.context->>'direct_raw_finish_reason', ''),
                   NULLIF(c.context #>> '{stages,0,stop_reason}', ''),
                   NULLIF(c.context #>> '{stats,termination_reason}', ''),
                   ''
               ) AS finish_reason,
               COALESCE(c.context #>> '{strategy_a,stop_reason}', '')
                   AS strategy_a_finish_reason,
               COALESCE(c.context #>> '{stages,0,stop_reason}', '')
                   AS stage0_finish_reason,
               COALESCE(c.context #>> '{stages,1,stop_reason}', '')
                   AS stage1_finish_reason,
               COALESCE(c.context #>> '{stages,1,prompt}', '')
                   AS final_stage_prompt,
               COALESCE(c.context #>> '{stages,1,completion}', '')
                   AS final_stage_completion,
               COALESCE((c.context #>> '{stats,stage2,truncated}')::boolean, FALSE)
                   AS final_stage_truncated,
               COALESCE(c.context #>> '{strategy_a,completion}', '') <> ''
                   AS has_strategy_a,
               COALESCE(c.context #>> '{stages,0,completion}', '') <> ''
                   AS has_stage0,
               COALESCE(c.context #>> '{stages,1,completion}', '') <> ''
                   AS has_stage1,
               e.answer, e.ref_answer, e.is_passed, e.fail_reason
        FROM completions c
        LEFT JOIN eval e ON e.completions_id = c.completions_id
        WHERE c.task_id = ANY(%s)
        ORDER BY c.task_id, c.sample_index, c.avg_repeat_index, c.pass_index
    """
    with psycopg.connect(_build_conninfo(DEFAULT_DB_CONFIG), row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (list(dict.fromkeys(args.task_ids)),))
            rows = list(cursor.fetchall())

    rows_by_task: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_task.setdefault(int(row["task_id"]), []).append(row)

    if args.summary:
        for task_id in dict.fromkeys(args.task_ids):
            task_rows = rows_by_task.get(task_id, [])
            finish_reasons = Counter(str(row["finish_reason"] or "") for row in task_rows)
            strategy_a_finish_reasons = Counter(
                str(row["strategy_a_finish_reason"] or "")
                for row in task_rows
                if row["has_strategy_a"]
            )
            stage0_finish_reasons = Counter(
                str(row["stage0_finish_reason"] or "")
                for row in task_rows
                if row["has_stage0"]
            )
            stage1_finish_reasons = Counter(
                str(row["stage1_finish_reason"] or "")
                for row in task_rows
                if row["has_stage1"]
            )
            fail_reasons = Counter(str(row["fail_reason"] or "") for row in task_rows)
            fail_reason_categories = Counter(
                _classify_fail_reason(str(row["fail_reason"] or ""))
                for row in task_rows
            )
            blank_completions = sum(not str(row["completion"] or "").strip() for row in task_rows)
            orphan_closes = sum(
                str(row["completion"] or "").lstrip().startswith("></think>")
                for row in task_rows
            )
            passed = sum(row["is_passed"] is True for row in task_rows)
            evaluated = sum(row["is_passed"] is not None for row in task_rows)
            summary = {
                        "type": "summary",
                        "task_id": task_id,
                        "completion_count": len(task_rows),
                        "evaluated_count": evaluated,
                        "passed_count": passed,
                        "pass_rate": passed / evaluated if evaluated else None,
                        "blank_completion_count": blank_completions,
                        "leading_orphan_close_count": orphan_closes,
                        "finish_reason_counts": dict(finish_reasons),
                        "strategy_a_finish_reason_counts": dict(
                            strategy_a_finish_reasons
                        ),
                        "stage0_finish_reason_counts": dict(stage0_finish_reasons),
                        "stage1_finish_reason_counts": dict(stage1_finish_reasons),
                        "strategy_a_count": sum(
                            bool(row["has_strategy_a"]) for row in task_rows
                        ),
                        "staged_count": sum(
                            bool(row["has_stage0"]) for row in task_rows
                        ),
                        "final_stage_count": sum(
                            bool(row["has_stage1"]) for row in task_rows
                        ),
                        "final_stage_truncated_count": sum(
                            bool(row["final_stage_truncated"]) for row in task_rows
                        ),
                        "fail_reason_category_counts": dict(fail_reason_categories),
                    }
            if args.detailed_fail_reasons:
                summary["fail_reason_counts"] = dict(fail_reasons)
            print(json.dumps(summary, ensure_ascii=False))

    seen: dict[int, int] = {}
    for row in rows:
        task_id = int(row["task_id"])
        if args.only_final_stage_truncated and not row["final_stage_truncated"]:
            continue
        count = seen.get(task_id, 0)
        if count >= args.limit:
            continue
        seen[task_id] = count + 1
        prompt = str(row["prompt"] or "")
        completion = str(row["completion"] or "")
        direct_raw_completion = str(row["direct_raw_completion"] or "")
        strategy_a_completion = str(row["strategy_a_completion"] or "")
        stage0_completion = str(row["stage0_completion"] or "")
        final_stage_prompt = str(row["final_stage_prompt"] or "")
        final_stage_completion = str(row["final_stage_completion"] or "")
        print(
            json.dumps(
                {
                    "task_id": task_id,
                    "sample_index": int(row["sample_index"]),
                    "avg_repeat_index": int(row["avg_repeat_index"]),
                    "prompt_tail": prompt[-180:],
                    "completion_head": completion[:240],
                    "completion_tail": completion[-180:],
                    "leading_orphan_close": completion.lstrip().startswith("></think>"),
                    "finish_reason": row["finish_reason"],
                    "strategy_a_finish_reason": row["strategy_a_finish_reason"],
                    "stage0_finish_reason": row["stage0_finish_reason"],
                    "stage1_finish_reason": row["stage1_finish_reason"],
                    "direct_raw_completion_head": direct_raw_completion[:240],
                    "direct_raw_completion_tail": direct_raw_completion[-180:],
                    "strategy_a_completion_head": strategy_a_completion[:240],
                    "strategy_a_completion_tail": strategy_a_completion[-180:],
                    "stage0_completion_head": stage0_completion[:240],
                    "stage0_completion_tail": stage0_completion[-180:],
                    "final_stage_prompt": final_stage_prompt,
                    "final_stage_truncated": bool(row["final_stage_truncated"]),
                    "final_stage_completion_head": final_stage_completion[:240],
                    "final_stage_completion_tail": final_stage_completion[-180:],
                    "answer": row["answer"],
                    "ref_answer": row["ref_answer"],
                    "is_passed": row["is_passed"],
                    "fail_reason": row["fail_reason"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
