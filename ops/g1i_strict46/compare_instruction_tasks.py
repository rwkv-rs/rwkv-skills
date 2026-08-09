"""Read-only per-sample comparison for two instruction-following tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from src.db.pool import _build_conninfo
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


DB_NAME = "chase_rwkv_skills_frontend46_20260804"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATHS = {
    "ifeval": REPO_ROOT / "data" / "ifeval" / "test.jsonl",
    "ifbench": REPO_ROOT / "data" / "ifbench" / "test.jsonl",
}


QUERY = """
SELECT
    c.task_id,
    c.sample_index,
    COUNT(*) AS attempts,
    COUNT(*) FILTER (WHERE e.is_passed) AS passed,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,0,stop_reason}', '')
              IN ('length', 'max_length', 'max_tokens')
    ) AS truncated,
    COUNT(*) FILTER (
        WHERE e.is_passed
          AND COALESCE(c.context #>> '{stages,0,stop_reason}', '')
              IN ('length', 'max_length', 'max_tokens')
    ) AS truncated_passed,
    COUNT(*) FILTER (
        WHERE COALESCE(c.context #>> '{stages,0,completion}', '')
              ~ '^[[:space:]]*>?</think>'
    ) AS leading_orphan_close,
    COUNT(*) FILTER (
        WHERE e.is_passed
          AND COALESCE(c.context #>> '{stages,0,completion}', '')
              ~ '^[[:space:]]*>?</think>'
    ) AS leading_orphan_close_passed,
    ARRAY_AGG(e.fail_reason) FILTER (
        WHERE NOT e.is_passed AND COALESCE(e.fail_reason, '') <> ''
    ) AS fail_reasons
FROM completions c
JOIN eval e ON e.completions_id = c.completions_id
WHERE c.task_id = ANY(%s)
GROUP BY c.task_id, c.sample_index
ORDER BY c.sample_index, c.task_id
"""


TASK_QUERY = """
SELECT t.task_id, m.model_name, b.benchmark_name, b.benchmark_split,
       t.status, t.sampling_config, s.metrics, s.created_at AS score_created_at
FROM task t
JOIN model m ON m.model_id = t.model_id
JOIN benchmark b ON b.benchmark_id = t.benchmark_id
LEFT JOIN scores s ON s.task_id = t.task_id
WHERE t.task_id = ANY(%s)
ORDER BY t.task_id
"""


def _dataset_path_for_tasks(tasks: list[dict[str, Any]]) -> Path:
    """Return the matching dataset for a same-benchmark task comparison."""

    benchmark_names = {str(task["benchmark_name"]).lower() for task in tasks}
    if len(benchmark_names) != 1:
        raise ValueError(
            "instruction task comparison requires one benchmark, got "
            + ", ".join(sorted(benchmark_names))
        )
    benchmark_name = next(iter(benchmark_names))
    try:
        return DATASET_PATHS[benchmark_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported instruction benchmark for comparison: {benchmark_name}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_a", type=int)
    parser.add_argument("task_b", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    task_ids = [args.task_a, args.task_b]

    config = replace(DEFAULT_DB_CONFIG, dbname=DB_NAME)
    with psycopg.connect(_build_conninfo(config), row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(TASK_QUERY, (task_ids,))
            tasks = [dict(row) for row in cursor.fetchall()]
            cursor.execute(QUERY, (task_ids,))
            rows = [dict(row) for row in cursor.fetchall()]

    by_task: dict[int, dict[int, dict[str, Any]]] = {task_id: {} for task_id in task_ids}
    reason_counts: dict[int, Counter[str]] = {task_id: Counter() for task_id in task_ids}
    for row in rows:
        task_id = int(row["task_id"])
        sample_index = int(row["sample_index"])
        by_task[task_id][sample_index] = row
        reason_counts[task_id].update(str(item) for item in (row.get("fail_reasons") or []))

    sample_indices = sorted(set(by_task[args.task_a]) | set(by_task[args.task_b]))
    dataset_path = _dataset_path_for_tasks(tasks)
    dataset_rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    deltas: list[dict[str, Any]] = []
    for sample_index in sample_indices:
        a = by_task[args.task_a].get(sample_index, {})
        b = by_task[args.task_b].get(sample_index, {})
        passed_a = int(a.get("passed") or 0)
        passed_b = int(b.get("passed") or 0)
        deltas.append(
            {
                "sample_index": sample_index,
                "task_a_passed": passed_a,
                "task_b_passed": passed_b,
                "passed_delta": passed_a - passed_b,
                "task_a_truncated": int(a.get("truncated") or 0),
                "task_b_truncated": int(b.get("truncated") or 0),
                "task_a_truncated_passed": int(a.get("truncated_passed") or 0),
                "task_b_truncated_passed": int(b.get("truncated_passed") or 0),
                "task_a_leading_orphan_close": int(a.get("leading_orphan_close") or 0),
                "task_b_leading_orphan_close": int(b.get("leading_orphan_close") or 0),
                "task_a_leading_orphan_close_passed": int(
                    a.get("leading_orphan_close_passed") or 0
                ),
                "task_b_leading_orphan_close_passed": int(
                    b.get("leading_orphan_close_passed") or 0
                ),
            }
        )

    total_a = sum(item["task_a_passed"] for item in deltas)
    total_b = sum(item["task_b_passed"] for item in deltas)
    family_totals: dict[str, dict[str, int]] = {}
    for item in deltas:
        sample_index = int(item["sample_index"])
        instruction_ids = dataset_rows[sample_index].get("instruction_id_list") or ["unknown"]
        for instruction_id in instruction_ids:
            family = str(instruction_id)
            totals = family_totals.setdefault(
                family,
                {"samples": 0, "attempts": 0, "task_a_passed": 0, "task_b_passed": 0},
            )
            totals["samples"] += 1
            totals["attempts"] += int(
                by_task[args.task_a].get(sample_index, {}).get("attempts") or 0
            )
            totals["task_a_passed"] += int(item["task_a_passed"])
            totals["task_b_passed"] += int(item["task_b_passed"])
    instruction_family_comparison = []
    for family, totals in family_totals.items():
        attempts = max(1, int(totals["attempts"]))
        instruction_family_comparison.append(
            {
                "instruction_id": family,
                **totals,
                "task_a_rate": totals["task_a_passed"] / attempts,
                "task_b_rate": totals["task_b_passed"] / attempts,
                "delta_pp": 100.0
                * (totals["task_a_passed"] - totals["task_b_passed"])
                / attempts,
            }
        )
    instruction_family_comparison.sort(
        key=lambda item: abs(float(item["delta_pp"])), reverse=True
    )
    result = {
        "database": DB_NAME,
        "task_a": next((task for task in tasks if int(task["task_id"]) == args.task_a), None),
        "task_b": next((task for task in tasks if int(task["task_id"]) == args.task_b), None),
        "sample_count": len(sample_indices),
        "passed": {str(args.task_a): total_a, str(args.task_b): total_b},
        "passed_delta": total_a - total_b,
        "sample_direction": {
            "task_a_higher": sum(item["passed_delta"] > 0 for item in deltas),
            "equal": sum(item["passed_delta"] == 0 for item in deltas),
            "task_b_higher": sum(item["passed_delta"] < 0 for item in deltas),
        },
        "truncation": {
            str(args.task_a): sum(item["task_a_truncated"] for item in deltas),
            str(args.task_b): sum(item["task_b_truncated"] for item in deltas),
            f"{args.task_a}_passed": sum(item["task_a_truncated_passed"] for item in deltas),
            f"{args.task_b}_passed": sum(item["task_b_truncated_passed"] for item in deltas),
        },
        "leading_orphan_close": {
            str(args.task_a): sum(item["task_a_leading_orphan_close"] for item in deltas),
            str(args.task_b): sum(item["task_b_leading_orphan_close"] for item in deltas),
            f"{args.task_a}_passed": sum(
                item["task_a_leading_orphan_close_passed"] for item in deltas
            ),
            f"{args.task_b}_passed": sum(
                item["task_b_leading_orphan_close_passed"] for item in deltas
            ),
        },
        "fail_reason_counts": {
            str(task_id): reason_counts[task_id].most_common(50) for task_id in task_ids
        },
        "instruction_family_comparison": instruction_family_comparison,
        "largest_task_a_advantages": sorted(
            deltas, key=lambda item: item["passed_delta"], reverse=True
        )[: args.top],
        "largest_task_b_advantages": sorted(
            deltas, key=lambda item: item["passed_delta"]
        )[: args.top],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
