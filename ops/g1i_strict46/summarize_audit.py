#!/usr/bin/env python3
"""Print a compact human-readable view of a strict-46 audit JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit", type=Path)
    parser.add_argument("--since-task-id", type=int, default=0)
    args = parser.parse_args()
    data = json.loads(args.audit.read_text(encoding="utf-8"))
    active = data.get("active_target_tasks", [])
    invalid_scored = data.get("invalid_scored_tasks", [])
    print(
        f"coverage={data['valid_complete']}/{data['target_cells']} "
        f"remaining={data['remaining']} active={len(active)} "
        f"invalid_scored={len(invalid_scored)}"
    )
    for model, status in data["models"].items():
        print(f"model {model}: complete={status['complete']} missing={status['missing']}")
    for task in data.get("valid_task_rows", []):
        if int(task["task_id"]) < args.since_task_id:
            continue
        print(
            "valid "
            f"task={task['task_id']} model={task['model_name']} "
            f"benchmark={task['benchmark']} metrics={task['metrics']} "
            f"completions={task['completion_count']} eval={task['eval_count']} "
            f"blank={task['blank_primary_generation_count']} "
            f"missing={task['missing_prediction_count']} "
            f"truncation={task['overall_truncation_count']} "
            f"orphan_close={task.get('leading_orphan_close_count', 0)}"
        )
        if task.get("raw_answer_counts"):
            print(
                f"  raw_answers={task['raw_answer_counts']} "
                f"predicted={task.get('predicted_label_counts', {})} "
                f"reference={task.get('reference_label_counts', {})}"
            )
    issues_by_task = {
        int(item["task_id"]): item.get("reasons") or []
        for item in data.get("active_protocol_issues", [])
    }
    for task in active:
        reasons = issues_by_task.get(int(task["task_id"]), [])
        print(
            "active "
            f"task={task['task_id']} model={task['model_name']} "
            f"benchmark={task['benchmark_name']}__{task['benchmark_split']} "
            f"completions={task['completion_count']} eval={task['eval_count']} "
            f"issues={','.join(reasons) if reasons else '-'}"
        )
    for signal in data.get("choice_bias_signals", []):
        if int(signal["task_id"]) < args.since_task_id:
            continue
        print(f"choice_bias {signal}")
    for comparison in data.get("reference_differences_over_5pp", []):
        if int(comparison["g1i_task_id"]) < args.since_task_id:
            continue
        print(f"reference_delta {comparison}")
    for comparison in data.get("curve_inversions_over_5pp", []):
        if max(
            int(comparison["smaller_task_id"]),
            int(comparison["larger_task_id"]),
        ) < args.since_task_id:
            continue
        print(f"curve_inversion {comparison}")


if __name__ == "__main__":
    main()
