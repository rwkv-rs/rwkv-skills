#!/usr/bin/env python3
"""Produce oracle answer-or-null outputs and exact/contain summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate oracle answer-or-null upper bound.")
    parser.add_argument("--chunks-jsonl", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(value: Any) -> str:
    return str(value).strip()


def main() -> None:
    args = parse_args()
    chunks_path = Path(args.chunks_jsonl)
    tasks_path = Path(args.tasks_jsonl)
    out_path = Path(args.out_jsonl)
    summary_path = Path(args.summary_json)

    chunks = load_jsonl(chunks_path)
    tasks = load_jsonl(tasks_path)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        positive_chunks = {int(chunk_id) for chunk_id in task["positive_chunks"]}
        for chunk in chunks:
            chunk_id = int(chunk["chunk_id"])
            label = normalize(task["answer"]) if chunk_id in positive_chunks else "null"
            rows.append(
                {
                    "task_id": task["id"],
                    "chunk_id": chunk_id,
                    "label": label,
                    "answer": label,
                    "is_positive_chunk": chunk_id in positive_chunks,
                    "source": "oracle_positive_chunks",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_task: dict[str, dict[str, Any]] = {}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_task.setdefault(str(row["task_id"]), []).append(row)

    for task in tasks:
        task_id = str(task["id"])
        candidates = [row["answer"] for row in rows_by_task.get(task_id, []) if row["answer"] != "null"]
        prediction = candidates[0] if candidates else "null"
        gold = normalize(task["answer"])
        by_task[task_id] = {
            "candidate_count": len(candidates),
            "prediction": prediction,
            "exact": prediction == gold,
            "contain": prediction != "null" and (gold in prediction or prediction in gold),
        }

    tp = fp = fn = tn = 0
    for row in rows:
        positive = bool(row["is_positive_chunk"])
        predicted = row["answer"] != "null"
        if positive and predicted:
            tp += 1
        elif positive and not predicted:
            fn += 1
        elif not positive and predicted:
            fp += 1
        else:
            tn += 1

    exact = sum(1 for row in by_task.values() if row["exact"])
    contain = sum(1 for row in by_task.values() if row["exact"] or row["contain"])
    summary = {
        "stage": "oracle_answer_or_null_upper_bound",
        "chunks_jsonl": str(chunks_path.resolve()),
        "tasks_jsonl": str(tasks_path.resolve()),
        "outputs_jsonl": str(out_path.resolve()),
        "chunk_count": len(chunks),
        "task_count": len(tasks),
        "task_chunk_pairs": len(rows),
        "positive_pairs": tp + fn,
        "negative_pairs": tn + fp,
        "per_chunk_label_accuracy": (tp + tn) / len(rows) if rows else 0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "final_first_candidate": {
            "exact": exact,
            "contain": contain,
            "task_count": len(tasks),
            "exact_rate": exact / len(tasks) if tasks else 0,
            "contain_rate": contain / len(tasks) if tasks else 0,
            "no_candidate_count": sum(1 for row in by_task.values() if row["candidate_count"] == 0),
        },
        "by_task": by_task,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
