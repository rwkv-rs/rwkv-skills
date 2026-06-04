#!/usr/bin/env python3
"""Build evidence-QA task JSONL from external task definitions and chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ALLOWED_ANSWER_FORMATS = {"scalar_string", "scalar_number_string"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute positive chunks for long-doc evidence QA tasks.")
    parser.add_argument("--chunks-jsonl", required=True)
    parser.add_argument("--task-defs-jsonl", required=True)
    parser.add_argument("--out-tasks-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--chunk-source", default="")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--allow-answer-missing", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def match_positive_rule(text: str, rule: dict[str, Any]) -> bool:
    if "all" in rule:
        return all(str(term) in text for term in rule["all"])
    if "any" in rule:
        return any(str(term) in text for term in rule["any"])
    if "not" in rule and isinstance(rule["not"], dict):
        return not match_positive_rule(text, rule["not"])
    raise ValueError(f"unsupported positive_rule: {rule!r}")


def validate_task(row: dict[str, Any], index: int) -> None:
    required = ["id", "question", "answer", "answer_format", "positive_rule"]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"task #{index} missing required keys: {missing}")
    if row["answer_format"] not in ALLOWED_ANSWER_FORMATS:
        raise ValueError(f"task {row['id']!r} unsupported answer_format: {row['answer_format']!r}")
    if not isinstance(row["positive_rule"], dict):
        raise ValueError(f"task {row['id']!r} positive_rule must be an object")


def main() -> None:
    args = parse_args()
    chunks = load_jsonl(args.chunks_jsonl)
    task_defs = load_jsonl(args.task_defs_jsonl)
    chunks_by_id = {int(row["chunk_id"]): row for row in chunks}

    output_tasks: list[dict[str, Any]] = []
    empty_positive: list[str] = []
    answer_missing: list[str] = []
    positive_counts: dict[str, int] = {}
    answer_hit_counts: dict[str, int] = {}

    for index, task in enumerate(task_defs, 1):
        validate_task(task, index)
        task_id = str(task["id"])
        rule = task["positive_rule"]
        positive_chunks = [
            int(chunk["chunk_id"])
            for chunk in chunks
            if match_positive_rule(str(chunk.get("text", "")), rule)
        ]
        positive_counts[task_id] = len(positive_chunks)
        if not positive_chunks:
            empty_positive.append(task_id)

        answer = str(task["answer"])
        answer_hits = [
            chunk_id
            for chunk_id in positive_chunks
            if answer in str(chunks_by_id[chunk_id].get("text", ""))
        ]
        answer_hit_counts[task_id] = len(answer_hits)
        if not answer_hits:
            answer_missing.append(task_id)

        output = dict(task)
        output["positive_chunks"] = positive_chunks
        output["null_rule"] = "chunk does not contain the positive_rule terms"
        output["chunking"] = {
            "source": args.chunk_source or str(Path(args.chunks_jsonl).name),
            "positive_chunks_recomputed_from": "positive_rule",
        }
        output_tasks.append(output)

    write_jsonl(args.out_tasks_jsonl, output_tasks)

    lengths = [int(row.get("char_count", len(str(row.get("text", ""))))) for row in chunks]
    summary = {
        "chunks_jsonl": str(Path(args.chunks_jsonl).resolve()),
        "task_defs_jsonl": str(Path(args.task_defs_jsonl).resolve()),
        "out_tasks_jsonl": str(Path(args.out_tasks_jsonl).resolve()),
        "chunk_count": len(chunks),
        "task_count": len(output_tasks),
        "chunk_char_count": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": (sum(lengths) / len(lengths)) if lengths else 0,
        },
        "empty_positive_tasks": empty_positive,
        "answer_missing_from_positive_tasks": answer_missing,
        "positive_chunk_count": {
            "min": min(positive_counts.values()) if positive_counts else 0,
            "max": max(positive_counts.values()) if positive_counts else 0,
            "avg": (sum(positive_counts.values()) / len(positive_counts)) if positive_counts else 0,
            "by_task": positive_counts,
        },
        "answer_hit_count_in_positive_chunks": {
            "min": min(answer_hit_counts.values()) if answer_hit_counts else 0,
            "max": max(answer_hit_counts.values()) if answer_hit_counts else 0,
            "avg": (sum(answer_hit_counts.values()) / len(answer_hit_counts)) if answer_hit_counts else 0,
            "by_task": answer_hit_counts,
        },
        "oracle_passed": not empty_positive and not answer_missing,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if empty_positive and not args.allow_empty:
        raise SystemExit(1)
    if answer_missing and not args.allow_answer_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
