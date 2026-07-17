#!/usr/bin/env python3
from __future__ import annotations

"""Restore knowledge/coding scores from already persisted eval rows.

Some long matrix runs can finish generation and deterministic eval, then block
in the optional LLM checker before recording the final score row. This recovery
script only uses existing eval rows; it does not call models or rerun code tests.
"""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True, slots=True)
class Candidate:
    task_id: int
    benchmark_name: str
    benchmark_split: str
    model_name: str
    evaluator: str
    completed_completions: int
    eval_rows: int
    score_count: int
    cot_mode: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Write score rows. Default is dry-run.")
    parser.add_argument("--include-scored", action="store_true", help="Also recompute tasks that already have scores.")
    parser.add_argument("--task-id", action="append", type=int, help="Limit to explicit task_id. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum tasks to process after filtering.")
    return parser.parse_args(argv)


def connect_from_env():
    from src.eval.env_config import load_env_file
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    load_env_file()
    return psycopg.connect(
        host=DEFAULT_DB_CONFIG.host,
        port=DEFAULT_DB_CONFIG.port,
        user=DEFAULT_DB_CONFIG.user,
        password=DEFAULT_DB_CONFIG.password,
        dbname=DEFAULT_DB_CONFIG.dbname,
        row_factory=dict_row,
    )


def load_candidates(opts: argparse.Namespace) -> list[Candidate]:
    where = [
        "(t.evaluator LIKE 'multi_choice%%' OR t.evaluator LIKE 'code_%%')",
        "(m.model_name LIKE 'rwkv7-g1f-%%' OR m.model_name LIKE 'rwkv7-g1g-%%' OR m.model_name LIKE 'rwkv7-g1h-%%')",
    ]
    params: list[Any] = []
    if not opts.include_scored:
        where.append("NOT EXISTS (SELECT 1 FROM scores s WHERE s.task_id = t.task_id)")
    if opts.task_id:
        where.append("t.task_id = ANY(%s)")
        params.append(list(opts.task_id))
    query = f"""
        SELECT
            t.task_id,
            b.benchmark_name,
            b.benchmark_split,
            m.model_name,
            t.evaluator,
            count(DISTINCT c.completions_id) FILTER (WHERE c.status = 'Completed') AS completed_completions,
            count(DISTINCT e.eval_id) AS eval_rows,
            count(DISTINCT s.score_id) AS score_count,
            coalesce(t.sampling_config->>'cot_mode', '') AS cot_mode
        FROM task t
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        JOIN model m ON m.model_id = t.model_id
        LEFT JOIN completions c ON c.task_id = t.task_id
        LEFT JOIN eval e ON e.completions_id = c.completions_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        WHERE {" AND ".join(where)}
        GROUP BY t.task_id, b.benchmark_name, b.benchmark_split, m.model_name, t.evaluator
        HAVING count(DISTINCT e.eval_id) > 0
        ORDER BY t.task_id
    """
    with connect_from_env() as conn:
        rows = list(conn.execute(query, params))
    return [
        Candidate(
            task_id=int(row["task_id"]),
            benchmark_name=str(row["benchmark_name"]),
            benchmark_split=str(row["benchmark_split"] or ""),
            model_name=str(row["model_name"]),
            evaluator=str(row["evaluator"]),
            completed_completions=int(row["completed_completions"] or 0),
            eval_rows=int(row["eval_rows"] or 0),
            score_count=int(row["score_count"] or 0),
            cot_mode=str(row["cot_mode"] or ""),
        )
        for row in rows
    ]


def choose_latest_per_key(candidates: Sequence[Candidate]) -> list[Candidate]:
    chosen: dict[tuple[str, str, str, str, str], Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.task_id, reverse=True):
        key = (
            candidate.benchmark_name,
            candidate.benchmark_split,
            candidate.model_name,
            candidate.evaluator,
            candidate.cot_mode,
        )
        chosen.setdefault(key, candidate)
    return sorted(chosen.values(), key=lambda item: item.task_id)


def dataset_for(candidate: Candidate):
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import make_dataset_slug

    slug = make_dataset_slug(candidate.benchmark_name, candidate.benchmark_split)
    return slug, resolve_or_prepare_dataset(slug, verbose=False)


def load_k_plan(candidate: Candidate, dataset_len: int):
    from src.eval.field_common import resolve_configured_k_plan

    return resolve_configured_k_plan(
        slug=dataset_slug(candidate),
        model_name=candidate.model_name,
        dataset_len=dataset_len,
        args=SimpleNamespace(pass_k=None, avg_k=None, max_samples=None),
    )


def dataset_slug(candidate: Candidate) -> str:
    from src.eval.scheduler.dataset_utils import make_dataset_slug

    return make_dataset_slug(candidate.benchmark_name, candidate.benchmark_split)


def eval_rows_for_task(task_id: int) -> list[dict[str, Any]]:
    with connect_from_env() as conn:
        rows = list(
            conn.execute(
                """
                SELECT
                    c.sample_index,
                    c.avg_repeat_index AS repeat_index,
                    c.pass_index,
                    e.is_passed,
                    e.answer,
                    e.ref_answer,
                    e.fail_reason
                FROM completions c
                JOIN eval e ON e.completions_id = c.completions_id
                WHERE c.task_id = %s
                ORDER BY c.sample_index, c.avg_repeat_index, c.pass_index, e.eval_id
                """,
                (int(task_id),),
            )
        )
    return [dict(row) for row in rows]


def rows_for_at_k(eval_rows: Sequence[dict[str, Any]]) -> list[tuple[int, int, bool]]:
    rows: list[tuple[int, int, bool]] = []
    for row in eval_rows:
        rows.append((int(row["sample_index"]), int(row["repeat_index"]), bool(row["is_passed"])))
    return rows


def build_metrics(candidate: Candidate, eval_rows: Sequence[dict[str, Any]]) -> tuple[dict[str, float], dict[str, object], int]:
    if candidate.evaluator.startswith("multi_choice"):
        return build_knowledge_metrics(candidate, eval_rows)
    if candidate.evaluator.startswith("code_"):
        return build_coding_metrics(candidate, eval_rows)
    raise ValueError(f"unsupported evaluator for restore: {candidate.evaluator}")


def build_knowledge_metrics(
    candidate: Candidate,
    eval_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, object], int]:
    from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
    from src.eval.k_values import filter_metrics_by_k
    from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k

    _slug, dataset_path = dataset_for(candidate)
    dataset = list(JsonlMultipleChoiceLoader(str(dataset_path)).load())
    k_plan = load_k_plan(candidate, len(dataset))
    at_rows = rows_for_at_k(eval_rows)
    pass_metrics_all = compute_pass_at_k(at_rows, k_plan.pass_k)
    avg_metrics_all = compute_avg_at_k(at_rows, k_plan.avg_k)

    metrics_payload: dict[str, float] = {}
    if not k_plan.report_pass_k and not k_plan.report_avg_k:
        total = len(eval_rows)
        correct = sum(1 for row in eval_rows if bool(row["is_passed"]))
        metrics_payload["accuracy"] = correct / total if total else 0.0
    pass_payload = filter_metrics_by_k(pass_metrics_all, k_plan.report_pass_k, "pass@")
    if k_plan.report_pass_k and not pass_payload:
        pass_payload = pass_metrics_all or {}
    metrics_payload.update(pass_payload)
    avg_payload = filter_metrics_by_k(avg_metrics_all, k_plan.report_avg_k, "avg@")
    if k_plan.report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    metrics_payload.update(avg_payload)

    subject_totals: dict[str | None, list[int]] = defaultdict(lambda: [0, 0])
    for row in eval_rows:
        sample_index = int(row["sample_index"])
        subject = dataset[sample_index].subject if 0 <= sample_index < len(dataset) else None
        totals = subject_totals[subject]
        totals[0] += 1
        if bool(row["is_passed"]):
            totals[1] += 1
    task_details: dict[str, object] = {
        "accuracy_by_subject": {
            subject: (hits / count if count else 0.0)
            for subject, (count, hits) in subject_totals.items()
        },
        "cot_mode": "cot" if candidate.cot_mode == "CoT" else "no_cot",
        "avg_k": k_plan.plan.avg_k,
        "sample_size": k_plan.plan.sample_size,
        "avg_repeat_count": k_plan.plan.repeat_count,
        "effective_sample_count": k_plan.plan.effective_sample_count,
    }
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all
    return metrics_payload, task_details, len(eval_rows)


def build_coding_metrics(
    candidate: Candidate,
    eval_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, object], int]:
    from src.eval.datasets.data_loader.code_generation import JsonlCodeGenerationLoader
    from src.eval.k_values import filter_metrics_by_k
    from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k

    _slug, dataset_path = dataset_for(candidate)
    dataset = list(JsonlCodeGenerationLoader(str(dataset_path)).load())
    k_plan = load_k_plan(candidate, len(dataset))
    at_rows = rows_for_at_k(eval_rows)
    eval_metrics = compute_pass_at_k(at_rows, k_plan.pass_k)
    avg_metrics_all = compute_avg_at_k(at_rows, k_plan.avg_k)

    metrics_payload: dict[str, float] = {}
    pass_payload = filter_metrics_by_k(eval_metrics, k_plan.report_pass_k, "pass@")
    if k_plan.report_pass_k and not pass_payload:
        pass_payload = eval_metrics or {}
    metrics_payload.update(pass_payload)
    avg_payload = filter_metrics_by_k(avg_metrics_all, k_plan.report_avg_k, "avg@")
    if k_plan.report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    metrics_payload.update(avg_payload)

    task_details: dict[str, object] = {
        "cot_mode": "cot" if candidate.cot_mode == "CoT" else "no_cot",
        "avg_k": k_plan.plan.avg_k,
        "sample_size": k_plan.plan.sample_size,
        "avg_repeat_count": k_plan.plan.repeat_count,
        "effective_sample_count": k_plan.plan.effective_sample_count,
    }
    if eval_metrics and pass_payload != eval_metrics:
        task_details["pass_curve"] = eval_metrics
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all
    return metrics_payload, task_details, len(dataset)


def restore_one(candidate: Candidate, *, execute: bool) -> tuple[str, str]:
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    eval_rows = eval_rows_for_task(candidate.task_id)
    if not eval_rows:
        return "skip_no_eval", f"{candidate.task_id} {dataset_slug(candidate)}"
    if candidate.completed_completions and len(eval_rows) < candidate.completed_completions:
        return (
            "skip_incomplete",
            f"{candidate.task_id} {dataset_slug(candidate)} eval_rows={len(eval_rows)} completions={candidate.completed_completions}",
        )

    metrics_payload, task_details, problem_count = build_metrics(candidate, eval_rows)
    message = (
        f"{candidate.task_id} {dataset_slug(candidate)} {candidate.model_name} {candidate.evaluator} "
        f"samples={len(eval_rows)} metrics={metrics_payload}"
    )
    if not execute:
        return "dry_run", message

    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    score_payload = make_score_payload(
        dataset_slug(candidate),
        is_cot=candidate.cot_mode == "CoT",
        model_name=candidate.model_name,
        metrics=metrics_payload,
        samples=len(eval_rows),
        problems=problem_count if candidate.evaluator.startswith("code_") else None,
        task=candidate.evaluator,
        task_details=task_details,
        extra={"cot_mode": task_details["cot_mode"]},
    )
    service.record_score_payload(payload=score_payload, task_id=str(candidate.task_id))
    return "restored", message


def main(argv: Sequence[str] | None = None) -> int:
    opts = parse_args(argv)
    from src.eval.env_config import load_env_file

    load_env_file()
    candidates = choose_latest_per_key(load_candidates(opts))
    if opts.limit and opts.limit > 0:
        candidates = candidates[: int(opts.limit)]
    counts: dict[str, int] = {}
    for candidate in candidates:
        status, message = restore_one(candidate, execute=bool(opts.execute))
        counts[status] = counts.get(status, 0) + 1
        print(f"{status}: {message}", flush=True)
    print("summary")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
