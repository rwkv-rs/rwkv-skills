#!/usr/bin/env python3
from __future__ import annotations

"""Restore maths scores from already persisted completion rows.

This is intended for recovery after score/eval rows were deleted while the
model generations remain in the DB.  The script uses the current TOML sampling
plan, filters persisted completions to the required attempt keys, re-runs the
official free-response evaluator, and writes eval/score rows back to Postgres.
"""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import psycopg
from psycopg.rows import dict_row

MATH_BENCHMARKS = (
    "aime24",
    "aime25",
    "algebra222",
    "answer_judge",
    "asdiv",
    "beyond_aime",
    "brumo25",
    "college_math",
    "gsm_plus",
    "hendrycks_math",
    "hmmt_feb25",
    "math_500",
    "math_odyssey",
    "mawps",
    "minerva_math",
    "omni_math",
    "polymath",
    "polymath_all",
    "simpleqa",
    "svamp",
    "amc23",
    "comp_math_24_25",
    "gaokao2023en",
    "gsm8k",
    "olympiadbench",
)

PRIMARY_EVALUATORS = ("free_response", "free_response_judge")


@dataclass(frozen=True)
class Candidate:
    task_id: int
    benchmark_name: str
    benchmark_split: str
    model_name: str
    evaluator: str
    completed_completions: int
    score_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Write eval/score rows. Default is dry-run.")
    parser.add_argument("--include-scored", action="store_true", help="Also recompute tasks that already have scores.")
    parser.add_argument("--task-id", action="append", type=int, help="Limit to explicit task_id. Repeatable.")
    parser.add_argument("--only-evaluator", choices=PRIMARY_EVALUATORS, help="Limit to one primary evaluator.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum task rows to process after filtering.")
    parser.add_argument("--judge-model", help="Override judge model.")
    parser.add_argument("--judge-api-key", help="Override judge API key.")
    parser.add_argument("--judge-base-url", help="Override judge base URL.")
    parser.add_argument("--judge-max-workers", type=int, help="Override judge worker count.")
    parser.add_argument("--judge-max-tokens", type=int, help="Override judge max completion tokens.")
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
        "b.benchmark_name = ANY(%s)",
        "(m.model_name LIKE 'rwkv7-g1f-%%' OR m.model_name LIKE 'rwkv7-g1g-%%')",
        "t.evaluator = ANY(%s)",
    ]
    params: list[Any] = [list(MATH_BENCHMARKS), list(PRIMARY_EVALUATORS)]
    if not opts.include_scored:
        where.append("NOT EXISTS (SELECT 1 FROM scores s WHERE s.task_id = t.task_id)")
    if opts.only_evaluator:
        where.append("t.evaluator = %s")
        params.append(str(opts.only_evaluator))
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
            count(c.completions_id) FILTER (WHERE c.status = 'Completed') AS completed_completions,
            count(DISTINCT s.score_id) AS score_count
        FROM task t
        JOIN benchmark b ON b.benchmark_id = t.benchmark_id
        JOIN model m ON m.model_id = t.model_id
        LEFT JOIN completions c ON c.task_id = t.task_id
        LEFT JOIN scores s ON s.task_id = t.task_id
        WHERE {" AND ".join(where)}
        GROUP BY t.task_id, b.benchmark_name, b.benchmark_split, m.model_name, t.evaluator
        HAVING count(c.completions_id) FILTER (WHERE c.status = 'Completed') > 0
        ORDER BY b.benchmark_name, b.benchmark_split, m.model_name, t.evaluator, t.task_id DESC
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
            score_count=int(row["score_count"] or 0),
        )
        for row in rows
    ]


def current_plan_for(candidate: Candidate):
    from src.eval.field_common import resolve_configured_k_plan
    from src.eval.maths.common import count_free_answer_records
    from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
    from src.eval.scheduler.dataset_utils import make_dataset_slug

    slug = make_dataset_slug(candidate.benchmark_name, candidate.benchmark_split)
    dataset_path = resolve_or_prepare_dataset(slug, verbose=False)
    total_records = count_free_answer_records(dataset_path, None)
    args = SimpleNamespace(pass_k=None, avg_k=None, max_samples=None)
    k_plan = resolve_configured_k_plan(
        slug=slug,
        model_name=candidate.model_name,
        dataset_len=total_records,
        args=args,
    )
    return slug, dataset_path, total_records, k_plan


def filter_completion_payloads(service, task_id: int, expected_keys: set[tuple[int, int, int]]) -> list[dict[str, Any]]:
    payloads = service.list_completion_payloads(task_id=str(task_id), status="Completed")
    selected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for payload in payloads:
        key = (
            int(payload.get("sample_index", -1)),
            int(payload.get("repeat_index", -1)),
            int(payload.get("pass_index", 0)),
        )
        if key in expected_keys and key not in selected:
            selected[key] = payload
    return [selected[key] for key in sorted(expected_keys) if key in selected]


def delete_dependent_rows(task_ids: Sequence[int]) -> dict[str, int]:
    if not task_ids:
        return {"checker": 0, "eval": 0, "scores": 0, "completions": 0, "task": 0}
    with connect_from_env() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH target_completions AS (
                    SELECT completions_id FROM completions WHERE task_id = ANY(%s)
                )
                DELETE FROM checker
                WHERE completions_id IN (SELECT completions_id FROM target_completions)
                """,
                (list(task_ids),),
            )
            checker = int(cur.rowcount)
            cur.execute(
                """
                WITH target_completions AS (
                    SELECT completions_id FROM completions WHERE task_id = ANY(%s)
                )
                DELETE FROM eval
                WHERE completions_id IN (SELECT completions_id FROM target_completions)
                """,
                (list(task_ids),),
            )
            eval_rows = int(cur.rowcount)
            cur.execute("DELETE FROM scores WHERE task_id = ANY(%s)", (list(task_ids),))
            scores = int(cur.rowcount)
            cur.execute("DELETE FROM completions WHERE task_id = ANY(%s)", (list(task_ids),))
            completions = int(cur.rowcount)
            cur.execute("DELETE FROM task WHERE task_id = ANY(%s)", (list(task_ids),))
            tasks = int(cur.rowcount)
        conn.commit()
    return {"checker": checker, "eval": eval_rows, "scores": scores, "completions": completions, "task": tasks}


def delete_primary_eval_rows(task_id: int) -> dict[str, int]:
    with connect_from_env() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH target_completions AS (
                    SELECT completions_id FROM completions WHERE task_id = %s
                )
                DELETE FROM checker
                WHERE completions_id IN (SELECT completions_id FROM target_completions)
                """,
                (int(task_id),),
            )
            checker = int(cur.rowcount)
            cur.execute(
                """
                WITH target_completions AS (
                    SELECT completions_id FROM completions WHERE task_id = %s
                )
                DELETE FROM eval
                WHERE completions_id IN (SELECT completions_id FROM target_completions)
                """,
                (int(task_id),),
            )
            eval_rows = int(cur.rowcount)
        conn.commit()
    return {"checker": checker, "eval": eval_rows}


def find_strategy_children(parent_task_id: int) -> list[int]:
    with connect_from_env() as conn:
        rows = list(
            conn.execute(
                """
                SELECT task_id
                FROM task
                WHERE evaluator LIKE '%%:strategy_%%'
                  AND "desc" LIKE %s
                ORDER BY task_id
                """,
                (f"%parent_task_id={int(parent_task_id)}%",),
            )
        )
    return [int(row["task_id"]) for row in rows]


def build_judge(candidate: Candidate, opts: argparse.Namespace, slug: str):
    if candidate.evaluator != "free_response_judge":
        return None
    from src.eval.benchmark_config import resolve_benchmark_model_config
    from src.eval.maths.common import build_llm_judge

    judge = build_llm_judge(
        judge_model=opts.judge_model,
        judge_api_key=opts.judge_api_key,
        judge_base_url=opts.judge_base_url,
        judge_max_workers=opts.judge_max_workers,
        judge_max_tokens=opts.judge_max_tokens,
        required=True,
    )
    root_config = resolve_benchmark_model_config(slug, candidate.model_name, stage=None)
    if root_config is not None and root_config.judge_prompt_template:
        judge.config.prompt_template = root_config.judge_prompt_template
    return judge


def restore_one(candidate: Candidate, opts: argparse.Namespace, *, execute: bool) -> tuple[str, str]:
    from src.db.database import init_db
    from src.db.eval_db_service import EvalDbService
    from src.eval.benchmark_registry import CoTMode
    from src.eval.execution_plan import build_attempt_keys
    from src.eval.field_common import build_plan_task_details
    from src.eval.metrics.free_response import (
        attach_strategy_task_ids,
        build_grouped_metrics_payload,
        evaluate_free_response,
    )
    from src.eval.results.payloads import make_score_payload
    from src.eval.scheduler.config import DEFAULT_DB_CONFIG

    slug, dataset_path, problem_count, k_plan = current_plan_for(candidate)
    expected_keys = {key.as_tuple() for key in build_attempt_keys(k_plan.plan, max_pass_k=1)}
    init_db(DEFAULT_DB_CONFIG)
    service = EvalDbService()
    filtered = filter_completion_payloads(service, candidate.task_id, expected_keys)
    if len(filtered) != len(expected_keys):
        return (
            "skip_incomplete",
            f"{candidate.task_id} {slug} {candidate.model_name} {candidate.evaluator} "
            f"have={len(filtered)} expected={len(expected_keys)}",
        )
    if not execute:
        return (
            "dry_run",
            f"{candidate.task_id} {slug} {candidate.model_name} {candidate.evaluator} "
            f"samples={len(filtered)} avg_k={k_plan.avg_k}",
        )

    child_task_ids = find_strategy_children(candidate.task_id)
    if child_task_ids:
        delete_dependent_rows(child_task_ids)
    delete_primary_eval_rows(candidate.task_id)
    judge = build_judge(candidate, opts, slug)
    evaluation = evaluate_free_response(filtered, dataset_path=dataset_path, judge=judge)
    strategy_task_ids = service.ingest_eval_payload_groups(
        task_id=str(candidate.task_id),
        completion_payloads=filtered,
        payloads_by_group=evaluation.payloads_by_group,
        primary_group=evaluation.primary_group,
    )
    metrics_payload, metric_details = build_grouped_metrics_payload(
        evaluation,
        pass_k=k_plan.pass_k,
        avg_k=k_plan.avg_k,
        report_pass_k=k_plan.report_pass_k,
        report_avg_k=k_plan.report_avg_k,
    )
    attach_strategy_task_ids(metrics_payload, strategy_task_ids)
    task_details = build_plan_task_details(k_plan.plan, cot_mode=CoTMode.COT.value, prompt_profile="normal")
    task_details.update(metric_details)
    score_payload = make_score_payload(
        slug,
        is_cot=True,
        model_name=candidate.model_name,
        metrics=metrics_payload,
        samples=evaluation.samples,
        problems=problem_count,
        task=candidate.evaluator,
        task_details=task_details,
        extra={"cot_mode": CoTMode.COT.value, "prompt_profile": "normal"},
    )
    service.record_score_payload(payload=score_payload, task_id=str(candidate.task_id))
    return (
        "restored",
        f"{candidate.task_id} {slug} {candidate.model_name} {candidate.evaluator} "
        f"samples={evaluation.samples} primary={evaluation.primary_group}",
    )


def choose_latest_per_key(candidates: Sequence[Candidate]) -> list[Candidate]:
    chosen: dict[tuple[str, str, str, str], Candidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.task_id, reverse=True):
        key = (
            candidate.benchmark_name,
            candidate.benchmark_split,
            candidate.model_name,
            candidate.evaluator,
        )
        chosen.setdefault(key, candidate)
    return sorted(chosen.values(), key=lambda item: item.task_id)


def main(argv: Sequence[str] | None = None) -> int:
    opts = parse_args(argv)
    from src.eval.env_config import load_env_file

    load_env_file()
    candidates = choose_latest_per_key(load_candidates(opts))
    if opts.limit and opts.limit > 0:
        candidates = candidates[: int(opts.limit)]
    counts: dict[str, int] = {}
    for candidate in candidates:
        status, message = restore_one(candidate, opts, execute=bool(opts.execute))
        counts[status] = counts.get(status, 0) + 1
        print(f"{status}: {message}", flush=True)
    print("summary")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
