from __future__ import annotations

"""Run LLM wrong-answer checker over eval results in database.

Typical usage:
  python -m src.bin.run_llm_checker --task-id 123
  python -m src.bin.run_llm_checker --dataset gsm8k --model rwkv7-g1-1.5b
"""

import argparse
from typing import Sequence

from src.eval.checkers.llm_checker import run_llm_checker_db
from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService
from src.eval.scheduler.config import DEFAULT_DB_CONFIG


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run llm_checker over eval results in database")
    parser.add_argument(
        "--task-id",
        help="Task ID to run checker on",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name (used with --model to find latest task)",
    )
    parser.add_argument(
        "--model",
        help="Model name (used with --dataset to find latest task)",
    )
    parser.add_argument(
        "--evaluator",
        help="Evaluator name to filter tasks (e.g., eval_free_response, eval_multi_choice_cot)",
    )
    parser.add_argument(
        "--checker-type",
        default="llm_checker",
        help="Checker type identifier (default: llm_checker)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show how many samples need checking; do not call the checker.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()

    task_id: str | None = args.task_id

    # 如果没有直接指定 task_id，通过 dataset + model 查找
    if task_id is None:
        if not args.dataset or not args.model:
            print("⚠️  Must specify --task-id or both --dataset and --model")
            return 1

        ctx = service.get_resume_context(
            dataset=args.dataset,
            model=args.model,
            is_param_search=False,
            evaluator=args.evaluator,
        )
        if ctx.task_id is None:
            print(f"⚠️  No task found for dataset={args.dataset} model={args.model}")
            return 1
        task_id = str(ctx.task_id)
        print(f"📍 Found task_id={task_id} for dataset={args.dataset} model={args.model}")

    if args.dry_run:
        count = service.count_failed_evals_for_checker(
            task_id=task_id,
            checker_type=args.checker_type,
        )
        print(f"🧪 dry-run: {count} samples need checking for task {task_id}")
        return 0

    updated = run_llm_checker_db(
        task_id=task_id,
        checker_type=args.checker_type,
    )
    return 0 if updated >= 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
