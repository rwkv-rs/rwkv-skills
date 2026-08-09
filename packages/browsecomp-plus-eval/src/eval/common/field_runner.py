"""Shared orchestration helpers for the field evaluation runners.

The knowledge / maths / coding / instruction-following runners share the same
CLI surface and the same ``TaskRunController`` stage boilerplate around their
field-specific pipelines. These helpers capture that duplication so each
runner's ``main()`` keeps only what is genuinely field-specific.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Iterator


def resolve_prompt_profile(raw: str | None, job_name: str) -> str:
    """Resolve the prompt-profile label from an explicit flag or the job name."""
    if raw:
        return raw
    if job_name.endswith("_naive"):
        return "naive"
    return "normal"


def add_common_runner_args(
    parser: argparse.ArgumentParser,
    *,
    batch_size_default: int,
    db_write_queue_default: int | None,
) -> None:
    """Register the CLI flags shared by every field runner.

    ``batch_size_default`` and ``db_write_queue_default`` differ per runner and
    are passed in so each runner keeps its exact prior default.
    """
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--batch-size", type=int, default=batch_size_default, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit source questions for quick runs")
    parser.add_argument(
        "--db-write-queue",
        type=int,
        default=db_write_queue_default,
        help="DB completion write queue max size",
    )
    parser.add_argument(
        "--avg-k",
        type=float,
        action="append",
        help="avg@k values to compute from generated samples (defaults come from configs/<benchmark>.toml)",
    )


@contextmanager
def attempt_stage(runtime, writer) -> Iterator[None]:
    """Guard the sampling/attempt stage, releasing the writer on failure."""
    try:
        yield
    except Exception:
        runtime.handle_attempt_stage_failure(writer)
        raise


@contextmanager
def scoring_stage(runtime) -> Iterator[None]:
    """Guard the evaluation/scoring stage, marking the task failed on error."""
    try:
        yield
    except Exception as exc:
        runtime.fail_task(error=str(exc))
        raise
