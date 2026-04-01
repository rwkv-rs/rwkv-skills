from __future__ import annotations

"""DB-backed wrong-answer checker orchestration."""

from typing import Any

from src.db.eval_db_service import EvalDbService
from src.eval.checkers.llm_checker import run_llm_checker_rows


def run_checker_for_task(
    *,
    service: EvalDbService,
    task_id: str,
    model_name: str,
) -> int:
    bundle = service.get_task_bundle(task_id=task_id)
    benchmark = bundle.get("benchmark") if isinstance(bundle, dict) else None
    benchmark_name = str((benchmark or {}).get("benchmark_name") or "")
    dataset_split = str((benchmark or {}).get("benchmark_split") or "")

    failed_rows = service.list_eval_records_for_space(
        task_id=task_id,
        only_wrong=True,
        include_context=True,
    )
    if not failed_rows:
        return 0
    existing_keys = service.list_checker_keys(task_id=task_id)

    checker_inputs: list[dict[str, Any]] = []
    for row in failed_rows:
        key = (
            int(row.get("sample_index", 0)),
            int(row.get("repeat_index", 0)),
            int(row.get("pass_index", 0)),
        )
        if key in existing_keys:
            continue
        checker_inputs.append(
            {
                "benchmark_name": benchmark_name,
                "dataset_split": dataset_split,
                "sample_index": int(row.get("sample_index", 0)),
                "repeat_index": int(row.get("repeat_index", 0)),
                "pass_index": int(row.get("pass_index", 0)),
                "context": row.get("context"),
                "answer": row.get("answer"),
                "ref_answer": row.get("ref_answer"),
                "model_name": model_name,
            }
        )

    checker_rows = run_llm_checker_rows(checker_inputs)
    if not checker_rows:
        return 0
    return service.ingest_checker_payloads(payloads=checker_rows, task_id=task_id)


__all__ = ["run_checker_for_task"]
