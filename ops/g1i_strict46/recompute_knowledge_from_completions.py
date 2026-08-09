#!/usr/bin/env python3
"""Read-only Knowledge score replay using the current raw-answer adapter.

This command never writes to Postgres.  It is intentionally suitable for
auditing historical generations: scores produced before the raw-completions
protocol cutoff remain diagnostic-only even when the new adapter can extract
more answers from them.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from src.db.database import init_db
from src.db.eval_db_service import EvalDbService
from src.eval.metrics.at_k import compute_avg_at_k
from src.eval.metrics.multi_choice import evaluate_multiple_choice
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import make_dataset_slug


DEFAULT_DB_NAME = "chase_rwkv_skills_frontend46_20260804"
RAW_PROTOCOL_CUTOFF = datetime(2026, 8, 6, 5, 10, 0)
EXPECTED_PROMPT_SUFFIX = "Assistant: <think></think>\nThe answer is"


def _metric_value(metrics: object) -> float | None:
    if not isinstance(metrics, dict):
        return None
    for key in ("avg@1", "avg@4", "avg@8", "avg@16", "avg@32", "avg@64", "accuracy"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", action="append", type=int)
    parser.add_argument(
        "--audit",
        type=Path,
        help=(
            "Optionally load task ids from audit.invalid_scored_tasks.  Only "
            "Knowledge rows whose sole invalid reason is the historical raw "
            "protocol cutoff are selected; generation/protocol failures stay "
            "excluded."
        ),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        help=(
            "Load latest present Knowledge task ids from audit_cot_addons.py "
            "output. Only rows belonging to --dbname are replayed."
        ),
    )
    parser.add_argument("--dbname", default=DEFAULT_DB_NAME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    task_ids = list(args.task_id or [])
    if args.audit:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        for row in audit.get("invalid_scored_tasks", []):
            if row.get("domain") != "knowledge":
                continue
            if row.get("invalid_reasons") != [
                "generation_predates_raw_completions_protocol_fix"
            ]:
                continue
            task_ids.append(int(row["task_id"]))
    if args.inventory:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        for cell in inventory.get("cells", []):
            latest = cell.get("latest")
            if not isinstance(latest, dict) or latest.get("database") != args.dbname:
                continue
            task_ids.append(int(latest["task_id"]))
    task_ids = sorted(set(task_ids))
    if not task_ids:
        parser.error("provide --task-id or --audit with replayable Knowledge rows")

    init_db(replace(DEFAULT_DB_CONFIG, dbname=args.dbname))
    service = EvalDbService()
    report: list[dict[str, object]] = []
    for task_id in task_ids:
        bundle = service.get_task_bundle(task_id=str(task_id))
        if not bundle:
            report.append({"task_id": task_id, "eligible": False, "reasons": ["task_not_found"]})
            continue
        task = bundle["task"]
        model = bundle["model"]
        benchmark = bundle["benchmark"]
        payloads = service.list_completion_payloads(task_id=str(task_id), status="Completed")
        source_score = service.get_score_payload(task_id=str(task_id)) or {}
        reasons: list[str] = []
        sampling_config = task.get("sampling_config") or {}
        cot_mode = str(sampling_config.get("cot_mode") or "").lower().replace("-", "_")
        is_cot = cot_mode in {"cot", "true", "1"}
        is_g1i = "g1i" in str(model.get("model_name") or "").lower()
        created_at = task.get("created_at")
        if (
            is_g1i
            and (not isinstance(created_at, datetime) or created_at < RAW_PROTOCOL_CUTOFF)
        ):
            reasons.append("generation_predates_raw_completions_protocol_fix")
        if "_naive" not in str(task.get("evaluator") or ""):
            reasons.append("evaluator_not_naive")
        if sampling_config.get("prompt_profile") != "naive":
            reasons.append("prompt_profile_not_naive")
        prompts = [
            str(payload.get("prompt1") or payload.get("strategy_a_prompt") or "")
            for payload in payloads
        ]
        if is_cot:
            if not prompts or any(not prompt.endswith("Assistant: <think") for prompt in prompts):
                reasons.append("prompt_not_strict_naive_cot")
            primary = [
                str(
                    payload.get("direct_raw_completion")
                    or payload.get("completion2")
                    or payload.get("completion1")
                    or payload.get("strategy_a_completion")
                    or ""
                )
                for payload in payloads
            ]
            if not primary or any(not value for value in primary):
                reasons.append("missing_primary_completion")
        else:
            expected_suffix = EXPECTED_PROMPT_SUFFIX if is_g1i else "Assistant: The answer is"
            if not prompts or any(not prompt.endswith(expected_suffix) for prompt in prompts):
                reasons.append("prompt_not_architecture_naive_nocot")
            primary = [
                str(
                    payload.get("direct_raw_completion")
                    or payload.get("completion1")
                    or payload.get("strategy_a_completion")
                    or ""
                )
                for payload in payloads
            ]
            if not primary or any(not value for value in primary):
                reasons.append("missing_primary_completion")
        expected = sampling_config.get("effective_sample_count")
        coordinates = {
            (
                int(payload["sample_index"]),
                int(payload["repeat_index"]),
                int(payload.get("pass_index", 0)),
            )
            for payload in payloads
        }
        if not isinstance(expected, int) or len(payloads) != expected or len(coordinates) != expected:
            reasons.append("incomplete_coordinates")

        slug = make_dataset_slug(
            str(benchmark["benchmark_name"]),
            str(benchmark.get("benchmark_split") or ""),
        )
        dataset_path = resolve_or_prepare_dataset(slug, verbose=False)
        evaluation = evaluate_multiple_choice(payloads, dataset_path=dataset_path)
        avg_k = int(float(sampling_config.get("avg_k") or 1))
        replay_metrics = compute_avg_at_k(evaluation.rows, (avg_k,))
        replay_value = float(replay_metrics.get(f"avg@{avg_k}", evaluation.accuracy))
        stored_value = _metric_value(source_score.get("metrics"))
        missing_predictions = sum(
            not bool(payload.get("answer")) for payload in evaluation.payloads
        )
        missing_rate = missing_predictions / len(evaluation.payloads) if evaluation.payloads else 1.0
        if missing_rate > 0.05:
            reasons.append("recomputed_missing_rate_gt_5pct")
        score_changed = stored_value is None or abs(replay_value - stored_value) > 1e-12
        report.append(
            {
                "task_id": task_id,
                "model": model["model_name"],
                "benchmark": slug,
                "mode": "cot" if is_cot else "no_cot",
                "created_at": created_at.isoformat(sep=" ") if isinstance(created_at, datetime) else None,
                "rows": len(payloads),
                "valid_predictions": sum(bool(payload.get("answer")) for payload in evaluation.payloads),
                "missing_predictions": missing_predictions,
                "missing_prediction_rate": missing_rate,
                "stored_score": stored_value,
                "recomputed_score": replay_value,
                "delta_pp": None if stored_value is None else (replay_value - stored_value) * 100.0,
                "stored_score_differs_from_replay": score_changed,
                "strict_reuse_eligible": not reasons,
                "replay_eligible_except_cutoff": reasons == [
                    "generation_predates_raw_completions_protocol_fix"
                ],
                "reasons": reasons,
            }
        )

    output = {
        "database": args.dbname,
        "adapter": "current_raw_multiple_choice_adapter",
        "read_only": True,
        "tasks": report,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary:
        print(
            json.dumps(
                {
                    "database": args.dbname,
                    "tasks": len(report),
                    "strict_reuse_eligible": sum(
                        bool(row.get("strict_reuse_eligible")) for row in report
                    ),
                    "replay_eligible_except_cutoff": sum(
                        bool(row.get("replay_eligible_except_cutoff")) for row in report
                    ),
                    "ineligible": sum(
                        not bool(row.get("strict_reuse_eligible"))
                        and not bool(row.get("replay_eligible_except_cutoff"))
                        for row in report
                    ),
                    "reason_counts": {
                        reason: sum(reason in row.get("reasons", []) for row in report)
                        for reason in sorted(
                            {
                                reason
                                for row in report
                                for reason in row.get("reasons", [])
                            }
                        )
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
