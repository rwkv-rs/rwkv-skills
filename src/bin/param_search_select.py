from __future__ import annotations

"""Select the best param-search grid points and promote scores in DB.

This script reads per-trial scores from the database (is_param_search=1),
selects the best shared grid point *per sampling mode* (normal/simple), and
promotes the corresponding scores into non-param-search records using suffixes:
- {benchmark} (best overall across modes; backward compatible)
- {benchmark}__ps_normal
- {benchmark}__ps_simple
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from src.eval.results.payloads import make_score_payload
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_utils import canonical_slug
from src.db.database import DatabaseManager
from src.db.eval_db_service import EvalDbService


DEFAULT_BENCHMARKS = ("gsm8k_test", "math_500_test")
_MODE_SUFFIXES: dict[str, str] = {"normal": "__ps_normal", "simple": "__ps_simple"}


def _objective_from_metrics(metrics: dict[str, Any]) -> float:
    judge = metrics.get("judge_accuracy")
    if isinstance(judge, (int, float)):
        return float(judge)
    exact = metrics.get("exact_accuracy")
    if isinstance(exact, (int, float)):
        return float(exact)
    return 0.0


def _sample_mode_from_key(key: str) -> str | None:
    try:
        params = json.loads(key)
    except json.JSONDecodeError:
        return None
    if not isinstance(params, dict):
        return None
    mode = params.get("sample_mode")
    if not isinstance(mode, str):
        return None
    mode = mode.strip().lower()
    return mode if mode in _MODE_SUFFIXES else None


def _param_key_from_payload(payload: dict[str, Any]) -> str | None:
    task_details = payload.get("task_details")
    if isinstance(task_details, dict):
        trial_info = task_details.get("param_search_trial")
        if isinstance(trial_info, dict):
            params = trial_info.get("params")
            if isinstance(params, dict):
                return json.dumps(params, sort_keys=True, ensure_ascii=False)
    return None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV param-search selector/promoter")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", help="Ignored (scheduler compatibility)")
    parser.add_argument("--device", help="Ignored (scheduler compatibility)")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to aggregate (default: gsm8k_test math_500_test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite promoted records in DB.",
    )
    return parser.parse_args(argv)


def _should_skip_promotion(
    service: EvalDbService,
    *,
    dataset: str,
    model_name: str,
    overwrite: bool,
) -> bool:
    if overwrite:
        return False
    existing = service.list_scores_by_dataset(dataset=dataset, model=model_name, is_param_search=False)
    return bool(existing)


def _promote_score(
    service: EvalDbService,
    *,
    source_payload: dict[str, Any],
    dest_dataset: str,
    model_name: str,
    overwrite: bool,
) -> None:
    if _should_skip_promotion(service, dataset=dest_dataset, model_name=model_name, overwrite=overwrite):
        return
    task_details = source_payload.get("task_details") if isinstance(source_payload.get("task_details"), dict) else {}
    task_details = dict(task_details)
    task_details["param_search_selected_from"] = {
        "version_id": source_payload.get("version_id"),
        "param_key": _param_key_from_payload(source_payload),
    }
    score_payload = make_score_payload(
        dest_dataset,
        is_cot=bool(source_payload.get("cot", True)),
        model_name=model_name,
        metrics=source_payload.get("metrics") if isinstance(source_payload.get("metrics"), dict) else {},
        samples=int(source_payload.get("samples", 0) or 0),
        problems=source_payload.get("problems"),
        task=str(source_payload.get("task") or "param_search_select"),
        task_details=task_details,
    )
    version_id = service.get_or_create_version(
        job_name="param_search_select",
        job_id=os.environ.get("RWKV_SKILLS_JOB_ID"),
        dataset=str(dest_dataset),
        model=model_name,
        is_param_search=False,
        allow_resume=False,
    )
    os.environ["RWKV_SKILLS_VERSION_ID"] = version_id
    service.record_score_payload(
        payload=score_payload,
        version_id=version_id,
        is_param_search=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not DEFAULT_DB_CONFIG.enabled:
        raise RuntimeError("DB 未启用：当前仅支持数据库写入模式。")
    db = DatabaseManager.instance()
    db.initialize(DEFAULT_DB_CONFIG)
    service = EvalDbService(db)
    model_name = Path(args.model_path).stem
    benchmarks = tuple(canonical_slug(b) for b in args.benchmarks if b)
    if len(benchmarks) < 2:
        print("❌ 需要至少 2 个 benchmark 才能进行综合选参。")
        return 2

    by_benchmark: dict[str, dict[str, tuple[float, dict[str, Any]]]] = {}
    for bench in benchmarks:
        rows = service.list_scores_by_dataset(dataset=bench, model=model_name, is_param_search=True)
        if not rows:
            print(f"❌ 缺少 param-search trial scores: {bench}")
            return 1
        mapping: dict[str, tuple[float, dict[str, Any]]] = {}
        for payload in rows:
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                continue
            key = _param_key_from_payload(payload)
            if key is None:
                continue
            objective = _objective_from_metrics(metrics)
            existing = mapping.get(key)
            if existing is None or objective > existing[0]:
                mapping[key] = (objective, payload)
        if not mapping:
            print(f"❌ 未能从 {bench} 解析出任何有效 trial score")
            return 1
        by_benchmark[bench] = mapping

    common_keys = set.intersection(*(set(m.keys()) for m in by_benchmark.values()))
    if not common_keys:
        print("❌ 各 benchmark 的 grid 点没有交集（params key 不一致）")
        return 1

    selections: dict[str, tuple[str, float, dict[str, float], dict[str, dict[str, Any]]]] = {}
    for mode in sorted(_MODE_SUFFIXES.keys()):
        mode_keys = {key for key in common_keys if _sample_mode_from_key(key) == mode}
        if not mode_keys:
            continue

        best_key: str | None = None
        best_sum: float | None = None
        best_detail: dict[str, float] = {}
        best_payloads: dict[str, dict[str, Any]] = {}
        for key in sorted(mode_keys):
            total = 0.0
            detail: dict[str, float] = {}
            payloads: dict[str, dict[str, Any]] = {}
            for bench, mapping in by_benchmark.items():
                score, payload = mapping[key]
                total += float(score)
                detail[bench] = float(score)
                payloads[bench] = payload
            if best_sum is None or total > best_sum:
                best_sum = total
                best_key = key
                best_detail = detail
                best_payloads = payloads

        if best_key is not None and best_sum is not None:
            selections[mode] = (best_key, best_sum, best_detail, best_payloads)

    if not selections:
        print("❌ 未找到可用的最佳 grid 点（normal/simple 均无可用交集）")
        return 1

    # Backward compatible: still promote the best overall selection into {benchmark}.
    best_overall_mode, best_overall = max(selections.items(), key=lambda item: item[1][1])
    best_key, best_sum, best_detail, best_payloads = best_overall

    print("✅ best param-search selections:")
    print(f"    model: {model_name}")
    for mode in ("normal", "simple"):
        if mode not in selections:
            continue
        _, total, detail, _ = selections[mode]
        print(f"    {mode}: total={total:.6f} ({', '.join(f'{b}={detail.get(b, 0.0):.6f}' for b in benchmarks)})")
    print(f"    promote default: {best_overall_mode} -> {{benchmark}}")
    for mode in ("normal", "simple"):
        if mode in selections:
            print(f"    promote: {mode} -> {{benchmark}}{_MODE_SUFFIXES[mode]}")

    # 1) Promote best overall into {benchmark}.
    for bench, payload in best_payloads.items():
        _promote_score(
            service,
            source_payload=payload,
            dest_dataset=bench,
            model_name=model_name,
            overwrite=bool(args.overwrite),
        )

    # 2) Promote per-mode best into {benchmark}__ps_{mode}.
    for mode, (mode_key, _, _, mode_payloads) in selections.items():
        suffix = _MODE_SUFFIXES[mode]
        for bench, payload in mode_payloads.items():
            _promote_score(
                service,
                source_payload=payload,
                dest_dataset=f"{bench}{suffix}",
                model_name=model_name,
                overwrite=bool(args.overwrite),
            )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
