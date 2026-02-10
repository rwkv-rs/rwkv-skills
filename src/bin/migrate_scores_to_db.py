from __future__ import annotations

"""Migrate local score JSON files into the evaluation database."""

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.db.eval_db_service import EvalDbService
from src.db.orm import init_orm
from src.eval.scheduler.config import DEFAULT_DB_CONFIG
from src.eval.scheduler.dataset_utils import canonical_slug


_DECIMAL_PARAM_RE = re.compile(r"(?P<whole>\d+)-(?P<frac>\d+)b")


@dataclass(slots=True)
class MigrationStats:
    total: int = 0
    inserted: int = 0
    skipped_existing: int = 0
    skipped_filtered: int = 0
    errors: int = 0


def _normalize_model_fallback(name: str) -> str:
    normalized = name.replace("_", "-")
    return _DECIMAL_PARAM_RE.sub(lambda m: f"{m.group('whole')}.{m.group('frac')}b", normalized)


def _infer_dataset_from_path(path: Path) -> str | None:
    stem = path.stem
    if stem.endswith("__cot"):
        stem = stem[: -len("__cot")]
    slug = canonical_slug(stem)
    return slug if slug else None


def _infer_model_from_path(path: Path) -> str | None:
    parent = path.parent.name
    if not parent:
        return None
    return _normalize_model_fallback(parent)


def _parse_created_at(raw: Any, *, fallback: datetime) -> datetime:
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None)
    if isinstance(raw, (int, float)):
        try:
            return datetime.utcfromtimestamp(raw)
        except (OverflowError, OSError, ValueError):
            return fallback
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            try:
                if text.endswith("Z"):
                    text = text[:-1] + "+00:00"
                return datetime.fromisoformat(text).replace(tzinfo=None)
            except ValueError:
                return fallback
    return fallback


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _iter_score_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.json") if p.is_file())


def _resolve_created_at(path: Path, payload: dict[str, Any]) -> datetime:
    fallback = datetime.utcfromtimestamp(path.stat().st_mtime)
    return _parse_created_at(payload.get("created_at"), fallback=fallback)


def _resolve_is_cot(path: Path, payload: dict[str, Any]) -> bool:
    raw = payload.get("cot")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    stem = path.stem.lower()
    return stem.endswith("__cot")


def _resolve_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def _resolve_log_path(payload: dict[str, Any]) -> str:
    log_path = payload.get("log_path")
    return str(log_path) if isinstance(log_path, str) else ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate scores/*.json into the RWKV evaluation database")
    parser.add_argument(
        "--scores-dir",
        default="scores",
        help="Root directory containing score JSON files (default: scores)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be migrated without writing to the database",
    )
    parser.add_argument(
        "--model-contains",
        help="Only migrate entries whose model name contains this substring",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N files (after sorting)",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Insert even if a score with the same dataset/model/cot/created_at already exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scores_dir = Path(args.scores_dir)
    files = list(_iter_score_files(scores_dir))
    if args.limit:
        files = files[: max(0, int(args.limit))]

    if not files:
        print(f"ERROR: no JSON files found under {scores_dir}")
        return 1

    init_orm(DEFAULT_DB_CONFIG)
    service = EvalDbService()

    existing_cache: dict[tuple[str, str, bool], set[tuple[bool, datetime]]] = {}
    stats = MigrationStats()

    for path in files:
        stats.total += 1
        payload = _read_json(path)
        if payload is None:
            stats.errors += 1
            print(f"WARN: failed to read {path}")
            continue

        dataset = payload.get("dataset") or _infer_dataset_from_path(path)
        if not isinstance(dataset, str) or not dataset.strip():
            stats.errors += 1
            print(f"WARN: missing dataset in {path}")
            continue
        dataset = canonical_slug(dataset)

        model = payload.get("model") or _infer_model_from_path(path)
        if not isinstance(model, str) or not model.strip():
            stats.errors += 1
            print(f"WARN: missing model in {path}")
            continue
        model = str(model)

        if args.model_contains and args.model_contains not in model:
            stats.skipped_filtered += 1
            continue

        is_cot = _resolve_is_cot(path, payload)
        metrics = _resolve_metrics(payload)
        created_at = _resolve_created_at(path, payload)
        log_path = _resolve_log_path(payload)
        task_name = payload.get("task") if isinstance(payload.get("task"), str) else "score_migration"

        samples = _parse_int(payload.get("samples"))
        problems = _parse_int(payload.get("problems"))
        num_samples = samples if samples is not None and samples > 0 else problems

        cache_key = (dataset, model, False)
        if not args.allow_duplicates:
            existing = existing_cache.get(cache_key)
            if existing is None:
                rows = service.list_scores_by_dataset(dataset=dataset, model=model, is_param_search=False)
                existing = set()
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row_created = row.get("created_at")
                    parsed = _parse_created_at(row_created, fallback=created_at)
                    existing.add((bool(row.get("cot", False)), parsed))
                existing_cache[cache_key] = existing
            if (is_cot, created_at) in existing:
                stats.skipped_existing += 1
                continue

        if args.dry_run:
            stats.inserted += 1
            if not args.allow_duplicates:
                existing_cache.setdefault(cache_key, set()).add((is_cot, created_at))
            continue

        if num_samples is not None and num_samples > 0:
            service.ensure_benchmark_num_samples(dataset=dataset, num_samples=num_samples)

        os.environ["RWKV_SKILLS_LOG_PATH"] = log_path
        os.environ.setdefault("RWKV_TASK_DESC", "score_migration")

        task_id = service.get_or_create_task(
            job_name=task_name,
            job_id=None,
            dataset=dataset,
            model=model,
            is_param_search=False,
            allow_resume=False,
        )

        payload["dataset"] = dataset
        payload["model"] = model
        payload["cot"] = is_cot
        payload["metrics"] = metrics
        payload["created_at"] = created_at

        service.record_score_payload(payload=payload, task_id=task_id)
        stats.inserted += 1
        if not args.allow_duplicates:
            existing_cache.setdefault(cache_key, set()).add((is_cot, created_at))

    print(
        "DONE: "
        f"files={stats.total} inserted={stats.inserted} "
        f"skipped_existing={stats.skipped_existing} "
        f"skipped_filtered={stats.skipped_filtered} errors={stats.errors}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
