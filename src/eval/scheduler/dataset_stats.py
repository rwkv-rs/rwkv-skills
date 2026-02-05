from __future__ import annotations

"""Dataset sample counting + DB recording helpers."""

from pathlib import Path

from src.db.orm import init_orm
from src.db.eval_db_service import EvalDbService

from .config import DEFAULT_DB_CONFIG
from .dataset_utils import canonical_slug, infer_dataset_slug_from_path


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def record_dataset_samples(dataset_path: Path, *, dataset_slug: str | None = None) -> None:
    """Persist dataset sample count into benchmark table."""
    slug = canonical_slug(dataset_slug or infer_dataset_slug_from_path(str(dataset_path)))
    if not slug:
        return
    init_orm(DEFAULT_DB_CONFIG)
    
    service = EvalDbService()
    existing = service.get_benchmark_num_samples(dataset=slug)
    if existing is not None and existing > 0:
        return
    try:
        samples = count_jsonl_records(dataset_path)
    except OSError as exc:
        print(f"⚠️  无法统计数据集样本数: {dataset_path} ({exc})")
        return
    if samples <= 0:
        print(f"⚠️  数据集样本数为 0: {dataset_path}")
        return
    service.ensure_benchmark_num_samples(dataset=slug, num_samples=samples)


__all__ = ["count_jsonl_records", "record_dataset_samples"]
