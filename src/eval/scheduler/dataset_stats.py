from __future__ import annotations

"""Dataset sample counting + DB recording helpers."""

import json
from pathlib import Path

from src.db.database import init_db
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


def _manifest_path_for_dataset(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def load_dataset_manifest_count(path: Path) -> int | None:
    manifest_path = _manifest_path_for_dataset(path)
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("row_count")
    if isinstance(value, int) and value > 0:
        return value
    return None


def record_dataset_samples(dataset_path: Path, *, dataset_slug: str | None = None) -> None:
    """Persist dataset sample count into benchmark table."""
    slug = canonical_slug(dataset_slug or infer_dataset_slug_from_path(str(dataset_path)))
    if not slug:
        return
    init_db(DEFAULT_DB_CONFIG)

    service = EvalDbService()
    existing = service.get_benchmark_num_samples(dataset=slug)
    if existing is not None and existing > 0:
        return
    try:
        samples = load_dataset_manifest_count(dataset_path) or count_jsonl_records(dataset_path)
    except OSError as exc:
        print(f"⚠️  无法统计数据集样本数: {dataset_path} ({exc})")
        return
    if samples <= 0:
        print(f"⚠️  数据集样本数为 0: {dataset_path}")
        return
    service.ensure_benchmark_num_samples(dataset=slug, num_samples=samples)


__all__ = ["count_jsonl_records", "load_dataset_manifest_count", "record_dataset_samples"]
