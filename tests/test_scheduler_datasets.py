from __future__ import annotations

from pathlib import Path

from src.eval.scheduler.datasets import find_dataset_file, refresh_dataset_index


def test_dataset_index_ignores_hidden_backup_dirs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    real_dir = data_root / "bfcl_v3"
    backup_dir = data_root / ".bak_resource_fix_20260627_025450" / "bfcl_v3"
    real_dir.mkdir(parents=True)
    backup_dir.mkdir(parents=True)

    real_path = real_dir / "test.jsonl"
    backup_path = backup_dir / "test.jsonl"
    real_path.write_text('{"task_id":"real"}\n', encoding="utf-8")
    backup_path.write_text('{"task_id":"backup"}\n', encoding="utf-8")

    refresh_dataset_index([data_root])

    assert find_dataset_file("bfcl_v3_test", [data_root]) == real_path.resolve()
