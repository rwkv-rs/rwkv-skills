from __future__ import annotations

from pathlib import Path

from src.eval.scheduler import dataset_stats
from src.eval.scheduler import dataset_resolver


def test_record_dataset_samples_does_not_block_when_db_unavailable(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "tau3_bench_mock" / "base.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text('{"task_id":"one"}\n', encoding="utf-8")

    monkeypatch.setattr(dataset_stats, "init_db", lambda _config: (_ for _ in ()).throw(RuntimeError("offline")))

    dataset_stats.record_dataset_samples(dataset)


def test_resolve_or_prepare_dataset_can_skip_probe_only_stats(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "tau3_bench_mock" / "base.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text('{"task_id":"one"}\n', encoding="utf-8")
    calls: list[Path] = []

    monkeypatch.setattr(dataset_resolver, "record_dataset_samples", lambda path: calls.append(path))

    resolved = dataset_resolver.resolve_or_prepare_dataset(str(dataset), record_stats=False)

    assert resolved == dataset
    assert calls == []
