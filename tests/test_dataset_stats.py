from __future__ import annotations

from pathlib import Path

from src.eval.scheduler import dataset_stats


def test_record_dataset_samples_skips_db_when_json_store(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text('{"id":"x"}\n', encoding="utf-8")
    called = False

    def _fail_init_db(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        nonlocal called
        called = True
        raise AssertionError("init_db should not be called in JSON store mode")

    monkeypatch.setenv("RWKV_EVAL_STORE", "json")
    monkeypatch.setattr(dataset_stats, "init_db", _fail_init_db)

    dataset_stats.record_dataset_samples(dataset_path)

    assert called is False
