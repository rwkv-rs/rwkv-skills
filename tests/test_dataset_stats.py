from __future__ import annotations

from pathlib import Path

from src.eval.scheduler import dataset_stats


def test_record_dataset_samples_uses_db_even_when_json_store_env_is_set(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "demo" / "test.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text('{"id":"x"}\n', encoding="utf-8")
    init_calls = 0
    recorded: list[tuple[str, int]] = []

    def _fake_init_db(*_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        nonlocal init_calls
        init_calls += 1

    class _FakeEvalDbService:
        def get_benchmark_num_samples(self, *, dataset: str) -> int | None:
            assert dataset == "demo_test"
            return None

        def ensure_benchmark_num_samples(self, *, dataset: str, num_samples: int) -> None:
            recorded.append((dataset, num_samples))

    monkeypatch.setenv("RWKV_EVAL_STORE", "json")
    monkeypatch.setattr(dataset_stats, "init_db", _fake_init_db)
    monkeypatch.setattr(dataset_stats, "EvalDbService", _FakeEvalDbService)

    dataset_stats.record_dataset_samples(dataset_path)

    assert init_calls == 1
    assert recorded == [("demo_test", 1)]
