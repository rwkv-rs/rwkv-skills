from __future__ import annotations

from datetime import datetime
from pathlib import Path

import scripts.export_wrong_eval_records as exporter
from src.space.data import ScoreEntry


def _entry(
    *,
    dataset: str = "aime24_test",
    model: str = "rwkv7-g1f-13.3b",
    cot: bool = True,
    task_id: int = 123,
) -> ScoreEntry:
    return ScoreEntry(
        task_id=task_id,
        dataset=dataset,
        model=model,
        metrics={"judge_accuracy": 0.5},
        samples=1,
        problems=None,
        created_at=datetime(2026, 1, 2, 3, 4, 5),
        log_path="",
        cot=cot,
        task=None,
        task_details=None,
        path=Path("<db>"),
        relative_path=Path(str(task_id)),
        domain="math reasoning系列",
        extra={},
        arch_version="RWKV7",
        data_version="G1F",
        num_params="13_3B",
    )


def test_matches_frontend_benchmark_and_model_keys() -> None:
    entry = _entry()

    assert exporter._matches_target(
        entry,
        exporter.ExportTarget(benchmark="aime24_cot", model="rwkv7-g1f-13.3b"),
    )
    assert exporter._matches_target(
        entry,
        exporter.ExportTarget(benchmark="aime24_test_cot", model="rwkv7-g1f-13.3b"),
    )
    assert not exporter._matches_target(
        entry,
        exporter.ExportTarget(benchmark="aime24_nocot", model="rwkv7-g1f-13.3b"),
    )


def test_target_output_path_uses_requested_directory_rule() -> None:
    path = exporter._target_output_path(Path("out"), _entry())

    assert path == Path("out/13.3b/g1f/aime24_cot.json")


def test_record_export_fields_default_to_requested_four_fields() -> None:
    record = {
        "sample_index": 7,
        "repeat_index": 2,
        "answer": "A",
        "ref_answer": "B",
        "fail_reason": "wrong",
        "context": {"stages": []},
    }

    assert exporter._record_for_export(record, include_indexes=False) == {
        "answer": "A",
        "ref-answer": "B",
        "fail-reason": "wrong",
        "context": {"stages": []},
    }


def test_export_wrong_records_writes_one_file_per_target(monkeypatch, tmp_path) -> None:
    entry = _entry()

    class FakeService:
        def list_eval_records_for_space(self, **_kwargs):
            return [
                {
                    "answer": "bad",
                    "ref_answer": "good",
                    "fail_reason": "incorrect",
                    "context": {"prompt": "p"},
                }
            ]

    monkeypatch.setattr(exporter, "_load_frontend_latest_entries", lambda **_kwargs: [entry])
    monkeypatch.setattr(exporter, "EvalDbService", FakeService)

    manifest = exporter.export_wrong_records(
        [exporter.ExportTarget(benchmark="aime24_cot", model="rwkv7-g1f-13.3b")],
        output_root=tmp_path,
    )

    output_path = tmp_path / "13.3b" / "g1f" / "aime24_cot.json"
    assert output_path.exists()
    assert manifest["targets"][0]["path"] == str(output_path)
