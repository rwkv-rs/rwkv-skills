from __future__ import annotations

from src.eval.evaluating import RunMode
from src.eval.scheduler import actions
from src.eval.scheduler.actions import QueueOptions
from src.eval.scheduler.state import CompletedKey


def test_action_queue_auto_filters_completed(monkeypatch, tmp_path) -> None:
    completed_key = CompletedKey(job="free_response", model_slug="rwkv", dataset_slug="gsm8k_test", is_cot=True)
    captured: dict[str, object] = {}

    monkeypatch.setattr(actions, "scan_completed_jobs", lambda _log_dir: ({completed_key}, {}))
    monkeypatch.setattr(actions, "load_running", lambda _pid_dir: {})
    monkeypatch.setattr(actions, "derive_question_counts", lambda _records: {})
    monkeypatch.setattr(actions, "sort_queue_items", lambda items, **_kwargs: items)
    monkeypatch.setattr(actions, "_print_queue_summary", lambda *_args, **_kwargs: None)

    def _fake_build_queue(**kwargs):
        captured["completed"] = kwargs["completed"]
        return []

    monkeypatch.setattr(actions, "build_queue", _fake_build_queue)

    actions.action_queue(
        QueueOptions(
            log_dir=tmp_path,
            pid_dir=tmp_path,
            job_order=("free_response",),
            run_mode=RunMode.AUTO,
        )
    )

    assert captured["completed"] == {completed_key}


def test_action_queue_rerun_ignores_completed_for_queue_building(monkeypatch, tmp_path) -> None:
    completed_key = CompletedKey(job="free_response", model_slug="rwkv", dataset_slug="gsm8k_test", is_cot=True)
    captured: dict[str, object] = {}

    monkeypatch.setattr(actions, "scan_completed_jobs", lambda _log_dir: ({completed_key}, {}))
    monkeypatch.setattr(actions, "load_running", lambda _pid_dir: {})
    monkeypatch.setattr(actions, "derive_question_counts", lambda _records: {})
    monkeypatch.setattr(actions, "sort_queue_items", lambda items, **_kwargs: items)
    monkeypatch.setattr(actions, "_print_queue_summary", lambda *_args, **_kwargs: None)

    def _fake_build_queue(**kwargs):
        captured["completed"] = kwargs["completed"]
        return []

    monkeypatch.setattr(actions, "build_queue", _fake_build_queue)

    actions.action_queue(
        QueueOptions(
            log_dir=tmp_path,
            pid_dir=tmp_path,
            job_order=("free_response",),
            run_mode=RunMode.RERUN,
        )
    )

    assert captured["completed"] == set()
