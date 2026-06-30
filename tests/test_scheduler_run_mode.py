from __future__ import annotations

from src.eval.evaluating import RunMode
from src.eval.scheduler import actions, actions_base
from src.eval.scheduler.actions import QueueOptions
from src.eval.scheduler.state import CompletedKey


def test_completed_for_queue_merges_or_resets_by_run_mode() -> None:
    historical = CompletedKey(job="free_response", model_slug="rwkv", dataset_slug="gsm8k_test", is_cot=True)
    current = CompletedKey(job="multi_choice_plain", model_slug="rwkv", dataset_slug="mmlu_test", is_cot=False)

    assert actions._completed_for_queue(
        run_mode=RunMode.AUTO,
        completed={historical},
        session_completed={current},
    ) == {historical, current}
    assert actions._completed_for_queue(
        run_mode=RunMode.RERUN,
        completed={historical},
        session_completed={current},
    ) == {current}
    assert actions._completed_for_queue(
        run_mode=RunMode.FRESH,
        completed={historical},
        session_completed={current},
    ) == {historical, current}


def test_action_queue_auto_filters_completed(monkeypatch, tmp_path) -> None:
    completed_key = CompletedKey(job="free_response", model_slug="rwkv", dataset_slug="gsm8k_test", is_cot=True)
    captured: dict[str, object] = {}

    monkeypatch.setattr(actions_base, "scan_completed_jobs", lambda: ({completed_key}, {}))
    monkeypatch.setattr(actions_base, "load_running", lambda _pid_dir: {})
    monkeypatch.setattr(actions_base, "derive_question_counts", lambda _records: {})
    monkeypatch.setattr(actions_base, "sort_queue_items", lambda items, **_kwargs: items)
    monkeypatch.setattr(actions_base, "_print_queue_summary", lambda *_args, **_kwargs: None)

    def _fake_build_queue(**kwargs):
        captured["completed"] = kwargs["completed"]
        return []

    monkeypatch.setattr(actions_base, "build_queue", _fake_build_queue)

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

    monkeypatch.setattr(actions_base, "scan_completed_jobs", lambda: ({completed_key}, {}))
    monkeypatch.setattr(actions_base, "load_running", lambda _pid_dir: {})
    monkeypatch.setattr(actions_base, "derive_question_counts", lambda _records: {})
    monkeypatch.setattr(actions_base, "sort_queue_items", lambda items, **_kwargs: items)
    monkeypatch.setattr(actions_base, "_print_queue_summary", lambda *_args, **_kwargs: None)

    def _fake_build_queue(**kwargs):
        captured["completed"] = kwargs["completed"]
        return []

    monkeypatch.setattr(actions_base, "build_queue", _fake_build_queue)

    actions.action_queue(
        QueueOptions(
            log_dir=tmp_path,
            pid_dir=tmp_path,
            job_order=("free_response",),
            run_mode=RunMode.RERUN,
        )
    )

    assert captured["completed"] == set()
