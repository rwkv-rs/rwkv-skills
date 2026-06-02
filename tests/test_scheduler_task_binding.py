from __future__ import annotations

import pytest

from src.db.eval_db_service import EvalDbService, ResumeContext


def test_create_task_from_context_uses_scheduler_task_id(monkeypatch) -> None:
    service = object.__new__(EvalDbService)
    repo = _BindingRepo(result=42)
    service._repo = repo
    monkeypatch.setattr("src.db.eval_db_service.get_session", lambda: _FakeSession())
    monkeypatch.setenv("RWKV_SESSION_ID", "session-1")
    monkeypatch.setenv("RWKV_SESSION_TASK_ID", "42")
    monkeypatch.setenv("RWKV_TASK_DESC", "job=free_response_judge, dataset=math_500_test")
    monkeypatch.setenv("RWKV_SKILLS_LOG_PATH", "results/logs/task.log")

    task_id = service.create_task_from_context(
        ctx=ResumeContext(benchmark_id=7, model_id=11),
        job_name="eval_free_response_judge",
        dataset="math_500_test",
        model="rwkv7-g1g-1.5b-20260526-ctx8192",
        is_param_search=False,
        sampling_config={"generation": {"temperature": 1.0}},
    )

    assert task_id == "42"
    assert repo.runtime_update == {
        "task_id": 42,
        "session_id": "session-1",
        "benchmark_id": 7,
        "model_id": 11,
        "is_param_search": False,
        "desc": "job=free_response_judge, dataset=math_500_test",
        "sampling_config": {"generation": {"temperature": 1.0}},
        "log_path": "results/logs/task.log",
    }
    assert repo.insert_task_called is False


def test_create_task_from_context_rejects_unmatched_scheduler_task_id(monkeypatch) -> None:
    service = object.__new__(EvalDbService)
    repo = _BindingRepo(result=None)
    service._repo = repo
    monkeypatch.setattr("src.db.eval_db_service.get_session", lambda: _FakeSession())
    monkeypatch.setenv("RWKV_SESSION_ID", "session-1")
    monkeypatch.setenv("RWKV_SESSION_TASK_ID", "42")

    with pytest.raises(RuntimeError, match="RWKV_SESSION_TASK_ID did not match"):
        service.create_task_from_context(
            ctx=ResumeContext(benchmark_id=7, model_id=11),
            job_name="eval_free_response_judge",
            dataset="math_500_test",
            model="rwkv7-g1g-1.5b-20260526-ctx8192",
            is_param_search=False,
            sampling_config=None,
        )

    assert repo.insert_task_called is False


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _BindingRepo:
    def __init__(self, *, result: int | None) -> None:
        self.result = result
        self.runtime_update: dict | None = None
        self.insert_task_called = False

    def update_session_task_runtime(self, _session, **kwargs):
        self.runtime_update = dict(kwargs)
        return self.result

    def insert_task(self, *_args, **_kwargs):
        self.insert_task_called = True
        raise AssertionError("scheduler task binding should not create a new task")
