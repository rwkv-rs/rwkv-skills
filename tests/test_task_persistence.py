from __future__ import annotations

import pytest

from src.db.eval_db_service import ResumeContext, TaskLookup
from src.eval.evaluating import RunMode, current_run_mode, prepare_task_execution


class _FakeService:
    def __init__(self, ctx: ResumeContext, *, task_id: str = "9001") -> None:
        self._ctx = ctx
        self.task_id = task_id
        self.get_calls: list[dict[str, object]] = []
        self.create_calls: list[dict[str, object]] = []

    def get_resume_context(self, **kwargs: object) -> ResumeContext:
        self.get_calls.append(dict(kwargs))
        return self._ctx

    def create_task_from_context(self, **kwargs: object) -> str:
        self.create_calls.append(dict(kwargs))
        return self.task_id


def test_current_run_mode_prefers_explicit_env_and_legacy_overwrite() -> None:
    assert current_run_mode({"RWKV_EVAL_RUN_MODE": "resume"}) is RunMode.RESUME
    assert current_run_mode({"RWKV_SCHEDULER_OVERWRITE": "1"}) is RunMode.RERUN
    assert current_run_mode({}) is RunMode.AUTO


def test_prepare_task_execution_auto_classifies_new_resume_and_rerun() -> None:
    new_state = prepare_task_execution(
        service=_FakeService(ResumeContext()),
        dataset="gsm8k_test",
        model="rwkv",
        is_param_search=False,
        job_name="free_response_judge",
    )
    assert new_state.run_mode is RunMode.NEW
    assert new_state.skip_keys == set()

    resume_state = prepare_task_execution(
        service=_FakeService(ResumeContext(task_id=7, can_resume=True, completed_keys={(1, 0, 0)})),
        dataset="gsm8k_test",
        model="rwkv",
        is_param_search=False,
        job_name="free_response_judge",
    )
    assert resume_state.run_mode is RunMode.RESUME
    assert resume_state.skip_keys == {(1, 0, 0)}

    rerun_state = prepare_task_execution(
        service=_FakeService(ResumeContext(task_id=8, can_resume=False)),
        dataset="gsm8k_test",
        model="rwkv",
        is_param_search=False,
        job_name="free_response_judge",
    )
    assert rerun_state.run_mode is RunMode.RERUN
    assert rerun_state.skip_keys == set()


def test_prepare_task_execution_new_refuses_existing_task() -> None:
    with pytest.raises(ValueError, match="run_mode=new refused"):
        prepare_task_execution(
            service=_FakeService(
                ResumeContext(
                    task_id=3,
                    can_resume=True,
                    matching_tasks=(TaskLookup(task_id=3, status="Running"),),
                    resumable_task_ids=(3,),
                )
            ),
            dataset="mmlu_test",
            model="rwkv",
            is_param_search=False,
            job_name="multi_choice_plain",
            run_mode=RunMode.NEW,
        )


def test_prepare_task_execution_resume_requires_resumable_task() -> None:
    with pytest.raises(ValueError, match="could not find a matching running/failed task"):
        prepare_task_execution(
            service=_FakeService(ResumeContext()),
            dataset="mmlu_test",
            model="rwkv",
            is_param_search=False,
            job_name="multi_choice_plain",
            run_mode="resume",
        )

    with pytest.raises(ValueError, match="matching completed task already exists"):
        prepare_task_execution(
            service=_FakeService(
                ResumeContext(
                    task_id=4,
                    can_resume=False,
                    matching_tasks=(TaskLookup(task_id=4, status="Completed"),),
                    completed_task_ids=(4,),
                )
            ),
            dataset="mmlu_test",
            model="rwkv",
            is_param_search=False,
            job_name="multi_choice_plain",
            run_mode=RunMode.RESUME,
        )


def test_prepare_task_execution_rerun_forces_new_task_lookup() -> None:
    service = _FakeService(ResumeContext(), task_id="42")
    state = prepare_task_execution(
        service=service,
        dataset="math_500_test",
        model="rwkv",
        is_param_search=False,
        job_name="free_response",
        run_mode=RunMode.RERUN,
    )

    assert state.task_id == "42"
    assert state.run_mode is RunMode.RERUN
    assert service.get_calls == [
        {
            "dataset": "math_500_test",
            "model": "rwkv",
            "is_param_search": False,
            "job_name": "free_response",
            "sampling_config": None,
            "force_new_task": True,
        }
    ]


def test_prepare_task_execution_resume_rejects_ambiguous_matches() -> None:
    with pytest.raises(ValueError, match="multiple matching running/failed tasks exist"):
        prepare_task_execution(
            service=_FakeService(
                ResumeContext(
                    matching_tasks=(
                        TaskLookup(task_id=11, status="Running"),
                        TaskLookup(task_id=12, status="Failed"),
                    ),
                    resumable_task_ids=(11, 12),
                )
            ),
            dataset="mmlu_test",
            model="rwkv",
            is_param_search=False,
            job_name="multi_choice_plain",
            run_mode=RunMode.RESUME,
        )
