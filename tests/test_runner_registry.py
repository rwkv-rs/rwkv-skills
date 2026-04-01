from __future__ import annotations

from src.eval.runner_registry import ALL_RUNNERS, RUNNERS_BY_GROUP, RunnerGroup
from src.eval.scheduler.jobs import JOB_CATALOGUE


def test_runner_registry_groups_jobs_by_field_like_rwkv_rs() -> None:
    assert [runner.name for runner in RUNNERS_BY_GROUP[RunnerGroup.KNOWLEDGE]] == [
        "multi_choice_plain",
        "multi_choice_fake_cot",
        "multi_choice_cot",
    ]
    assert [runner.name for runner in RUNNERS_BY_GROUP[RunnerGroup.MATHS]] == [
        "free_response",
        "free_response_judge",
    ]
    assert [runner.name for runner in RUNNERS_BY_GROUP[RunnerGroup.INSTRUCTION_FOLLOWING]] == [
        "instruction_following",
    ]
    assert [runner.name for runner in RUNNERS_BY_GROUP[RunnerGroup.FUNCTION_CALLING]] == [
        "function_browsecomp",
        "function_mcp_bench",
        "function_tau_bench",
        "function_tau2_bench",
    ]


def test_scheduler_job_catalogue_is_derived_from_runner_registry() -> None:
    assert tuple(runner.name for runner in ALL_RUNNERS) == tuple(JOB_CATALOGUE.keys())

    for runner in ALL_RUNNERS:
        job = JOB_CATALOGUE[runner.name]
        assert job.module == runner.module
        assert job.domain == runner.scheduler_domain
        assert job.runner_group is runner.group
        assert job.extra_args == runner.extra_args


def test_knowledge_runners_share_unified_module() -> None:
    knowledge_modules = {
        runner.module
        for runner in RUNNERS_BY_GROUP[RunnerGroup.KNOWLEDGE]
    }
    assert knowledge_modules == {"src.eval.knowledge.runner"}


def test_maths_runners_share_unified_module() -> None:
    maths_modules = {
        runner.module
        for runner in RUNNERS_BY_GROUP[RunnerGroup.MATHS]
    }
    assert maths_modules == {"src.eval.maths.runner"}


def test_coding_runners_share_unified_module() -> None:
    coding_modules = {
        runner.module
        for runner in RUNNERS_BY_GROUP[RunnerGroup.CODING]
    }
    assert coding_modules == {"src.eval.coding.runner"}


def test_instruction_following_runners_share_unified_module() -> None:
    modules = {
        runner.module
        for runner in RUNNERS_BY_GROUP[RunnerGroup.INSTRUCTION_FOLLOWING]
    }
    assert modules == {"src.eval.instruction_following.runner"}


def test_function_calling_runners_share_unified_module() -> None:
    modules = {
        runner.module
        for runner in RUNNERS_BY_GROUP[RunnerGroup.FUNCTION_CALLING]
    }
    assert modules == {"src.eval.function_calling.runner"}
