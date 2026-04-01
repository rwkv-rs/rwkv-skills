from __future__ import annotations

"""Field-oriented runner registry aligned with rwkv-rs benchmark domains."""

from dataclasses import dataclass
from enum import Enum

from src.eval.execution_plan import TARGET_EVAL_ATTEMPTS
from src.eval.scheduler.dataset_utils import canonical_slug


class RunnerGroup(str, Enum):
    KNOWLEDGE = "knowledge"
    MATHS = "maths"
    CODING = "coding"
    INSTRUCTION_FOLLOWING = "instruction_following"
    FUNCTION_CALLING = "function_calling"
    PARAM_SEARCH = "param_search"


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    name: str
    group: RunnerGroup
    scheduler_domain: str
    module: str
    is_cot: bool
    fallback_dataset_slugs: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    batch_flag: str | None = None
    probe_flag: str | None = None
    probe_max_generate_flag: str | None = None
    probe_dataset_required: bool = False
    probe_extra_args: tuple[str, ...] = ()
    probe_samples_per_task: int = 1
    probe_question_floor: int = 0


def _runner(
    name: str,
    *,
    group: RunnerGroup,
    scheduler_domain: str,
    module: str,
    is_cot: bool,
    fallback_dataset_slugs: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
    batch_flag: str | None = None,
    probe_flag: str | None = None,
    probe_max_generate_flag: str | None = None,
    probe_dataset_required: bool = False,
    probe_extra_args: tuple[str, ...] = (),
    probe_samples_per_task: int = 1,
    probe_question_floor: int = 0,
) -> RunnerSpec:
    return RunnerSpec(
        name=name,
        group=group,
        scheduler_domain=scheduler_domain,
        module=module,
        is_cot=is_cot,
        fallback_dataset_slugs=tuple(canonical_slug(slug) for slug in fallback_dataset_slugs),
        extra_args=extra_args,
        batch_flag=batch_flag,
        probe_flag=probe_flag,
        probe_max_generate_flag=probe_max_generate_flag,
        probe_dataset_required=probe_dataset_required,
        probe_extra_args=probe_extra_args,
        probe_samples_per_task=probe_samples_per_task,
        probe_question_floor=probe_question_floor,
    )


KNOWLEDGE_RUNNERS: tuple[RunnerSpec, ...] = (
    _runner(
        "multi_choice_plain",
        group=RunnerGroup.KNOWLEDGE,
        scheduler_domain="multi_choice",
        module="src.bin.knowledge_runner",
        is_cot=False,
        extra_args=("--cot-mode", "no_cot"),
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "multi_choice_fake_cot",
        group=RunnerGroup.KNOWLEDGE,
        scheduler_domain="multi_choice",
        module="src.bin.knowledge_runner",
        is_cot=True,
        extra_args=("--cot-mode", "fake_cot"),
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "multi_choice_cot",
        group=RunnerGroup.KNOWLEDGE,
        scheduler_domain="multi_choice",
        module="src.bin.knowledge_runner",
        is_cot=True,
        extra_args=("--cot-mode", "cot", "--no-param-search"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
)

MATHS_RUNNERS: tuple[RunnerSpec, ...] = (
    _runner(
        "free_response",
        group=RunnerGroup.MATHS,
        scheduler_domain="free_response",
        module="src.bin.maths_runner",
        is_cot=True,
        extra_args=("--judge-mode", "exact"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "free_response_judge",
        group=RunnerGroup.MATHS,
        scheduler_domain="free_response",
        module="src.bin.maths_runner",
        is_cot=True,
        extra_args=("--judge-mode", "llm"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
)

PARAM_SEARCH_RUNNERS: tuple[RunnerSpec, ...] = (
    _runner(
        "param_search_free_response",
        group=RunnerGroup.PARAM_SEARCH,
        scheduler_domain="param_search",
        module="src.bin.param_search_free_response",
        is_cot=True,
        fallback_dataset_slugs=("math_500_test",),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
    ),
    _runner(
        "param_search_free_response_judge",
        group=RunnerGroup.PARAM_SEARCH,
        scheduler_domain="param_search",
        module="src.bin.param_search_free_response_judge",
        is_cot=True,
        fallback_dataset_slugs=("gsm8k_test",),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
    ),
    _runner(
        "param_search_select",
        group=RunnerGroup.PARAM_SEARCH,
        scheduler_domain="param_search",
        module="src.bin.param_search_select",
        is_cot=True,
        fallback_dataset_slugs=("gsm8k_test",),
    ),
)

CODING_RUNNERS: tuple[RunnerSpec, ...] = (
    _runner(
        "code_human_eval",
        group=RunnerGroup.CODING,
        scheduler_domain="code",
        module="src.bin.coding_runner",
        is_cot=False,
        fallback_dataset_slugs=("human_eval_test",),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_max_generate_flag="--max-tokens",
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "code_mbpp",
        group=RunnerGroup.CODING,
        scheduler_domain="code",
        module="src.bin.coding_runner",
        is_cot=False,
        fallback_dataset_slugs=("mbpp_test",),
        extra_args=("--cot-mode", "no_cot"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_max_generate_flag="--max-tokens",
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "code_mbpp_fake_cot",
        group=RunnerGroup.CODING,
        scheduler_domain="code",
        module="src.bin.coding_runner",
        is_cot=True,
        fallback_dataset_slugs=("mbpp_test",),
        extra_args=("--cot-mode", "fake_cot"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_max_generate_flag="--max-tokens",
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "code_mbpp_cot",
        group=RunnerGroup.CODING,
        scheduler_domain="code",
        module="src.bin.coding_runner",
        is_cot=True,
        fallback_dataset_slugs=("mbpp_test",),
        extra_args=("--cot-mode", "cot"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_max_generate_flag="--max-tokens",
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "code_livecodebench",
        group=RunnerGroup.CODING,
        scheduler_domain="code",
        module="src.bin.coding_runner",
        is_cot=True,
        fallback_dataset_slugs=("livecodebench_test",),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_max_generate_flag="--max-tokens",
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
)

INSTRUCTION_FOLLOWING_RUNNERS: tuple[RunnerSpec, ...] = (
    _runner(
        "instruction_following",
        group=RunnerGroup.INSTRUCTION_FOLLOWING,
        scheduler_domain="instruction_following",
        module="src.bin.instruction_following_runner",
        is_cot=False,
        extra_args=("--no-param-search",),
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
)

FUNCTION_CALLING_RUNNERS: tuple[RunnerSpec, ...] = (
    _runner(
        "function_browsecomp",
        group=RunnerGroup.FUNCTION_CALLING,
        scheduler_domain="function_calling",
        module="src.bin.function_calling_runner",
        is_cot=True,
        fallback_dataset_slugs=("browsecomp_test", "browsecomp_zh_test"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "function_mcp_bench",
        group=RunnerGroup.FUNCTION_CALLING,
        scheduler_domain="function_calling",
        module="src.bin.function_calling_runner",
        is_cot=True,
        fallback_dataset_slugs=("mcp_bench_test",),
        probe_flag="--probe-only",
        probe_dataset_required=True,
    ),
    _runner(
        "function_tau_bench",
        group=RunnerGroup.FUNCTION_CALLING,
        scheduler_domain="function_calling",
        module="src.bin.function_calling_runner",
        is_cot=True,
        fallback_dataset_slugs=("tau_bench_airline_test", "tau_bench_retail_test", "tau_bench_telecom_test"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
    _runner(
        "function_tau2_bench",
        group=RunnerGroup.FUNCTION_CALLING,
        scheduler_domain="function_calling",
        module="src.bin.function_calling_runner",
        is_cot=True,
        fallback_dataset_slugs=("tau2_bench_airline_base", "tau2_bench_retail_base", "tau2_bench_telecom_base"),
        batch_flag="--batch-size",
        probe_flag="--probe-only",
        probe_dataset_required=True,
        probe_question_floor=TARGET_EVAL_ATTEMPTS,
    ),
)

ALL_RUNNERS: tuple[RunnerSpec, ...] = (
    KNOWLEDGE_RUNNERS
    + MATHS_RUNNERS
    + PARAM_SEARCH_RUNNERS
    + CODING_RUNNERS
    + FUNCTION_CALLING_RUNNERS
    + INSTRUCTION_FOLLOWING_RUNNERS
)

RUNNERS_BY_GROUP: dict[RunnerGroup, tuple[RunnerSpec, ...]] = {
    RunnerGroup.KNOWLEDGE: KNOWLEDGE_RUNNERS,
    RunnerGroup.MATHS: MATHS_RUNNERS,
    RunnerGroup.CODING: CODING_RUNNERS,
    RunnerGroup.INSTRUCTION_FOLLOWING: INSTRUCTION_FOLLOWING_RUNNERS,
    RunnerGroup.FUNCTION_CALLING: FUNCTION_CALLING_RUNNERS,
    RunnerGroup.PARAM_SEARCH: PARAM_SEARCH_RUNNERS,
}

RUNNER_BY_NAME: dict[str, RunnerSpec] = {runner.name: runner for runner in ALL_RUNNERS}


def resolve_runner(name: str) -> RunnerSpec:
    return RUNNER_BY_NAME[name]


__all__ = [
    "ALL_RUNNERS",
    "CODING_RUNNERS",
    "FUNCTION_CALLING_RUNNERS",
    "INSTRUCTION_FOLLOWING_RUNNERS",
    "KNOWLEDGE_RUNNERS",
    "MATHS_RUNNERS",
    "PARAM_SEARCH_RUNNERS",
    "RUNNERS_BY_GROUP",
    "RUNNER_BY_NAME",
    "RunnerGroup",
    "RunnerSpec",
    "resolve_runner",
]
