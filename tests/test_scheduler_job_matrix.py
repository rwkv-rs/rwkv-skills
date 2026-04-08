from __future__ import annotations

import importlib
import sys

from src.eval.runner_registry import RunnerGroup
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import (
    CODE_DATASET_SLUGS,
    DATASET_PREP_SPECS,
    INSTRUCTION_FOLLOWING_DATASET_SLUGS,
    JOB_CATALOGUE,
    detect_job_from_dataset,
)


def test_job_catalogue_exposes_fake_cot_and_mbpp_variants() -> None:
    assert "multi_choice_fake_cot" in JOB_CATALOGUE
    assert "code_mbpp_fake_cot" in JOB_CATALOGUE
    assert "code_mbpp_cot" in JOB_CATALOGUE

    assert JOB_CATALOGUE["multi_choice_plain"].runner_group is RunnerGroup.KNOWLEDGE
    assert JOB_CATALOGUE["free_response"].runner_group is RunnerGroup.MATHS
    assert JOB_CATALOGUE["code_mbpp"].runner_group is RunnerGroup.CODING
    assert JOB_CATALOGUE["function_mcp_bench"].runner_group is RunnerGroup.FUNCTION_CALLING
    assert JOB_CATALOGUE["multi_choice_plain"].module == "src.eval.knowledge.runner"
    assert JOB_CATALOGUE["multi_choice_fake_cot"].module == "src.eval.knowledge.runner"
    assert JOB_CATALOGUE["multi_choice_cot"].module == "src.eval.knowledge.runner"
    assert JOB_CATALOGUE["free_response"].module == "src.eval.maths.runner"
    assert JOB_CATALOGUE["free_response_judge"].module == "src.eval.maths.runner"
    assert JOB_CATALOGUE["code_human_eval"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["code_mbpp"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["code_mbpp_fake_cot"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["code_mbpp_cot"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["code_livecodebench"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["instruction_following"].module == "src.eval.instruction_following.runner"
    assert JOB_CATALOGUE["function_browsecomp"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_mcp_bench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_tau_bench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_tau2_bench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["multi_choice_plain"].extra_args == ("--cot-mode", "no_cot")
    assert JOB_CATALOGUE["multi_choice_fake_cot"].extra_args == ("--cot-mode", "fake_cot")
    assert JOB_CATALOGUE["multi_choice_cot"].extra_args == ("--cot-mode", "cot")
    assert JOB_CATALOGUE["free_response"].extra_args == ("--judge-mode", "exact")
    assert JOB_CATALOGUE["free_response_judge"].extra_args == ("--judge-mode", "llm")
    assert JOB_CATALOGUE["code_mbpp"].extra_args == ("--cot-mode", "no_cot")
    assert JOB_CATALOGUE["code_mbpp_fake_cot"].extra_args == ("--cot-mode", "fake_cot")
    assert JOB_CATALOGUE["code_mbpp_cot"].extra_args == ("--cot-mode", "cot")
    assert JOB_CATALOGUE["instruction_following"].extra_args == ()


def test_instruction_following_matrix_includes_all_supported_datasets() -> None:
    assert canonical_slug("ifeval_test") in INSTRUCTION_FOLLOWING_DATASET_SLUGS
    assert canonical_slug("arena_hard_test") in INSTRUCTION_FOLLOWING_DATASET_SLUGS
    assert canonical_slug("wmt24pp_test") in INSTRUCTION_FOLLOWING_DATASET_SLUGS
    assert canonical_slug("flores200_devtest") in INSTRUCTION_FOLLOWING_DATASET_SLUGS


def test_scheduler_matrix_uses_metadata_default_splits() -> None:
    assert canonical_slug("include_test") in JOB_CATALOGUE["multi_choice_plain"].dataset_slugs
    assert canonical_slug("gpqa_main") in JOB_CATALOGUE["multi_choice_plain"].dataset_slugs
    assert canonical_slug("gpqa_extended") in JOB_CATALOGUE["multi_choice_plain"].dataset_slugs
    assert canonical_slug("gpqa_diamond") in JOB_CATALOGUE["multi_choice_plain"].dataset_slugs
    assert canonical_slug("simpleqa_verified") in JOB_CATALOGUE["free_response"].dataset_slugs
    assert canonical_slug("polymath_all") in JOB_CATALOGUE["free_response"].dataset_slugs
    assert canonical_slug("gsm8k_test") in JOB_CATALOGUE["free_response_judge"].dataset_slugs


def test_dataset_prep_specs_follow_benchmark_metadata_splits() -> None:
    gpqa_spec = DATASET_PREP_SPECS[canonical_slug("gpqa_diamond")]
    include_spec = DATASET_PREP_SPECS[canonical_slug("include_test")]
    polymath_spec = DATASET_PREP_SPECS[canonical_slug("polymath_all")]
    tau2_spec = DATASET_PREP_SPECS[canonical_slug("tau2_bench_airline_base")]

    assert gpqa_spec.dataset == "gpqa"
    assert gpqa_spec.split == "diamond"
    assert include_spec.dataset == "include"
    assert include_spec.split == "test"
    assert polymath_spec.dataset == "polymath"
    assert polymath_spec.split == "all"
    assert tau2_spec.dataset == "tau2_bench_airline"
    assert tau2_spec.split == "base"
    assert canonical_slug("tau2_bench_airline_base") in CODE_DATASET_SLUGS


def test_function_calling_jobs_cover_browsecomp_and_mcp_bench() -> None:
    assert "function_browsecomp" in JOB_CATALOGUE
    assert "function_mcp_bench" in JOB_CATALOGUE
    assert "function_tau_bench" in JOB_CATALOGUE
    assert "function_tau2_bench" in JOB_CATALOGUE

    browsecomp_slugs = JOB_CATALOGUE["function_browsecomp"].dataset_slugs
    mcp_slugs = JOB_CATALOGUE["function_mcp_bench"].dataset_slugs
    tau_slugs = JOB_CATALOGUE["function_tau_bench"].dataset_slugs
    tau2_slugs = JOB_CATALOGUE["function_tau2_bench"].dataset_slugs

    assert canonical_slug("browsecomp_test") in browsecomp_slugs
    assert canonical_slug("browsecomp_zh_test") in browsecomp_slugs
    assert canonical_slug("mcp_bench_test") in mcp_slugs
    assert canonical_slug("tau_bench_retail_test") in tau_slugs
    assert canonical_slug("tau_bench_airline_test") in tau_slugs
    assert canonical_slug("tau_bench_telecom_test") in tau_slugs
    assert canonical_slug("tau2_bench_retail_base") in tau2_slugs
    assert canonical_slug("tau2_bench_airline_base") in tau2_slugs
    assert canonical_slug("tau2_bench_telecom_base") in tau2_slugs

    assert detect_job_from_dataset(canonical_slug("browsecomp_test"), is_cot=True) == "function_browsecomp"
    assert detect_job_from_dataset(canonical_slug("mcp_bench_test"), is_cot=True) == "function_mcp_bench"
    assert detect_job_from_dataset(canonical_slug("tau_bench_retail_test"), is_cot=True) == "function_tau_bench"
    assert detect_job_from_dataset(canonical_slug("tau2_bench_telecom_base"), is_cot=True) == "function_tau2_bench"


def test_jobs_module_does_not_eagerly_import_data_manager() -> None:
    module_name = "src.eval.scheduler.jobs"
    data_manager_name = "src.eval.datasets.data_prepper.data_manager"

    sys.modules.pop(module_name, None)
    sys.modules.pop(data_manager_name, None)

    importlib.import_module(module_name)

    assert data_manager_name not in sys.modules
