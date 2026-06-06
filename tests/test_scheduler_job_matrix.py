from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from src.eval.datasets.runtime import read_jsonl_items
from src.eval.runner_registry import RunnerGroup
from src.eval.scheduler.datasets import refresh_dataset_index
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import (
    CODE_DATASET_SLUGS,
    DATASET_PREP_SPECS,
    INSTRUCTION_FOLLOWING_DATASET_SLUGS,
    JOB_CATALOGUE,
    detect_job_from_dataset,
    locate_dataset,
)


def test_job_catalogue_exposes_legacy_aligned_coding_jobs() -> None:
    assert "multi_choice_fake_cot" in JOB_CATALOGUE
    assert "code_mbpp_fake_cot" not in JOB_CATALOGUE
    assert "code_mbpp_cot" not in JOB_CATALOGUE
    assert "code_swe_bench" in JOB_CATALOGUE

    assert JOB_CATALOGUE["multi_choice_plain"].runner_group is RunnerGroup.KNOWLEDGE
    assert JOB_CATALOGUE["free_response"].runner_group is RunnerGroup.MATHS
    assert JOB_CATALOGUE["code_mbpp"].runner_group is RunnerGroup.CODING
    assert JOB_CATALOGUE["code_swe_bench"].runner_group is RunnerGroup.CODING
    assert JOB_CATALOGUE["function_mcp_bench"].runner_group is RunnerGroup.FUNCTION_CALLING
    assert JOB_CATALOGUE["multi_choice_plain"].module == "src.eval.knowledge.runner"
    assert JOB_CATALOGUE["multi_choice_fake_cot"].module == "src.eval.knowledge.runner"
    assert JOB_CATALOGUE["multi_choice_cot"].module == "src.eval.knowledge.runner"
    assert JOB_CATALOGUE["free_response"].module == "src.eval.maths.runner"
    assert JOB_CATALOGUE["free_response_judge"].module == "src.eval.maths.runner"
    assert JOB_CATALOGUE["code_human_eval"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["code_mbpp"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["code_livecodebench"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["code_swe_bench"].module == "src.eval.coding.runner"
    assert JOB_CATALOGUE["instruction_following"].module == "src.eval.instruction_following.runner"
    assert JOB_CATALOGUE["function_browsecomp"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_longcodebench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_mcp_bench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_bfcl_v3"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_bfcl_ast"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_bfcl_exec"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_toolalpaca"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_complexfuncbench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_tau_bench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_tau2_bench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["function_tau3_bench"].module == "src.eval.function_calling.runner"
    assert JOB_CATALOGUE["multi_choice_plain"].extra_args == ("--cot-mode", "no_cot")
    assert JOB_CATALOGUE["multi_choice_fake_cot"].extra_args == ("--cot-mode", "fake_cot")
    assert JOB_CATALOGUE["multi_choice_cot"].extra_args == ("--cot-mode", "cot")
    assert JOB_CATALOGUE["free_response"].extra_args == ("--judge-mode", "exact")
    assert JOB_CATALOGUE["free_response_judge"].extra_args == ("--judge-mode", "llm")
    assert JOB_CATALOGUE["code_mbpp"].extra_args == ("--cot-mode", "no_cot")
    assert canonical_slug("swe_bench_lite_test") in JOB_CATALOGUE["code_swe_bench"].dataset_slugs
    assert detect_job_from_dataset(canonical_slug("swe_bench_lite_test"), is_cot=True) == "code_swe_bench"
    assert JOB_CATALOGUE["instruction_following"].extra_args == ()


def test_instruction_following_matrix_includes_rule_scored_datasets_only() -> None:
    assert canonical_slug("ifeval_test") in INSTRUCTION_FOLLOWING_DATASET_SLUGS
    assert canonical_slug("ifbench_test") in INSTRUCTION_FOLLOWING_DATASET_SLUGS
    assert canonical_slug("arena_hard_test") not in INSTRUCTION_FOLLOWING_DATASET_SLUGS
    assert canonical_slug("wmt24pp_test") not in INSTRUCTION_FOLLOWING_DATASET_SLUGS
    assert canonical_slug("flores200_devtest") not in INSTRUCTION_FOLLOWING_DATASET_SLUGS


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
    tau3_spec = DATASET_PREP_SPECS[canonical_slug("tau3_bench_banking_knowledge_base")]
    complex_spec = DATASET_PREP_SPECS[canonical_slug("complexfuncbench_official_test")]

    assert gpqa_spec.dataset == "gpqa"
    assert gpqa_spec.split == "diamond"
    assert include_spec.dataset == "include"
    assert include_spec.split == "test"
    assert polymath_spec.dataset == "polymath"
    assert polymath_spec.split == "all"
    assert tau2_spec.dataset == "tau2_bench_airline"
    assert tau2_spec.split == "base"
    assert tau3_spec.dataset == "tau3_bench_banking_knowledge"
    assert tau3_spec.split == "base"
    assert complex_spec.dataset == "complexfuncbench_official"
    assert complex_spec.split == "test"
    assert canonical_slug("tau2_bench_airline_base") in CODE_DATASET_SLUGS
    assert canonical_slug("tau3_bench_banking_knowledge_base") in CODE_DATASET_SLUGS
    assert canonical_slug("complexfuncbench_official_test") in CODE_DATASET_SLUGS


def test_locate_dataset_prefers_existing_registered_artifact_without_source_prepare(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "data"
    dataset_dir = output_root / "bfcl_exec_simple"
    dataset_dir.mkdir(parents=True)
    stale_path = dataset_dir / "test.jsonl"
    stale_path.write_text(
        json.dumps(
            {
                "task_id": "exec_simple_0",
                "instruction": "Find a probability.",
                "tools": [],
                "expected_tool_calls": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, Path, str]] = []

    def _prepare_dataset(name: str, root: Path, split: str) -> list[Path]:
        calls.append((name, root, split))
        stale_path.write_text(
            json.dumps(
                {
                    "task_id": "exec_simple_0",
                    "instruction": "Find a probability.",
                    "tools": [],
                    "expected_executable_calls": ["calc_binomial_probability(n=20, k=5, p=0.6)"],
                    "execution_result_type": ["exact_match"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return [stale_path]

    monkeypatch.setattr("src.eval.datasets.data_prepper.data_manager.prepare_dataset", _prepare_dataset)
    monkeypatch.setattr("src.eval.scheduler.dataset_stats.record_dataset_samples", lambda *_args, **_kwargs: None)

    refresh_dataset_index([output_root])
    found = locate_dataset("bfcl_exec_simple_test", search=[output_root], output_root=output_root)

    assert found == stale_path
    assert calls == []
    [row] = read_jsonl_items(found)
    assert "expected_executable_calls" not in row


def test_locate_dataset_can_refresh_registered_default_artifact_when_requested(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "data"
    dataset_dir = output_root / "bfcl_exec_simple"
    dataset_dir.mkdir(parents=True)
    stale_path = dataset_dir / "test.jsonl"
    stale_path.write_text(
        json.dumps(
            {
                "task_id": "exec_simple_0",
                "instruction": "Find a probability.",
                "tools": [],
                "expected_tool_calls": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, Path, str]] = []

    def _prepare_dataset(name: str, root: Path, split: str) -> list[Path]:
        calls.append((name, root, split))
        stale_path.write_text(
            json.dumps(
                {
                    "task_id": "exec_simple_0",
                    "instruction": "Find a probability.",
                    "tools": [],
                    "expected_executable_calls": ["calc_binomial_probability(n=20, k=5, p=0.6)"],
                    "execution_result_type": ["exact_match"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return [stale_path]

    monkeypatch.setenv("RWKV_EVAL_REFRESH_DATASET", "1")
    monkeypatch.setattr("src.eval.datasets.data_prepper.data_manager.prepare_dataset", _prepare_dataset)
    monkeypatch.setattr("src.eval.scheduler.dataset_stats.record_dataset_samples", lambda *_args, **_kwargs: None)

    refresh_dataset_index([output_root])
    found = locate_dataset("bfcl_exec_simple_test", search=[output_root], output_root=output_root)

    assert found == stale_path
    assert calls == [("bfcl_exec_simple", output_root, "test")]
    [row] = read_jsonl_items(found)
    assert row["expected_executable_calls"] == ["calc_binomial_probability(n=20, k=5, p=0.6)"]


def test_function_calling_jobs_cover_browsecomp_and_mcp_bench() -> None:
    assert "function_browsecomp" in JOB_CATALOGUE
    assert "function_longbench" in JOB_CATALOGUE
    assert "function_longcodebench" in JOB_CATALOGUE
    assert "function_mcp_bench" in JOB_CATALOGUE
    assert "function_api_bank" in JOB_CATALOGUE
    assert "function_agentbench" in JOB_CATALOGUE
    assert "function_bfcl_v3" in JOB_CATALOGUE
    assert "function_bfcl_ast" in JOB_CATALOGUE
    assert "function_bfcl_exec" in JOB_CATALOGUE
    assert "function_toolalpaca" in JOB_CATALOGUE
    assert "function_complexfuncbench" in JOB_CATALOGUE
    assert "function_tau_bench" in JOB_CATALOGUE
    assert "function_tau2_bench" in JOB_CATALOGUE
    assert "function_tau3_bench" in JOB_CATALOGUE

    browsecomp_slugs = JOB_CATALOGUE["function_browsecomp"].dataset_slugs
    longbench_slugs = JOB_CATALOGUE["function_longbench"].dataset_slugs
    longcodebench_slugs = JOB_CATALOGUE["function_longcodebench"].dataset_slugs
    mcp_slugs = JOB_CATALOGUE["function_mcp_bench"].dataset_slugs
    api_bank_slugs = JOB_CATALOGUE["function_api_bank"].dataset_slugs
    agentbench_slugs = JOB_CATALOGUE["function_agentbench"].dataset_slugs
    bfcl_slugs = JOB_CATALOGUE["function_bfcl_v3"].dataset_slugs
    bfcl_ast_slugs = JOB_CATALOGUE["function_bfcl_ast"].dataset_slugs
    bfcl_exec_slugs = JOB_CATALOGUE["function_bfcl_exec"].dataset_slugs
    toolalpaca_slugs = JOB_CATALOGUE["function_toolalpaca"].dataset_slugs
    tau_slugs = JOB_CATALOGUE["function_tau_bench"].dataset_slugs
    complex_slugs = JOB_CATALOGUE["function_complexfuncbench"].dataset_slugs
    tau2_slugs = JOB_CATALOGUE["function_tau2_bench"].dataset_slugs
    tau3_slugs = JOB_CATALOGUE["function_tau3_bench"].dataset_slugs

    assert canonical_slug("browsecomp_test") in browsecomp_slugs
    assert canonical_slug("browsecomp_zh_test") in browsecomp_slugs
    assert canonical_slug("longbench_test") in longbench_slugs
    assert canonical_slug("longbench_qa_test") in longbench_slugs
    assert canonical_slug("longbench_qa_balanced_test") in longbench_slugs
    assert canonical_slug("longcodeqa_test") in longcodebench_slugs
    assert canonical_slug("mcp_bench_test") in mcp_slugs
    assert canonical_slug("apibank_level1_test") in api_bank_slugs
    assert canonical_slug("apibank_level2_test") in api_bank_slugs
    assert canonical_slug("agentbench_db_test") in agentbench_slugs
    assert canonical_slug("agentbench_kg_test") in agentbench_slugs
    assert canonical_slug("bfcl_v3_test") in bfcl_slugs
    assert canonical_slug("bfcl_simple_python_test") in bfcl_ast_slugs
    assert canonical_slug("bfcl_exec_simple_ast_test") in bfcl_ast_slugs
    assert canonical_slug("bfcl_multiple_test") in bfcl_ast_slugs
    assert canonical_slug("bfcl_exec_multiple_ast_test") in bfcl_ast_slugs
    assert canonical_slug("bfcl_exec_simple_test") in bfcl_exec_slugs
    assert canonical_slug("bfcl_exec_multiple_test") in bfcl_exec_slugs
    assert canonical_slug("bfcl_exec_parallel_test") in bfcl_exec_slugs
    assert canonical_slug("bfcl_exec_parallel_multiple_test") in bfcl_exec_slugs
    assert canonical_slug("toolalpaca_eval_simulated_test") in toolalpaca_slugs
    assert canonical_slug("toolalpaca_eval_real_test") in toolalpaca_slugs
    assert canonical_slug("tau_bench_retail_test") in tau_slugs
    assert canonical_slug("tau_bench_airline_test") in tau_slugs
    assert canonical_slug("tau_bench_telecom_test") in tau_slugs
    assert canonical_slug("complexfuncbench_official_test") in complex_slugs
    assert canonical_slug("tau2_bench_retail_base") in tau2_slugs
    assert canonical_slug("tau2_bench_airline_base") in tau2_slugs
    assert canonical_slug("tau2_bench_telecom_base") in tau2_slugs
    assert canonical_slug("tau3_bench_retail_base") in tau3_slugs
    assert canonical_slug("tau3_bench_airline_base") in tau3_slugs
    assert canonical_slug("tau3_bench_telecom_base") in tau3_slugs
    assert canonical_slug("tau3_bench_banking_knowledge_base") in tau3_slugs
    assert canonical_slug("tau3_bench_mock_base") in tau3_slugs
    assert canonical_slug("tau3_bench_mock_long_context_base") in tau3_slugs

    assert detect_job_from_dataset(canonical_slug("browsecomp_test"), is_cot=True) == "function_browsecomp"
    assert detect_job_from_dataset(canonical_slug("longbench_qa_test"), is_cot=True) == "function_longbench"
    assert detect_job_from_dataset(canonical_slug("longbench_qa_balanced_test"), is_cot=True) == "function_longbench"
    assert detect_job_from_dataset(canonical_slug("longcodeqa_test"), is_cot=True) == "function_longcodebench"
    assert detect_job_from_dataset(canonical_slug("mcp_bench_test"), is_cot=True) == "function_mcp_bench"
    assert detect_job_from_dataset(canonical_slug("apibank_level1_test"), is_cot=True) == "function_api_bank"
    assert detect_job_from_dataset(canonical_slug("agentbench_db_test"), is_cot=True) == "function_agentbench"
    assert detect_job_from_dataset(canonical_slug("bfcl_v3_test"), is_cot=True) == "function_bfcl_v3"
    assert detect_job_from_dataset(canonical_slug("bfcl_multiple_test"), is_cot=True) == "function_bfcl_ast"
    assert detect_job_from_dataset(canonical_slug("bfcl_exec_simple_ast_test"), is_cot=True) == "function_bfcl_ast"
    assert detect_job_from_dataset(canonical_slug("bfcl_exec_simple_test"), is_cot=True) == "function_bfcl_exec"
    assert detect_job_from_dataset(canonical_slug("toolalpaca_eval_real_test"), is_cot=True) == "function_toolalpaca"
    assert detect_job_from_dataset(canonical_slug("tau_bench_retail_test"), is_cot=True) == "function_tau_bench"
    assert detect_job_from_dataset(canonical_slug("tau2_bench_telecom_base"), is_cot=True) == "function_tau2_bench"
    assert (
        detect_job_from_dataset(canonical_slug("complexfuncbench_official_test"), is_cot=True)
        == "function_complexfuncbench"
    )
    assert (
        detect_job_from_dataset(canonical_slug("tau3_bench_banking_knowledge_base"), is_cot=True)
        == "function_tau3_bench"
    )
    assert (
        detect_job_from_dataset(canonical_slug("tau3_bench_mock_long_context_base"), is_cot=True)
        == "function_tau3_bench"
    )


def test_jobs_module_does_not_eagerly_import_data_manager() -> None:
    module_name = "src.eval.scheduler.jobs"
    data_manager_name = "src.eval.datasets.data_prepper.data_manager"

    sys.modules.pop(module_name, None)
    sys.modules.pop(data_manager_name, None)

    importlib.import_module(module_name)

    assert data_manager_name not in sys.modules
