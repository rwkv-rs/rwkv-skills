from __future__ import annotations

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.benchmark_sources import (
    AGENT_TOOL_CALL_BENCHMARK_SOURCES,
    REQUESTED_BENCHMARK_SOURCES,
)
from src.eval.datasets.data_prepper.data_manager import (
    available_code_generation_datasets,
    available_free_answer_datasets,
    available_function_calling_datasets,
    available_multiple_choice_datasets,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset


_REQUESTED_DISPLAY_NAMES = {
    "SWE-bench Multilingual",
    "SWE-bench Verified",
    "SWE-bench Pro",
    "Terminal-Bench 2.1",
    "NL2Repo",
    "DeepSWE",
    "Hy-Backend 2.0",
    "Hy-SWE Max",
    "Hy-CompanyBench",
    "BrowseComp",
    "WideSearch",
    "DeepSearchQA",
    "MCP Atlas",
    "Toolathlon",
    "APEX-Agents",
    "ClawEval",
    "WildClawBench",
    "SkillsBench",
    "e-bench",
    "Hy-FinModelBench",
    "ProdBench",
    "Hy-SkillsWorld",
    "HLE",
    "Hy-Euler pro",
    "GPQA Diamond",
    "FrontierScience-Research",
    "FrontierScience-Olympiad",
    "USAMO 2026",
    "MathArena Apex",
    "ArxivMath",
    "HorizonMath",
    "Hy-Math",
    "PHYBench",
    "CMT-Benchmark",
    "IMOAnswerBench",
    "SuperChem",
    "CL-bench",
    "CL-bench life",
    "AA-LCR",
}


def test_requested_benchmark_source_catalog_matches_user_table() -> None:
    names = {item.display_name for item in REQUESTED_BENCHMARK_SOURCES}

    assert names == _REQUESTED_DISPLAY_NAMES
    assert len(REQUESTED_BENCHMARK_SOURCES) == len(_REQUESTED_DISPLAY_NAMES)
    assert len({item.benchmark_name for item in REQUESTED_BENCHMARK_SOURCES}) == len(_REQUESTED_DISPLAY_NAMES)
    assert len({item.dataset_slug for item in REQUESTED_BENCHMARK_SOURCES}) == len(_REQUESTED_DISPLAY_NAMES)


def test_requested_benchmark_sources_resolve_to_scheduler_jobs() -> None:
    for item in REQUESTED_BENCHMARK_SOURCES:
        metadata = resolve_benchmark_metadata(item.dataset_slug)
        assert metadata.name == item.benchmark_name
        assert item.scheduler_job in JOB_CATALOGUE
        assert canonical_slug(item.dataset_slug) in JOB_CATALOGUE[item.scheduler_job].dataset_slugs
        assert detect_job_from_dataset(item.dataset_slug, is_cot=True) == item.scheduler_job


def test_requested_benchmark_sources_have_matching_dataset_preppers() -> None:
    code_datasets = set(available_code_generation_datasets())
    free_answer_datasets = set(available_free_answer_datasets())
    function_datasets = set(available_function_calling_datasets())
    multiple_choice_datasets = set(available_multiple_choice_datasets())

    for item in REQUESTED_BENCHMARK_SOURCES:
        if item.integration == "coding_swe_bench":
            assert item.benchmark_name in code_datasets
        elif item.integration in {"agent_tool_call", "function_browsecomp"}:
            assert item.benchmark_name in function_datasets
        elif item.integration == "free_answer":
            assert item.benchmark_name in free_answer_datasets
        elif item.integration == "multiple_choice":
            assert item.benchmark_name.split("_", 1)[0] in multiple_choice_datasets
        else:
            raise AssertionError(f"unknown integration kind: {item.integration}")


def test_requested_benchmark_source_fields_are_intentional() -> None:
    fields_by_integration = {
        "agent_tool_call": BenchmarkField.FUNCTION_CALLING,
        "function_browsecomp": BenchmarkField.FUNCTION_CALLING,
        "coding_swe_bench": BenchmarkField.CODING,
        "free_answer": BenchmarkField.MATHS,
        "multiple_choice": BenchmarkField.KNOWLEDGE,
    }

    for item in REQUESTED_BENCHMARK_SOURCES:
        metadata = resolve_benchmark_metadata(item.dataset_slug)
        assert metadata.field is fields_by_integration[item.integration]


def test_agent_tool_call_source_catalog_is_derived_from_requested_benchmarks() -> None:
    assert {item.benchmark_name for item in AGENT_TOOL_CALL_BENCHMARK_SOURCES} == {
        item.benchmark_name
        for item in REQUESTED_BENCHMARK_SOURCES
        if item.integration == "agent_tool_call"
    }


def test_agent_tool_call_benchmarks_have_function_call_sampling_fallback() -> None:
    sampling = resolve_sampling_config(
        "widesearch_test",
        "demo-model",
        stage="tool",
        fallback_templates="function_call_default",
    )

    assert sampling is not None
    assert sampling.max_generate_tokens == 2048
    assert sampling.top_k == 200
