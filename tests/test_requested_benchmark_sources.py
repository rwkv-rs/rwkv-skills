from __future__ import annotations

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.benchmark_sources import (
    AGENT_LOOP_BENCHMARK_SOURCES,
    AGENT_TOOL_CALL_BENCHMARK_SOURCES,
    REQUESTED_BENCHMARK_SOURCES,
)
from src.eval.datasets.data_prepper.data_manager import (
    available_function_calling_datasets,
    available_multiple_choice_datasets,
)
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset


_REQUESTED_DISPLAY_NAMES = {
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
    "HLE with tools",
    "Hy-Euler pro",
    "GPQA Diamond",
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
    function_datasets = set(available_function_calling_datasets())
    multiple_choice_datasets = set(available_multiple_choice_datasets())

    for item in REQUESTED_BENCHMARK_SOURCES:
        if item.integration in {"agent_tool_call", "agent_loop", "function_browsecomp"}:
            assert item.benchmark_name in function_datasets
        elif item.integration == "multiple_choice":
            assert item.benchmark_name.split("_", 1)[0] in multiple_choice_datasets
        else:
            raise AssertionError(f"unknown integration kind: {item.integration}")


def test_requested_benchmark_source_fields_are_intentional() -> None:
    fields_by_integration = {
        "agent_tool_call": BenchmarkField.FUNCTION_CALLING,
        "agent_loop": BenchmarkField.FUNCTION_CALLING,
        "function_browsecomp": BenchmarkField.FUNCTION_CALLING,
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


def test_agent_loop_source_catalog_is_derived_from_requested_benchmarks() -> None:
    assert {item.benchmark_name for item in AGENT_LOOP_BENCHMARK_SOURCES} == {
        item.benchmark_name
        for item in REQUESTED_BENCHMARK_SOURCES
        if item.integration == "agent_loop"
    }


def test_agent_loop_benchmarks_have_explicit_instruction_sampling_config() -> None:
    sampling = resolve_sampling_config(
        "widesearch_test",
        "demo-model",
        stage="tool",
        fallback_templates="function_call_default",
    )

    assert sampling is not None
    assert sampling.max_generate_tokens == 4096
    assert sampling.top_k == 50


def test_benchmark_domains_partition_the_requested_set() -> None:
    from src.eval.benchmark_sources import BENCHMARKS_BY_DOMAIN, BENCHMARK_DOMAINS

    assert set(BENCHMARK_DOMAINS) == {"knowledge", "math", "code", "agent"}
    partitioned = [item.benchmark_name for items in BENCHMARKS_BY_DOMAIN.values() for item in items]
    assert sorted(partitioned) == sorted(item.benchmark_name for item in REQUESTED_BENCHMARK_SOURCES)

    assert {item.benchmark_name for item in BENCHMARKS_BY_DOMAIN["code"]} == set()
    assert {item.benchmark_name for item in BENCHMARKS_BY_DOMAIN["agent"]} == {
        "terminal_bench_2_1",
        "nl2repo",
        "deepswe",
        "browsecomp",
        "widesearch",
        "deepsearchqa",
        "mcp_atlas",
        "toolathlon",
        "apex_agents",
        "claweval",
        "wildclawbench",
        "skillsbench",
        "hle_with_tools",
        "hy_backend_2_0",
        "hy_swe_max",
        "hy_companybench",
        "e_bench",
        "hy_finmodelbench",
        "prodbench",
        "hy_skillsworld",
        "hy_euler_pro",
    }
    assert {item.benchmark_name for item in BENCHMARKS_BY_DOMAIN["math"]} == set()
    assert {item.benchmark_name for item in BENCHMARKS_BY_DOMAIN["knowledge"]} == {"gpqa_diamond"}

    integrations_by_domain = {
        "agent": {"agent_loop", "function_browsecomp"},
        "code": set(),
        "math": set(),
        "knowledge": {"multiple_choice"},
    }
    for item in REQUESTED_BENCHMARK_SOURCES:
        assert item.integration in integrations_by_domain[item.domain], item.benchmark_name
