from __future__ import annotations

from src.eval.benchmark_config import resolve_sampling_config
from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.benchmark_sources import (
    AGENT_LOOP_BENCHMARK_SOURCES,
    AGENT_TOOL_CALL_BENCHMARK_SOURCES,
    FREE_ANSWER_BENCHMARK_SOURCES,
    REQUESTED_BENCHMARK_SOURCES,
)
from src.eval.datasets.data_prepper.data_manager import (
    available_code_generation_datasets,
    available_free_answer_datasets,
    available_function_calling_datasets,
    available_multiple_choice_datasets,
)
from src.eval.datasets.data_prepper.free_answer.requested_sets import normalize_free_answer_row
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
    "HLE with tools",
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
        elif item.integration in {"agent_tool_call", "agent_loop", "function_browsecomp"}:
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
        "agent_loop": BenchmarkField.FUNCTION_CALLING,
        "function_browsecomp": BenchmarkField.FUNCTION_CALLING,
        "coding_swe_bench": BenchmarkField.CODING,
        "free_answer": BenchmarkField.MATHS,
        "multiple_choice": BenchmarkField.KNOWLEDGE,
    }

    for item in REQUESTED_BENCHMARK_SOURCES:
        metadata = resolve_benchmark_metadata(item.dataset_slug)
        assert metadata.field is fields_by_integration[item.integration]


def test_free_answer_context_is_materialized_separately_from_problem() -> None:
    payload = normalize_free_answer_row(
        {
            "question": "What is the answer?",
            "answer": "42",
            "context": "SECRET_CONTEXT",
        },
        dataset_name="aa_lcr",
        index=0,
        source_path="unit.jsonl",
    )

    assert payload["problem"] == "What is the answer?"
    assert payload["context"] == "SECRET_CONTEXT"
    assert "SECRET_CONTEXT" not in payload["problem"]


def test_free_answer_messages_row_does_not_extract_context_separately() -> None:
    payload = normalize_free_answer_row(
        {
            "messages": [{"role": "user", "content": "Read DOC_BODY and answer."}],
            "answer": "42",
            "context": "DOC_BODY",
        },
        dataset_name="cl_bench",
        index=0,
        source_path="unit.jsonl",
    )

    assert "DOC_BODY" in payload["problem"]
    assert "context" not in payload


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
    assert {item.benchmark_name for item in FREE_ANSWER_BENCHMARK_SOURCES} == {
        item.benchmark_name
        for item in REQUESTED_BENCHMARK_SOURCES
        if item.integration == "free_answer"
    }
    # Free-answer entries route to the maths runner with the officially matching grading mode.
    for item in FREE_ANSWER_BENCHMARK_SOURCES:
        assert item.scheduler_job in {"free_response", "free_response_judge"}


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

    assert {item.benchmark_name for item in BENCHMARKS_BY_DOMAIN["code"]} == {
        "swe_bench_multilingual",
        "swe_bench_verified",
        "swe_bench_pro",
    }
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
    assert {item.benchmark_name for item in BENCHMARKS_BY_DOMAIN["math"]} == {
        "usamo_2026",
        "matharena_apex",
        "arxivmath",
        "horizonmath",
        "hy_math",
        "imoanswerbench",
        "phybench",
        "cmt_benchmark",
    }
    assert {item.benchmark_name for item in BENCHMARKS_BY_DOMAIN["knowledge"]} == {
        "hle",
        "gpqa_diamond",
        "frontierscience_research",
        "frontierscience_olympiad",
        "superchem",
        "cl_bench",
        "cl_bench_life",
        "aa_lcr",
    }

    integrations_by_domain = {
        "agent": {"agent_loop", "function_browsecomp"},
        "code": {"coding_swe_bench"},
        "math": {"free_answer"},
        "knowledge": {"free_answer", "multiple_choice"},
    }
    for item in REQUESTED_BENCHMARK_SOURCES:
        assert item.integration in integrations_by_domain[item.domain], item.benchmark_name


def test_dashboard_groups_agent_domain_separately() -> None:
    from src.dashboard.core.domains import DOMAIN_AGENT, DOMAIN_FUNCTION_CALL, domain_for_benchmark_field

    assert domain_for_benchmark_field(BenchmarkField.FUNCTION_CALLING, dataset_slug="widesearch_test") == DOMAIN_AGENT
    assert (
        domain_for_benchmark_field(BenchmarkField.FUNCTION_CALLING, dataset_slug="hle_with_tools_test")
        == DOMAIN_AGENT
    )
    assert (
        domain_for_benchmark_field(BenchmarkField.FUNCTION_CALLING, dataset_slug="bfcl_v3_test")
        == DOMAIN_FUNCTION_CALL
    )


def test_free_answer_prepper_normalizes_qa_rubric_and_message_rows() -> None:
    from src.eval.datasets.data_prepper.free_answer.requested_sets import normalize_free_answer_row

    qa = normalize_free_answer_row(
        {"id": "m-1", "problem": "1+1?", "answer": 2},
        dataset_name="matharena_apex",
        index=0,
        source_path="src.jsonl",
    )
    assert qa["problem"] == "1+1?"
    assert qa["expected_answer"] == "2"
    assert qa["source_benchmark"] == "matharena_apex"

    rubric = normalize_free_answer_row(
        {
            "id": "cl-1",
            "messages": [
                {"role": "system", "content": "You learn the rules from context."},
                {"role": "user", "content": "Apply rule R to case C."},
            ],
            "rubrics": ["states the rule", "applies it to C"],
        },
        dataset_name="cl_bench",
        index=0,
        source_path="src.jsonl",
    )
    assert rubric["problem"].startswith("System: You learn the rules")
    assert rubric["rubrics"] == ["states the rule", "applies it to C"]
    assert "states the rule" in rubric["expected_answer"]

    context = normalize_free_answer_row(
        {"id": "aa-1", "question": "Q?", "context": "long doc", "answer": "A"},
        dataset_name="aa_lcr",
        index=0,
        source_path="src.jsonl",
    )
    assert context["problem"] == "Q?"
    assert context["context"] == "long doc"

    usamo = normalize_free_answer_row(
        {"problem_idx": 1, "problem": "Prove P.", "sample_solution": "Proof.", "grading_scheme": "7 point rubric"},
        dataset_name="usamo_2026",
        index=0,
        source_path="hf",
    )
    assert usamo["id"] == "usamo_2026__00000"
    assert usamo["expected_answer"] == "Proof."
    assert usamo["rubrics"] == ["7 point rubric"]

    phybench = normalize_free_answer_row(
        {"id": 495, "content": "Physics problem", "answer": "42", "solution": "derive 42"},
        dataset_name="phybench",
        index=0,
        source_path="hf",
    )
    assert phybench["problem"] == "Physics problem"
    assert phybench["expected_answer"] == "42"
