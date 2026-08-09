from __future__ import annotations

"""Source catalog for the user-requested benchmark integration set."""

from dataclasses import dataclass


BENCHMARK_DOMAINS: tuple[str, ...] = ("knowledge", "math", "code", "agent")


@dataclass(frozen=True, slots=True)
class RequestedBenchmarkSource:
    display_name: str
    benchmark_name: str
    dataset_slug: str
    integration: str
    source_kind: str
    source_url: str | None
    scheduler_job: str
    domain: str

    def __post_init__(self) -> None:
        if self.domain not in BENCHMARK_DOMAINS:
            raise ValueError(f"{self.benchmark_name}: unknown benchmark domain {self.domain!r}")


REQUESTED_BENCHMARK_SOURCES: tuple[RequestedBenchmarkSource, ...] = (
    RequestedBenchmarkSource(
        "SWE-bench Multilingual",
        "swe_bench_multilingual",
        "swe_bench_multilingual_test",
        "coding_swe_bench",
        "github",
        "https://github.com/swe-bench/SWE-bench",
        "code_swe_bench",
        domain="code",
    ),
    RequestedBenchmarkSource(
        "SWE-bench Verified",
        "swe_bench_verified",
        "swe_bench_verified_test",
        "coding_swe_bench",
        "github",
        "https://github.com/swe-bench/SWE-bench",
        "code_swe_bench",
        domain="code",
    ),
    RequestedBenchmarkSource(
        "SWE-bench Pro",
        "swe_bench_pro",
        "swe_bench_pro_test",
        "coding_swe_bench",
        "github",
        "https://github.com/scaleapi/SWE-bench_Pro-os",
        "code_swe_bench",
        domain="code",
    ),
    RequestedBenchmarkSource(
        "Terminal-Bench 2.1",
        "terminal_bench_2_1",
        "terminal_bench_2_1_test",
        "agent_loop",
        "github",
        "https://github.com/harbor-framework/terminal-bench-2-1",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "NL2Repo",
        "nl2repo",
        "nl2repo_test",
        "agent_loop",
        "github",
        "https://github.com/multimodal-art-projection/NL2RepoBench",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "DeepSWE",
        "deepswe",
        "deepswe_test",
        "agent_loop",
        "github",
        "https://github.com/datacurve-ai/deep-swe",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "Hy-Backend 2.0",
        "hy_backend_2_0",
        "hy_backend_2_0_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "Hy-SWE Max",
        "hy_swe_max",
        "hy_swe_max_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "Hy-CompanyBench",
        "hy_companybench",
        "hy_companybench_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "BrowseComp",
        "browsecomp",
        "browsecomp_test",
        "function_browsecomp",
        "github",
        "https://github.com/openai/simple-evals",
        "function_browsecomp",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "WideSearch",
        "widesearch",
        "widesearch_test",
        "agent_loop",
        "github",
        "https://github.com/ByteDance-Seed/WideSearch",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "DeepSearchQA",
        "deepsearchqa",
        "deepsearchqa_test",
        "agent_loop",
        "hf_dataset",
        "https://huggingface.co/datasets/google/deepsearchqa",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "MCP Atlas",
        "mcp_atlas",
        "mcp_atlas_test",
        "agent_loop",
        "github",
        "https://github.com/scaleapi/mcp-atlas",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "Toolathlon",
        "toolathlon",
        "toolathlon_test",
        "agent_loop",
        "github",
        "https://github.com/hkust-nlp/Toolathlon",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "APEX-Agents",
        "apex_agents",
        "apex_agents_test",
        "agent_loop",
        "github",
        "https://github.com/Mercor-Intelligence/archipelago",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "ClawEval",
        "claweval",
        "claweval_test",
        "agent_loop",
        "github",
        "https://github.com/claw-eval/claw-eval",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "WildClawBench",
        "wildclawbench",
        "wildclawbench_test",
        "agent_loop",
        "github",
        "https://github.com/InternLM/WildClawBench",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "SkillsBench",
        "skillsbench",
        "skillsbench_test",
        "agent_loop",
        "github",
        "https://github.com/benchflow-ai/skillsbench",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "e-bench",
        "e_bench",
        "e_bench_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "Hy-FinModelBench",
        "hy_finmodelbench",
        "hy_finmodelbench_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "ProdBench",
        "prodbench",
        "prodbench_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "Hy-SkillsWorld",
        "hy_skillsworld",
        "hy_skillsworld_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "HLE",
        "hle",
        "hle_all",
        "free_answer",
        "github",
        "https://github.com/centerforaisafety/hle",
        "free_response",
        domain="knowledge",
    ),
    RequestedBenchmarkSource(
        "HLE with tools",
        "hle_with_tools",
        "hle_with_tools_test",
        "agent_loop",
        "hf_dataset",
        "https://huggingface.co/datasets/cais/hle",
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "Hy-Euler pro",
        "hy_euler_pro",
        "hy_euler_pro_test",
        "agent_loop",
        "internal",
        None,
        "function_agent_loop",
        domain="agent",
    ),
    RequestedBenchmarkSource(
        "GPQA Diamond",
        "gpqa_diamond",
        "gpqa_diamond",
        "multiple_choice",
        "github",
        "https://github.com/idavidrein/gpqa",
        "multi_choice_cot",
        domain="knowledge",
    ),
    RequestedBenchmarkSource(
        "FrontierScience-Research",
        "frontierscience_research",
        "frontierscience_research_test",
        "free_answer",
        "official_page",
        "https://openai.com/index/frontierscience/",
        "free_response_judge",
        domain="knowledge",
    ),
    RequestedBenchmarkSource(
        "FrontierScience-Olympiad",
        "frontierscience_olympiad",
        "frontierscience_olympiad_test",
        "free_answer",
        "official_page",
        "https://openai.com/index/frontierscience/",
        "free_response_judge",
        domain="knowledge",
    ),
    RequestedBenchmarkSource(
        "USAMO 2026",
        "usamo_2026",
        "usamo_2026_test",
        "free_answer",
        "github",
        "https://github.com/eth-sri/matharena",
        "free_response_judge",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "MathArena Apex",
        "matharena_apex",
        "matharena_apex_test",
        "free_answer",
        "github",
        "https://github.com/eth-sri/matharena",
        "free_response",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "ArxivMath",
        "arxivmath",
        "arxivmath_test",
        "free_answer",
        "github",
        "https://github.com/eth-sri/matharena",
        "free_response",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "HorizonMath",
        "horizonmath",
        "horizonmath_test",
        "free_answer",
        "github",
        "https://github.com/ewang26/HorizonMath",
        "free_response",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "Hy-Math",
        "hy_math",
        "hy_math_test",
        "free_answer",
        "internal",
        None,
        "free_response",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "PHYBench",
        "phybench",
        "phybench_test",
        "free_answer",
        "github",
        "https://github.com/phybench-official/phybench",
        "free_response_judge",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "CMT-Benchmark",
        "cmt_benchmark",
        "cmt_benchmark_test",
        "free_answer",
        "github",
        "https://github.com/JamesRoggeveen/cmt_benchmark_data",
        "free_response_judge",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "IMOAnswerBench",
        "imoanswerbench",
        "imoanswerbench_test",
        "free_answer",
        "github",
        "https://github.com/google-deepmind/superhuman",
        "free_response",
        domain="math",
    ),
    RequestedBenchmarkSource(
        "SuperChem",
        "superchem",
        "superchem_test",
        "free_answer",
        "github",
        "https://github.com/catalystforyou/SUPERChem_eval",
        "free_response_judge",
        domain="knowledge",
    ),
    RequestedBenchmarkSource(
        "CL-bench",
        "cl_bench",
        "cl_bench_test",
        "free_answer",
        "github",
        "https://github.com/Tencent-Hunyuan/CL-bench",
        "free_response_judge",
        domain="knowledge",
    ),
    RequestedBenchmarkSource(
        "CL-bench life",
        "cl_bench_life",
        "cl_bench_life_test",
        "free_answer",
        "github",
        "https://github.com/Tencent-Hunyuan/CL-bench",
        "free_response_judge",
        domain="knowledge",
    ),
    RequestedBenchmarkSource(
        "AA-LCR",
        "aa_lcr",
        "aa_lcr_test",
        "free_answer",
        "hf_dataset",
        "https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR",
        "free_response_judge",
        domain="knowledge",
    ),
)


AGENT_TOOL_CALL_BENCHMARK_SOURCES: tuple[RequestedBenchmarkSource, ...] = tuple(
    item for item in REQUESTED_BENCHMARK_SOURCES if item.integration == "agent_tool_call"
)

AGENT_LOOP_BENCHMARK_SOURCES: tuple[RequestedBenchmarkSource, ...] = tuple(
    item for item in REQUESTED_BENCHMARK_SOURCES if item.integration == "agent_loop"
)

FREE_ANSWER_BENCHMARK_SOURCES: tuple[RequestedBenchmarkSource, ...] = tuple(
    item for item in REQUESTED_BENCHMARK_SOURCES if item.integration == "free_answer"
)

REQUESTED_BENCHMARKS_BY_NAME: dict[str, RequestedBenchmarkSource] = {
    item.benchmark_name: item for item in REQUESTED_BENCHMARK_SOURCES
}

BENCHMARKS_BY_DOMAIN: dict[str, tuple[RequestedBenchmarkSource, ...]] = {
    domain: tuple(item for item in REQUESTED_BENCHMARK_SOURCES if item.domain == domain)
    for domain in BENCHMARK_DOMAINS
}


__all__ = [
    "AGENT_LOOP_BENCHMARK_SOURCES",
    "AGENT_TOOL_CALL_BENCHMARK_SOURCES",
    "BENCHMARKS_BY_DOMAIN",
    "BENCHMARK_DOMAINS",
    "FREE_ANSWER_BENCHMARK_SOURCES",
    "REQUESTED_BENCHMARKS_BY_NAME",
    "REQUESTED_BENCHMARK_SOURCES",
    "RequestedBenchmarkSource",
]
