from __future__ import annotations

from src.eval.benchmark_registry import (
    ALL_BENCHMARKS,
    AUTO_TARGET_ATTEMPTS,
    BENCHMARK_ALIASES,
    BENCHMARKS_BY_FIELD,
    BenchmarkField,
    CoTMode,
    expand_benchmark_alias,
    get_benchmarks_with_field,
    resolve_benchmark_metadata,
    supports_cot_mode,
)


def test_mmlu_metadata_is_two_mode_knowledge_zeroshot() -> None:
    metadata = resolve_benchmark_metadata("mmlu_test")

    assert metadata.name == "mmlu"
    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.cot_modes == (CoTMode.NO_COT, CoTMode.COT)
    assert metadata.default_split == "test"
    assert metadata.scheduler_jobs == (
        "multi_choice_plain",
        "multi_choice_cot",
        "multi_choice_plain_naive",
        "multi_choice_cot_naive",
    )
    assert metadata.n_shots == (0,)
    assert metadata.avg_ks == ()
    assert metadata.pass_ks == ()
    assert metadata.target_eval_attempts == AUTO_TARGET_ATTEMPTS


def test_include_defaults_to_knowledge_test_split() -> None:
    metadata = resolve_benchmark_metadata("include_test")

    assert metadata.name == "include"
    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.default_split == "test"
    assert metadata.scheduler_jobs == (
        "multi_choice_plain",
        "multi_choice_cot",
        "multi_choice_plain_naive",
        "multi_choice_cot_naive",
    )


def test_gpqa_variants_use_explicit_catalog_entries_with_shared_dataset_source() -> None:
    metadata = resolve_benchmark_metadata("gpqa_diamond_test")

    assert metadata.name == "gpqa_diamond"
    assert metadata.dataset == "gpqa"
    assert metadata.default_split == "diamond"
    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.cot_modes == (CoTMode.NO_COT, CoTMode.COT)
    assert metadata.scheduler_jobs == (
        "multi_choice_plain",
        "multi_choice_cot",
        "multi_choice_plain_naive",
        "multi_choice_cot_naive",
    )
    assert metadata.n_shots == (0,)


def test_human_eval_family_is_no_cot_only() -> None:
    metadata = resolve_benchmark_metadata("human_eval_plus_test")

    assert metadata.field is BenchmarkField.CODING
    assert metadata.cot_modes == (CoTMode.NO_COT,)
    assert metadata.scheduler_jobs == ("code_human_eval", "code_human_eval_naive")
    assert supports_cot_mode("human_eval_plus_test", CoTMode.NO_COT)
    assert not supports_cot_mode("human_eval_plus_test", CoTMode.COT)


def test_mbpp_family_is_legacy_no_cot_only() -> None:
    metadata = resolve_benchmark_metadata("mbpp_plus_test")

    assert metadata.field is BenchmarkField.CODING
    assert metadata.cot_modes == (CoTMode.NO_COT,)
    assert metadata.scheduler_jobs == ("code_mbpp", "code_mbpp_naive")
    assert supports_cot_mode("mbpp_plus_test", CoTMode.NO_COT)
    assert not supports_cot_mode("mbpp_plus_test", CoTMode.COT)


def test_swe_bench_family_is_coding_cot_only_with_naive_variant() -> None:
    metadata = resolve_benchmark_metadata("swe_bench_lite_test")

    assert metadata.field is BenchmarkField.CODING
    assert metadata.cot_modes == (CoTMode.COT,)
    assert metadata.scheduler_jobs == ("code_swe_bench", "code_swe_bench_naive")
    assert supports_cot_mode("swe_bench_lite_test", CoTMode.COT)
    assert not supports_cot_mode("swe_bench_lite_test", CoTMode.NO_COT)


def test_function_calling_benchmarks_are_cot_only() -> None:
    browsecomp = resolve_benchmark_metadata("browsecomp_zh_test")
    complexfuncbench = resolve_benchmark_metadata("complexfuncbench_official_test")
    complexfuncbench_subset = resolve_benchmark_metadata("complexfuncbench_subset_test")
    longbench = resolve_benchmark_metadata("longbench_qa_test")
    longbench_balanced = resolve_benchmark_metadata("longbench_qa_balanced_test")
    longcodeqa = resolve_benchmark_metadata("longcodeqa_test")
    mcp_bench = resolve_benchmark_metadata("mcp_bench_test")
    mcp_single = resolve_benchmark_metadata("mcp_bench_single_test")
    mcp_multi = resolve_benchmark_metadata("mcp_bench_multi_2server_test")
    apibank_alias = resolve_benchmark_metadata("apibank_l1_test")
    apibank = resolve_benchmark_metadata("apibank_level1_test")
    agentbench = resolve_benchmark_metadata("agentbench_db_test")
    bfcl_ast = resolve_benchmark_metadata("bfcl_simple_python_test")
    bfcl_exec_ast = resolve_benchmark_metadata("bfcl_exec_simple_ast_test")
    bfcl_exec = resolve_benchmark_metadata("bfcl_exec_simple_test")
    bfcl_v3 = resolve_benchmark_metadata("bfcl_v3_test")
    toolalpaca = resolve_benchmark_metadata("toolalpaca_eval_simulated_test")
    complexfuncbench = resolve_benchmark_metadata("complexfuncbench_official_test")
    tau_bench = resolve_benchmark_metadata("tau_bench_airline_test")
    tau2_bench = resolve_benchmark_metadata("tau2_bench_retail_base")
    tau3_bench = resolve_benchmark_metadata("tau3_bench_banking_knowledge_base")
    tau3_mock = resolve_benchmark_metadata("tau3_bench_mock_long_context_base")

    assert browsecomp.field is BenchmarkField.FUNCTION_CALLING
    assert browsecomp.cot_modes == (CoTMode.COT,)
    assert browsecomp.scheduler_jobs == ("function_browsecomp",)
    assert complexfuncbench.field is BenchmarkField.FUNCTION_CALLING
    assert complexfuncbench.cot_modes == (CoTMode.COT,)
    assert complexfuncbench.scheduler_jobs == ("function_complexfuncbench",)
    assert complexfuncbench_subset.field is BenchmarkField.FUNCTION_CALLING
    assert complexfuncbench_subset.scheduler_jobs == ("function_complexfuncbench",)
    assert longbench.field is BenchmarkField.FUNCTION_CALLING
    assert longbench.cot_modes == (CoTMode.COT,)
    assert longbench.scheduler_jobs == ("function_longbench",)
    assert longbench_balanced.field is BenchmarkField.FUNCTION_CALLING
    assert longbench_balanced.cot_modes == (CoTMode.COT,)
    assert longbench_balanced.scheduler_jobs == ("function_longbench",)
    assert longcodeqa.field is BenchmarkField.FUNCTION_CALLING
    assert longcodeqa.cot_modes == (CoTMode.COT,)
    assert longcodeqa.scheduler_jobs == ("function_longcodebench",)
    assert mcp_bench.field is BenchmarkField.FUNCTION_CALLING
    assert mcp_bench.scheduler_jobs == ("function_mcp_bench",)
    assert mcp_single.field is BenchmarkField.FUNCTION_CALLING
    assert mcp_single.scheduler_jobs == ("function_mcp_bench",)
    assert mcp_multi.field is BenchmarkField.FUNCTION_CALLING
    assert mcp_multi.scheduler_jobs == ("function_mcp_bench",)
    assert apibank_alias.field is BenchmarkField.FUNCTION_CALLING
    assert apibank_alias.scheduler_jobs == ("function_api_bank",)
    assert apibank.field is BenchmarkField.FUNCTION_CALLING
    assert apibank.scheduler_jobs == ("function_api_bank",)
    assert agentbench.field is BenchmarkField.FUNCTION_CALLING
    assert agentbench.scheduler_jobs == ("function_agentbench",)
    assert bfcl_ast.field is BenchmarkField.FUNCTION_CALLING
    assert bfcl_ast.scheduler_jobs == ("function_bfcl_ast",)
    assert bfcl_exec_ast.field is BenchmarkField.FUNCTION_CALLING
    assert bfcl_exec_ast.scheduler_jobs == ("function_bfcl_ast",)
    assert bfcl_exec_ast.name == "bfcl_exec_simple_ast"
    assert bfcl_exec.field is BenchmarkField.FUNCTION_CALLING
    assert bfcl_exec.scheduler_jobs == ("function_bfcl_exec",)
    assert bfcl_exec.name == "bfcl_exec_simple"
    assert bfcl_v3.field is BenchmarkField.FUNCTION_CALLING
    assert bfcl_v3.scheduler_jobs == ("function_bfcl_v3",)
    assert toolalpaca.field is BenchmarkField.FUNCTION_CALLING
    assert toolalpaca.scheduler_jobs == ("function_toolalpaca",)
    assert complexfuncbench.field is BenchmarkField.FUNCTION_CALLING
    assert complexfuncbench.scheduler_jobs == ("function_complexfuncbench",)
    assert tau_bench.field is BenchmarkField.FUNCTION_CALLING
    assert tau_bench.scheduler_jobs == ("function_tau_bench",)
    assert tau2_bench.default_split == "base"
    assert tau2_bench.scheduler_jobs == ("function_tau2_bench",)
    assert tau3_bench.default_split == "base"
    assert tau3_bench.scheduler_jobs == ("function_tau3_bench",)
    assert tau3_mock.field is BenchmarkField.FUNCTION_CALLING
    assert tau3_mock.default_split == "base"
    assert tau3_mock.scheduler_jobs == ("function_tau3_bench",)


def test_instruction_following_benchmarks_are_no_cot_only() -> None:
    metadata = resolve_benchmark_metadata("ifbench_test")

    assert metadata.field is BenchmarkField.INSTRUCTION_FOLLOWING
    assert metadata.cot_modes == (CoTMode.NO_COT,)
    assert metadata.scheduler_jobs == ("instruction_following", "instruction_following_naive")


def test_instruction_following_data_only_benchmarks_do_not_use_ifeval_scorer() -> None:
    flores = resolve_benchmark_metadata("flores200_devtest")
    arena = resolve_benchmark_metadata("arena_hard_test")
    wmt = resolve_benchmark_metadata("wmt24pp_test")

    assert flores.field is BenchmarkField.INSTRUCTION_FOLLOWING
    assert flores.default_split == "devtest"
    assert flores.scheduler_jobs == ()
    assert arena.field is BenchmarkField.INSTRUCTION_FOLLOWING
    assert arena.scheduler_jobs == ()
    assert wmt.field is BenchmarkField.INSTRUCTION_FOLLOWING
    assert wmt.scheduler_jobs == ()


def test_benchmark_aliases_expand_rwkv_rs_style_group_names() -> None:
    assert expand_benchmark_alias("gpqa") == (
        "gpqa_main",
        "gpqa_extended",
        "gpqa_diamond",
    )
    assert BENCHMARK_ALIASES["arena_hard"] == ("arena_hard_v2",)
    assert expand_benchmark_alias("tau_bench") == (
        "tau_bench_retail",
        "tau_bench_airline",
        "tau_bench_telecom",
    )
    assert expand_benchmark_alias("tau3_bench") == (
        "tau3_bench_retail",
        "tau3_bench_airline",
        "tau3_bench_telecom",
        "tau3_bench_banking_knowledge",
        "tau3_bench_mock",
        "tau3_bench_mock_long_context",
    )
    assert expand_benchmark_alias("apibank") == (
        "apibank_l1",
        "apibank_l2",
        "apibank_level1",
        "apibank_level2",
    )


def test_simpleqa_defaults_to_cot_only_maths() -> None:
    metadata = resolve_benchmark_metadata("simpleqa_verified")

    assert metadata.field is BenchmarkField.MATHS
    assert metadata.cot_modes == (CoTMode.COT,)
    assert metadata.default_split == "verified"
    assert metadata.scheduler_jobs == ("free_response", "free_response_naive")
    assert metadata.n_shots == (0,)
    assert metadata.pass_ks == ()


def test_polymath_defaults_to_all_split_maths() -> None:
    metadata = resolve_benchmark_metadata("polymath_all")

    assert metadata.name == "polymath"
    assert metadata.field is BenchmarkField.MATHS
    assert metadata.default_split == "all"
    assert metadata.scheduler_jobs == ("free_response", "free_response_naive")


def test_judge_only_math_benchmarks_route_to_judge_runner() -> None:
    metadata = resolve_benchmark_metadata("gsm8k_test")

    assert metadata.field is BenchmarkField.MATHS
    assert metadata.scheduler_jobs == ("free_response_judge", "free_response_judge_naive")


def test_catalog_names_are_not_polluted_by_dataset_slug_aliases() -> None:
    assert resolve_benchmark_metadata("mbpp_test").name == "mbpp"
    assert resolve_benchmark_metadata("cmmlu_test").name == "cmmlu"
    assert resolve_benchmark_metadata("mmmlu_test").name == "mmmlu"


def test_benchmarks_are_grouped_by_field_like_rwkv_rs() -> None:
    assert ALL_BENCHMARKS == tuple(sorted(ALL_BENCHMARKS, key=lambda item: (item.field.value, item.name)))

    knowledge = get_benchmarks_with_field(BenchmarkField.KNOWLEDGE)
    maths = get_benchmarks_with_field(BenchmarkField.MATHS)
    coding = BENCHMARKS_BY_FIELD[BenchmarkField.CODING]

    assert any(item.name == "mmlu" for item in knowledge)
    assert any(item.name == "gpqa_main" for item in knowledge)
    assert any(item.name == "gsm8k" for item in maths)
    assert any(item.name == "human_eval" for item in coding)
    assert all(item.field is BenchmarkField.KNOWLEDGE for item in knowledge)
