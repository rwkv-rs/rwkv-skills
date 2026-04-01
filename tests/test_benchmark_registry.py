from __future__ import annotations

from src.eval.benchmark_registry import (
    ALL_BENCHMARKS,
    AUTO_TARGET_ATTEMPTS,
    BENCHMARKS_BY_FIELD,
    BenchmarkField,
    CoTMode,
    get_benchmarks_with_field,
    resolve_benchmark_metadata,
    supports_cot_mode,
)


def test_mmlu_metadata_is_three_mode_knowledge_zeroshot() -> None:
    metadata = resolve_benchmark_metadata("mmlu_test")

    assert metadata.name == "mmlu"
    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.cot_modes == (CoTMode.NO_COT, CoTMode.FAKE_COT, CoTMode.COT)
    assert metadata.default_split == "test"
    assert metadata.scheduler_jobs == (
        "multi_choice_plain",
        "multi_choice_fake_cot",
        "multi_choice_cot",
    )
    assert metadata.n_shots == (0,)
    assert metadata.avg_ks == ()
    assert metadata.pass_ks == ()
    assert metadata.target_eval_attempts == AUTO_TARGET_ATTEMPTS


def test_gpqa_variant_uses_prefix_fallback() -> None:
    metadata = resolve_benchmark_metadata("gpqa_diamond_test")

    assert metadata.name == "gpqa_diamond"
    assert metadata.field is BenchmarkField.KNOWLEDGE
    assert metadata.cot_modes == (CoTMode.NO_COT, CoTMode.FAKE_COT, CoTMode.COT)
    assert metadata.scheduler_jobs == (
        "multi_choice_plain",
        "multi_choice_fake_cot",
        "multi_choice_cot",
    )
    assert metadata.n_shots == (0,)


def test_human_eval_family_is_no_cot_only() -> None:
    metadata = resolve_benchmark_metadata("human_eval_plus_test")

    assert metadata.field is BenchmarkField.CODING
    assert metadata.cot_modes == (CoTMode.NO_COT,)
    assert metadata.scheduler_jobs == ("code_human_eval",)
    assert supports_cot_mode("human_eval_plus_test", CoTMode.NO_COT)
    assert not supports_cot_mode("human_eval_plus_test", CoTMode.COT)


def test_function_calling_benchmarks_are_cot_only() -> None:
    browsecomp = resolve_benchmark_metadata("browsecomp_zh_test")
    mcp_bench = resolve_benchmark_metadata("mcp_bench_test")
    tau_bench = resolve_benchmark_metadata("tau_bench_airline_test")
    tau2_bench = resolve_benchmark_metadata("tau2_bench_retail_base")

    assert browsecomp.field is BenchmarkField.FUNCTION_CALLING
    assert browsecomp.cot_modes == (CoTMode.COT,)
    assert browsecomp.scheduler_jobs == ("function_browsecomp",)
    assert mcp_bench.field is BenchmarkField.FUNCTION_CALLING
    assert mcp_bench.scheduler_jobs == ("function_mcp_bench",)
    assert tau_bench.field is BenchmarkField.FUNCTION_CALLING
    assert tau_bench.scheduler_jobs == ("function_tau_bench",)
    assert tau2_bench.default_split == "base"
    assert tau2_bench.scheduler_jobs == ("function_tau2_bench",)


def test_instruction_following_benchmarks_are_no_cot_only() -> None:
    metadata = resolve_benchmark_metadata("flores200_devtest")

    assert metadata.field is BenchmarkField.INSTRUCTION_FOLLOWING
    assert metadata.cot_modes == (CoTMode.NO_COT,)
    assert metadata.default_split == "devtest"
    assert metadata.scheduler_jobs == ("instruction_following",)


def test_simpleqa_defaults_to_cot_only_maths() -> None:
    metadata = resolve_benchmark_metadata("simpleqa_verified")

    assert metadata.field is BenchmarkField.MATHS
    assert metadata.cot_modes == (CoTMode.COT,)
    assert metadata.default_split == "verified"
    assert metadata.scheduler_jobs == ("free_response",)
    assert metadata.n_shots == (0,)
    assert metadata.pass_ks == ()


def test_judge_only_math_benchmarks_route_to_judge_runner() -> None:
    metadata = resolve_benchmark_metadata("gsm8k_test")

    assert metadata.field is BenchmarkField.MATHS
    assert metadata.scheduler_jobs == ("free_response_judge",)


def test_benchmarks_are_grouped_by_field_like_rwkv_rs() -> None:
    assert ALL_BENCHMARKS == tuple(sorted(ALL_BENCHMARKS, key=lambda item: (item.field.value, item.name)))

    knowledge = get_benchmarks_with_field(BenchmarkField.KNOWLEDGE)
    maths = get_benchmarks_with_field(BenchmarkField.MATHS)
    coding = BENCHMARKS_BY_FIELD[BenchmarkField.CODING]

    assert any(item.name == "mmlu" for item in knowledge)
    assert any(item.name == "gsm8k" for item in maths)
    assert any(item.name == "human_eval" for item in coding)
    assert all(item.field is BenchmarkField.KNOWLEDGE for item in knowledge)
