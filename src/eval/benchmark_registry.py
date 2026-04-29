from __future__ import annotations

"""Benchmark metadata aligned with rwkv-rs' evaluator matrix."""

from dataclasses import dataclass, replace
from enum import Enum

from src.eval.execution_plan import TARGET_EVAL_ATTEMPTS
from src.eval.scheduler.dataset_utils import canonical_slug, make_dataset_slug, safe_slug, split_benchmark_and_split


AUTO_TARGET_ATTEMPTS = TARGET_EVAL_ATTEMPTS


class BenchmarkField(str, Enum):
    KNOWLEDGE = "knowledge"
    MATHS = "maths"
    CODING = "coding"
    INSTRUCTION_FOLLOWING = "instruction_following"
    FUNCTION_CALLING = "function_calling"


class CoTMode(str, Enum):
    NO_COT = "no_cot"
    FAKE_COT = "fake_cot"
    COT = "cot"

    @property
    def is_cot(self) -> bool:
        return self is not CoTMode.NO_COT


@dataclass(frozen=True, slots=True)
class BenchmarkMetadata:
    name: str
    field: BenchmarkField
    cot_modes: tuple[CoTMode, ...]
    default_split: str = "test"
    dataset_name: str | None = None
    scheduler_jobs: tuple[str, ...] = ()
    n_shots: tuple[int, ...] = (0,)
    # Empty avg_ks means "derive avg@k automatically from dataset size":
    # run the benchmark once, unless it is larger than target_eval_attempts.
    avg_ks: tuple[float, ...] = ()
    pass_ks: tuple[int, ...] = ()
    target_eval_attempts: int = AUTO_TARGET_ATTEMPTS

    @property
    def dataset(self) -> str:
        return self.dataset_name or self.name


_THREE_MODE_KNOWLEDGE = (CoTMode.NO_COT, CoTMode.FAKE_COT, CoTMode.COT)
_THREE_MODE_CODE = (CoTMode.NO_COT, CoTMode.FAKE_COT, CoTMode.COT)
_COT_ONLY = (CoTMode.COT,)
_NO_COT_ONLY = (CoTMode.NO_COT,)

_MULTI_CHOICE_JOBS = ("multi_choice_plain", "multi_choice_fake_cot", "multi_choice_cot")
_FREE_RESPONSE_JOBS = ("free_response",)
_FREE_RESPONSE_JUDGE_JOBS = ("free_response_judge",)
_HUMAN_EVAL_JOBS = ("code_human_eval",)
_MBPP_JOBS = ("code_mbpp", "code_mbpp_fake_cot", "code_mbpp_cot")
_LIVECODEBENCH_JOBS = ("code_livecodebench",)
_INSTRUCTION_FOLLOWING_JOBS = ("instruction_following",)
_BROWSECOMP_JOBS = ("function_browsecomp",)
_MCP_BENCH_JOBS = ("function_mcp_bench",)
_BFCL_V3_JOBS = ("function_bfcl_v3",)
_TAU_BENCH_JOBS = ("function_tau_bench",)
_TAU2_BENCH_JOBS = ("function_tau2_bench",)


def _metadata(
    name: str,
    *,
    field: BenchmarkField,
    cot_modes: tuple[CoTMode, ...],
    default_split: str = "test",
    dataset_name: str | None = None,
    scheduler_jobs: tuple[str, ...] = (),
    n_shots: tuple[int, ...] = (0,),
    avg_ks: tuple[float, ...] = (),
    pass_ks: tuple[int, ...] = (),
) -> BenchmarkMetadata:
    return BenchmarkMetadata(
        name=safe_slug(name).lower(),
        field=field,
        cot_modes=cot_modes,
        default_split=default_split,
        dataset_name=safe_slug(dataset_name).lower() if dataset_name else None,
        scheduler_jobs=scheduler_jobs,
        n_shots=n_shots,
        avg_ks=avg_ks,
        pass_ks=pass_ks,
    )


def _knowledge(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.KNOWLEDGE,
        cot_modes=_THREE_MODE_KNOWLEDGE,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=_MULTI_CHOICE_JOBS,
    )


def _math(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
    scheduler_jobs: tuple[str, ...] = _FREE_RESPONSE_JOBS,
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.MATHS,
        cot_modes=_COT_ONLY,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=scheduler_jobs,
    )


def _coding_human_eval(name: str, *, dataset_name: str | None = None) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.CODING,
        cot_modes=_NO_COT_ONLY,
        dataset_name=dataset_name,
        scheduler_jobs=_HUMAN_EVAL_JOBS,
    )


def _coding_mbpp(name: str, *, dataset_name: str | None = None) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.CODING,
        cot_modes=_THREE_MODE_CODE,
        dataset_name=dataset_name,
        scheduler_jobs=_MBPP_JOBS,
    )


def _coding_livecodebench(name: str, *, dataset_name: str | None = None) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.CODING,
        cot_modes=_COT_ONLY,
        dataset_name=dataset_name,
        scheduler_jobs=_LIVECODEBENCH_JOBS,
    )


def _instruction_following(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.INSTRUCTION_FOLLOWING,
        cot_modes=_NO_COT_ONLY,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=_INSTRUCTION_FOLLOWING_JOBS,
    )


def _function_calling(
    name: str,
    *,
    default_split: str = "test",
    dataset_name: str | None = None,
    scheduler_jobs: tuple[str, ...],
) -> BenchmarkMetadata:
    return _metadata(
        name,
        field=BenchmarkField.FUNCTION_CALLING,
        cot_modes=_COT_ONLY,
        default_split=default_split,
        dataset_name=dataset_name,
        scheduler_jobs=scheduler_jobs,
    )


_EXPLICIT_METADATA: dict[str, BenchmarkMetadata] = {
    # Knowledge
    canonical_slug("include"): _knowledge("include"),
    canonical_slug("mmlu"): _knowledge("mmlu"),
    canonical_slug("cmmlu"): _knowledge("cmmlu"),
    canonical_slug("ceval"): _knowledge("ceval"),
    canonical_slug("mmlu_pro"): _knowledge("mmlu_pro"),
    canonical_slug("mmlu_redux"): _knowledge("mmlu_redux"),
    canonical_slug("mmmlu"): _knowledge("mmmlu"),
    canonical_slug("gpqa_main"): _knowledge("gpqa_main", dataset_name="gpqa", default_split="main"),
    canonical_slug("gpqa_extended"): _knowledge("gpqa_extended", dataset_name="gpqa", default_split="extended"),
    canonical_slug("gpqa_diamond"): _knowledge("gpqa_diamond", dataset_name="gpqa", default_split="diamond"),
    canonical_slug("supergpqa"): _knowledge("supergpqa"),
    # Maths / free response
    canonical_slug("aime24"): _math("aime24"),
    canonical_slug("aime25"): _math("aime25"),
    canonical_slug("algebra222"): _math("algebra222"),
    canonical_slug("amc23"): _math("amc23", scheduler_jobs=_FREE_RESPONSE_JUDGE_JOBS),
    canonical_slug("answer_judge"): _math("answer_judge", scheduler_jobs=_FREE_RESPONSE_JUDGE_JOBS),
    canonical_slug("asdiv"): _math("asdiv"),
    canonical_slug("beyond_aime"): _math("beyond_aime"),
    canonical_slug("brumo25"): _math("brumo25"),
    canonical_slug("college_math"): _math("college_math"),
    canonical_slug("comp_math_24_25"): _math("comp_math_24_25", scheduler_jobs=_FREE_RESPONSE_JUDGE_JOBS),
    canonical_slug("gaokao2023en"): _math("gaokao2023en", scheduler_jobs=_FREE_RESPONSE_JUDGE_JOBS),
    canonical_slug("gsm_plus"): _math("gsm_plus"),
    canonical_slug("gsm8k"): _math("gsm8k", scheduler_jobs=_FREE_RESPONSE_JUDGE_JOBS),
    canonical_slug("hendrycks_math"): _math("hendrycks_math"),
    canonical_slug("hle"): _math("hle", default_split="all"),
    canonical_slug("hmmt_feb25"): _math("hmmt_feb25"),
    canonical_slug("math_500"): _math("math_500", scheduler_jobs=_FREE_RESPONSE_JUDGE_JOBS),
    canonical_slug("math_odyssey"): _math("math_odyssey"),
    canonical_slug("mawps"): _math("mawps"),
    canonical_slug("minerva_math"): _math("minerva_math", scheduler_jobs=_FREE_RESPONSE_JUDGE_JOBS),
    canonical_slug("olympiadbench"): _math("olympiadbench"),
    canonical_slug("omni_math"): _math("omni_math"),
    canonical_slug("polymath"): _math("polymath", default_split="all"),
    canonical_slug("simpleqa"): _math("simpleqa", default_split="verified"),
    canonical_slug("svamp"): _math("svamp"),
    # Coding
    canonical_slug("human_eval"): _coding_human_eval("human_eval"),
    canonical_slug("human_eval_cn"): _coding_human_eval("human_eval_cn"),
    canonical_slug("human_eval_fix"): _coding_human_eval("human_eval_fix"),
    canonical_slug("human_eval_plus"): _coding_human_eval("human_eval_plus"),
    canonical_slug("mbpp"): _coding_mbpp("mbpp"),
    canonical_slug("mbpp_plus"): _coding_mbpp("mbpp_plus"),
    canonical_slug("livecodebench"): _coding_livecodebench("livecodebench"),
    # Instruction following
    canonical_slug("ifeval"): _instruction_following("ifeval"),
    canonical_slug("ifbench"): _instruction_following("ifbench"),
    canonical_slug("arena_hard_v2"): _instruction_following("arena_hard_v2", dataset_name="arena_hard"),
    canonical_slug("wmt24pp"): _instruction_following("wmt24pp"),
    canonical_slug("flores200"): _instruction_following("flores200", default_split="devtest"),
    # Function calling
    canonical_slug("browsecomp"): _function_calling("browsecomp", scheduler_jobs=_BROWSECOMP_JOBS),
    canonical_slug("browsecomp_zh"): _function_calling("browsecomp_zh", scheduler_jobs=_BROWSECOMP_JOBS),
    canonical_slug("mcp_bench"): _function_calling("mcp_bench", scheduler_jobs=_MCP_BENCH_JOBS),
    canonical_slug("bfcl_v3"): _function_calling("bfcl_v3", scheduler_jobs=_BFCL_V3_JOBS),
    canonical_slug("tau_bench_retail"): _function_calling("tau_bench_retail", scheduler_jobs=_TAU_BENCH_JOBS),
    canonical_slug("tau_bench_airline"): _function_calling("tau_bench_airline", scheduler_jobs=_TAU_BENCH_JOBS),
    canonical_slug("tau_bench_telecom"): _function_calling("tau_bench_telecom", scheduler_jobs=_TAU_BENCH_JOBS),
    canonical_slug("tau2_bench_retail"): _function_calling(
        "tau2_bench_retail",
        default_split="base",
        scheduler_jobs=_TAU2_BENCH_JOBS,
    ),
    canonical_slug("tau2_bench_airline"): _function_calling(
        "tau2_bench_airline",
        default_split="base",
        scheduler_jobs=_TAU2_BENCH_JOBS,
    ),
    canonical_slug("tau2_bench_telecom"): _function_calling(
        "tau2_bench_telecom",
        default_split="base",
        scheduler_jobs=_TAU2_BENCH_JOBS,
    ),
}

BENCHMARK_ALIASES: dict[str, tuple[str, ...]] = {
    canonical_slug("gpqa"): (
        canonical_slug("gpqa_main"),
        canonical_slug("gpqa_extended"),
        canonical_slug("gpqa_diamond"),
    ),
    canonical_slug("arena_hard"): (canonical_slug("arena_hard_v2"),),
    canonical_slug("tau_bench"): (
        canonical_slug("tau_bench_retail"),
        canonical_slug("tau_bench_airline"),
        canonical_slug("tau_bench_telecom"),
    ),
    canonical_slug("tau2_bench"): (
        canonical_slug("tau2_bench_retail"),
        canonical_slug("tau2_bench_airline"),
        canonical_slug("tau2_bench_telecom"),
    ),
}

_PREFIX_FALLBACKS: tuple[tuple[tuple[str, ...], BenchmarkMetadata], ...] = (
    (
        ("mmlu", "cmmlu", "mmmlu", "ceval", "supergpqa"),
        _knowledge("knowledge"),
    ),
    (
        ("human_eval",),
        _coding_human_eval("human_eval"),
    ),
    (
        ("mbpp",),
        _coding_mbpp("mbpp"),
    ),
    (
        ("livecodebench",),
        _coding_livecodebench("livecodebench"),
    ),
    (("ifeval", "ifbench", "wmt24pp", "flores200"), _instruction_following("instruction_following")),
    (
        ("browsecomp",),
        _function_calling("browsecomp", scheduler_jobs=_BROWSECOMP_JOBS),
    ),
    (
        ("mcp_bench",),
        _function_calling("mcp_bench", scheduler_jobs=_MCP_BENCH_JOBS),
    ),
)

ALL_BENCHMARKS: tuple[BenchmarkMetadata, ...] = tuple(
    sorted(_EXPLICIT_METADATA.values(), key=lambda item: (item.field.value, item.name))
)

BENCHMARKS_BY_FIELD: dict[BenchmarkField, tuple[BenchmarkMetadata, ...]] = {
    field: tuple(item for item in ALL_BENCHMARKS if item.field is field)
    for field in BenchmarkField
}

_EXPLICIT_BY_DATASET_SLUG: dict[str, BenchmarkMetadata] = {
    canonical_slug(make_dataset_slug(item.dataset, item.default_split)): item
    for item in ALL_BENCHMARKS
}


def _resolve_single_alias(name: str) -> BenchmarkMetadata | None:
    target_names = BENCHMARK_ALIASES.get(name)
    if not target_names or len(target_names) != 1:
        return None
    return _EXPLICIT_METADATA[target_names[0]]


def expand_benchmark_alias(raw_name: str) -> tuple[str, ...]:
    slug = canonical_slug(raw_name)
    if slug in _EXPLICIT_METADATA:
        return (slug,)
    if slug in _EXPLICIT_BY_DATASET_SLUG:
        return (_EXPLICIT_BY_DATASET_SLUG[slug].name,)
    targets = BENCHMARK_ALIASES.get(slug)
    if targets:
        return targets

    benchmark_name, _ = split_benchmark_and_split(slug)
    if benchmark_name in _EXPLICIT_METADATA:
        return (benchmark_name,)
    if benchmark_name in _EXPLICIT_BY_DATASET_SLUG:
        return (_EXPLICIT_BY_DATASET_SLUG[benchmark_name].name,)
    targets = BENCHMARK_ALIASES.get(benchmark_name)
    if targets:
        return targets
    return tuple()


def resolve_benchmark_metadata(dataset_slug: str) -> BenchmarkMetadata:
    slug = canonical_slug(dataset_slug)
    explicit = _EXPLICIT_METADATA.get(slug)
    if explicit is not None:
        return explicit

    alias_target = _resolve_single_alias(slug)
    if alias_target is not None:
        return alias_target

    explicit = _EXPLICIT_BY_DATASET_SLUG.get(slug)
    if explicit is not None:
        return explicit

    benchmark_name, _ = split_benchmark_and_split(slug)
    explicit = _EXPLICIT_METADATA.get(benchmark_name)
    if explicit is not None:
        return explicit

    alias_target = _resolve_single_alias(benchmark_name)
    if alias_target is not None:
        return alias_target

    for prefixes, template in _PREFIX_FALLBACKS:
        if benchmark_name.startswith(prefixes):
            return replace(template, name=benchmark_name)

    return _math(benchmark_name)


def default_split_for_benchmark(dataset_slug: str) -> str:
    return resolve_benchmark_metadata(dataset_slug).default_split


def get_benchmarks_with_field(field: BenchmarkField) -> tuple[BenchmarkMetadata, ...]:
    return BENCHMARKS_BY_FIELD.get(field, ())


def scheduler_jobs_for_benchmark(dataset_slug: str) -> tuple[str, ...]:
    return resolve_benchmark_metadata(dataset_slug).scheduler_jobs


def supports_cot_mode(dataset_slug: str, cot_mode: CoTMode) -> bool:
    return cot_mode in resolve_benchmark_metadata(dataset_slug).cot_modes


__all__ = [
    "AUTO_TARGET_ATTEMPTS",
    "ALL_BENCHMARKS",
    "BENCHMARK_ALIASES",
    "BenchmarkField",
    "BenchmarkMetadata",
    "BENCHMARKS_BY_FIELD",
    "CoTMode",
    "default_split_for_benchmark",
    "expand_benchmark_alias",
    "get_benchmarks_with_field",
    "resolve_benchmark_metadata",
    "scheduler_jobs_for_benchmark",
    "supports_cot_mode",
]
