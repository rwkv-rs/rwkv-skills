"""Shared dashboard domain labels and benchmark-field helpers."""

from __future__ import annotations

from src.eval.benchmark_registry import BenchmarkField, resolve_benchmark_metadata
from src.eval.benchmark_sources import REQUESTED_BENCHMARKS_BY_NAME
from src.eval.scheduler.dataset_utils import canonical_slug


DOMAIN_MMLU = "mmlu系列"
DOMAIN_MULTI_CHOICE = "multi-choice系列"
DOMAIN_OTHER = "其他"
DOMAIN_MATH = "math reasoning系列"
DOMAIN_CODING = "coding系列"
DOMAIN_INSTRUCTION_FOLLOWING = "instruction following系列"
DOMAIN_FUNCTION_CALL = "function_call系列"
DOMAIN_FUNCTION_CALL_LEGACY = "function_call"
DOMAIN_AGENT = "agent系列"

KNOWLEDGE_GROUP_DOMAINS = frozenset({DOMAIN_MMLU, DOMAIN_MULTI_CHOICE, DOMAIN_OTHER})
MULTI_CHOICE_DOMAINS = frozenset({DOMAIN_MMLU, DOMAIN_MULTI_CHOICE})
MATH_DOMAINS = frozenset({DOMAIN_MATH})
CODING_DOMAINS = frozenset({DOMAIN_CODING})
INSTRUCTION_FOLLOWING_DOMAINS = frozenset({DOMAIN_INSTRUCTION_FOLLOWING})
FUNCTION_CALL_DOMAINS = frozenset({DOMAIN_FUNCTION_CALL, DOMAIN_FUNCTION_CALL_LEGACY, DOMAIN_AGENT})
AGENT_DOMAINS = frozenset({DOMAIN_AGENT})


def _is_agent_domain_slug(slug: str) -> bool:
    try:
        benchmark_name = resolve_benchmark_metadata(slug).name
    except Exception:  # noqa: BLE001 - unknown slugs keep the generic label
        return False
    source = REQUESTED_BENCHMARKS_BY_NAME.get(benchmark_name)
    return source is not None and source.domain == "agent"


def domain_for_benchmark_field(field: BenchmarkField, *, dataset_slug: str) -> str:
    slug = canonical_slug(dataset_slug)
    if field is BenchmarkField.KNOWLEDGE:
        return DOMAIN_MMLU if slug.startswith("mmlu") else DOMAIN_MULTI_CHOICE
    if field is BenchmarkField.MATHS:
        return DOMAIN_MATH
    if field is BenchmarkField.CODING:
        return DOMAIN_CODING
    if field is BenchmarkField.INSTRUCTION_FOLLOWING:
        return DOMAIN_INSTRUCTION_FOLLOWING
    if field is BenchmarkField.FUNCTION_CALLING:
        return DOMAIN_AGENT if _is_agent_domain_slug(slug) else DOMAIN_FUNCTION_CALL
    return DOMAIN_OTHER


def is_knowledge_group_domain(domain: str | None) -> bool:
    return bool(domain) and domain in KNOWLEDGE_GROUP_DOMAINS


def is_multi_choice_domain(domain: str | None) -> bool:
    return bool(domain) and domain in MULTI_CHOICE_DOMAINS


def is_math_domain(domain: str | None) -> bool:
    return bool(domain) and domain in MATH_DOMAINS


def is_coding_domain(domain: str | None) -> bool:
    return bool(domain) and domain in CODING_DOMAINS


def is_instruction_following_domain(domain: str | None) -> bool:
    return bool(domain) and domain in INSTRUCTION_FOLLOWING_DOMAINS


def is_function_call_domain(domain: str | None) -> bool:
    return bool(domain) and domain in FUNCTION_CALL_DOMAINS


def is_agent_domain(domain: str | None) -> bool:
    return bool(domain) and domain in AGENT_DOMAINS


__all__ = [
    "AGENT_DOMAINS",
    "CODING_DOMAINS",
    "DOMAIN_AGENT",
    "DOMAIN_CODING",
    "DOMAIN_FUNCTION_CALL",
    "DOMAIN_FUNCTION_CALL_LEGACY",
    "DOMAIN_INSTRUCTION_FOLLOWING",
    "DOMAIN_MATH",
    "DOMAIN_MMLU",
    "DOMAIN_MULTI_CHOICE",
    "DOMAIN_OTHER",
    "FUNCTION_CALL_DOMAINS",
    "INSTRUCTION_FOLLOWING_DOMAINS",
    "KNOWLEDGE_GROUP_DOMAINS",
    "MATH_DOMAINS",
    "MULTI_CHOICE_DOMAINS",
    "domain_for_benchmark_field",
    "is_agent_domain",
    "is_coding_domain",
    "is_function_call_domain",
    "is_instruction_following_domain",
    "is_knowledge_group_domain",
    "is_math_domain",
    "is_multi_choice_domain",
]
