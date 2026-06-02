from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FunctionCallingSubtype = Literal["one_step", "agent"]

ONE_STEP_METRIC_KEYS: tuple[str, ...] = (
    "avg@1",
    "success_rate",
    "name_match",
    "argument_match",
    "missing_call",
    "extra_call",
)
COMPLEXFUNC_METRIC_KEYS: tuple[str, ...] = (
    "official_score",
    "success_rate",
    "call_accuracy",
    "parse_error_rate",
    "missing_call",
    "extra_call",
)
AGENT_METRIC_KEYS: tuple[str, ...] = (
    "avg@1",
    "success_rate",
    "official_score",
    "avg_steps",
    "invalid_action_rate",
    "timeout_rate",
    "parse_error_rate",
)


@dataclass(frozen=True, slots=True)
class FunctionCallingBenchmarkSpec:
    job_name: str
    subtype: FunctionCallingSubtype
    benchmark: str
    metric_keys: tuple[str, ...]


FUNCTION_CALLING_BENCHMARK_SPECS: dict[str, FunctionCallingBenchmarkSpec] = {
    "function_one_step_bfcl_ast": FunctionCallingBenchmarkSpec(
        "function_one_step_bfcl_ast", "one_step", "bfcl", ONE_STEP_METRIC_KEYS
    ),
    "function_one_step_bfcl_exec": FunctionCallingBenchmarkSpec(
        "function_one_step_bfcl_exec", "one_step", "bfcl", ONE_STEP_METRIC_KEYS
    ),
    "function_one_step_toolalpaca": FunctionCallingBenchmarkSpec(
        "function_one_step_toolalpaca", "one_step", "toolalpaca", ONE_STEP_METRIC_KEYS
    ),
    "function_one_step_apibank_l1": FunctionCallingBenchmarkSpec(
        "function_one_step_apibank_l1", "one_step", "apibank", ONE_STEP_METRIC_KEYS
    ),
    "function_one_step_apibank_l2": FunctionCallingBenchmarkSpec(
        "function_one_step_apibank_l2", "one_step", "apibank", ONE_STEP_METRIC_KEYS
    ),
    "function_one_step_complexfuncbench_subset": FunctionCallingBenchmarkSpec(
        "function_one_step_complexfuncbench_subset",
        "one_step",
        "complexfuncbench",
        COMPLEXFUNC_METRIC_KEYS,
    ),
    "function_agent_apibank_l2": FunctionCallingBenchmarkSpec(
        "function_agent_apibank_l2", "agent", "apibank", AGENT_METRIC_KEYS
    ),
    "function_agent_browsecomp_plus": FunctionCallingBenchmarkSpec(
        "function_agent_browsecomp_plus",
        "agent",
        "browsecomp_plus",
        AGENT_METRIC_KEYS,
    ),
    "function_agent_agentbench_db": FunctionCallingBenchmarkSpec(
        "function_agent_agentbench_db", "agent", "agentbench", AGENT_METRIC_KEYS
    ),
    "function_agent_agentbench_kg": FunctionCallingBenchmarkSpec(
        "function_agent_agentbench_kg", "agent", "agentbench", AGENT_METRIC_KEYS
    ),
}

FUNCTION_CALLING_JOB_NAMES: frozenset[str] = frozenset(FUNCTION_CALLING_BENCHMARK_SPECS)
FUNCTION_CALLING_DATASET_PREFIXES: tuple[str, ...] = (
    "bfcl_",
    "toolalpaca_",
    "apibank_",
    "agentbench_",
    "complexfuncbench_",
    "browsecomp_plus",
)
FUNCTION_CALLING_EXPLICIT_ONLY_JOBS: tuple[str, ...] = (
    "function_agent_apibank_l2",
    "function_agent_agentbench_db",
    "function_agent_agentbench_kg",
)


def function_calling_benchmark_spec(
    job_name: str | None,
) -> FunctionCallingBenchmarkSpec | None:
    if not job_name:
        return None
    return FUNCTION_CALLING_BENCHMARK_SPECS.get(str(job_name))


def is_function_calling_job(job_name: str | None) -> bool:
    return function_calling_benchmark_spec(job_name) is not None


__all__ = [
    "AGENT_METRIC_KEYS",
    "COMPLEXFUNC_METRIC_KEYS",
    "FUNCTION_CALLING_BENCHMARK_SPECS",
    "FUNCTION_CALLING_DATASET_PREFIXES",
    "FUNCTION_CALLING_EXPLICIT_ONLY_JOBS",
    "FUNCTION_CALLING_JOB_NAMES",
    "FunctionCallingBenchmarkSpec",
    "FunctionCallingSubtype",
    "ONE_STEP_METRIC_KEYS",
    "function_calling_benchmark_spec",
    "is_function_calling_job",
]
