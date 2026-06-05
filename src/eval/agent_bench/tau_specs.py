from __future__ import annotations

"""Shared TAU benchmark dataset/job metadata."""

from dataclasses import dataclass
from typing import Literal

TauBenchmarkVersion = Literal["tau_v2", "tau_v3"]
TauJobName = Literal["function_tau2_bench", "function_tau3_bench"]


@dataclass(frozen=True, slots=True)
class TauBenchSpec:
    dataset_name: str
    task_set_name: str
    domain: str
    benchmark_version: TauBenchmarkVersion
    job_name: TauJobName
    task_split: str | None = "base"
    retrieval_config: str | None = None

    @property
    def dataset_slug(self) -> str:
        return self.dataset_name

    @property
    def prep_split(self) -> str:
        return self.task_split or ""


TAU_BENCH_SPECS: tuple[TauBenchSpec, ...] = (
    TauBenchSpec("tau2_bench_airline", "airline", "airline", "tau_v2", "function_tau2_bench"),
    TauBenchSpec("tau2_bench_retail", "retail", "retail", "tau_v2", "function_tau2_bench"),
    TauBenchSpec("tau2_bench_telecom", "telecom", "telecom", "tau_v2", "function_tau2_bench"),
    TauBenchSpec("tau3_bench_airline", "airline", "airline", "tau_v3", "function_tau3_bench"),
    TauBenchSpec(
        "tau3_bench_banking_knowledge",
        "banking_knowledge",
        "banking_knowledge",
        "tau_v3",
        "function_tau3_bench",
        None,
        "bm25",
    ),
    TauBenchSpec("tau3_bench_retail", "retail", "retail", "tau_v3", "function_tau3_bench"),
    TauBenchSpec("tau3_bench_telecom", "telecom", "telecom", "tau_v3", "function_tau3_bench"),
)

TAU_BENCH_DEFAULT_SPLITS: dict[str, str] = {
    spec.dataset_name: "" for spec in TAU_BENCH_SPECS
}
TAU_BENCH_SPEC_BY_DATASET: dict[str, TauBenchSpec] = {
    spec.dataset_name: spec for spec in TAU_BENCH_SPECS
}
TAU_BENCH_SPEC_BY_SLUG: dict[str, TauBenchSpec] = {
    spec.dataset_slug: spec for spec in TAU_BENCH_SPECS
}
TAU_AGENT_JOB_BY_DATASET: dict[str, str] = {
    spec.dataset_slug: spec.job_name for spec in TAU_BENCH_SPECS
}
TAU2_BENCH_DATASET_SLUGS: tuple[str, ...] = tuple(
    spec.dataset_slug for spec in TAU_BENCH_SPECS if spec.job_name == "function_tau2_bench"
)
TAU3_BENCH_DATASET_SLUGS: tuple[str, ...] = tuple(
    spec.dataset_slug for spec in TAU_BENCH_SPECS if spec.job_name == "function_tau3_bench"
)


def tau_bench_specs_for_job(job_name: str) -> tuple[TauBenchSpec, ...]:
    return tuple(spec for spec in TAU_BENCH_SPECS if spec.job_name == job_name)


__all__ = [
    "TAU2_BENCH_DATASET_SLUGS",
    "TAU3_BENCH_DATASET_SLUGS",
    "TAU_AGENT_JOB_BY_DATASET",
    "TAU_BENCH_DEFAULT_SPLITS",
    "TAU_BENCH_SPECS",
    "TAU_BENCH_SPEC_BY_DATASET",
    "TAU_BENCH_SPEC_BY_SLUG",
    "TauBenchSpec",
    "TauBenchmarkVersion",
    "TauJobName",
    "tau_bench_specs_for_job",
]
