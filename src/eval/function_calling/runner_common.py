from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from src.eval.benchmark_config import resolve_benchmark_model_config
from src.eval.execution_plan import build_auto_avg_k_execution_plan, build_avg_k_execution_plan
from src.infer.backend import InferenceBackend

if TYPE_CHECKING:
    from src.eval.evaluating.contracts import RunContext

_TEMPLATE_LEAK_MARKERS = (
    "<system message>",
    "</system message>",
    "<assistant>",
    "</assistant>",
    "<user_input>",
    "</user_input>",
)


class FunctionCallingBenchmarkKind(str, Enum):
    AUTO = "auto"
    BROWSECOMP = "browsecomp"
    BROWSECOMP_PLUS = "browsecomp_plus"
    COMPLEXFUNCBENCH = "complexfuncbench"
    LONGBENCH = "longbench"
    LONGCODEBENCH = "longcodebench"
    MCP_BENCH = "mcp_bench"
    API_BANK = "api_bank"
    AGENTBENCH = "agentbench"
    BFCL_AST = "bfcl_ast"
    BFCL_EXEC = "bfcl_exec"
    BFCL_V3 = "bfcl_v3"
    TOOLALPACA = "toolalpaca"
    TAU_BENCH = "tau_bench"
    TAU2_BENCH = "tau2_bench"
    TAU3_BENCH = "tau3_bench"


@dataclass(slots=True)
class ResolvedFunctionCallingRun:
    benchmark_kind: FunctionCallingBenchmarkKind
    dataset_path: Path
    dataset_slug: str
    benchmark_name: str
    dataset_split: str
    model_name: str
    engine: InferenceBackend


def _resolve_job_name(default_job_name: str, *, run_context: "RunContext | None" = None) -> str:
    if run_context is not None:
        return run_context.job_name
    return os.environ.get("RWKV_SKILLS_JOB_NAME", default_job_name)


def _looks_like_template_leak(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if "<system message>" in lowered and "you are a helpful assistant" in lowered:
        return True
    marker_hits = sum(lowered.count(marker) for marker in _TEMPLATE_LEAK_MARKERS)
    return marker_hits >= 3


def _resolve_function_calling_plan(
    dataset_slug: str,
    dataset_len: int,
    *,
    avg_ks: Sequence[float] | None,
    model_name: str | None = None,
    config_defaults: bool = False,
):
    explicit = tuple(float(item) for item in (avg_ks or ()))
    if explicit:
        if len(explicit) != 1:
            rendered = ", ".join(str(item) for item in explicit)
            raise ValueError(f"function-calling runner accepts exactly one avg_k override, got: {rendered}")
        return build_avg_k_execution_plan(dataset_slug, dataset_len, avg_k=explicit[0])

    if config_defaults:
        configured = ()
        if model_name:
            config = resolve_benchmark_model_config(dataset_slug, model_name, stage=None)
            configured = tuple(config.avg_k or ()) if config is not None else ()
        if configured:
            if len(configured) != 1:
                rendered = ", ".join(str(item) for item in configured)
                raise ValueError(f"function-calling runner accepts exactly one configured avg_k value, got: {rendered}")
            return build_avg_k_execution_plan(dataset_slug, dataset_len, avg_k=float(configured[0]))
        return build_avg_k_execution_plan(dataset_slug, dataset_len, avg_k=1.0)

    return build_auto_avg_k_execution_plan(dataset_slug, dataset_len)


def _resolve_function_calling_sample_limit(
    dataset_slug: str,
    model_name: str,
    *,
    max_samples: int | None,
) -> int | None:
    if max_samples is not None:
        value = int(max_samples)
        return value if value > 0 else None
    config = resolve_benchmark_model_config(dataset_slug, model_name, stage=None)
    if config is None or config.max_samples is None:
        return None
    value = int(config.max_samples)
    return value if value > 0 else None
