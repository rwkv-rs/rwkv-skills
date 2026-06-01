from __future__ import annotations

import os

FUNCTION_CALL_JOB_NAMES_BY_BENCHMARK: dict[str, tuple[str, ...]] = {
    "bfcl_exec_simple": ("function_one_step_bfcl_exec",),
    "bfcl_exec_multiple": ("function_one_step_bfcl_exec",),
    "bfcl_exec_parallel": ("function_one_step_bfcl_exec",),
    "bfcl_exec_parallel_multiple": ("function_one_step_bfcl_exec",),
    "bfcl_simple_python": ("function_one_step_bfcl_ast",),
    "bfcl_multiple": ("function_one_step_bfcl_ast",),
    "toolalpaca_eval_simulated": ("function_one_step_toolalpaca",),
    "toolalpaca_eval_real": ("function_one_step_toolalpaca",),
    "apibank_level1": ("function_one_step_apibank_l1",),
    "apibank_l1": ("function_one_step_apibank_l1",),
    "apibank_l2": ("function_one_step_apibank_l2",),
    "complexfuncbench_subset": ("function_one_step_complexfuncbench_subset",),
}


def simple_tool_call_job_name(slug: str, *, scheduled_job: str | None = None) -> str | None:
    benchmark = str(slug).rsplit("_", 1)[0]
    allowed = FUNCTION_CALL_JOB_NAMES_BY_BENCHMARK.get(benchmark)
    if not allowed:
        return None
    scheduled = scheduled_job if scheduled_job is not None else os.environ.get("RWKV_SKILLS_JOB_NAME")
    if scheduled in allowed:
        return scheduled
    return allowed[0]


__all__ = [
    "FUNCTION_CALL_JOB_NAMES_BY_BENCHMARK",
    "simple_tool_call_job_name",
]
