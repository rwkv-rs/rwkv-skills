from __future__ import annotations

from src.space.data import _infer_domain


def test_infer_domain_recognizes_function_calling_jobs() -> None:
    assert _infer_domain("browsecomp_test", is_cot=True, task="function_browsecomp") == "function_call系列"
    assert _infer_domain("mcp_bench_test", is_cot=True, task="function_mcp_bench") == "function_call系列"
    assert _infer_domain("tau_bench_airline_test", is_cot=True, task="function_tau_bench") == "function_call系列"
    assert _infer_domain("tau2_bench_telecom_base", is_cot=True, task="function_tau2_bench") == "function_call系列"
