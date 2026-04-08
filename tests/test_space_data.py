from __future__ import annotations

import json
from pathlib import Path

from src.eval.benchmark_registry import BenchmarkField
from src.space.data import _infer_domain, load_scores
from src.space.domains import (
    DOMAIN_CODING,
    DOMAIN_FUNCTION_CALL,
    DOMAIN_INSTRUCTION_FOLLOWING,
    DOMAIN_MATH,
    DOMAIN_MMLU,
    DOMAIN_OTHER,
    domain_for_benchmark_field,
    is_coding_domain,
    is_knowledge_group_domain,
)


def test_infer_domain_recognizes_function_calling_jobs() -> None:
    assert _infer_domain("browsecomp_test", is_cot=True, task="function_browsecomp") == DOMAIN_FUNCTION_CALL
    assert _infer_domain("mcp_bench_test", is_cot=True, task="function_mcp_bench") == DOMAIN_FUNCTION_CALL
    assert _infer_domain("tau_bench_airline_test", is_cot=True, task="function_tau_bench") == DOMAIN_FUNCTION_CALL
    assert _infer_domain("tau2_bench_telecom_base", is_cot=True, task="function_tau2_bench") == DOMAIN_FUNCTION_CALL


def test_infer_domain_prefers_benchmark_metadata_field_mapping() -> None:
    assert _infer_domain("mmlu_test", is_cot=False, task=None) == DOMAIN_MMLU
    assert _infer_domain("math_500_test", is_cot=True, task=None) == DOMAIN_MATH
    assert _infer_domain("ifeval_test", is_cot=False, task=None) == DOMAIN_INSTRUCTION_FOLLOWING
    assert _infer_domain("human_eval_test", is_cot=False, task=None) == DOMAIN_CODING


def test_domain_helpers_follow_benchmark_field_mapping() -> None:
    assert domain_for_benchmark_field(BenchmarkField.KNOWLEDGE, dataset_slug="mmlu_test") == DOMAIN_MMLU
    assert domain_for_benchmark_field(BenchmarkField.MATHS, dataset_slug="gsm8k_test") == DOMAIN_MATH
    assert domain_for_benchmark_field(BenchmarkField.INSTRUCTION_FOLLOWING, dataset_slug="ifeval_test") == DOMAIN_INSTRUCTION_FOLLOWING
    assert domain_for_benchmark_field(BenchmarkField.FUNCTION_CALLING, dataset_slug="browsecomp_test") == DOMAIN_FUNCTION_CALL
    assert is_knowledge_group_domain(DOMAIN_OTHER)
    assert is_coding_domain(DOMAIN_CODING)


def test_load_scores_reads_latest_rows_from_score_index(tmp_path: Path, monkeypatch) -> None:
    index_path = tmp_path / "score_index.jsonl"
    monkeypatch.setenv("RWKV_SPACE_SCORE_INDEX", str(index_path))

    rows = [
        {
            "task_id": 11,
            "dataset": "math_500_test",
            "model": "rwkv7-g1-1_5b",
            "cot": True,
            "task": "free_response",
            "metrics": {"accuracy": 0.6},
            "samples": 32,
            "created_at": "2026-04-03T10:00:00",
        },
        {
            "task_id": 12,
            "dataset": "math_500_test",
            "model": "rwkv7-g1-1_5b",
            "cot": True,
            "task": "free_response",
            "metrics": {"accuracy": 0.8},
            "samples": 32,
            "created_at": "2026-04-03T12:00:00",
        },
        {
            "task_id": 99,
            "dataset": "math_500_test",
            "model": "rwkv7-g1-1_5b",
            "cot": True,
            "task": "param_search_free_response",
            "metrics": {"accuracy": 1.0},
            "samples": 8,
            "created_at": "2026-04-03T13:00:00",
            "is_param_search": True,
        },
    ]
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    entries = load_scores()

    assert len(entries) == 1
    entry = entries[0]
    assert entry.task_id == 12
    assert entry.metrics["accuracy"] == 0.8
    assert entry.path == index_path
