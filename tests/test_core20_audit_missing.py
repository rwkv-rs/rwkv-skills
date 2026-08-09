import pytest

from ops.g1i_strict46.audit_core20_dual import (
    STATS_QUERY,
    _quality_reasons,
    _quality_warnings,
)
from ops.g1i_strict46.run_core20_audit_missing import (
    audit_cell_to_dataset,
    job_for_cell,
    recovery_plan,
)


MODEL = "rwkv7-g1i-1.5b-20260805-ctx16384"


def _cell(benchmark, domain, mode, state="missing"):
    return {
        "model": MODEL,
        "benchmark": benchmark,
        "domain": domain,
        "mode": mode,
        "state": state,
    }


def test_core20_dataset_mapping() -> None:
    assert audit_cell_to_dataset("mmlu__test") == "mmlu"
    assert audit_cell_to_dataset("gpqa__diamond") == "gpqa_diamond"


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (_cell("mmlu__test", "knowledge", "NoCoT"), "multi_choice_plain_naive"),
        (_cell("mmlu__test", "knowledge", "CoT"), "multi_choice_cot_naive"),
        (_cell("math_500__test", "math", "CoT"), "free_response_naive"),
        (_cell("math_500__test", "math", "NoCoT"), "free_response_plain_naive"),
        (_cell("amc23__test", "math", "CoT"), "free_response_judge_naive"),
        (_cell("amc23__test", "math", "NoCoT"), "free_response_judge_plain_naive"),
        (_cell("human_eval_plus__test", "coding", "NoCoT"), "code_human_eval_naive"),
        (_cell("mbpp_plus__test", "coding", "NoCoT"), "code_mbpp_naive"),
        (_cell("livecodebench__test", "coding", "NoCoT"), "code_livecodebench_plain_naive"),
        (_cell("ifbench__test", "instruction_following", "NoCoT"), "instruction_following_naive"),
    ],
)
def test_job_for_cell_uses_global_protocol_mapping(cell, expected) -> None:
    assert job_for_cell(cell) == expected


def test_recovery_plan_only_includes_missing_and_invalid_cells() -> None:
    audit = {
        "cells": [
            _cell("mmlu__test", "knowledge", "NoCoT", "valid"),
            _cell("mmlu__test", "knowledge", "CoT", "running"),
            _cell("math_500__test", "math", "NoCoT", "missing"),
            _cell("amc23__test", "math", "NoCoT", "invalid"),
            {**_cell("mmlu__test", "knowledge", "NoCoT"), "model": "other"},
        ]
    }
    assert recovery_plan(audit, MODEL) == {
        "free_response_judge_plain_naive": ["amc23"],
        "free_response_plain_naive": ["math_500"],
    }


def test_recovery_plan_requires_selected_model() -> None:
    with pytest.raises(ValueError, match="absent"):
        recovery_plan({"cells": []}, MODEL)


def _scored_row(evaluator: str) -> dict:
    return {
        "status": "Completed",
        "evaluator": evaluator,
        "cot_mode": "CoT",
        "sampling_config": {
            "effective_sample_count": 4,
            "prompt_profile": "naive",
        },
        "metrics": {"avg@4": 0.25},
    }


def _complete_stats(**overrides) -> dict:
    stats = {
        "completion_count": 4,
        "completed_completion_count": 4,
        "distinct_coordinates": 4,
        "eval_count": 4,
        "missing_prediction_count": 0,
        "blank_primary_count": 0,
        "noncompleted_completion_count": 0,
        "leading_orphan_close_count": 0,
        "blank_eval_answer_count": 0,
        "overall_truncation_count": 0,
        "final_stage_truncation_count": 0,
    }
    stats.update(overrides)
    return stats


def test_blank_final_answer_is_valid_wrong_outcome_outside_choice_tasks() -> None:
    stats = _complete_stats(
        missing_prediction_count=2,
        blank_eval_answer_count=2,
        final_stage_truncation_count=2,
    )
    reasons = _quality_reasons(
        _scored_row("free_response_naive"),
        stats,
        domain="math",
        benchmark="math_500",
        mode="CoT",
    )
    assert reasons == []
    assert _quality_warnings(stats, domain="math") == [
        "final_stage_truncation_count:2",
        "model_missing_prediction:2",
        "model_blank_answer:2",
    ]


def test_blank_choice_answer_still_invalidates_score_evidence() -> None:
    stats = _complete_stats(
        missing_prediction_count=2,
        blank_eval_answer_count=2,
    )
    reasons = _quality_reasons(
        _scored_row("multi_choice_cot_naive"),
        stats,
        domain="knowledge",
        benchmark="mmlu",
        mode="CoT",
    )
    assert "missing_prediction:2" in reasons
    assert "blank_eval_answers:2" in reasons


def test_blank_primary_sql_trims_whitespace() -> None:
    assert "NULLIF(BTRIM(c.context->>'direct_raw_completion'), '')" in STATS_QUERY
    assert "NULLIF(BTRIM(c.context #>> '{stages,0,completion}'), '')" in STATS_QUERY
