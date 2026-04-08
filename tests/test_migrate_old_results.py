from __future__ import annotations

from pathlib import Path

from src.bin.migrate_old_results import (
    _build_metrics,
    _infer_type_from_name,
    _prefix_type,
)


def test_prefix_type_maps_legacy_single_choice_to_canonical_job_names() -> None:
    task, is_cot = _prefix_type(
        {
            "benchmark": "single_choice_plain_mmlu_test",
            "dataset_slug": "mmlu_test",
        }
    )
    assert task == "multi_choice_plain"
    assert is_cot is False

    task, is_cot = _prefix_type(
        {
            "benchmark": "single_choice_cot_mmlu_test",
            "dataset_slug": "mmlu_test",
        }
    )
    assert task == "multi_choice_cot"
    assert is_cot is True


def test_infer_type_from_name_uses_canonical_scheduler_job_names() -> None:
    assert _infer_type_from_name(Path("cot_llm_judge_gsm8k_test.json"), "gsm8k_test") == (
        "free_response_judge",
        True,
    )
    assert _infer_type_from_name(Path("cot_mmlu_test.json"), "mmlu_test") == (
        "multi_choice_cot",
        True,
    )
    assert _infer_type_from_name(Path("results_mmlu_test.json"), "mmlu_test") == (
        "multi_choice_plain",
        False,
    )
    assert _infer_type_from_name(Path("results_mbpp_test.json"), "mbpp_test") == (
        "code_mbpp",
        False,
    )


def test_build_metrics_accepts_canonical_multi_choice_task_names() -> None:
    metrics, task_details = _build_metrics(
        "multi_choice_cot",
        {
            "accuracy": 0.75,
            "score_by_subject": {
                "history": {"accuracy": 0.5},
                "math": {"accuracy": 1.0},
            },
        },
    )

    assert metrics == {"accuracy": 0.75}
    assert task_details == {
        "accuracy_by_subject": {
            "history": 0.5,
            "math": 1.0,
        }
    }
