from __future__ import annotations

from src.eval.benchmark_registry import CoTMode
from src.eval.execution_plan import (
    avg_k_metric_key,
    build_attempt_keys,
    build_auto_avg_k_execution_plan,
)
from src.eval.field_common import build_task_sampling_config
from src.eval.evaluators.common import sample_repeat_seed


def test_auto_avg_k_plan_downsamples_large_datasets_to_ratio() -> None:
    plan = build_auto_avg_k_execution_plan("mmlu_test", 12_500)

    assert plan.repeat_count == 1
    assert plan.sample_size == 5_000
    assert plan.effective_sample_count == 5_000
    assert 0.0 < plan.avg_k < 1.0
    assert avg_k_metric_key(plan.avg_k).startswith("avg@0.")


def test_auto_avg_k_plan_uses_small_datasets_once() -> None:
    plan = build_auto_avg_k_execution_plan("math_500_test", 500)

    assert plan.avg_k == 1.0
    assert plan.repeat_count == 1
    assert plan.sample_size == 500
    assert plan.effective_sample_count == 500
    assert plan.sample_indices[:3] == (0, 1, 2)


def test_build_task_sampling_config_uses_rwkv_rs_shape() -> None:
    payload = build_task_sampling_config(
        cot_mode=CoTMode.COT,
        avg_k=4.0,
        sampling_config={"stage1": {"temperature": 0.7}},
        effective_sample_count=20,
        pass_ks=(1, 4, 4),
        judger_model_name="judge-v1",
    )

    assert payload == {
        "cot_mode": "CoT",
        "n_shot": 0,
        "avg_k": 4.0,
        "sample_limit": None,
        "effective_sample_count": 20,
        "pass_ks": [1, 4],
        "sampling_config": {"stage1": {"temperature": 0.7}},
        "judger_model_name": "judge-v1",
        "checker_model_name": None,
    }


def test_build_attempt_keys_tracks_pass_index_in_identity() -> None:
    plan = build_auto_avg_k_execution_plan("math_500_test", 500)

    keys = build_attempt_keys(plan, max_pass_k=2)

    assert keys[0].as_tuple() == (0, 0, 0)
    assert keys[1].as_tuple() == (0, 0, 1)
    assert keys[2].as_tuple() == (1, 0, 0)


def test_sample_repeat_seed_changes_when_pass_index_changes() -> None:
    base = sample_repeat_seed(7, 3, stage=2)
    alternate = sample_repeat_seed(7, 3, pass_index=1, stage=2)

    assert base != alternate
    assert base == sample_repeat_seed(7, 3, pass_index=0, stage=2)
