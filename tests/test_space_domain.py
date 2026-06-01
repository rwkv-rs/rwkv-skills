from datetime import datetime
from pathlib import Path

from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset
from src.space.constants import TableCellMeta
from src.space.data import ScoreEntry, _infer_domain, _score_entry_from_db
from src.space.metrics import (
    _cell_metric_value,
    _cell_numeric_value,
    _detail_rows_for_entry,
    _field_primary_score,
    _metric_score,
)
from src.space.tables import _split_rows_by_suspect


def test_bfcl_and_toolalpaca_domains_are_function_call() -> None:
    assert _infer_domain("bfcl_exec_multiple_nocot", is_cot=False, task=None) == "function_call系列"
    assert (
        _infer_domain("bfcl_exec_multiple_test", is_cot=True, task="function_one_step_bfcl_ast")
        == "function_call系列"
    )
    assert (
        _infer_domain("toolalpaca_eval_real_test", is_cot=True, task="function_one_step_toolalpaca")
        == "function_call系列"
    )


def test_function_call_task_domains_are_function_call() -> None:
    assert _infer_domain("unknown_test", is_cot=True, task="function_one_step_bfcl_ast") == "function_call系列"
    assert _infer_domain("unknown_test", is_cot=True, task="function_agent_apibank_l2") == "function_call系列"
    assert _infer_domain("unknown_test", is_cot=False, task="eval_function_call") == "function_call系列"


def test_bfcl_and_toolalpaca_scheduler_jobs_are_nocot() -> None:
    assert JOB_CATALOGUE["function_one_step_bfcl_ast"].is_cot is False
    assert JOB_CATALOGUE["function_one_step_bfcl_exec"].is_cot is False
    assert JOB_CATALOGUE["function_one_step_toolalpaca"].is_cot is False
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=False) == "function_one_step_bfcl_exec"
    assert detect_job_from_dataset("toolalpaca_eval_real_test", is_cot=False) == "function_one_step_toolalpaca"
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=True) is None


def _db_score_payload(dataset: str, task: str, metrics: dict[str, float]) -> dict[str, object]:
    return {
        "task_id": 1,
        "dataset": dataset,
        "model": "rwkv7-g1f-13.3b-20260415-ctx8192",
        "metrics": metrics,
        "samples": 100,
        "problems": 100,
        "created_at": datetime(2026, 1, 1),
        "log_path": "",
        "cot": True,
        "task": task,
    }


def test_space_db_loader_keeps_only_current_free_response_route() -> None:
    college_exact = _score_entry_from_db(
        _db_score_payload("college_math_test", "eval_free_response", {"exact_accuracy": 0.8, "avg@2": 0.8}),
        errors=None,
    )
    assert college_exact is not None
    assert _score_entry_from_db(
        _db_score_payload("college_math_test", "eval_free_response_judge", {"judge_accuracy": 0.9, "avg@2": 0.9}),
        errors=None,
    ) is None

    assert _score_entry_from_db(
        _db_score_payload("math_500_test", "eval_free_response", {"exact_accuracy": 0.7, "avg@4": 0.7}),
        errors=None,
    ) is None
    assert _score_entry_from_db(
        _db_score_payload("math_500_test", "eval_free_response_judge", {"judge_accuracy": 0.75, "avg@4": 0.75}),
        errors=None,
    ) is not None


def test_college_math_space_detail_uses_exact_method() -> None:
    entry = _score_entry_from_db(
        _db_score_payload(
            "college_math_test",
            "eval_free_response",
            {"exact_accuracy": 0.8, "judge_accuracy": 0.9, "avg@2": 0.85},
        ),
        errors=None,
    )
    assert entry is not None

    rows = _detail_rows_for_entry(entry)

    assert rows == [("college_math_cot", "exact_match", "avg@2", 0.8)]


def test_split_rows_by_suspect_remaps_cell_meta() -> None:
    rows = [
        ["normal", "10.0", ("+1.0", "cell-delta-pos")],
        ["suspect", "11.0", ("-2.0", "cell-delta-suspect")],
        ["normal2", "12.0", ("0.0", "cell-delta-zero")],
    ]
    meta = TableCellMeta(
        cell_id="cell-test",
        task_id=123,
        benchmark_name="suspect",
        eval_method="cot",
        k_metric="pass@1",
        column_label="delta",
        model="rwkv7-test",
        tooltip=None,
        clickable=True,
    )

    (normal_rows, normal_meta), (suspect_rows, suspect_meta) = _split_rows_by_suspect(rows, {(1, 2): meta})

    assert [row[0] for row in normal_rows] == ["normal", "normal2"]
    assert [row[0] for row in suspect_rows] == ["suspect"]
    assert normal_meta == {}
    assert suspect_meta == {(0, 2): meta}


def _function_call_score_entry(metrics: dict[str, float]) -> ScoreEntry:
    return ScoreEntry(
        task_id=123,
        dataset="bfcl_exec_parallel_multiple_test",
        model="rwkv7-g1f-13.3b-20260415-ctx8192",
        metrics=metrics,
        samples=40,
        problems=40,
        created_at=datetime(2026, 5, 28),
        log_path="",
        cot=False,
        task="function_one_step_bfcl_exec",
        task_details=None,
        path=Path("<test>"),
        relative_path=Path("<test>"),
        domain="function_call系列",
        extra={},
        arch_version="RWKV7",
        data_version="G1F",
        num_params="13.3B",
    )


def test_function_call_scores_prefer_avg_at_1_over_success_rate() -> None:
    entry = _function_call_score_entry({"success_rate": 0.0, "avg@1": 0.5})

    assert _cell_metric_value(entry, dataset_base="bfcl_exec_parallel_multiple") == "avg@1 50.0%"
    assert _cell_numeric_value(entry, dataset_base="bfcl_exec_parallel_multiple") == 0.5
    assert _metric_score(entry) == 0.5
    assert _field_primary_score(entry) == 0.5
