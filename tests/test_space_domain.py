from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset
from src.space.constants import TableCellMeta
from src.space.data import _infer_domain
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
