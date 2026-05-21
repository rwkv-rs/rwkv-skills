from src.eval.scheduler.jobs import JOB_CATALOGUE, detect_job_from_dataset
from src.space.data import _infer_domain
from src.space.metrics import _benchmark_name


def test_bfcl_and_toolalpaca_domains_are_function_call() -> None:
    assert _infer_domain("bfcl_exec_multiple_nocot", is_cot=False, task=None) == "function_call系列"
    assert _infer_domain("bfcl_exec_multiple_test", is_cot=True, task="function_bfcl_ast") == "function_call系列"
    assert (
        _infer_domain("toolalpaca_eval_real_test", is_cot=True, task="function_toolalpaca")
        == "function_call系列"
    )


def test_function_call_task_domains_are_function_call() -> None:
    assert _infer_domain("unknown_test", is_cot=True, task="function_bfcl_ast") == "function_call系列"
    assert _infer_domain("unknown_test", is_cot=False, task="eval_function_call") == "function_call系列"


def test_bfcl_and_toolalpaca_scheduler_jobs_are_nocot() -> None:
    assert JOB_CATALOGUE["function_bfcl_ast"].is_cot is False
    assert JOB_CATALOGUE["function_bfcl_exec"].is_cot is False
    assert JOB_CATALOGUE["function_toolalpaca"].is_cot is False
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=False) == "function_bfcl_exec"
    assert detect_job_from_dataset("toolalpaca_eval_real_test", is_cot=False) == "function_toolalpaca"
    assert detect_job_from_dataset("bfcl_exec_multiple_test", is_cot=True) is None


def test_function_call_display_forces_nocot_for_legacy_cot_rows() -> None:
    from datetime import datetime
    from pathlib import Path

    from src.space.data import ScoreEntry

    entry = ScoreEntry(
        task_id=1,
        dataset="bfcl_exec_multiple_test",
        model="rwkv7-g1f-13.3b",
        metrics={"success_rate": 1.0},
        samples=50,
        problems=50,
        created_at=datetime(2026, 1, 1),
        log_path="",
        cot=True,
        task="function_bfcl_ast",
        task_details=None,
        path=Path("<test>"),
        relative_path=Path("<test>"),
        domain="function_call系列",
        extra={},
        arch_version="RWKV7",
        data_version="G1F",
        num_params="13_3b",
    )

    assert _benchmark_name(entry) == "bfcl_exec_multiple_nocot"
