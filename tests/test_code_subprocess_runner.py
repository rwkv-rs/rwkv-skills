from __future__ import annotations

import multiprocessing
import time

from src.eval.metrics.code_generation.human_eval.execution import (
    _format_plus_assertions,
    check_correctness as check_human_eval,
)
from src.eval.metrics.code_generation.livecodebench.execution import (
    check_correctness as check_livecodebench,
)
from src.eval.metrics.code_generation.mbpp.execution import (
    check_correctness as check_mbpp,
)
from src.eval.metrics.code_generation.subprocess_runner import run_isolated


def _emit(value: str, sink) -> None:
    sink.append(value)


def _hang(sink) -> None:
    del sink
    time.sleep(10)


def _exit_without_result(sink) -> None:
    del sink


def test_run_isolated_returns_child_result() -> None:
    assert run_isolated(_emit, ("passed",), timeout=1.0) == "passed"


def test_run_isolated_times_out_and_reaps_child() -> None:
    before = {child.pid for child in multiprocessing.active_children()}
    started = time.monotonic()
    assert run_isolated(_hang, (), timeout=0.05) == "timed out"
    assert time.monotonic() - started < 2.0
    after = {child.pid for child in multiprocessing.active_children()}
    assert after <= before


def test_run_isolated_handles_child_without_result() -> None:
    assert run_isolated(_exit_without_result, (), timeout=0.2) == "timed out"


def test_human_eval_uses_shared_isolated_runner() -> None:
    problem = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    }
    result = check_human_eval(problem, "    return a + b\n", 1.0, completion_id=7)
    assert result == {
        "task_id": "HumanEval/0",
        "passed": True,
        "result": "passed",
        "completion_id": 7,
    }


def test_human_eval_plus_large_integer_is_supported_by_direct_call() -> None:
    huge = 10**5000
    problem = {
        "task_id": "HumanEvalPlus/large-int",
        "prompt": "def identity(value):\n",
        "entry_point": "identity",
        "canonical_solution": "    return value\n",
        "plus_input": [[huge]],
        "test": "def check(candidate):\n    assert candidate(1) == 1\n",
    }
    result = check_human_eval(problem, "    return value\n", 1.0)
    assert result["passed"] is True

    cached = _format_plus_assertions(problem)
    cached_result = check_human_eval(
        problem,
        "    return value\n",
        1.0,
        plus_block=cached,
    )
    assert cached_result == result


def test_mbpp_uses_shared_isolated_runner() -> None:
    problem = {"task_id": "mbpp/0", "test_list": ["assert add(2, 3) == 5"]}
    result = check_mbpp(problem, "def add(a, b):\n    return a + b\n", 1.0, completion_id=3)
    assert result["passed"] is True
    assert result["result"] == "passed"


def test_livecodebench_uses_shared_isolated_runner() -> None:
    sample = {
        "input_output": '{"inputs":["2\\n3"],"outputs":["5"],"fn_name":"add"}'
    }
    result = check_livecodebench(
        0,
        0,
        sample,
        "def add(a, b):\n    return a + b\n",
        1.0,
    )
    assert result["passed"] is True
    assert result["result"] == "passed"
