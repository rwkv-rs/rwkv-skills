from __future__ import annotations

from copy import deepcopy

from src.eval.metrics.code_generation.human_eval import evaluation


def test_plus_assertions_are_precomputed_once_per_problem(monkeypatch, tmp_path):
    problems = {
        "HumanEval/0": {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        }
    }
    samples = [
        {"task_id": "HumanEval/0", "completion": "    return a + b\n"},
        {"task_id": "HumanEval/0", "completion": "    return b + a\n"},
    ]
    assertion_calls: list[str] = []
    check_calls: list[object] = []

    monkeypatch.setattr(evaluation, "read_problems", lambda _path: problems)
    monkeypatch.setattr(
        evaluation,
        "stream_jsonl",
        lambda _path: iter(deepcopy(samples)),
    )

    def fake_assertions(problem):
        assertion_calls.append(problem["task_id"])
        return "    assert candidate(1, 2) == 3"

    def fake_check(problem, completion, timeout, completion_id, plus_block):
        del completion, timeout
        check_calls.append(plus_block)
        return {
            "task_id": problem["task_id"],
            "completion_id": completion_id,
            "passed": True,
            "result": "passed",
        }

    written = []
    monkeypatch.setattr(evaluation, "_format_plus_assertions", fake_assertions)
    monkeypatch.setattr(evaluation, "check_correctness", fake_check)
    monkeypatch.setattr(
        evaluation,
        "write_jsonl",
        lambda _path, rows: written.extend(rows),
    )

    metrics, _ = evaluation.evaluate_functional_correctness(
        str(tmp_path / "samples.jsonl"),
        k=(1,),
        n_workers=2,
        timeout=1.0,
        problem_file=str(tmp_path / "problems.jsonl"),
    )

    assert assertion_calls == ["HumanEval/0"]
    assert check_calls == [
        "    assert candidate(1, 2) == 3",
        "    assert candidate(1, 2) == 3",
    ]
    assert metrics == {"pass@1": 1.0}
    assert len(written) == 2
