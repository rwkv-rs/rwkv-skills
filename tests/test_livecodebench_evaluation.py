from __future__ import annotations

from pathlib import Path

from src.eval.datasets.data_struct.code_generation import CodeGenerationRecord
from src.eval.metrics.code_generation.livecodebench import evaluation


def test_evaluate_livecodebench_dataset_accepts_generator_completions(monkeypatch, tmp_path: Path) -> None:
    dataset_path = tmp_path / "livecodebench_test.jsonl"
    dataset_path.write_text("", encoding="utf-8")

    records = [
        CodeGenerationRecord(
            task_id="task-0",
            prompt="solve 0",
            metadata={
                "metadata": {"func_name": "solve"},
                "public_test_cases": [{"input": "1", "output": "1"}],
                "private_test_cases": [],
                "difficulty": "easy",
            },
        ),
        CodeGenerationRecord(
            task_id="task-1",
            prompt="solve 1",
            metadata={
                "metadata": {"func_name": "solve"},
                "public_test_cases": [{"input": "2", "output": "2"}],
                "private_test_cases": [],
                "difficulty": "hard",
            },
        ),
    ]

    class _FakeLoader:
        def __init__(self, _path: str) -> None:
            self._path = _path

        def load(self) -> list[CodeGenerationRecord]:
            return records

    def _fake_check_correctness(
        sample_index: int,
        repeat_index: int,
        sample: dict,
        code: str,
        timeout: float,
    ) -> dict[str, object]:
        del sample, code, timeout
        return {
            "sample_index": sample_index,
            "repeat_index": repeat_index,
            "passed": sample_index == 0,
            "result": "incorrect output" if sample_index != 0 else "accepted",
        }

    monkeypatch.setattr(evaluation, "JsonlCodeGenerationLoader", _FakeLoader)
    monkeypatch.setattr(evaluation, "check_correctness", _fake_check_correctness)

    completions = (
        {
            "benchmark_name": "livecodebench",
            "dataset_split": "test",
            "sample_index": idx,
            "repeat_index": 0,
            "pass_index": 0,
            "prompt0": f"prompt-{idx}",
            "completion0": f"```python\nprint({idx})\n```",
        }
        for idx in range(2)
    )

    metrics, eval_payloads = evaluation.evaluate_livecodebench_dataset(
        completions,
        dataset_path=dataset_path,
        pass_k=(1,),
        n_workers=1,
    )

    assert metrics["pass@1"] == 0.5
    assert metrics["pass@1_easy"] == 1.0
    assert metrics["pass@1_hard"] == 0.0
    assert len(eval_payloads) == 2
    assert eval_payloads[0]["is_passed"] is True
    assert eval_payloads[1]["is_passed"] is False
    assert eval_payloads[1]["fail_reason"] == "incorrect output"
