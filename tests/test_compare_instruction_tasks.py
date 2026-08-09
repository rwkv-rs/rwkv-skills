from pathlib import Path

import pytest

from ops.g1i_strict46.compare_instruction_tasks import _dataset_path_for_tasks


def test_selects_ifeval_dataset_for_ifeval_tasks() -> None:
    path = _dataset_path_for_tasks(
        [{"benchmark_name": "ifeval"}, {"benchmark_name": "IFEVAL"}]
    )
    assert path == Path(__file__).resolve().parents[1] / "data" / "ifeval" / "test.jsonl"


def test_selects_ifbench_dataset_for_ifbench_tasks() -> None:
    path = _dataset_path_for_tasks(
        [{"benchmark_name": "ifbench"}, {"benchmark_name": "ifbench"}]
    )
    assert path == Path(__file__).resolve().parents[1] / "data" / "ifbench" / "test.jsonl"


def test_rejects_cross_benchmark_comparison() -> None:
    with pytest.raises(ValueError, match="requires one benchmark"):
        _dataset_path_for_tasks(
            [{"benchmark_name": "ifeval"}, {"benchmark_name": "ifbench"}]
        )


def test_rejects_unknown_instruction_benchmark() -> None:
    with pytest.raises(ValueError, match="unsupported instruction benchmark"):
        _dataset_path_for_tasks([{"benchmark_name": "unknown"}])
