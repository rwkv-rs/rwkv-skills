from __future__ import annotations

import pytest

from src.eval.scheduler.dataset_utils import (
    canonicalize_benchmark_list,
    infer_dataset_slug_from_path,
    split_benchmark_and_split,
)


def test_canonicalize_benchmark_list_resolves_aliases_and_base_names() -> None:
    known_slugs = {
        "hendrycks_math_test",
        "ifeval_test",
        "livecodebench_test",
        "math_500_test",
    }

    resolved = canonicalize_benchmark_list(
        ["math", "math500", "lcb", "ifeval"],
        known_slugs=known_slugs,
    )

    assert resolved == (
        "hendrycks_math_test",
        "ifeval_test",
        "livecodebench_test",
        "math_500_test",
    )


def test_canonicalize_benchmark_list_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="未知的 benchmark 名称: unknown_bench"):
        canonicalize_benchmark_list(["unknown-bench"], known_slugs={"mmlu_test"})


def test_infer_dataset_slug_from_path_handles_split_files_and_aliases() -> None:
    assert infer_dataset_slug_from_path("/tmp/data/mmlu/test.jsonl") == "mmlu_test"
    assert infer_dataset_slug_from_path("/tmp/data/hle/math.jsonl") == "hle_math"
    assert infer_dataset_slug_from_path("/tmp/data/lcb.jsonl") == "livecodebench_test"


def test_split_benchmark_and_split_strips_cot_suffix() -> None:
    assert split_benchmark_and_split("gpqa_main__cot") == ("gpqa", "main")
    assert split_benchmark_and_split("human_eval_plus_test") == ("human_eval_plus", "test")
