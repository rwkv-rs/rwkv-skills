from __future__ import annotations

import pytest

from src.eval.benchmark_registry import CoTMode
from src.eval.coding import runner as coding_runner


def test_coding_runner_parser_accepts_benchmark_kind_and_cot_mode() -> None:
    args = coding_runner.parse_args(
        [
            "--model-path",
            "model.pth",
            "--dataset",
            "dataset.jsonl",
            "--benchmark-kind",
            "mbpp",
            "--cot-mode",
            "fake_cot",
            "--probe-only",
        ]
    )
    assert args.benchmark_kind == "mbpp"
    assert args.cot_mode == "fake_cot"
    assert args.probe_only is True


def test_coding_runner_rejects_non_legacy_mbpp_cot_modes() -> None:
    with pytest.raises(ValueError, match="mbpp legacy-aligned runner"):
        coding_runner._resolve_cot_mode(coding_runner.CodingBenchmarkKind.MBPP, CoTMode.FAKE_COT.value)


def test_coding_runner_parser_accepts_swebench_options() -> None:
    args = coding_runner.parse_args(
        [
            "--model-path",
            "model.pth",
            "--dataset",
            "dataset.jsonl",
            "--benchmark-kind",
            "swe_bench",
            "--cot-mode",
            "cot",
            "--swebench-run-harness",
            "--swebench-dataset-name",
            "princeton-nlp/SWE-bench_Lite",
            "--swebench-max-context-chars",
            "12000",
            "--swebench-harness-timeout-s",
            "3600",
            "--long-doc-mode",
            "model_parallel",
            "--long-doc-max-evidence-chars",
            "3000",
            "--long-doc-model-parallel-batch-size",
            "8",
        ]
    )
    assert args.benchmark_kind == "swe_bench"
    assert args.cot_mode == "cot"
    assert args.swebench_run_harness is True
    assert args.swebench_dataset_name == "princeton-nlp/SWE-bench_Lite"
    assert args.swebench_max_context_chars == 12000
    assert args.swebench_harness_timeout_s == 3600
    assert args.long_doc_mode == "model_parallel"
    assert args.long_doc_max_evidence_chars == 3000
    assert args.long_doc_model_parallel_batch_size == 8
