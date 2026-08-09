from __future__ import annotations

import pytest

from src.eval.benchmark_registry import CoTMode
from src.eval.tasks.coding import runner as coding_runner
from src.infer.sampling import SamplingConfig


def test_coding_runner_parser_accepts_benchmark_kind_and_cot_mode() -> None:
    args = coding_runner.parse_args(
        [
            "--dataset",
            "dataset.jsonl",
            "--benchmark-kind",
            "mbpp",
            "--cot-mode",
            "no_cot",
            "--probe-only",
        ]
    )
    assert args.benchmark_kind == "mbpp"
    assert args.cot_mode == "no_cot"
    assert args.probe_only is True
    assert args.run_checker is False


def test_coding_runner_checker_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_CODING_RUN_CHECKER", raising=False)
    monkeypatch.delenv("RWKV_SKILLS_DISABLE_CHECKER", raising=False)
    monkeypatch.delenv("DISABLE_CHECKER", raising=False)

    args = coding_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert coding_runner._should_run_checker(args) is False

    args = coding_runner.parse_args(["--dataset", "dataset.jsonl", "--run-checker"])
    assert coding_runner._should_run_checker(args) is True

    args = coding_runner.parse_args(["--dataset", "dataset.jsonl"])
    monkeypatch.setenv("RWKV_CODING_RUN_CHECKER", "1")
    assert coding_runner._should_run_checker(args) is True

    monkeypatch.setenv("RWKV_SKILLS_DISABLE_CHECKER", "1")
    assert coding_runner._should_run_checker(args) is False


def test_coding_runner_rejects_non_legacy_mbpp_cot_modes() -> None:
    with pytest.raises(ValueError, match="mbpp legacy-aligned runner"):
        coding_runner._resolve_cot_mode(coding_runner.CodingBenchmarkKind.MBPP, CoTMode.COT.value)


def test_livecodebench_accepts_direct_nocot_mode() -> None:
    assert (
        coding_runner._resolve_cot_mode(
            coding_runner.CodingBenchmarkKind.LIVECODEBENCH,
            CoTMode.NO_COT.value,
        )
        is CoTMode.NO_COT
    )

    sampling = SamplingConfig(max_generate_tokens=32)
    payload = coding_runner._sampling_payload(
        coding_runner.CodingBenchmarkKind.LIVECODEBENCH,
        CoTMode.NO_COT,
        cot_sampling=sampling,
    )
    assert set(payload) == {"stage1"}


@pytest.mark.parametrize(
    "kind, expected",
    [
        (coding_runner.CodingBenchmarkKind.HUMAN_EVAL, True),
        (coding_runner.CodingBenchmarkKind.MBPP, True),
        (coding_runner.CodingBenchmarkKind.LIVECODEBENCH, True),
    ],
)
def test_coding_runner_uses_raw_completions_for_literal_prefill(kind, expected) -> None:
    assert coding_runner._requires_completion_style_remote(kind) is expected


@pytest.mark.parametrize(
    "dataset_slug",
    [
        "human_eval_test",
        "human_eval_cn_test",
        "human_eval_fix_test",
        "human_eval_plus_test",
    ],
)
def test_coding_runner_treats_human_eval_variants_as_human_eval(dataset_slug: str) -> None:
    assert (
        coding_runner._resolve_benchmark_kind(dataset_slug, coding_runner.CodingBenchmarkKind.AUTO)
        is coding_runner.CodingBenchmarkKind.HUMAN_EVAL
    )
    assert (
        coding_runner._resolve_benchmark_kind(dataset_slug, coding_runner.CodingBenchmarkKind.HUMAN_EVAL)
        is coding_runner.CodingBenchmarkKind.HUMAN_EVAL
    )
