from __future__ import annotations

from pathlib import Path

from src.eval.scheduler import jobs, queue
from src.eval.scheduler.cli import build_parser


def _prepare_single_latest_model(monkeypatch, tmp_path: Path) -> Path:
    model_path = tmp_path / "weights" / "rwkv7-g1a-2.9b-20250728-ctx4096.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.touch()

    monkeypatch.setattr(queue, "expand_model_paths", lambda _patterns: [model_path])
    monkeypatch.setattr(queue, "filter_model_paths", lambda paths, *_args: list(paths))
    monkeypatch.setattr(queue, "RESULTS_ROOT", tmp_path / "results")
    return model_path


def test_detect_job_from_dataset_prioritizes_judge_and_agent_benchmarks() -> None:
    assert jobs.detect_job_from_dataset("math_500_test", True) == "free_response_judge"
    assert jobs.detect_job_from_dataset("hendrycks_math_test", True) == "free_response"
    assert jobs.detect_job_from_dataset("tau_bench_retail_test", True) == "function_tau_bench"
    assert jobs.detect_job_from_dataset("tau2_bench_telecom_base", True) == "function_tau2_bench"


def test_build_queue_replaces_latest_2_9b_math_benchmarks_with_param_search(monkeypatch, tmp_path: Path) -> None:
    _prepare_single_latest_model(monkeypatch, tmp_path)
    monkeypatch.setattr(queue, "_param_search_done", lambda *_args: False)

    items = queue.build_queue(
        model_globs=("weights/*.pth",),
        job_order=(
            "free_response",
            "free_response_judge",
            "param_search_free_response",
            "param_search_free_response_judge",
            "param_search_select",
        ),
        completed=(),
        failed=(),
        running=(),
        skip_dataset_slugs=(),
        only_dataset_slugs=("gsm8k_test", "math_500_test"),
        model_select="all",
        min_param_b=None,
        max_param_b=None,
        enable_param_search=True,
        model_name_patterns=(),
    )

    assert [(item.job_name, item.dataset_slug) for item in items] == [
        ("param_search_free_response", "math_500_test"),
        ("param_search_free_response_judge", "gsm8k_test"),
    ]


def test_build_queue_adds_param_search_select_after_trials_finish(monkeypatch, tmp_path: Path) -> None:
    _prepare_single_latest_model(monkeypatch, tmp_path)
    monkeypatch.setattr(queue, "_param_search_done", lambda *_args: True)

    items = queue.build_queue(
        model_globs=("weights/*.pth",),
        job_order=(
            "free_response",
            "free_response_judge",
            "param_search_free_response",
            "param_search_free_response_judge",
            "param_search_select",
        ),
        completed=(),
        failed=(),
        running=(),
        skip_dataset_slugs=(),
        only_dataset_slugs=("gsm8k_test", "math_500_test"),
        model_select="all",
        min_param_b=None,
        max_param_b=None,
        enable_param_search=True,
        model_name_patterns=(),
    )

    assert [(item.job_name, item.dataset_slug) for item in items] == [
        ("param_search_select", "gsm8k_test"),
    ]


def test_cli_supports_rwkv_rs_style_benchmark_selection_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "queue",
            "--run-mode",
            "rerun",
            "--benchmark-fields",
            "knowledge",
            "--extra-benchmarks",
            "gsm8k",
        ]
    )

    assert args.run_mode == "rerun"
    assert args.benchmark_fields == ["knowledge"]
    assert args.extra_benchmarks == ["gsm8k"]
