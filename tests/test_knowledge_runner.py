from __future__ import annotations

from pathlib import Path

from src.eval.tasks.knowledge import runner as knowledge_runner


def test_knowledge_runner_parser_accepts_all_modes() -> None:
    args = knowledge_runner.parse_args(
        [
            "--dataset",
            "dataset.jsonl",
            "--cot-mode",
            "cot",
            "--probe-only",
        ]
    )
    assert args.cot_mode == "cot"
    assert args.probe_only is True
    assert args.run_checker is False


def test_knowledge_runner_parser_long_doc_flags_default_to_none() -> None:
    args = knowledge_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert args.prompt_max_chars is None
    assert args.long_doc_mode is None

    args = knowledge_runner.parse_args(
        ["--dataset", "dataset.jsonl", "--long-doc-mode", "lexical", "--prompt-max-chars", "4096"]
    )
    assert args.long_doc_mode == "lexical"
    assert args.prompt_max_chars == 4096


def test_knowledge_runner_checker_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("RWKV_KNOWLEDGE_RUN_CHECKER", raising=False)
    monkeypatch.delenv("RWKV_SKILLS_DISABLE_CHECKER", raising=False)
    monkeypatch.delenv("DISABLE_CHECKER", raising=False)

    args = knowledge_runner.parse_args(["--dataset", "dataset.jsonl"])
    assert knowledge_runner._should_run_checker(args) is False

    args = knowledge_runner.parse_args(["--dataset", "dataset.jsonl", "--run-checker"])
    assert knowledge_runner._should_run_checker(args) is True

    args = knowledge_runner.parse_args(["--dataset", "dataset.jsonl"])
    monkeypatch.setenv("RWKV_KNOWLEDGE_RUN_CHECKER", "1")
    assert knowledge_runner._should_run_checker(args) is True

    monkeypatch.setenv("RWKV_SKILLS_DISABLE_CHECKER", "1")
    assert knowledge_runner._should_run_checker(args) is False


def test_knowledge_cot_strategy_can_be_forced_without_oracle_cascade(monkeypatch) -> None:
    class Config:
        knowledge_cot_strategy = "cascade_a_b"

    monkeypatch.delenv("RWKV_KNOWLEDGE_COT_STRATEGY", raising=False)
    assert knowledge_runner._resolve_knowledge_cot_strategy(Config()) == "cascade_a_b"

    monkeypatch.setenv("RWKV_KNOWLEDGE_COT_STRATEGY", "two_stage")
    assert knowledge_runner._resolve_knowledge_cot_strategy(Config()) == "two_stage"


def test_knowledge_runner_resolves_cot_stage_sampling(monkeypatch) -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs" / "g1h"
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", str(config_root))

    sampling = knowledge_runner._resolve_cot_sampling_config(
        "gpqa_diamond_test",
        "rwkv7-g1h-7.2b-20260710-ctx10240",
    )

    assert sampling is not None
    assert sampling.temperature == 0.96
    assert sampling.top_p == 0.76
    assert sampling.top_k == 32
    assert sampling.alpha_presence == 1.0
    assert sampling.alpha_frequency == 0.1
    assert sampling.alpha_decay == 0.988


def test_g1h_multi_choice_template_uses_single_stage_answer_extraction(monkeypatch) -> None:
    from src.eval.benchmark_config import resolve_benchmark_model_config, resolve_sampling_config

    config_root = Path(__file__).resolve().parents[1] / "configs" / "g1h"
    monkeypatch.setenv("RWKV_BENCHMARK_CONFIG_ROOT", str(config_root))

    config = resolve_benchmark_model_config(
        "mmlu_test",
        "rwkv7-g1h-7.2b-20260710-ctx10240",
    )

    assert config is not None
    assert config.knowledge_cot_strategy == "cascade_a_b"
    sampling = resolve_sampling_config(
        "mmlu_test",
        "rwkv7-g1h-7.2b-20260710-ctx10240",
        stage="cot",
    )
    assert sampling is not None
    assert sampling.stop_tokens == (0,)
