from __future__ import annotations

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
