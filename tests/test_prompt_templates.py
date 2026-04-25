from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_free_response_default_prompt_is_legacy_boxed_format() -> None:
    prompt = (ROOT / "src/eval/evaluators/free_response.py").read_text(encoding="utf-8")

    assert "User: <Q>" in prompt
    assert r"\\boxed{" in prompt


def test_multi_choice_default_prompt_is_legacy_template() -> None:
    prompt = (ROOT / "src/eval/evaluators/multi_choice.py").read_text(encoding="utf-8")

    assert "You are a very talented expert" in prompt
    assert "Choose the single best option" not in prompt
    assert "唯一最佳选项" not in prompt


def test_prompt_trials_file_records_candidate_prompts() -> None:
    prompt_trials = (ROOT / "prompt_trials.toml").read_text(encoding="utf-8")

    assert "[trial.multi_choice_zh_cot_v1]" in prompt_trials
    assert "[trial.multi_choice_gpqa_cot_v1]" in prompt_trials
    assert "[trial.free_response_math500_v1]" in prompt_trials
