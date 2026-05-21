from __future__ import annotations

from pathlib import Path

from src.eval.evaluators.coding import _format_lcb_cot_prompt, _format_prompt_no_echo
from src.infer.engine import _normalize_prompt


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


def test_formal_generation_prompts_do_not_insert_space_after_assistant_colon() -> None:
    files = [
        ROOT / "src/eval/evaluators/free_response.py",
        ROOT / "src/eval/evaluators/multi_choice.py",
        ROOT / "src/eval/evaluators/instruction_following.py",
        ROOT / "src/eval/evaluators/function_call.py",
    ]
    prompts = [_format_prompt_no_echo("Write a function that returns 1.")]
    prompts.append(_format_lcb_cot_prompt("Write a program that prints 1.", None))
    prompts.extend(path.read_text(encoding="utf-8") for path in files)

    for prompt in prompts:
        assert "Assistant: <think" not in prompt
        assert "Assistant: The answer is" not in prompt
        assert "Assistant: 正确答案是" not in prompt


def test_inference_prompt_normalization_strips_terminal_assistant_space() -> None:
    assert _normalize_prompt("User: hello\n\nAssistant: ") == "User: hello\n\nAssistant:"
