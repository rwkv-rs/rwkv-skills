from __future__ import annotations

from src.eval.tasks.coding.pipeline import (
    _format_lcb_cot_prompt,
    _format_lcb_final_prompt,
    _format_prompt,
    _format_prompt_no_echo,
    _format_signature_prompt,
)
from src.eval.results.schema import prompt_delta


def test_humaneval_prompt_uses_legacy_echo_format() -> None:
    prompt = '''
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers are closer than the threshold."""
'''

    formatted = _format_prompt(prompt)

    assert formatted.endswith("Assistant: <think>\n</think>\n```python")
    assert "def has_close_elements(numbers: List[float], threshold: float) -> bool:" in formatted
    assert "Function signature:" not in formatted


def test_mbpp_prompt_uses_legacy_no_echo_format() -> None:
    prompt = '''"""
Write a function to find the n'th star number.
assert find_star_num(3) == 37
"""'''

    formatted = _format_signature_prompt(prompt, "def find_star_num(n):")

    assert formatted == (
        "User: You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        "\"\"\"\n"
        "Write a function to find the n'th star number.\n"
        "assert find_star_num(3) == 37\n"
        "\"\"\"\n"
        "Function signature: def find_star_num(n):\n"
        "Write the full function definition.\n\n"
        "Assistant: <think></think>\n"
        "```python"
    )


def test_mbpp_prompt_without_signature_uses_legacy_no_echo_format() -> None:
    formatted = _format_prompt_no_echo("Write a function that returns 1.")

    assert formatted.endswith("Assistant: <think></think>\n```python")
    assert "Complete the following code without any additional text or explanation" in formatted


def test_livecodebench_prompt_uses_legacy_two_stage_format() -> None:
    cot_prompt = _format_lcb_cot_prompt(
        "Write a program that prints the sum.",
        "def solve() -> None:\n    pass",
    )
    final_prompt = _format_lcb_final_prompt(cot_prompt, ">First reason about the input format.")
    delta = prompt_delta(final_prompt, f"{cot_prompt}>First reason about the input format.")

    assert cot_prompt.endswith("Assistant: <think")
    assert delta == "\n</think>\n```python\n"
    assert "Therefore, the correct code is" not in final_prompt
