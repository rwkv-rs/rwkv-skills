from __future__ import annotations

import unittest

from src.eval.evaluators.coding import (
    _format_lcb_cot_prompt,
    _format_lcb_final_prompt,
    _format_prompt,
    _format_prompt_no_echo,
)
from src.eval.results.schema import prompt_delta


class LiveCodeBenchPromptFormatTest(unittest.TestCase):
    def test_livecodebench_final_stage_switches_directly_to_python_block(self) -> None:
        cot_prompt = _format_lcb_cot_prompt(
            "Write a program that prints the sum.",
            "def solve() -> None:\n    pass",
        )
        cot_completion = ">First reason about the input format."

        final_prompt = _format_lcb_final_prompt(cot_prompt, cot_completion)
        delta = prompt_delta(final_prompt, f"{cot_prompt}{cot_completion}")

        self.assertEqual(delta, "\n</think>\n```python\n")
        self.assertNotIn("Therefore, the correct code is", final_prompt)

    def test_livecodebench_final_stage_matches_rwkv_code_fence_style(self) -> None:
        mbpp_prompt = _format_prompt_no_echo("Write a function that returns 1.")

        self.assertTrue(mbpp_prompt.endswith("Assistant: <think></think>\n```python"))
        self.assertIn("You are a top-level code master", mbpp_prompt)
        self.assertIn("Complete the following code without any additional text or explanation", mbpp_prompt)

    def test_mbpp_single_assert_prompt_discourages_formula_memorization(self) -> None:
        prompt = '''"""
Write a function to find the n'th star number.
assert find_star_num(3) == 37
"""'''

        formatted = _format_prompt_no_echo(prompt)

        self.assertIn("assert find_star_num(3) == 37", formatted)
        self.assertTrue(formatted.endswith("Assistant: <think></think>\n```python"))

    def test_humaneval_prompt_uses_legacy_echo_format(self) -> None:
        prompt = '''
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each other than given threshold."""
'''

        formatted = _format_prompt(prompt)

        self.assertIn("Assistant:from typing import List", formatted)
        self.assertIn("def has_close_elements(numbers: List[float], threshold: float) -> bool:", formatted)
        self.assertNotIn("Function signature:", formatted)


if __name__ == "__main__":
    unittest.main()
