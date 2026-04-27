from __future__ import annotations


def _compress_newlines(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _format_prompt(prompt: str) -> str:
    """HumanEval prompt aligned with the MBPP no-echo code-block format."""

    signature = _extract_function_signature(prompt)
    if signature:
        return _format_signature_prompt(prompt, signature)
    return _format_prompt_no_echo(prompt)


def _format_prompt_no_echo(prompt: str) -> str:
    """Variant without echoing prompt after Assistant (used for bug-fix style prompts)."""

    clean = _compress_newlines(prompt).strip()
    return (
        "User: You are a top-level code master. Complete the following code without any additional text or explanation:\n"
        f"{clean}\n\nAssistant: <think></think>\n```python"
    )


def _extract_function_signature(code: str | None) -> str | None:
    if not code:
        return None
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def ") and stripped.endswith(":"):
            return stripped
    return None


def _format_signature_prompt(prompt: str, signature: str) -> str:
    prompt = f"{prompt}\nFunction signature: {signature}\nWrite the full function definition."
    return _format_prompt_no_echo(prompt)


_LCB_SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)
_LCB_FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
)
_LCB_FORMAT_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def _format_lcb_body(question: str, starter_code: str | None) -> str:
    clean = (question or "").strip()
    body = f"### Question:\n{clean}\n\n"
    if starter_code and starter_code.strip():
        body += f"### Format: {_LCB_FORMAT_WITH_STARTER}\n"
        body += f"```python\n{starter_code}\n```\n\n"
    else:
        body += f"### Format: {_LCB_FORMAT_WITHOUT_STARTER}\n"
        body += "```python\n# YOUR CODE HERE\n```\n\n"
    body += "### Answer: (use the provided format with backticks)\n\n"
    return body


# Match the existing RWKV code prompts: finish the reasoning span and enter a
# Python code block directly, instead of adding an extra natural-language bridge.
_LCB_FINAL_ANSWER_PREFIX = "\n</think>\n```python\n"


def _format_lcb_cot_prompt(question: str, starter_code: str | None) -> str:
    body = _format_lcb_body(question, starter_code)
    return f"User: {_LCB_SYSTEM_MESSAGE}\n{body}Assistant: <think"


def _format_lcb_final_prompt(cot_prompt: str, cot_completion: str) -> str:
    return f"{cot_prompt}{cot_completion}{_LCB_FINAL_ANSWER_PREFIX}"


__all__ = [
    "_extract_function_signature",
    "_format_lcb_cot_prompt",
    "_format_lcb_final_prompt",
    "_format_prompt",
    "_format_prompt_no_echo",
    "_format_signature_prompt",
]
