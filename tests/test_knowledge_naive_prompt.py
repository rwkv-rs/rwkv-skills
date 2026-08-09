from src.eval.tasks.knowledge.runner import _naive_direct_prompt_template


def test_naive_direct_prompt_closes_empty_think_before_answer_slot() -> None:
    """NoCoT must not leave room for a model-generated reasoning turn."""
    prompt = _naive_direct_prompt_template()
    assert "Assistant: <think></think>" in prompt
    assert prompt.endswith("The answer is")
