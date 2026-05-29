from __future__ import annotations

import pytest

from src.eval.checkers.llm_checker import LLMCheckerConfig, LLMCheckerFailure, _call_llm_checker


def test_llm_checker_treats_string_response_as_checker_failure() -> None:
    class FakeCompletions:
        def create(self, **_kwargs: object) -> str:
            return "data: [DONE]"

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    config = LLMCheckerConfig(api_key="key", model="judge", max_retries=0)

    with pytest.raises(LLMCheckerFailure):
        _call_llm_checker(FakeClient(), config=config, prompt="check this")
