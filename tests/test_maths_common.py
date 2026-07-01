from __future__ import annotations

from src.eval.tasks.maths.common import build_llm_judge


def test_build_llm_judge_uses_timeout_env(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("JUDGE_TIMEOUT_S", "12.5")

    judge = build_llm_judge(required=True)

    assert judge.config.timeout_s == 12.5
