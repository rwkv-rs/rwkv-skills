from __future__ import annotations

import pytest

from src.eval.env_config import resolve_judge_model_config, resolve_required_user_model_config


_ENV_KEYS = (
    "USER_API_KEY",
    "USER_MODEL_NAME",
    "USER_BASE_URL",
    "API_KEY",
    "OPENAI_API_KEY",
    "model_name",
    "MODEL_NAME",
    "OPENAI_BASE_URL",
    "API_BASE",
    "BASE_URL",
    "JUDGE_API_KEY",
    "JUDGE_MODEL",
    "judge_model_name",
    "LLM_JUDGE_MODEL",
    "JUDGE_BASE_URL",
)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_user_model_config_prefers_user_specific_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("API_KEY", "shared-key")
    monkeypatch.setenv("model_name", "gpt-5.4")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://shared.example/v1")
    monkeypatch.setenv("USER_API_KEY", "user-key")
    monkeypatch.setenv("USER_MODEL_NAME", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("USER_BASE_URL", "https://user.example/v1/chat/completions")

    cfg = resolve_required_user_model_config()

    assert cfg.api_key == "user-key"
    assert cfg.model_name == "gpt-4.1-2025-04-14"
    assert cfg.base_url == "https://user.example/v1"


def test_user_model_config_falls_back_to_shared_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("API_KEY", "shared-key")
    monkeypatch.setenv("MODEL_NAME", "gpt-5.4")
    monkeypatch.setenv("API_BASE", "https://shared.example/v1")

    cfg = resolve_required_user_model_config()

    assert cfg.api_key == "shared-key"
    assert cfg.model_name == "gpt-5.4"
    assert cfg.base_url == "https://shared.example/v1"


def test_judge_model_config_uses_judge_specific_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("API_KEY", "shared-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://shared.example/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4.1-2025-04-14")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example/v1/chat/completions")

    cfg = resolve_judge_model_config(default_model="gpt-5.4")

    assert cfg is not None
    assert cfg.api_key == "judge-key"
    assert cfg.model_name == "gpt-4.1-2025-04-14"
    assert cfg.base_url == "https://judge.example/v1"
