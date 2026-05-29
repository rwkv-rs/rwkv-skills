from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        return False


@dataclass(slots=True)
class OpenAIModelConfig:
    api_key: str
    model_name: str
    base_url: str | None = None


def load_env_file(path: Path | str = ".env") -> None:
    target = Path(path).expanduser()
    if not target.exists():
        return
    load_dotenv(dotenv_path=target, override=False, encoding="utf-8")


def resolve_required_user_model_config() -> OpenAIModelConfig:
    api_key = _first_env("USER_API_KEY", "API_KEY", "OPENAI_API_KEY")
    model_name = _first_env("USER_MODEL_NAME", "model_name", "MODEL_NAME")
    base_url = _first_env("USER_BASE_URL", "OPENAI_BASE_URL", "API_BASE", "BASE_URL")

    missing: list[str] = []
    if not api_key:
        missing.append("USER_API_KEY (or API_KEY / OPENAI_API_KEY)")
    if not model_name:
        missing.append("USER_MODEL_NAME (or model_name / MODEL_NAME)")
    if missing:
        detail = ", ".join(missing)
        raise ValueError(f"Missing required .env fields for user simulator: {detail}")

    return OpenAIModelConfig(
        api_key=api_key,
        model_name=model_name,
        base_url=normalize_openai_base_url(base_url),
    )


def resolve_judge_model_config(default_model: str | None = None) -> OpenAIModelConfig | None:
    model_name = _first_env("judge_model_name", "JUDGE_MODEL", "LLM_JUDGE_MODEL") or default_model
    if not model_name:
        return None
    api_key = _first_env("JUDGE_API_KEY", "API_KEY", "OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing judge API key: set JUDGE_API_KEY or API_KEY / OPENAI_API_KEY in .env")
    base_url = _first_env("JUDGE_BASE_URL", "OPENAI_BASE_URL", "API_BASE")
    return OpenAIModelConfig(api_key=api_key, model_name=model_name, base_url=normalize_openai_base_url(base_url))


def apply_openai_env(config: OpenAIModelConfig) -> None:
    os.environ["OPENAI_API_KEY"] = config.api_key
    os.environ["API_KEY"] = config.api_key
    base_url = normalize_openai_base_url(config.base_url)
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["API_BASE"] = base_url


def normalize_openai_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().rstrip("/")
    if not text:
        return None
    suffix = "/chat/completions"
    if text.endswith(suffix):
        text = text[: -len(suffix)].rstrip("/")
    return text or None


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        text = value.strip()
        if text:
            return text
    return None


__all__ = [
    "OpenAIModelConfig",
    "load_env_file",
    "resolve_required_user_model_config",
    "resolve_judge_model_config",
    "apply_openai_env",
    "normalize_openai_base_url",
]
