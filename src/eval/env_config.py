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


JUDGE_API_KEY_ENV = "JUDGE_API_KEY"
JUDGE_MODEL_ENV = "JUDGE_MODEL"
JUDGE_BASE_URL_ENV = "JUDGE_BASE_URL"
JUDGE_MAX_WORKERS_ENV = "JUDGE_MAX_WORKERS"
JUDGE_MAX_TOKENS_ENV = "JUDGE_MAX_TOKENS"
JUDGE_TIMEOUT_S_ENV = "JUDGE_TIMEOUT_S"


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


def resolve_judge_model_config(
    default_model: str | None = None,
    *,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    default_api_key: str | None = None,
    default_base_url: str | None = None,
) -> OpenAIModelConfig | None:
    explicit_model_name = _first_value(model_name, _first_env(JUDGE_MODEL_ENV))
    resolved_model_name = _first_value(explicit_model_name, default_model)
    if not resolved_model_name:
        return None
    using_default_model = explicit_model_name is None
    api_key = _first_value(
        api_key,
        _first_env(JUDGE_API_KEY_ENV),
        default_api_key if using_default_model else None,
    )
    if not api_key:
        raise ValueError("Missing judge API key: set JUDGE_API_KEY in .env or pass --judge-api-key")
    base_url = _first_value(
        base_url,
        _first_env(JUDGE_BASE_URL_ENV),
        default_base_url if using_default_model else None,
    )
    return OpenAIModelConfig(
        api_key=api_key,
        model_name=resolved_model_name,
        base_url=normalize_openai_base_url(base_url),
    )


def resolve_judge_max_workers(value: int | None = None, *, default: int = 16) -> int:
    return _positive_int_value(value, field_name="judge max workers") or _positive_int_env(
        JUDGE_MAX_WORKERS_ENV,
        default=default,
    )


def resolve_judge_max_tokens(value: int | None = None) -> int | None:
    return _positive_int_value(value, field_name="judge max tokens") or _positive_int_env(
        JUDGE_MAX_TOKENS_ENV,
        default=None,
    )


def resolve_judge_timeout_s(value: float | None = None, *, default: float = 60.0) -> float:
    return _positive_float_value(value, field_name="judge timeout") or _positive_float_env(
        JUDGE_TIMEOUT_S_ENV,
        default=default,
    )


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


def _first_value(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = value.strip()
        if text:
            return text
    return None


def _positive_int_env(name: str, *, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = _positive_int_value(raw, field_name=name)
    if value is None:
        return default
    return value


def _positive_int_value(value: int | str | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer: {value!r}")
    return parsed


def _positive_float_env(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = _positive_float_value(raw, field_name=name)
    if value is None:
        return default
    return value


def _positive_float_value(value: float | str | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive number: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive number: {value!r}")
    return parsed


__all__ = [
    "OpenAIModelConfig",
    "JUDGE_API_KEY_ENV",
    "JUDGE_BASE_URL_ENV",
    "JUDGE_MAX_TOKENS_ENV",
    "JUDGE_TIMEOUT_S_ENV",
    "JUDGE_MAX_WORKERS_ENV",
    "JUDGE_MODEL_ENV",
    "load_env_file",
    "resolve_required_user_model_config",
    "resolve_judge_model_config",
    "resolve_judge_max_tokens",
    "resolve_judge_timeout_s",
    "resolve_judge_max_workers",
    "apply_openai_env",
    "normalize_openai_base_url",
]
