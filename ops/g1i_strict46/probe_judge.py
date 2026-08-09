from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from src.eval.env_config import (
    load_env_file,
    resolve_judge_max_tokens,
    resolve_judge_max_workers,
    resolve_judge_model_config,
    resolve_judge_timeout_s,
)
from src.eval.metrics.free_response import LLMJudge, LLMJudgeConfig


def _safe_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.hostname:
        return "configured"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _relay_base_url() -> str | None:
    value = os.environ.get("RWKV_JUDGE_RELAY_BASE_URL", "").strip()
    if value:
        return value
    relay_file = Path(
        os.environ.get("RWKV_JUDGE_RELAY_BASE_URL_FILE", ".judge-relay-base-url")
    )
    try:
        return relay_file.read_text(encoding="utf-8").strip() or None
    except FileNotFoundError:
        return None


def _sanitize(value: object, *, secret: str) -> object:
    if isinstance(value, str):
        return value.replace(secret, "<redacted>") if secret else value
    if isinstance(value, list):
        return [_sanitize(item, secret=secret) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item, secret=secret) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only connectivity and response-format probe for the configured judge."
    )
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    load_env_file(args.env_file)
    resolved = resolve_judge_model_config()
    if resolved is None:
        print(json.dumps({"ok": False, "error": "judge model is not configured"}))
        return 2

    relay = _relay_base_url()
    config = LLMJudgeConfig(
        api_key=resolved.api_key,
        model=resolved.model_name,
        base_url=resolved.base_url,
        timeout_s=resolve_judge_timeout_s(default=60.0),
        max_workers=min(2, resolve_judge_max_workers(default=2)),
        max_completion_tokens=resolve_judge_max_tokens(),
        max_retries=1,
        recovery_rounds=0,
    )
    judge = LLMJudge(config)
    results = judge.judge(
        [
            ("What is 2 + 2?", "4", "4"),
            ("What is 2 + 2?", "4", "5"),
        ]
    )
    stats = judge.last_run_stats.as_dict() if judge.last_run_stats else {}
    payload = {
        "ok": results == [True, False] and stats.get("parsed_count") == 2,
        "model": resolved.model_name,
        "configured_endpoint": _safe_endpoint(resolved.base_url),
        "relay_endpoint": _safe_endpoint(relay),
        "effective_endpoint": _safe_endpoint(relay or resolved.base_url),
        "api_key_present": bool(resolved.api_key),
        "expected": [True, False],
        "actual": results,
        "stats": _sanitize(stats, secret=resolved.api_key),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
