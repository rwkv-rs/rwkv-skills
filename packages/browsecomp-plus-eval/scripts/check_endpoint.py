#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url: str, *, api_key: str, payload: dict[str, Any] | None = None, timeout: float) -> Any:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except HTTPError as exc:
        body = exc.read(1000).decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise SystemExit(f"Cannot reach {url}: {exc.reason}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe an OpenAI-compatible RWKV-vLLM endpoint")
    parser.add_argument("--base-url", required=True, help="Base URL including /v1")
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    models = request_json(f"{base_url}/models", api_key=args.api_key, timeout=args.timeout)
    completion = request_json(
        f"{base_url}/completions",
        api_key=args.api_key,
        timeout=args.timeout,
        payload={"model": args.model, "prompt": "User: reply OK\nAssistant:", "max_tokens": 1, "temperature": 0},
    )
    model_ids = [str(item.get("id")) for item in models.get("data", []) if isinstance(item, dict)]
    choices = completion.get("choices", []) if isinstance(completion, dict) else []
    if args.model not in model_ids:
        raise SystemExit(f"Model {args.model!r} is absent from /models: {model_ids}")
    if not choices:
        raise SystemExit("/completions returned no choices")
    print(json.dumps({"models_ok": True, "completion_ok": True, "model": args.model}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
