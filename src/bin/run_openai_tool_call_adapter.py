from __future__ import annotations

"""Proxy OpenAI chat responses and normalize text JSON calls into tool_calls."""

import argparse
import asyncio
import json
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import uvicorn

from src.eval.function_calling.rwkv_prompt import extract_json_call_value_text
from src.infer.backend import normalize_api_base


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenAI tool-call response format adapter")
    parser.add_argument("--upstream-base-url", required=True, help="Upstream OpenAI-compatible base URL")
    parser.add_argument("--model", action="append", default=None, help="Model id to expose from /v1/models")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=19082, help="Bind port")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Upstream request timeout")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser.parse_args(argv)


def create_app(
    upstream_base_url: str,
    *,
    model_ids: Sequence[str] = (),
    timeout_s: float = 600.0,
) -> FastAPI:
    upstream = normalize_api_base(upstream_base_url)
    exposed_models = tuple(str(item).strip() for item in model_ids if str(item).strip())
    app = FastAPI(title="RWKV OpenAI Tool Call Adapter", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "upstream": upstream, "models": list(exposed_models)}

    @app.get("/v1/models")
    @app.get("/openai/v1/models")
    async def list_models() -> Response:
        if exposed_models:
            payload = {
                "object": "list",
                "data": [{"id": model, "object": "model"} for model in exposed_models],
            }
            return Response(content=json.dumps(payload), media_type="application/json")
        return await asyncio.to_thread(_get_upstream, f"{upstream}/models", timeout_s)

    @app.post("/v1/chat/completions")
    @app.post("/openai/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.body()
        request_payload = _decode_json_object(body, label="request body")
        response = await asyncio.to_thread(
            _post_upstream,
            f"{upstream}/chat/completions",
            body,
            request.headers.get("authorization"),
            request.headers.get("content-type") or "application/json",
            timeout_s,
        )
        if _is_streaming_request(request_payload) or response.status_code >= 400:
            return response
        if "json" not in (response.media_type or ""):
            return response
        response_payload = _decode_json_object(response.body, label="upstream response")
        normalized = normalize_chat_completion_response(response_payload, request_payload=request_payload)
        return Response(
            content=json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            status_code=response.status_code,
            media_type="application/json",
        )

    @app.post("/v1/completions")
    @app.post("/openai/v1/completions")
    async def completions(request: Request) -> Response:
        body = await request.body()
        return await asyncio.to_thread(
            _post_upstream,
            f"{upstream}/completions",
            body,
            request.headers.get("authorization"),
            request.headers.get("content-type") or "application/json",
            timeout_s,
        )

    return app


def normalize_chat_completion_response(
    response_payload: Mapping[str, Any],
    *,
    request_payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(response_payload)
    if not request_payload.get("tools"):
        return normalized
    choices = normalized.get("choices")
    if not isinstance(choices, list):
        return normalized
    new_choices: list[Any] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            new_choices.append(choice)
            continue
        new_choice = dict(choice)
        message = new_choice.get("message")
        if not isinstance(message, Mapping) or message.get("tool_calls"):
            new_choices.append(new_choice)
            continue
        content = message.get("content")
        tool_calls = parse_text_tool_calls(content)
        if not tool_calls:
            new_choices.append(new_choice)
            continue
        new_message = dict(message)
        new_message["content"] = None
        new_message["tool_calls"] = tool_calls
        new_choice["message"] = new_message
        new_choice["finish_reason"] = "tool_calls"
        new_choices.append(new_choice)
    normalized["choices"] = new_choices
    return normalized


def parse_text_tool_calls(content: object) -> list[dict[str, Any]]:
    if not isinstance(content, str) or not content.strip():
        return []
    text = _strip_model_output_wrappers(content)
    try:
        payload = json.loads(extract_json_call_value_text(text))
    except (json.JSONDecodeError, ValueError):
        return []
    return _coerce_tool_calls(payload)


def _coerce_tool_calls(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _valid_tool_calls(payload)
    if not isinstance(payload, Mapping):
        return []
    parsed_type = payload.get("type")
    if parsed_type == "message":
        return []
    if isinstance(payload.get("tool_calls"), list):
        return _valid_tool_calls(payload["tool_calls"])
    if "name" in payload or isinstance(payload.get("function"), Mapping):
        return _valid_tool_calls([payload])
    return []


def _valid_tool_calls(items: Sequence[Any]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        try:
            tool_calls.append(_openai_tool_call(index, item))
        except ValueError:
            continue
    return tool_calls


def _openai_tool_call(index: int, item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError("tool call item must be a JSON object")
    function = item.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name") or item.get("name") or "").strip()
        raw_arguments = function.get("arguments", item.get("arguments", {}))
    else:
        name = str(item.get("name") or item.get("tool_name") or item.get("tool") or "").strip()
        raw_arguments = item.get("arguments", item.get("input", item.get("parameters", {})))
    if not name:
        raise ValueError("tool call name is required")
    return {
        "id": str(item.get("id") or f"call_{index}"),
        "type": "function",
        "function": {
            "name": name,
            "arguments": _arguments_json(raw_arguments),
        },
    }


def _arguments_json(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return json.dumps(value, ensure_ascii=False)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(",", ":"))


def _strip_model_output_wrappers(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("Assistant:"):
        stripped = stripped[len("Assistant:") :].strip()
    if stripped.startswith("<think>"):
        close = stripped.find("</think>")
        if close >= 0:
            stripped = stripped[close + len("</think>") :].strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json", "```js", "```javascript"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must be JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail=f"{label} must be a JSON object")
    return payload


def _is_streaming_request(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("stream"))


def _post_upstream(
    url: str,
    body: bytes,
    authorization: str | None,
    content_type: str,
    timeout_s: float,
) -> Response:
    headers = {"Content-Type": content_type, "Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    req = urllib_request.Request(url, data=body, method="POST", headers=headers)
    return _open_upstream(req, timeout_s=timeout_s)


def _get_upstream(url: str, timeout_s: float) -> Response:
    req = urllib_request.Request(url, method="GET", headers={"Accept": "application/json"})
    return _open_upstream(req, timeout_s=timeout_s)


def _open_upstream(req: urllib_request.Request, *, timeout_s: float) -> Response:
    try:
        with urllib_request.urlopen(req, timeout=max(float(timeout_s), 1.0)) as resp:
            return Response(
                content=resp.read(),
                status_code=int(resp.status),
                media_type=resp.headers.get_content_type() or "application/json",
            )
    except urllib_error.HTTPError as exc:
        return Response(
            content=exc.read(),
            status_code=int(exc.code),
            media_type=exc.headers.get_content_type() if exc.headers else "application/json",
        )
    except urllib_error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"upstream request failed: {exc.reason}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = create_app(
        args.upstream_base_url,
        model_ids=tuple(args.model or ()),
        timeout_s=float(args.timeout_s),
    )
    uvicorn.run(
        app,
        host=str(args.host),
        port=int(args.port),
        log_level=str(args.log_level),
        access_log=False,
    )
    return 0


__all__ = [
    "create_app",
    "main",
    "normalize_chat_completion_response",
    "parse_args",
    "parse_text_tool_calls",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
