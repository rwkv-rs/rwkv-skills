from __future__ import annotations

"""Route OpenAI-compatible infer requests to per-model backend services."""

import argparse
import asyncio
import json
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import uvicorn

from src.infer.backend import normalize_api_base


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible model router for RWKV infer services")
    parser.add_argument(
        "--route",
        action="append",
        required=True,
        metavar="MODEL=BASE_URL",
        help="Route one model to one infer service, e.g. model=http://127.0.0.1:18081",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=19081, help="Bind port")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Backend request timeout")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser.parse_args(argv)


RouteMap = dict[str, tuple[str, ...]]


def parse_routes(raw_routes: Sequence[str]) -> RouteMap:
    routes: dict[str, list[str]] = {}
    for raw in raw_routes:
        model, sep, base_url = str(raw).partition("=")
        model = model.strip()
        base_url = base_url.strip()
        if not sep or not model or not base_url:
            raise ValueError(f"route must be MODEL=BASE_URL, got {raw!r}")
        routes.setdefault(model, []).append(normalize_api_base(base_url))
    return {model: tuple(urls) for model, urls in routes.items()}


def normalize_routes(routes: Mapping[str, str | Sequence[str]]) -> RouteMap:
    normalized: dict[str, list[str]] = {}
    for model, raw_urls in routes.items():
        model = str(model).strip()
        if not model:
            raise ValueError("route model name cannot be empty")
        if isinstance(raw_urls, str):
            urls = (raw_urls,)
        else:
            urls = tuple(str(url) for url in raw_urls)
        if not urls:
            raise ValueError(f"route for model {model!r} cannot be empty")
        normalized[model] = [normalize_api_base(url) for url in urls]
    return {model: tuple(urls) for model, urls in normalized.items()}


def _next_backend_url(model: str, routes: RouteMap, offsets: dict[str, int]) -> str:
    urls = routes.get(model)
    if urls is None:
        available = ", ".join(sorted(routes))
        raise HTTPException(status_code=400, detail=f"unknown model {model!r}; available models: {available}")
    index = offsets.get(model, 0) % len(urls)
    offsets[model] = index + 1
    return urls[index]


def _backend_urls_for_model(model: str, routes: RouteMap) -> tuple[str, ...]:
    urls = routes.get(model)
    if urls is None:
        available = ", ".join(sorted(routes))
        raise HTTPException(status_code=400, detail=f"unknown model {model!r}; available models: {available}")
    return urls


def create_app(routes: Mapping[str, str | Sequence[str]], *, timeout_s: float = 600.0) -> FastAPI:
    route_map = normalize_routes(routes)
    route_offsets = {model: 0 for model in route_map}
    app = FastAPI(title="RWKV Skills Infer Router", version="0.1.0")

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "models": sorted(route_map),
            "route_counts": {model: len(urls) for model, urls in sorted(route_map.items())},
        }

    @app.get("/v1/models")
    @app.get("/openai/v1/models")
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": model, "object": "model"} for model in sorted(route_map)],
        }

    @app.post("/v1/chat/completions")
    @app.post("/openai/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _forward_json_request(
            request,
            routes=route_map,
            route_offsets=route_offsets,
            backend_path="chat/completions",
            timeout_s=timeout_s,
        )

    @app.post("/v1/completions")
    @app.post("/openai/v1/completions")
    async def completions(request: Request) -> Response:
        return await _forward_json_request(
            request,
            routes=route_map,
            route_offsets=route_offsets,
            backend_path="completions",
            timeout_s=timeout_s,
        )

    return app


async def _forward_json_request(
    request: Request,
    *,
    routes: RouteMap,
    route_offsets: dict[str, int],
    backend_path: str,
    timeout_s: float,
) -> Response:
    body = await request.body()
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="request body must be JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")
    model = str(payload.get("model") or "").strip()
    if backend_path == "chat/completions" and isinstance(payload.get("contents"), list):
        urls = _backend_urls_for_model(model, routes)
        if len(urls) > 1:
            return await _forward_contents_batch_request(
                payload,
                urls=urls,
                route_offsets=route_offsets,
                authorization=request.headers.get("authorization"),
                content_type=request.headers.get("content-type") or "application/json",
                timeout_s=timeout_s,
            )
    base_url = _next_backend_url(model, routes, route_offsets)
    target_url = f"{base_url}/{backend_path}"
    authorization = request.headers.get("authorization")
    content_type = request.headers.get("content-type") or "application/json"
    return await asyncio.to_thread(
        _post_bytes,
        target_url,
        body,
        authorization,
        content_type,
        timeout_s,
    )


async def _forward_contents_batch_request(
    payload: dict[str, Any],
    *,
    urls: Sequence[str],
    route_offsets: dict[str, int],
    authorization: str | None,
    content_type: str,
    timeout_s: float,
) -> Response:
    model = str(payload.get("model") or "").strip()
    subrequests = _split_contents_payload(payload, urls=urls, start_offset=route_offsets.get(model, 0))
    route_offsets[model] = route_offsets.get(model, 0) + len(subrequests)
    results = await asyncio.gather(
        *(
            asyncio.to_thread(
                _post_raw,
                f"{base_url}/chat/completions",
                json.dumps(subpayload, ensure_ascii=False).encode("utf-8"),
                authorization,
                content_type,
                timeout_s,
            )
            for base_url, _indices, subpayload in subrequests
        )
    )
    for status_code, raw, media_type in results:
        if int(status_code) >= 400:
            return Response(content=raw, status_code=int(status_code), media_type=media_type)
    merged = _merge_contents_batch_responses(payload, subrequests=subrequests, results=results)
    return Response(
        content=json.dumps(merged, ensure_ascii=False).encode("utf-8"),
        status_code=200,
        media_type="application/json",
    )


def _split_contents_payload(
    payload: dict[str, Any],
    *,
    urls: Sequence[str],
    start_offset: int = 0,
) -> list[tuple[str, tuple[int, ...], dict[str, Any]]]:
    contents = payload.get("contents")
    if not isinstance(contents, list):
        raise HTTPException(status_code=400, detail="contents must be a list")
    if not contents:
        raise HTTPException(status_code=400, detail="contents cannot be empty")
    if not urls:
        raise HTTPException(status_code=502, detail="no backend urls configured")
    worker_count = min(len(contents), len(urls))
    start = int(start_offset) % len(urls)
    base_size, remainder = divmod(len(contents), worker_count)
    subrequests: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    cursor = 0
    for worker_index in range(worker_count):
        size = base_size + (1 if worker_index < remainder else 0)
        indices = tuple(range(cursor, cursor + size))
        subpayload = dict(payload)
        subpayload["contents"] = [contents[index] for index in indices]
        subrequests.append((urls[(start + worker_index) % len(urls)], indices, subpayload))
        cursor += size
    return subrequests


def _merge_contents_batch_responses(
    payload: dict[str, Any],
    *,
    subrequests: Sequence[tuple[str, tuple[int, ...], dict[str, Any]]],
    results: Sequence[tuple[int, bytes, str]],
) -> dict[str, Any]:
    merged_choices: list[dict[str, Any]] = []
    template: dict[str, Any] | None = None
    for (_base_url, indices, _subpayload), (_status_code, raw, _media_type) in zip(subrequests, results, strict=True):
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=502, detail="backend response must be JSON") from exc
        if not isinstance(response, dict):
            raise HTTPException(status_code=502, detail="backend response must be a JSON object")
        if template is None:
            template = dict(response)
        choices = response.get("choices")
        if not isinstance(choices, list):
            raise HTTPException(status_code=502, detail="backend response missing choices")
        for fallback_index, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise HTTPException(status_code=502, detail="backend response choice must be an object")
            local_index = int(choice.get("index", fallback_index))
            if local_index < 0 or local_index >= len(indices):
                raise HTTPException(status_code=502, detail="backend response choice index out of range")
            rewritten = dict(choice)
            rewritten["index"] = indices[local_index]
            merged_choices.append(rewritten)
    if template is None:
        template = {
            "object": "chat.completion",
            "model": str(payload.get("model") or ""),
        }
    template["choices"] = sorted(merged_choices, key=lambda choice: int(choice.get("index", 0)))
    template["model"] = str(payload.get("model") or template.get("model") or "")
    return template


def _post_raw(
    url: str,
    body: bytes,
    authorization: str | None,
    content_type: str,
    timeout_s: float,
) -> tuple[int, bytes, str]:
    headers = {
        "Content-Type": content_type,
        "Accept": "application/json",
    }
    if authorization:
        headers["Authorization"] = authorization
    req = urllib_request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib_request.urlopen(req, timeout=max(float(timeout_s), 1.0)) as resp:
            raw = resp.read()
            media_type = resp.headers.get_content_type() or "application/json"
            return int(resp.status), raw, media_type
    except urllib_error.HTTPError as exc:
        raw = exc.read()
        media_type = exc.headers.get_content_type() if exc.headers else "application/json"
        return int(exc.code), raw, media_type
    except urllib_error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"backend request failed: {exc.reason}") from exc


def _post_bytes(
    url: str,
    body: bytes,
    authorization: str | None,
    content_type: str,
    timeout_s: float,
) -> Response:
    status_code, raw, media_type = _post_raw(url, body, authorization, content_type, timeout_s)
    return Response(content=raw, status_code=status_code, media_type=media_type)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    routes = parse_routes(args.route)
    app = create_app(routes, timeout_s=float(args.timeout_s))
    uvicorn.run(
        app,
        host=str(args.host),
        port=int(args.port),
        log_level=str(args.log_level),
        access_log=False,
    )
    return 0


__all__ = ["create_app", "main", "parse_args", "parse_routes"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
