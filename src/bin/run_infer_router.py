from __future__ import annotations

"""Route OpenAI-compatible infer requests to per-model backend services."""

import argparse
import asyncio
import json
from typing import Mapping, Sequence
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


def _post_bytes(
    url: str,
    body: bytes,
    authorization: str | None,
    content_type: str,
    timeout_s: float,
) -> Response:
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
            return Response(content=raw, status_code=int(resp.status), media_type=media_type)
    except urllib_error.HTTPError as exc:
        raw = exc.read()
        media_type = exc.headers.get_content_type() if exc.headers else "application/json"
        return Response(content=raw, status_code=int(exc.code), media_type=media_type)
    except urllib_error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"backend request failed: {exc.reason}") from exc


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
