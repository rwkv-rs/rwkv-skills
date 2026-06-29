from __future__ import annotations

"""Route OpenAI-compatible infer requests to per-model backend services."""

import argparse
import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
import json
from typing import Any, Mapping, Sequence

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
import httpx
import uvicorn


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
    parser.add_argument(
        "--choice-logits-timeout-s",
        type=float,
        default=None,
        help="Backend request timeout for /v1/choice_logits; defaults to --timeout-s",
    )
    parser.add_argument(
        "--forward-max-workers",
        type=int,
        default=256,
        help="Max blocking HTTP forwarding threads; keep above aggregate eval-side request concurrency",
    )
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    return parser.parse_args(argv)


RouteMap = dict[str, tuple[str, ...]]


def normalize_api_root(base_url: str) -> str:
    base = str(base_url or "").strip()
    if not base:
        raise ValueError("infer base URL cannot be empty")
    if "://" not in base:
        base = f"http://{base}"
    base = base.rstrip("/")
    if base.endswith("/v1") or base.endswith("/v2"):
        return base.rsplit("/", 1)[0]
    return base


def normalize_api_base(base_url: str) -> str:
    return f"{normalize_api_root(base_url)}/v1"


def _backend_url_for_path(base_url: str, backend_path: str) -> str:
    if (
        backend_path.startswith("v1/")
        or backend_path.startswith("v2/")
        or backend_path.startswith("high_throughput/")
    ):
        return f"{normalize_api_root(base_url)}/{backend_path}"
    return f"{base_url}/{backend_path}"


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


def create_app(
    routes: Mapping[str, str | Sequence[str]],
    *,
    timeout_s: float = 600.0,
    choice_logits_timeout_s: float | None = None,
    forward_max_workers: int = 256,
) -> FastAPI:
    route_map = normalize_routes(routes)
    route_offsets = {model: 0 for model in route_map}
    request_timeout_s = float(timeout_s)
    choice_timeout_s = (
        request_timeout_s if choice_logits_timeout_s is None else float(choice_logits_timeout_s)
    )
    forward_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(forward_max_workers)),
        thread_name_prefix="infer-router-forward",
    )
    forward_http_client = _build_forward_http_client(
        max_connections=max(1, int(forward_max_workers)),
        timeout_s=max(request_timeout_s, choice_timeout_s),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.forward_executor = forward_executor
        app.state.forward_http_client = forward_http_client
        try:
            yield
        finally:
            forward_http_client.close()
            forward_executor.shutdown(wait=False, cancel_futures=False)

    app = FastAPI(title="RWKV Skills Infer Router", version="0.1.0", lifespan=lifespan)
    app.state.forward_executor = forward_executor
    app.state.forward_http_client = forward_http_client

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "models": sorted(route_map),
            "route_counts": {model: len(urls) for model, urls in sorted(route_map.items())},
        }

    @app.get("/v1/backpressure")
    @app.get("/openai/v1/backpressure")
    async def backpressure(request: Request) -> dict[str, object]:
        return await _collect_backpressure(
            route_map,
            forward_executor=request.app.state.forward_executor,
            http_client=request.app.state.forward_http_client,
            authorization=request.headers.get("authorization"),
            timeout_s=min(float(timeout_s), 5.0),
        )

    @app.get("/v1/batch-metrics")
    @app.get("/openai/v1/batch-metrics")
    async def batch_metrics(request: Request) -> dict[str, object]:
        return await _collect_batch_metrics(
            route_map,
            forward_executor=request.app.state.forward_executor,
            http_client=request.app.state.forward_http_client,
            authorization=request.headers.get("authorization"),
            timeout_s=min(float(timeout_s), 5.0),
        )

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

    @app.post("/v2/chat/completions")
    @app.post("/openai/v2/chat/completions")
    async def chat_completions_v2(request: Request) -> Response:
        return await _forward_json_request(
            request,
            routes=route_map,
            route_offsets=route_offsets,
            backend_path="v2/chat/completions",
            timeout_s=timeout_s,
        )

    @app.post("/high_throughput/chat/completions")
    async def high_throughput_chat_completions(request: Request) -> Response:
        return await _forward_json_request(
            request,
            routes=route_map,
            route_offsets=route_offsets,
            backend_path="high_throughput/chat/completions",
            timeout_s=timeout_s,
        )

    @app.post("/v1/choice_logits")
    @app.post("/openai/v1/choice_logits")
    async def choice_logits(request: Request) -> Response:
        return await _forward_json_request(
            request,
            routes=route_map,
            route_offsets=route_offsets,
            backend_path="choice_logits",
            timeout_s=choice_timeout_s,
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
    should_split_contents_batch = (
        isinstance(payload.get("contents"), list)
        and (backend_path.endswith("chat/completions") or backend_path == "choice_logits")
    )
    if should_split_contents_batch:
        urls = _backend_urls_for_model(model, routes)
        if len(urls) > 1:
            return await _forward_contents_batch_request(
                payload,
                urls=urls,
                backend_path=backend_path,
                route_offsets=route_offsets,
                forward_executor=request.app.state.forward_executor,
                http_client=request.app.state.forward_http_client,
                authorization=request.headers.get("authorization"),
                content_type=request.headers.get("content-type") or "application/json",
                timeout_s=timeout_s,
            )
    base_url = _next_backend_url(model, routes, route_offsets)
    target_url = _backend_url_for_path(base_url, backend_path)
    authorization = request.headers.get("authorization")
    content_type = request.headers.get("content-type") or "application/json"
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        request.app.state.forward_executor,
        _post_bytes,
        target_url,
        body,
        authorization,
        content_type,
        timeout_s,
        request.app.state.forward_http_client,
    )


async def _forward_contents_batch_request(
    payload: dict[str, Any],
    *,
    urls: Sequence[str],
    backend_path: str,
    route_offsets: dict[str, int],
    forward_executor: concurrent.futures.Executor,
    http_client: httpx.Client,
    authorization: str | None,
    content_type: str,
    timeout_s: float,
) -> Response:
    model = str(payload.get("model") or "").strip()
    subrequests = _split_contents_payload(payload, urls=urls, start_offset=route_offsets.get(model, 0))
    route_offsets[model] = route_offsets.get(model, 0) + len(subrequests)
    loop = asyncio.get_running_loop()
    results = await asyncio.gather(
        *(
            loop.run_in_executor(
                forward_executor,
                _post_raw,
                _backend_url_for_path(base_url, backend_path),
                json.dumps(subpayload, ensure_ascii=False).encode("utf-8"),
                authorization,
                content_type,
                timeout_s,
                http_client,
            )
            for base_url, _indices, subpayload in subrequests
        )
    )
    for status_code, raw, media_type in results:
        if int(status_code) >= 400:
            return Response(content=raw, status_code=int(status_code), media_type=media_type)
    if backend_path == "choice_logits":
        merged = _merge_choice_logits_batch_responses(payload, subrequests=subrequests, results=results)
    else:
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


def _merge_choice_logits_batch_responses(
    payload: dict[str, Any],
    *,
    subrequests: Sequence[tuple[str, tuple[int, ...], dict[str, Any]]],
    results: Sequence[tuple[int, bytes, str]],
) -> dict[str, Any]:
    merged_results: list[dict[str, Any]] = []
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
        response_results = response.get("results")
        if isinstance(response_results, list):
            local_results = response_results
        elif "choice_logits" in response:
            local_results = [_single_choice_logits_response_to_result(response)]
        else:
            raise HTTPException(status_code=502, detail="backend choice_logits response missing results")
        for fallback_index, result in enumerate(local_results):
            if not isinstance(result, dict):
                raise HTTPException(status_code=502, detail="backend choice_logits result must be an object")
            local_index = int(result.get("index", fallback_index))
            if local_index < 0 or local_index >= len(indices):
                raise HTTPException(status_code=502, detail="backend choice_logits result index out of range")
            rewritten = dict(result)
            rewritten["index"] = indices[local_index]
            merged_results.append(rewritten)
    response_model = str(payload.get("model") or (template or {}).get("model") or "")
    merged: dict[str, Any] = {
        "id": str((template or {}).get("id") or "router-choice-logits"),
        "object": "choice.logits.batch",
        "model": response_model,
        "results": sorted(merged_results, key=lambda result: int(result.get("index", 0))),
    }
    if template is not None and template.get("created") is not None:
        merged["created"] = template["created"]
    return merged


def _single_choice_logits_response_to_result(response: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in response.items()
        if key not in {"id", "object", "created", "model"}
    }
    result.setdefault("index", 0)
    return result


async def _collect_backpressure(
    routes: RouteMap,
    *,
    forward_executor: concurrent.futures.Executor,
    http_client: httpx.Client,
    authorization: str | None,
    timeout_s: float,
) -> dict[str, object]:
    results = await _collect_backend_batch_metrics(
        routes,
        forward_executor=forward_executor,
        http_client=http_client,
        authorization=authorization,
        timeout_s=timeout_s,
    )
    return _build_backpressure_payload(routes, results)


async def _collect_batch_metrics(
    routes: RouteMap,
    *,
    forward_executor: concurrent.futures.Executor,
    http_client: httpx.Client,
    authorization: str | None,
    timeout_s: float,
) -> dict[str, object]:
    results = await _collect_backend_batch_metrics(
        routes,
        forward_executor=forward_executor,
        http_client=http_client,
        authorization=authorization,
        timeout_s=timeout_s,
    )
    return _build_batch_metrics_payload(routes, results)


async def _collect_backend_batch_metrics(
    routes: RouteMap,
    *,
    forward_executor: concurrent.futures.Executor,
    http_client: httpx.Client,
    authorization: str | None,
    timeout_s: float,
) -> dict[str, list[tuple[str, int, object]]]:
    loop = asyncio.get_running_loop()
    pending: list[tuple[str, str, asyncio.Future[tuple[int, object]]]] = []
    for model, urls in sorted(routes.items()):
        for base_url in urls:
            future = loop.run_in_executor(
                forward_executor,
                _get_json,
                f"{base_url}/batch-metrics",
                authorization,
                timeout_s,
                http_client,
            )
            pending.append((model, base_url, future))
    results: dict[str, list[tuple[str, int, object]]] = {model: [] for model in routes}
    for model, base_url, future in pending:
        status_code, payload = await future
        results.setdefault(model, []).append((base_url, status_code, payload))
    return results


def _build_backpressure_payload(
    routes: RouteMap,
    backend_results: Mapping[str, Sequence[tuple[str, int, object]]],
) -> dict[str, object]:
    models: dict[str, object] = {}
    for model, urls in sorted(routes.items()):
        backend_entries = [
            _backend_backpressure_entry(base_url, status_code, payload)
            for base_url, status_code, payload in backend_results.get(model, ())
        ]
        known_urls = {str(entry.get("base_url")) for entry in backend_entries if isinstance(entry, dict)}
        for base_url in urls:
            if base_url not in known_urls:
                backend_entries.append(
                    {
                        "base_url": base_url,
                        "status": "error",
                        "error": "missing backend metrics result",
                    }
                )
        aggregate = _aggregate_backpressure(backend_entries)
        models[model] = {
            "model": model,
            "status": aggregate["status"],
            "route_count": len(urls),
            "aggregate": aggregate,
            "backends": backend_entries,
        }
    overall_status = "ok"
    if any(isinstance(entry, dict) and entry.get("status") != "ok" for entry in models.values()):
        overall_status = "degraded"
    return {"status": overall_status, "models": models}


def _build_batch_metrics_payload(
    routes: RouteMap,
    backend_results: Mapping[str, Sequence[tuple[str, int, object]]],
) -> dict[str, object]:
    models: dict[str, object] = {}
    for model, urls in sorted(routes.items()):
        backend_entries = [
            _backend_batch_metrics_entry(base_url, status_code, payload)
            for base_url, status_code, payload in backend_results.get(model, ())
        ]
        known_urls = {str(entry.get("base_url")) for entry in backend_entries if isinstance(entry, dict)}
        for base_url in urls:
            if base_url not in known_urls:
                backend_entries.append(
                    {
                        "base_url": base_url,
                        "status": "error",
                        "error": "missing backend metrics result",
                    }
                )
        aggregate = _aggregate_live_metrics(backend_entries)
        models[model] = {
            "model": model,
            "status": aggregate["status"],
            "route_count": len(urls),
            "aggregate": aggregate,
            "backends": backend_entries,
        }
    overall_status = "ok"
    if any(isinstance(entry, dict) and entry.get("status") != "ok" for entry in models.values()):
        overall_status = "degraded"
    return {"status": overall_status, "models": models}


def _backend_backpressure_entry(base_url: str, status_code: int, payload: object) -> dict[str, object]:
    if int(status_code) == 404:
        return _backend_missing_metrics_fallback_entry(base_url, status_code, payload)
    if not isinstance(payload, dict):
        return {
            "base_url": base_url,
            "status": "error",
            "http_status": int(status_code),
            "error": str(payload),
        }
    pending = payload.get("pending") if isinstance(payload.get("pending"), dict) else {}
    totals = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
    status = "ok" if int(status_code) and int(status_code) < 400 else "error"
    entry: dict[str, object] = {
        "base_url": base_url,
        "status": status,
        "http_status": int(status_code),
        "model_name": payload.get("model_name"),
        "max_batch_size": _optional_int(payload.get("max_batch_size")),
        "batch_collect_ms": _optional_int(payload.get("batch_collect_ms")),
        "pending_queue": _int_or_zero(pending.get("pending_queue")),
        "service_queue": _int_or_zero(pending.get("service_queue")),
        "engine_inbox": _int_or_zero(pending.get("engine_inbox")),
        "active_records": _int_or_zero(pending.get("active_records")),
        "scheduler_waiting": _int_or_zero(pending.get("scheduler_waiting")),
        "scheduler_running": _int_or_zero(pending.get("scheduler_running")),
        "prefill_reserved_bsz": _int_or_zero(pending.get("prefill_reserved_bsz")),
        "total_batches": _int_or_zero(totals.get("total_batches")),
        "total_requests": _int_or_zero(totals.get("total_requests")),
        "completed_requests": _int_or_zero(totals.get("completed_requests")),
        "error_requests": _int_or_zero(totals.get("error_requests")),
        "failed_batches": _int_or_zero(totals.get("failed_batches")),
        "last_total_tok_s": _optional_float(totals.get("last_total_tok_s")),
        "last_output_tok_s": _optional_float(totals.get("last_output_tok_s")),
        "avg_batch_size": _optional_float(totals.get("avg_batch_size")),
    }
    if status != "ok":
        entry["error"] = payload.get("error") or payload.get("detail") or "backend metrics request failed"
    return entry


def _backend_missing_metrics_fallback_entry(base_url: str, status_code: int, payload: object) -> dict[str, object]:
    return {
        "base_url": base_url,
        "status": "ok",
        "http_status": int(status_code),
        "model_name": None,
        "max_batch_size": 512,
        "batch_collect_ms": 0,
        "pending_queue": 0,
        "service_queue": 0,
        "engine_inbox": 0,
        "active_records": 0,
        "scheduler_waiting": 0,
        "scheduler_running": 0,
        "prefill_reserved_bsz": 0,
        "total_batches": 0,
        "total_requests": 0,
        "completed_requests": 0,
        "error_requests": 0,
        "failed_batches": 0,
        "last_total_tok_s": None,
        "last_output_tok_s": None,
        "avg_batch_size": None,
        "metrics_fallback": "backend_missing_batch_metrics",
        "backend_live": {"protocol": "lightning-high-throughput", "metrics_fallback": True},
        "recent_batches": [],
        "totals": {},
        "pending": {},
        "error": str(payload.get("error") if isinstance(payload, dict) else payload),
    }


def _backend_batch_metrics_entry(base_url: str, status_code: int, payload: object) -> dict[str, object]:
    entry = _backend_backpressure_entry(base_url, status_code, payload)
    if isinstance(payload, dict):
        backend_live = payload.get("backend_live")
        recent_batches = payload.get("recent_batches")
        entry["backend_live"] = backend_live if isinstance(backend_live, dict) else {}
        entry["recent_batches"] = recent_batches if isinstance(recent_batches, list) else []
        entry["totals"] = payload.get("totals") if isinstance(payload.get("totals"), dict) else {}
        entry["pending"] = payload.get("pending") if isinstance(payload.get("pending"), dict) else {}
    return entry


def _aggregate_backpressure(backends: Sequence[Mapping[str, object]]) -> dict[str, object]:
    aggregate = _aggregate_live_metrics(backends)
    return {
        "status": aggregate["status"],
        "route_count": aggregate["route_count"],
        "ok_route_count": aggregate["ok_route_count"],
        "pending_queue": aggregate["pending_queue"],
        "service_queue": aggregate["service_queue"],
        "engine_inbox": aggregate["engine_inbox"],
        "active_records": aggregate["active_records"],
        "scheduler_waiting": aggregate["scheduler_waiting"],
        "scheduler_running": aggregate["scheduler_running"],
        "prefill_reserved_bsz": aggregate["prefill_reserved_bsz"],
        "max_batch_size": aggregate["max_batch_size"],
        "failed_batches": aggregate["failed_batches"],
        "total_batches": aggregate["total_batches"],
        "total_requests": aggregate["total_requests"],
        "completed_requests": aggregate["completed_requests"],
        "error_requests": aggregate["error_requests"],
        "last_total_tok_s": aggregate["last_total_tok_s"],
    }


def _aggregate_live_metrics(backends: Sequence[Mapping[str, object]]) -> dict[str, object]:
    ok_backends = [entry for entry in backends if entry.get("status") == "ok"]
    route_count = len(backends)
    ok_count = len(ok_backends)
    if ok_count == route_count:
        status = "ok"
    elif ok_count > 0:
        status = "degraded"
    else:
        status = "error"
    max_batch_values = [
        int(value)
        for entry in ok_backends
        if (value := _optional_int(entry.get("max_batch_size"))) is not None
    ]
    last_total_tok_s_values = [
        float(value)
        for entry in ok_backends
        if (value := _optional_float(entry.get("last_total_tok_s"))) is not None
    ]
    return {
        "status": status,
        "route_count": route_count,
        "ok_route_count": ok_count,
        "pending_queue": sum(_int_or_zero(entry.get("pending_queue")) for entry in ok_backends),
        "service_queue": sum(_int_or_zero(entry.get("service_queue")) for entry in ok_backends),
        "engine_inbox": sum(_int_or_zero(entry.get("engine_inbox")) for entry in ok_backends),
        "active_records": sum(_int_or_zero(entry.get("active_records")) for entry in ok_backends),
        "scheduler_waiting": sum(_int_or_zero(entry.get("scheduler_waiting")) for entry in ok_backends),
        "scheduler_running": sum(_int_or_zero(entry.get("scheduler_running")) for entry in ok_backends),
        "prefill_reserved_bsz": sum(_int_or_zero(entry.get("prefill_reserved_bsz")) for entry in ok_backends),
        "max_batch_size": sum(max_batch_values) if max_batch_values else None,
        "failed_batches": sum(_int_or_zero(entry.get("failed_batches")) for entry in ok_backends),
        "total_batches": sum(_int_or_zero(entry.get("total_batches")) for entry in ok_backends),
        "total_requests": sum(_int_or_zero(entry.get("total_requests")) for entry in ok_backends),
        "completed_requests": sum(_int_or_zero(entry.get("completed_requests")) for entry in ok_backends),
        "error_requests": sum(_int_or_zero(entry.get("error_requests")) for entry in ok_backends),
        "last_total_tok_s": sum(last_total_tok_s_values) if last_total_tok_s_values else None,
    }


def _get_json(
    url: str,
    authorization: str | None,
    timeout_s: float,
    http_client: httpx.Client,
) -> tuple[int, object]:
    headers = {"Accept": "application/json"}
    if authorization:
        headers["Authorization"] = authorization
    try:
        response = http_client.get(url, headers=headers, timeout=max(float(timeout_s), 0.1))
        return int(response.status_code), _decode_json_or_text(response.content)
    except httpx.RequestError as exc:
        return 0, {"error": str(exc)}


def _decode_json_or_text(raw: bytes) -> object:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", errors="replace")


def _post_raw(
    url: str,
    body: bytes,
    authorization: str | None,
    content_type: str,
    timeout_s: float,
    http_client: httpx.Client,
) -> tuple[int, bytes, str]:
    headers = {
        "Content-Type": content_type,
        "Accept": "application/json",
    }
    if authorization:
        headers["Authorization"] = authorization
    try:
        response = http_client.post(
            url,
            content=body,
            headers=headers,
            timeout=max(float(timeout_s), 1.0),
        )
        return int(response.status_code), response.content, _response_media_type(response)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"backend request failed: {exc}") from exc


def _post_bytes(
    url: str,
    body: bytes,
    authorization: str | None,
    content_type: str,
    timeout_s: float,
    http_client: httpx.Client,
) -> Response:
    status_code, raw, media_type = _post_raw(url, body, authorization, content_type, timeout_s, http_client)
    return Response(content=raw, status_code=status_code, media_type=media_type)


def _build_forward_http_client(*, max_connections: int, timeout_s: float) -> httpx.Client:
    timeout = max(float(timeout_s), 1.0)
    connections = max(1, int(max_connections))
    return httpx.Client(
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=connections,
            max_keepalive_connections=connections,
            keepalive_expiry=60.0,
        ),
        timeout=httpx.Timeout(timeout),
    )


def _response_media_type(response: httpx.Response) -> str:
    content_type = str(response.headers.get("content-type") or "application/json")
    media_type = content_type.split(";", 1)[0].strip()
    return media_type or "application/json"


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: object) -> int:
    parsed = _optional_int(value)
    return 0 if parsed is None else parsed


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    routes = parse_routes(args.route)
    app = create_app(
        routes,
        timeout_s=float(args.timeout_s),
        choice_logits_timeout_s=args.choice_logits_timeout_s,
        forward_max_workers=int(args.forward_max_workers),
    )
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
