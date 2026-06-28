from __future__ import annotations

import asyncio
import http.client
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Target:
    host: str
    port: int
    strip_prefix: str = ""
    add_prefix: str = ""


OLD_TARGET = Target(
    host=os.environ.get("RWKV_OLD_FRONTEND_HOST", "127.0.0.1"),
    port=int(os.environ.get("RWKV_OLD_FRONTEND_PORT", "7861")),
)
NEW_TARGET = Target(
    host=os.environ.get("RWKV_NEW_FRONTEND_HOST", "127.0.0.1"),
    port=int(os.environ.get("RWKV_NEW_FRONTEND_PORT", "7862")),
    strip_prefix=os.environ.get("RWKV_NEW_FRONTEND_PREFIX", "/new-eval").rstrip("/"),
    add_prefix=os.environ.get("RWKV_NEW_FRONTEND_PREFIX", "/new-eval").rstrip("/"),
)

HOP_BY_HOP = {
    b"connection",
    b"keep-alive",
    b"proxy-authenticate",
    b"proxy-authorization",
    b"te",
    b"trailers",
    b"transfer-encoding",
    b"upgrade",
}


def _target_for(path: str) -> tuple[Target, str]:
    prefix = NEW_TARGET.strip_prefix
    if path == prefix or path.startswith(f"{prefix}/"):
        stripped = path[len(prefix) :] or "/"
        return NEW_TARGET, stripped
    return OLD_TARGET, path or "/"


def _headers_for_upstream(headers: Iterable[tuple[bytes, bytes]], target: Target) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for raw_name, raw_value in headers:
        lower = raw_name.lower()
        if lower in HOP_BY_HOP or lower in {b"host", b"content-length"}:
            continue
        resolved[raw_name.decode("latin1")] = raw_value.decode("latin1")
    resolved["Host"] = f"{target.host}:{target.port}"
    return resolved


def _headers_for_client(headers: Iterable[tuple[str, str]], target: Target) -> list[tuple[bytes, bytes]]:
    resolved: list[tuple[bytes, bytes]] = []
    for name, value in headers:
        lower = name.lower().encode("latin1")
        if lower in HOP_BY_HOP:
            continue
        if name.lower() == "location" and target.add_prefix and value.startswith("/") and not value.startswith("//"):
            value = f"{target.add_prefix}{value}"
        resolved.append((name.encode("latin1"), value.encode("latin1")))
    return resolved


def _proxy_request(
    *,
    target: Target,
    method: str,
    path: str,
    query: bytes,
    headers: Iterable[tuple[bytes, bytes]],
    body: bytes,
) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
    request_path = path
    if query:
        request_path = f"{request_path}?{query.decode('latin1')}"
    conn = http.client.HTTPConnection(target.host, target.port, timeout=60)
    try:
        conn.request(
            method,
            request_path,
            body=body if body else None,
            headers=_headers_for_upstream(headers, target),
        )
        response = conn.getresponse()
        data = response.read()
        return response.status, _headers_for_client(response.getheaders(), target), data
    finally:
        conn.close()


async def app(scope, receive, send) -> None:  # type: ignore[no-untyped-def]
    if scope["type"] != "http":
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})
        return

    body = bytearray()
    while True:
        event = await receive()
        if event["type"] != "http.request":
            continue
        body.extend(event.get("body", b""))
        if not event.get("more_body", False):
            break

    target, upstream_path = _target_for(scope.get("path") or "/")
    try:
        status, response_headers, response_body = await asyncio.to_thread(
            _proxy_request,
            target=target,
            method=scope["method"],
            path=upstream_path,
            query=scope.get("query_string", b""),
            headers=scope.get("headers", []),
            body=bytes(body),
        )
    except Exception as exc:  # noqa: BLE001
        status = 502
        response_headers = [(b"content-type", b"text/plain; charset=utf-8")]
        response_body = f"frontend gateway upstream error: {type(exc).__name__}: {exc}".encode()

    await send({"type": "http.response.start", "status": status, "headers": response_headers})
    await send({"type": "http.response.body", "body": response_body})
