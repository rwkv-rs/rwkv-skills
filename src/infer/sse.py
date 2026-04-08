from __future__ import annotations

"""Helpers for encoding Server-Sent Events payloads."""

from collections.abc import Iterable, Iterator

from pydantic import BaseModel


def iter_sse_payloads(payloads: Iterable[BaseModel | str]) -> Iterator[bytes]:
    for payload in payloads:
        if isinstance(payload, str):
            data = payload
        else:
            data = payload.model_dump_json(by_alias=True, exclude_none=True)
        yield f"data: {data}\n\n".encode("utf-8")


__all__ = ["iter_sse_payloads"]
