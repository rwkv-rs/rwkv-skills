from __future__ import annotations

"""FastAPI app for the standalone RWKV infer service."""

import asyncio
from contextlib import asynccontextmanager
from queue import Empty

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from .api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)
from .openai_service import (
    build_chat_completion_response,
    build_chat_completion_stream_finish_chunks,
    build_chat_completion_stream_role_chunk,
    build_chat_completion_stream_token_chunks,
    build_completion_stream_chunk,
    build_completion_stream_finish_chunk,
    prepare_chat_completion_request,
    new_chat_completion_stream_state,
    new_completion_stream_state,
    StreamResponseMetadata,
)
from .service import InferenceService
from .sse import encode_sse_comment, iter_sse_payloads


SSE_KEEPALIVE_INTERVAL_S = 10.0


def create_app(service: InferenceService, *, api_key: str | None = None) -> FastAPI:
    expected_api_key = str(api_key or "").strip()

    def _authorize(authorization: str | None = Header(default=None)) -> None:
        if not expected_api_key:
            return
        expected = f"Bearer {expected_api_key}"
        if authorization != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="RWKV Skills Infer", version="0.1.0", lifespan=lifespan)

    async def _encode_sse_stream(payload_stream) -> object:
        async for payload in payload_stream:
            if isinstance(payload, bytes):
                yield payload
                continue
            for encoded in iter_sse_payloads([payload]):
                yield encoded
        for encoded in iter_sse_payloads(["[DONE]"]):
            yield encoded

    async def _next_stream_item(token_queue):
        try:
            return await asyncio.to_thread(token_queue.get, True, SSE_KEEPALIVE_INTERVAL_S)
        except Empty:
            return encode_sse_comment()

    def _streaming_response(payload_stream) -> StreamingResponse:
        return StreamingResponse(
            _encode_sse_stream(payload_stream),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "model": service.model_name,
            "max_batch_size": service.max_batch_size,
            "batch_collect_ms": service.batch_collect_ms,
        }

    @app.get("/v1/models", dependencies=[Depends(_authorize)])
    async def list_models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": service.model_name, "object": "model"}],
        }

    @app.post("/openai/v1/completions", dependencies=[Depends(_authorize)], response_model=CompletionResponse)
    @app.post("/v1/completions", dependencies=[Depends(_authorize)], response_model=CompletionResponse)
    async def completions(request: CompletionRequest) -> CompletionResponse | StreamingResponse:
        if request.stream and request.is_choice_scoring_request():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="streaming is not supported for choice-scoring requests",
            )
        if request.model != service.model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown model {request.model!r}; available model is {service.model_name!r}",
            )
        if request.stream:
            handle = service.submit_streaming_completion(request)
            metadata = StreamResponseMetadata(
                id=handle.response_id,
                created=handle.created,
                model=service.model_name,
            )

            async def _iter_completion_stream():
                state = new_completion_stream_state()
                while True:
                    item = await _next_stream_item(handle.token_queue)
                    if isinstance(item, bytes):
                        yield item
                        continue
                    if item is None:
                        break
                    yield build_completion_stream_chunk(
                        metadata,
                        state,
                        item,
                        requested_top_logprobs=max(int(request.logprobs or 0), 0),
                    )
                response = await asyncio.wrap_future(handle.future)
                finish_reason = response.choices[0].finish_reason if response.choices else None
                yield build_completion_stream_finish_chunk(metadata, finish_reason=finish_reason)

            return _streaming_response(_iter_completion_stream())
        future = service.submit_completion(request)
        try:
            return await asyncio.wrap_future(future)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised through integration
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @app.post(
        "/openai/v1/chat/completions",
        dependencies=[Depends(_authorize)],
        response_model=ChatCompletionResponse,
    )
    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(_authorize)],
        response_model=ChatCompletionResponse,
    )
    async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse | StreamingResponse:
        try:
            prepared = prepare_chat_completion_request(request)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if prepared.completion_request.stream and prepared.completion_request.is_choice_scoring_request():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="streaming is not supported for choice-scoring requests",
            )
        if request.model != service.model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown model {request.model!r}; available model is {service.model_name!r}",
            )
        if prepared.completion_request.stream:
            handle = service.submit_streaming_completion(prepared.completion_request)
            metadata = StreamResponseMetadata(
                id=handle.response_id.replace("cmpl-", "chatcmpl-", 1),
                created=handle.created,
                model=service.model_name,
            )

            async def _iter_chat_stream():
                state = new_chat_completion_stream_state()
                yield build_chat_completion_stream_role_chunk(metadata)
                while True:
                    item = await _next_stream_item(handle.token_queue)
                    if isinstance(item, bytes):
                        yield item
                        continue
                    if item is None:
                        break
                    for chunk in build_chat_completion_stream_token_chunks(
                        request,
                        prepared,
                        metadata,
                        state,
                        item,
                    ):
                        yield chunk
                response = await asyncio.wrap_future(handle.future)
                finish_reason = response.choices[0].finish_reason if response.choices else None
                for chunk in build_chat_completion_stream_finish_chunks(
                    prepared,
                    metadata,
                    state,
                    finish_reason=finish_reason,
                ):
                    yield chunk

            return _streaming_response(_iter_chat_stream())
        future = service.submit_completion(prepared.completion_request)
        try:
            response = await asyncio.wrap_future(future)
            return build_chat_completion_response(request, prepared, response)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised through integration
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return app


__all__ = ["create_app"]
