from __future__ import annotations

"""FastAPI app for the standalone RWKV infer service."""

import asyncio
from contextlib import asynccontextmanager

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
    build_chat_completion_stream_responses,
    build_completion_stream_responses,
    prepare_chat_completion_request,
)
from .service import InferenceService
from .sse import iter_sse_payloads


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

    def _streaming_response(payloads) -> StreamingResponse:
        return StreamingResponse(
            iter_sse_payloads((*payloads, "[DONE]")),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
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
        future = service.submit_completion(request)
        try:
            response = await asyncio.wrap_future(future)
            if request.stream:
                return _streaming_response(build_completion_stream_responses(response))
            return response
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
        future = service.submit_completion(prepared.completion_request)
        try:
            response = await asyncio.wrap_future(future)
            if prepared.completion_request.stream:
                return _streaming_response(
                    build_chat_completion_stream_responses(request, prepared, response)
                )
            return build_chat_completion_response(request, prepared, response)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised through integration
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return app


__all__ = ["create_app"]
