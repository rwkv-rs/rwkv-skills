from __future__ import annotations

"""FastAPI app for the standalone RWKV infer service."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status

from .api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)
from .service import InferenceService


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

    @app.post("/v1/completions", dependencies=[Depends(_authorize)], response_model=CompletionResponse)
    async def completions(request: CompletionRequest) -> CompletionResponse:
        if request.stream:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="streaming is not supported")
        if request.model != service.model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown model {request.model!r}; available model is {service.model_name!r}",
            )
        future = service.submit_completion(request)
        try:
            return await asyncio.wrap_future(future)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised through integration
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(_authorize)],
        response_model=ChatCompletionResponse,
    )
    async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            completion_request = request.to_completion_request()
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        if completion_request.stream:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="streaming is not supported")
        if request.model != service.model_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown model {request.model!r}; available model is {service.model_name!r}",
            )
        future = service.submit_completion(completion_request)
        try:
            response = await asyncio.wrap_future(future)
            return ChatCompletionResponse.from_completion_response(response)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - exercised through integration
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return app


__all__ = ["create_app"]
